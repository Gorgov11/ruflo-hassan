/**
 * Local server replacing Supabase edge functions.
 * Routes POST /functions/v1/<name> to match Supabase SDK's invoke() format.
 * LLM calls go to Ollama (http://localhost:11434) with OpenAI-compatible API.
 * Research sessions and findings are persisted to local PostgreSQL.
 *
 * Start: node local-server.js
 * Requires: PostgreSQL running, optionally Ollama running.
 */

import express from 'express';
import cors from 'cors';
import pg from 'pg';

const { Pool } = pg;
const app = express();
const PORT = 54321;

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'qwen2.5:0.5b';
const DB_URL = process.env.DATABASE_URL || 'postgresql://root@/ruflo?host=/var/run/postgresql';

const pool = new Pool({ connectionString: DB_URL });

app.use(cors({ origin: '*', allowedHeaders: ['authorization', 'x-client-info', 'apikey', 'content-type'] }));
app.use(express.json({ limit: '4mb' }));

// ── LLM call via Ollama OpenAI-compatible API ─────────────────────────────────
async function callOllama(messages, toolName, toolSchema) {
  const body = {
    model: OLLAMA_MODEL,
    messages,
    tools: [{ type: 'function', function: { name: toolName, ...toolSchema } }],
    tool_choice: { type: 'function', function: { name: toolName } },
    stream: false,
  };

  let res;
  try {
    res = await fetch(`${OLLAMA_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    // Ollama not running — return mock data
    console.warn('[local-server] Ollama unreachable, returning mock response');
    return { mock: true };
  }

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Ollama error ${res.status}: ${text}`);
  }

  const data = await res.json();
  const toolCall = data.choices?.[0]?.message?.tool_calls?.[0];
  if (!toolCall) throw new Error('No tool call in Ollama response');
  return JSON.parse(toolCall.function.arguments);
}

// ── Mock fallback for when Ollama is not running ──────────────────────────────
function mockResearchItems(stepTitle, count = 3) {
  return Array.from({ length: count }, (_, i) => ({
    id: `mock-${Date.now()}-${i}`,
    title: `[Mock] Finding ${i + 1} for: ${stepTitle}`,
    content: `This is a placeholder finding generated because Ollama is not running. Install Ollama and pull a model (e.g. ollama pull qwen2.5:0.5b) to get real AI-generated research results.`,
    source: 'Local Mock (no LLM)',
    confidence: 0.5,
    timestamp: new Date().toISOString(),
  }));
}

// ── /functions/v1/research-step ───────────────────────────────────────────────
app.post('/functions/v1/research-step', async (req, res) => {
  const { goal, stepTitle, stepDescription, stepType, config, previousStepsData } = req.body;
  console.log('[research-step]', { goal: goal?.slice(0, 60), stepTitle, stepType });

  const systemPrompt = `You are a senior research analyst. Generate substantive research findings with specific data, metrics, and real-world examples. Do not describe tasks — provide actual findings.`;

  let previousContext = '';
  if (previousStepsData?.length) {
    previousContext = '\n\nPREVIOUS FINDINGS (build on these):\n';
    previousStepsData.forEach(s => {
      previousContext += `\n${s.stepTitle}:\n`;
      s.data?.forEach(d => { previousContext += `• ${d.title}: ${d.content}\n`; });
    });
  }

  const userPrompt = `GOAL: ${goal}\nSTEP: ${stepTitle}\nOBJECTIVE: ${stepDescription}\n${previousContext}\nGenerate 3-5 specific, data-rich research findings directly relevant to this goal.`;

  const toolSchema = {
    description: 'Generate research data items',
    parameters: {
      type: 'object',
      properties: {
        items: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              title: { type: 'string' },
              content: { type: 'string' },
              source: { type: 'string' },
              confidence: { type: 'number', minimum: 0.5, maximum: 0.95 },
            },
            required: ['title', 'content', 'source', 'confidence'],
          },
        },
      },
      required: ['items'],
    },
  };

  try {
    const result = await callOllama(
      [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }],
      'generate_research_data',
      toolSchema
    );

    const items = result.mock
      ? mockResearchItems(stepTitle)
      : result.items.map((item, i) => ({
          id: `${stepType}-${Date.now()}-${i}`,
          title: item.title,
          content: item.content,
          source: item.source || 'Local LLM',
          confidence: item.confidence,
          timestamp: new Date().toISOString(),
        }));

    // Persist to PostgreSQL (best-effort)
    pool.query(
      'INSERT INTO research_findings(step_type, step_title, data) VALUES($1,$2,$3)',
      [stepType, stepTitle, JSON.stringify(items)]
    ).catch(e => console.warn('[pg] insert finding:', e.message));

    res.json(items);
  } catch (err) {
    console.error('[research-step] error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── /functions/v1/generate-research-goal ─────────────────────────────────────
app.post('/functions/v1/generate-research-goal', async (req, res) => {
  const { category, customContext } = req.body;
  console.log('[generate-research-goal]', { category });

  const userPrompt = `Generate 3 diverse, innovative research goals for the category: "${category}".${customContext ? ` Context: ${customContext}` : ''} Each goal should be specific, actionable, and forward-thinking.`;

  const toolSchema = {
    description: 'Generate research goals',
    parameters: {
      type: 'object',
      properties: {
        goals: {
          type: 'array',
          minItems: 3,
          maxItems: 3,
          items: {
            type: 'object',
            properties: {
              title: { type: 'string' },
              description: { type: 'string' },
              complexity: { type: 'string', enum: ['low', 'medium', 'high'] },
            },
            required: ['title', 'description', 'complexity'],
          },
        },
      },
      required: ['goals'],
    },
  };

  try {
    const result = await callOllama(
      [{ role: 'system', content: 'You are a research consultant generating specific, innovative research goals.' }, { role: 'user', content: userPrompt }],
      'generate_research_goals',
      toolSchema
    );

    const goals = result.mock
      ? [
          { title: `[Mock] ${category} Research Goal 1`, description: 'Ollama not running. Install Ollama for real AI goals.', complexity: 'medium' },
          { title: `[Mock] ${category} Research Goal 2`, description: 'Ollama not running. Install Ollama for real AI goals.', complexity: 'high' },
          { title: `[Mock] ${category} Research Goal 3`, description: 'Ollama not running. Install Ollama for real AI goals.', complexity: 'low' },
        ]
      : result.goals;

    res.json(goals);
  } catch (err) {
    console.error('[generate-research-goal] error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── /functions/v1/generate-action-items ──────────────────────────────────────
app.post('/functions/v1/generate-action-items', async (req, res) => {
  const { goal, researchContext, totalSteps, totalDataPoints } = req.body;
  console.log('[generate-action-items]', { goal: goal?.slice(0, 60), totalSteps, totalDataPoints });

  let summary = '';
  researchContext?.forEach(step => {
    summary += `\n${step.stepTitle}:\n`;
    step.findings?.forEach(f => { summary += `• ${f.title}: ${f.content}\n`; });
  });

  const userPrompt = `RESEARCH GOAL: ${goal}\n\nRESEARCH SUMMARY:${summary}\n\nGenerate 5-7 specific, actionable next steps based on these research findings.`;

  const toolSchema = {
    description: 'Generate action items',
    parameters: {
      type: 'object',
      properties: {
        actionItems: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              title: { type: 'string' },
              description: { type: 'string' },
              priority: { type: 'string', enum: ['high', 'medium', 'low'] },
              category: { type: 'string' },
              estimatedTime: { type: 'string' },
            },
            required: ['title', 'description', 'priority', 'category'],
          },
        },
      },
      required: ['actionItems'],
    },
  };

  try {
    const result = await callOllama(
      [{ role: 'system', content: 'You are a research strategist generating specific, actionable next steps.' }, { role: 'user', content: userPrompt }],
      'generate_action_items',
      toolSchema
    );

    const items = result.mock
      ? [{ title: '[Mock] Action Item', description: 'Ollama not running. Install Ollama for real AI action items.', priority: 'medium', category: 'Research', estimatedTime: '1 hour' }]
      : result.actionItems;

    res.json(items);
  } catch (err) {
    console.error('[generate-action-items] error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── /functions/v1/optimize-research-config ────────────────────────────────────
app.post('/functions/v1/optimize-research-config', async (req, res) => {
  const { preset, currentGoal } = req.body;
  console.log('[optimize-research-config]', { preset, currentGoal: currentGoal?.slice(0, 60) });

  const userPrompt = `Generate optimized research configuration for preset: "${preset}". Goal: "${currentGoal || 'general research'}". Return specific parameters for depth, sources, confidence thresholds, and GOAP settings.`;

  const toolSchema = {
    description: 'Generate optimized research config',
    parameters: {
      type: 'object',
      properties: {
        config: {
          type: 'object',
          properties: {
            researchGuidance: {
              type: 'object',
              properties: {
                depth: { type: 'string', enum: ['surface', 'moderate', 'deep'] },
                perspective: { type: 'string' },
                focusAreas: { type: 'array', items: { type: 'string' } },
                excludeTopics: { type: 'array', items: { type: 'string' } },
                timeframe: { type: 'string' },
              },
            },
            parameters: {
              type: 'object',
              properties: {
                maxSources: { type: 'number' },
                minConfidence: { type: 'number' },
                maxSteps: { type: 'number' },
                parallelAgents: { type: 'number' },
                timeout: { type: 'number' },
              },
            },
          },
          required: ['researchGuidance', 'parameters'],
        },
        description: { type: 'string' },
      },
      required: ['config', 'description'],
    },
  };

  try {
    const result = await callOllama(
      [{ role: 'system', content: 'You are a research workflow architect optimizing configuration settings.' }, { role: 'user', content: userPrompt }],
      'generate_optimized_config',
      toolSchema
    );

    const payload = result.mock
      ? {
          config: { researchGuidance: { depth: 'moderate', focusAreas: [], excludeTopics: [], perspective: '', timeframe: 'recent' }, parameters: { maxSources: 5, minConfidence: 70, maxSteps: 6, parallelAgents: 2, timeout: 30 } },
          description: 'Default config (Ollama not running)',
        }
      : result;

    res.json(payload);
  } catch (err) {
    console.error('[optimize-research-config] error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── /functions/v1/research-api ────────────────────────────────────────────────
app.post('/functions/v1/research-api', async (req, res) => {
  // Delegates to research-step internally for compatibility
  req.url = '/functions/v1/research-step';
  app._router.handle(req, res, () => {});
});

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/health', async (_req, res) => {
  let ollamaOk = false;
  let pgOk = false;

  try {
    const r = await fetch(`${OLLAMA_BASE_URL}/api/tags`, { signal: AbortSignal.timeout(2000) });
    ollamaOk = r.ok;
  } catch {}

  try {
    await pool.query('SELECT 1');
    pgOk = true;
  } catch {}

  res.json({ status: 'ok', ollama: ollamaOk ? 'connected' : 'not running', postgres: pgOk ? 'connected' : 'error', model: OLLAMA_MODEL });
});

app.listen(PORT, () => {
  console.log(`\n[local-server] Running on http://localhost:${PORT}`);
  console.log(`[local-server] Ollama:   ${OLLAMA_BASE_URL} (model: ${OLLAMA_MODEL})`);
  console.log(`[local-server] Postgres: ${DB_URL}`);
  console.log(`[local-server] Health:   http://localhost:${PORT}/health\n`);
});
