/**
 * Local server replacing Supabase edge functions.
 * - Auto-detects installed Ollama models and ranks by capability
 * - Selects best model per task type (coding, research, config, fast)
 * - Falls back through available models if one fails, then to mock
 * - Persists findings to local PostgreSQL
 * - Re-scans for new models every 30s
 *
 * Start: node local-server.js
 */

import express from 'express';
import cors from 'cors';
import pg from 'pg';

const { Pool } = pg;
const app = express();
const PORT = 54321;

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const DB_URL = process.env.DATABASE_URL || 'postgresql://root@/ruflo?host=/var/run/postgresql';

const pool = new Pool({ connectionString: DB_URL });

app.use(cors({ origin: '*', allowedHeaders: ['authorization', 'x-client-info', 'apikey', 'content-type'] }));
app.use(express.json({ limit: '4mb' }));

// ── Model capability registry ─────────────────────────────────────────────────
// Higher score = better quality. Checked against model name substrings.
const MODEL_SCORES = [
  // Large / flagship
  { pattern: /llama3\.[12]:70b|llama3\.1:70b/,       score: 100, tags: ['research', 'coding', 'config'] },
  { pattern: /mixtral:8x22b/,                         score: 95,  tags: ['research', 'coding', 'config'] },
  { pattern: /deepseek-r1:70b/,                       score: 93,  tags: ['research', 'coding'] },
  { pattern: /qwen2\.5:72b/,                          score: 90,  tags: ['research', 'coding', 'config'] },
  // Mid-range
  { pattern: /llama3\.[12]:(8b|3b)/,                 score: 80,  tags: ['research', 'coding', 'config'] },
  { pattern: /llama3(:latest)?$/,                     score: 78,  tags: ['research', 'config'] },
  { pattern: /mistral(:latest|:7b)?$/,                score: 75,  tags: ['research', 'config'] },
  { pattern: /deepseek-coder/,                        score: 78,  tags: ['coding'] },
  { pattern: /codellama/,                             score: 76,  tags: ['coding'] },
  { pattern: /qwen2\.5-coder/,                        score: 74,  tags: ['coding'] },
  { pattern: /qwen2\.5:7b/,                           score: 72,  tags: ['research', 'config'] },
  { pattern: /gemma2/,                                score: 70,  tags: ['research'] },
  { pattern: /phi[34]/,                               score: 65,  tags: ['research', 'fast'] },
  // Small / fast
  { pattern: /qwen2\.5:3b/,                           score: 55,  tags: ['research', 'fast'] },
  { pattern: /qwen2\.5:1\.5b/,                        score: 45,  tags: ['fast'] },
  { pattern: /phi3:mini/,                             score: 44,  tags: ['fast'] },
  { pattern: /qwen2\.5:0\.5b/,                        score: 40,  tags: ['fast'] },
  { pattern: /smollm/,                                score: 30,  tags: ['fast'] },
  { pattern: /tinyllama/,                             score: 20,  tags: ['fast'] },
];

// Task type → preferred tag priority order
const TASK_TAGS = {
  'coding':   ['coding', 'research', 'fast'],
  'research': ['research', 'coding', 'fast'],
  'config':   ['config', 'research', 'fast'],
  'fast':     ['fast', 'research', 'coding'],
  'default':  ['research', 'coding', 'fast'],
};

// Live state — updated every 30s
let availableModels = [];   // [{ name, score, tags }]
let ollamaOnline = false;

function scoreModel(name) {
  const lower = name.toLowerCase();
  for (const entry of MODEL_SCORES) {
    if (entry.pattern.test(lower)) return { score: entry.score, tags: entry.tags };
  }
  return { score: 35, tags: ['research', 'fast'] }; // unknown model — assume usable
}

async function detectModels() {
  try {
    const res = await fetch(`${OLLAMA_BASE_URL}/api/tags`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) { ollamaOnline = false; availableModels = []; return; }
    const { models = [] } = await res.json();
    ollamaOnline = true;
    availableModels = models
      .map(m => ({ name: m.name, ...scoreModel(m.name) }))
      .sort((a, b) => b.score - a.score);

    const names = availableModels.map(m => `${m.name}(${m.score})`).join(', ');
    console.log(`[models] Detected ${availableModels.length} model(s): ${names || 'none'}`);
  } catch {
    ollamaOnline = false;
    availableModels = [];
    console.log('[models] Ollama not reachable');
  }
}

// Initial scan + refresh every 30s
await detectModels();
setInterval(detectModels, 30_000);

// ── Model selector ────────────────────────────────────────────────────────────
function selectModel(taskType = 'default', exclude = []) {
  if (!ollamaOnline || availableModels.length === 0) return null;
  const priority = TASK_TAGS[taskType] || TASK_TAGS.default;

  for (const tag of priority) {
    const match = availableModels.find(m => m.tags.includes(tag) && !exclude.includes(m.name));
    if (match) return match.name;
  }
  // Fallback: any available model not excluded
  return availableModels.find(m => !exclude.includes(m.name))?.name ?? null;
}

// ── LLM call with auto-fallback ───────────────────────────────────────────────
async function callOllama(messages, toolName, toolSchema, taskType = 'default') {
  const tried = [];

  while (true) {
    const model = selectModel(taskType, tried);
    if (!model) {
      console.warn('[llm] No usable model found, returning mock');
      return { mock: true };
    }

    const body = {
      model,
      messages,
      tools: [{ type: 'function', function: { name: toolName, ...toolSchema } }],
      tool_choice: { type: 'function', function: { name: toolName } },
      stream: false,
    };

    console.log(`[llm] ${taskType} → ${model}`);

    let res;
    try {
      res = await fetch(`${OLLAMA_BASE_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(60_000),
      });
    } catch (err) {
      console.warn(`[llm] ${model} unreachable (${err.message}), trying next`);
      tried.push(model);
      continue;
    }

    if (!res.ok) {
      const text = await res.text();
      console.warn(`[llm] ${model} returned ${res.status}: ${text.slice(0, 120)}, trying next`);
      tried.push(model);
      continue;
    }

    const data = await res.json();
    const toolCall = data.choices?.[0]?.message?.tool_calls?.[0];
    if (!toolCall) {
      console.warn(`[llm] ${model} returned no tool call, trying next`);
      tried.push(model);
      continue;
    }

    try {
      return JSON.parse(toolCall.function.arguments);
    } catch {
      console.warn(`[llm] ${model} returned unparseable JSON, trying next`);
      tried.push(model);
      continue;
    }
  }
}

// ── Mock fallback ─────────────────────────────────────────────────────────────
function mockItems(stepTitle, count = 3) {
  const hint = availableModels.length === 0
    ? 'No models found. Run: ollama pull qwen2.5:0.5b'
    : `Models detected but all failed. Try: ollama pull llama3.2:3b`;
  return Array.from({ length: count }, (_, i) => ({
    id: `mock-${Date.now()}-${i}`,
    title: `[Mock] Finding ${i + 1} for: ${stepTitle}`,
    content: hint,
    source: 'Local Mock',
    confidence: 0.5,
    timestamp: new Date().toISOString(),
  }));
}

// ── GET /models — list detected models and current selection ──────────────────
app.get('/models', (_req, res) => {
  res.json({
    ollama: ollamaOnline ? 'connected' : 'not running',
    count: availableModels.length,
    selected: {
      research: selectModel('research'),
      coding:   selectModel('coding'),
      config:   selectModel('config'),
      fast:     selectModel('fast'),
    },
    models: availableModels,
  });
});

app.get('/functions/v1/models', (_req, res) => res.redirect('/models'));

// ── /functions/v1/research-step ───────────────────────────────────────────────
app.post('/functions/v1/research-step', async (req, res) => {
  const { goal, stepTitle, stepDescription, stepType, config, previousStepsData } = req.body;
  console.log('[research-step]', { goal: goal?.slice(0, 60), stepTitle, stepType });

  const taskType = /code|implement|program|develop/i.test(goal + stepTitle) ? 'coding' : 'research';

  const systemPrompt = `You are a senior research analyst. Generate substantive findings with specific data, metrics, and real-world examples. Never describe tasks — provide actual findings with numbers and citations.`;

  let previousContext = '';
  if (previousStepsData?.length) {
    previousContext = '\n\nPREVIOUS FINDINGS (extend these):\n';
    previousStepsData.forEach(s => {
      previousContext += `\n${s.stepTitle}:\n`;
      s.data?.forEach(d => { previousContext += `• ${d.title}: ${d.content}\n`; });
    });
  }

  const userPrompt = `GOAL: ${goal}\nSTEP: ${stepTitle}\nOBJECTIVE: ${stepDescription}\n${previousContext}\nGenerate 3-5 specific, data-rich findings directly relevant to this goal.`;

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
              title:      { type: 'string' },
              content:    { type: 'string' },
              source:     { type: 'string' },
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
      'generate_research_data', toolSchema, taskType
    );

    const items = result.mock
      ? mockItems(stepTitle)
      : result.items.map((item, i) => ({
          id: `${stepType}-${Date.now()}-${i}`,
          title: item.title,
          content: item.content,
          source: item.source || 'Local LLM',
          confidence: item.confidence,
          timestamp: new Date().toISOString(),
        }));

    pool.query(
      'INSERT INTO research_findings(step_type, step_title, data) VALUES($1,$2,$3)',
      [stepType, stepTitle, JSON.stringify(items)]
    ).catch(e => console.warn('[pg]', e.message));

    res.json(items);
  } catch (err) {
    console.error('[research-step]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── /functions/v1/generate-research-goal ─────────────────────────────────────
app.post('/functions/v1/generate-research-goal', async (req, res) => {
  const { category, customContext } = req.body;
  console.log('[generate-research-goal]', { category });

  const userPrompt = `Generate 3 diverse, innovative research goals for: "${category}".${customContext ? ` Context: ${customContext}` : ''} Each must be specific, actionable, and forward-thinking.`;

  const toolSchema = {
    description: 'Generate research goals',
    parameters: {
      type: 'object',
      properties: {
        goals: {
          type: 'array', minItems: 3, maxItems: 3,
          items: {
            type: 'object',
            properties: {
              title:       { type: 'string' },
              description: { type: 'string' },
              complexity:  { type: 'string', enum: ['low', 'medium', 'high'] },
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
      'generate_research_goals', toolSchema, 'fast'
    );

    const goals = result.mock
      ? [1, 2, 3].map(n => ({ title: `[Mock] ${category} Goal ${n}`, description: 'Install Ollama for real goals.', complexity: 'medium' }))
      : result.goals;

    res.json(goals);
  } catch (err) {
    console.error('[generate-research-goal]', err.message);
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

  const userPrompt = `RESEARCH GOAL: ${goal}\n\nFINDINGS:${summary}\n\nGenerate 5-7 specific, actionable next steps based on these findings.`;

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
              title:         { type: 'string' },
              description:   { type: 'string' },
              priority:      { type: 'string', enum: ['high', 'medium', 'low'] },
              category:      { type: 'string' },
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
      'generate_action_items', toolSchema, 'research'
    );

    const items = result.mock
      ? [{ title: '[Mock] Action Item', description: 'Install Ollama for real suggestions.', priority: 'medium', category: 'Research', estimatedTime: '1 hour' }]
      : result.actionItems;

    res.json(items);
  } catch (err) {
    console.error('[generate-action-items]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── /functions/v1/optimize-research-config ────────────────────────────────────
app.post('/functions/v1/optimize-research-config', async (req, res) => {
  const { preset, currentGoal } = req.body;
  console.log('[optimize-research-config]', { preset });

  const userPrompt = `Generate optimized research config for preset: "${preset}". Goal: "${currentGoal || 'general research'}". Return specific depth, source, confidence, and GOAP parameters.`;

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
                depth:         { type: 'string', enum: ['surface', 'moderate', 'deep'] },
                perspective:   { type: 'string' },
                focusAreas:    { type: 'array', items: { type: 'string' } },
                excludeTopics: { type: 'array', items: { type: 'string' } },
                timeframe:     { type: 'string' },
              },
            },
            parameters: {
              type: 'object',
              properties: {
                maxSources:     { type: 'number' },
                minConfidence:  { type: 'number' },
                maxSteps:       { type: 'number' },
                parallelAgents: { type: 'number' },
                timeout:        { type: 'number' },
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
      [{ role: 'system', content: 'You are a research workflow architect optimizing config settings.' }, { role: 'user', content: userPrompt }],
      'generate_optimized_config', toolSchema, 'config'
    );

    const payload = result.mock
      ? {
          config: { researchGuidance: { depth: 'moderate', focusAreas: [], excludeTopics: [], perspective: '', timeframe: 'recent' }, parameters: { maxSources: 5, minConfidence: 70, maxSteps: 6, parallelAgents: 2, timeout: 30 } },
          description: `Default config — ${ollamaOnline ? 'all models failed, try pulling a larger model' : 'Ollama not running'}`,
        }
      : result;

    res.json(payload);
  } catch (err) {
    console.error('[optimize-research-config]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── /functions/v1/research-api ────────────────────────────────────────────────
app.post('/functions/v1/research-api', (req, res, next) => {
  req.url = '/functions/v1/research-step';
  app._router.handle(req, res, next);
});

// ── Health ────────────────────────────────────────────────────────────────────
app.get('/health', async (_req, res) => {
  let pgOk = false;
  try { await pool.query('SELECT 1'); pgOk = true; } catch {}

  res.json({
    status: 'ok',
    ollama: ollamaOnline ? 'connected' : 'not running',
    postgres: pgOk ? 'connected' : 'error',
    modelsAvailable: availableModels.length,
    activeModels: {
      research: selectModel('research') ?? 'none',
      coding:   selectModel('coding')   ?? 'none',
      fast:     selectModel('fast')     ?? 'none',
    },
  });
});

app.listen(PORT, () => {
  console.log(`\n[local-server] Running on http://localhost:${PORT}`);
  console.log(`[local-server] Ollama:   ${OLLAMA_BASE_URL}`);
  console.log(`[local-server] Postgres: ${DB_URL}`);
  console.log(`[local-server] Models:   http://localhost:${PORT}/models`);
  console.log(`[local-server] Health:   http://localhost:${PORT}/health\n`);
});
