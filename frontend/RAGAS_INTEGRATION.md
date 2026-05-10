# RAGAS Integration Guide

## What was added

### Backend (`src/evaluation/`)

| File | Purpose |
|---|---|
| `ragas_evaluator.py` | Core evaluator — computes 5 RAGAS metrics locally (no external RAGAS lib) |
| `__init__.py` | Package export |

### API (`api.py`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/evaluate` | POST | On-demand evaluation of any Q/A/context triple |
| `/api/ragas/history` | GET | Last N evaluations for history table |
| `/api/ragas/summary` | GET | Session averages for toolbar badge |
| `/api/query/stream` | SSE | Now emits `{type: "ragas", ...}` event after `done` |
| `/api/query` | POST | Non-streaming — now includes `ragas` field in response |

### Frontend components

| File | Purpose |
|---|---|
| `RagasBadge.jsx` | Inline `87% grounded · Good ↗` badge below each message |
| `RagasToolbarButton.jsx` | `● RAGAS 87%` pill button in chat toolbar |
| `RagasPanel.jsx` | Full slide-over drawer with gauges, metrics, chunk table, history |

## Integration steps in `rag_ui.jsx` / `App.jsx`

### 1. State
```jsx
const [ragasOpen,    setRagasOpen]    = useState(false);
const [latestRagas,  setLatestRagas]  = useState(null);
const [sessionAvg,   setSessionAvg]   = useState(null);
```

### 2. SSE handler — catch the `ragas` event
```js
if (event.type === 'ragas') {
  setLatestRagas(event);          // attach to the last message
  // update session avg
  fetch('/api/ragas/summary')
    .then(r => r.json())
    .then(d => setSessionAvg(d.avg_faithfulness));
}
```

### 3. Attach badge to each message
```jsx
// inside the assistant message render:
<RagasBadge ragas={msg.ragas} onOpenPanel={() => setRagasOpen(true)} />
```

### 4. Toolbar button (next to mode switcher)
```jsx
<RagasToolbarButton onClick={() => setRagasOpen(true)} sessionAvg={sessionAvg} />
```

### 5. Panel (root level)
```jsx
<RagasPanel
  open={ragasOpen}
  onClose={() => setRagasOpen(false)}
  latestRagas={latestRagas}
  latestQuestion={lastUserMessage}
  latestAnswer={lastAssistantMessage}
  latestChunks={lastRetrievedChunks}
/>
```

## Metrics explained

| Metric | Range | What it means |
|---|---|---|
| **Faithfulness** | 0–1 | % of answer sentences backed by retrieved context (shown inline as grounding score) |
| **Answer Relevance** | 0–1 | Cosine similarity between question and answer embeddings |
| **Context Precision** | 0–1 | % of retrieved chunks that actually contributed to the answer |
| **Context Recall** | 0–1 | Only when ground truth provided — coverage of GT by context |
| **Answer Similarity** | 0–1 | Only when ground truth provided — semantic similarity to GT |
| **Overall Score** | 0–1 | Weighted composite (faith 40% + relev 35% + prec 25% without GT) |

## No external dependencies

All metrics are computed with:
- `sentence-transformers` (already in requirements.txt) for embeddings
- Token-overlap NLI approximation as the grounding backbone
- Pure Python — no `ragas` PyPI package required
