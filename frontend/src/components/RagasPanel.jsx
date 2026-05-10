/**
 * RagasPanel.jsx  —  Full RAGAS Evaluation Dashboard
 *
 * Rendered inside a slide-over drawer triggered by the "RAGAS" button
 * in the chat toolbar.  Shows:
 *   - Session averages (4 metric gauges)
 *   - Last-evaluated detail (per-chunk contribution table)
 *   - History table (last 20 evaluations)
 *   - Re-evaluate button (fires POST /api/evaluate on the latest message)
 *
 * Props
 * -----
 *   open          : bool
 *   onClose       : () => void
 *   latestRagas   : object | null   — the ragas object from the last SSE event
 *   latestQuestion: string
 *   latestAnswer  : string
 *   latestChunks  : array
 */
import React, { useEffect, useState, useCallback } from 'react';

const API = 'http://localhost:8000/api';

// ── helpers ──────────────────────────────────────────────────────────────────

const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);

function gradeColor(grade) {
  if (grade === 'Excellent') return '#10b981';
  if (grade === 'Good')      return '#3b82f6';
  if (grade === 'Fair')      return '#f59e0b';
  return '#ef4444';
}

function ScoreGauge({ label, value, color }) {
  const radius = 28;
  const circ   = 2 * Math.PI * radius;
  const filled = value == null ? 0 : circ * value;
  return (
    <div style={{ textAlign: 'center', minWidth: 90 }}>
      <svg width={70} height={70} viewBox="0 0 70 70">
        <circle cx={35} cy={35} r={radius} fill="none"
          stroke="var(--color-surface-offset)" strokeWidth={6} />
        <circle cx={35} cy={35} r={radius} fill="none"
          stroke={color} strokeWidth={6}
          strokeDasharray={`${filled} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 35 35)"
          style={{ transition: 'stroke-dasharray 0.6s ease' }} />
        <text x={35} y={39} textAnchor="middle"
          fill="var(--color-text)" fontSize={13} fontWeight={600}>
          {value == null ? '—' : `${Math.round(value * 100)}`}
        </text>
      </svg>
      <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 2 }}>
        {label}
      </div>
    </div>
  );
}

function MetricRow({ label, value, description }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '8px 0', borderBottom: '1px solid var(--color-divider)',
    }}>
      <div>
        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--color-text)' }}>{label}</div>
        <div style={{ fontSize: 11, color: 'var(--color-text-muted)', maxWidth: 260 }}>{description}</div>
      </div>
      <div style={{
        fontVariantNumeric: 'tabular-nums',
        fontSize: 15, fontWeight: 600,
        color: value >= 0.7 ? '#10b981' : value >= 0.5 ? '#f59e0b' : '#ef4444',
        minWidth: 44, textAlign: 'right',
      }}>
        {pct(value)}
      </div>
    </div>
  );
}

// ── main panel ────────────────────────────────────────────────────────────────

export default function RagasPanel({ open, onClose, latestRagas, latestQuestion, latestAnswer, latestChunks }) {
  const [history, setHistory]       = useState([]);
  const [avg, setAvg]               = useState({});
  const [detail, setDetail]         = useState(latestRagas);
  const [reEvalLoading, setReEval]  = useState(false);
  const [activeTab, setActiveTab]   = useState('detail'); // 'detail' | 'history'

  // Sync latest ragas from parent
  useEffect(() => {
    if (latestRagas) setDetail(latestRagas);
  }, [latestRagas]);

  const fetchHistory = useCallback(async () => {
    try {
      const r = await fetch(`${API}/ragas/history?limit=20`);
      const d = await r.json();
      setHistory(d.history || []);
      setAvg(d.avg || {});
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    if (open) fetchHistory();
  }, [open, fetchHistory, latestRagas]);

  const reEvaluate = async () => {
    if (!latestQuestion || !latestAnswer) return;
    setReEval(true);
    try {
      const r = await fetch(`${API}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: latestQuestion,
          answer:   latestAnswer,
          context_chunks: latestChunks || [],
        }),
      });
      const d = await r.json();
      setDetail(d);
      fetchHistory();
    } catch { /* ignore */ } finally {
      setReEval(false);
    }
  };

  if (!open) return null;

  return (
    <>
      {/* backdrop */}
      <div onClick={onClose} style={{
        position: 'fixed', inset: 0,
        background: 'oklch(0 0 0 / 0.45)',
        zIndex: 999,
        backdropFilter: 'blur(2px)',
      }} />

      {/* drawer */}
      <div style={{
        position: 'fixed', top: 0, right: 0, bottom: 0,
        width: 'min(480px, 100vw)',
        background: 'var(--color-surface)',
        borderLeft: '1px solid var(--color-border)',
        zIndex: 1000,
        display: 'flex', flexDirection: 'column',
        overflowY: 'auto',
        boxShadow: 'var(--shadow-lg)',
      }}>

        {/* header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '16px 20px',
          borderBottom: '1px solid var(--color-divider)',
          background: 'var(--color-surface-2)',
          position: 'sticky', top: 0, zIndex: 1,
        }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--color-text)' }}>
              RAGAS Evaluation
            </div>
            <div style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>
              Faithfulness · Relevance · Precision · Recall
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={reEvaluate}
              disabled={reEvalLoading || !latestQuestion}
              style={{
                padding: '5px 12px', borderRadius: 6,
                background: 'var(--color-primary)', color: '#fff',
                fontSize: 12, fontWeight: 600, border: 'none',
                opacity: reEvalLoading ? 0.6 : 1, cursor: 'pointer',
              }}
            >
              {reEvalLoading ? 'Evaluating…' : '↺ Re-evaluate'}
            </button>
            <button
              onClick={onClose}
              style={{
                width: 30, height: 30, borderRadius: 6,
                background: 'var(--color-surface-offset)', border: 'none',
                fontSize: 18, color: 'var(--color-text-muted)', cursor: 'pointer',
              }}
            >
              ×
            </button>
          </div>
        </div>

        {/* session averages */}
        <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--color-divider)' }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-muted)',
            textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
            Session Averages
          </div>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'space-around', flexWrap: 'wrap' }}>
            <ScoreGauge label="Faithfulness"  value={avg.faithfulness}      color="#10b981" />
            <ScoreGauge label="Relevance"     value={avg.answer_relevance}  color="#3b82f6" />
            <ScoreGauge label="Precision"     value={avg.context_precision} color="#8b5cf6" />
            <ScoreGauge label="Overall"       value={avg.overall_score}     color="#f59e0b" />
          </div>
        </div>

        {/* tabs */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--color-divider)' }}>
          {['detail', 'history'].map(tab => (
            <button key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                flex: 1, padding: '10px 0', border: 'none',
                background: activeTab === tab ? 'var(--color-surface-2)' : 'transparent',
                borderBottom: activeTab === tab
                  ? '2px solid var(--color-primary)' : '2px solid transparent',
                fontSize: 13, fontWeight: activeTab === tab ? 600 : 400,
                color: activeTab === tab ? 'var(--color-primary)' : 'var(--color-text-muted)',
                cursor: 'pointer', textTransform: 'capitalize',
              }}
            >
              {tab === 'detail' ? 'Last Response' : 'History'}
            </button>
          ))}
        </div>

        <div style={{ padding: '16px 20px', flex: 1 }}>
          {/* DETAIL TAB */}
          {activeTab === 'detail' && (
            detail ? (
              <>
                {/* Grade banner */}
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '10px 14px', borderRadius: 8, marginBottom: 16,
                  background: `${gradeColor(detail.grade)}18`,
                  border: `1px solid ${gradeColor(detail.grade)}40`,
                }}>
                  <div style={{
                    width: 10, height: 10, borderRadius: '50%',
                    background: gradeColor(detail.grade), flexShrink: 0,
                  }} />
                  <div>
                    <span style={{ fontSize: 14, fontWeight: 700, color: gradeColor(detail.grade) }}>
                      {detail.grade}
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--color-text-muted)', marginLeft: 8 }}>
                      Overall {pct(detail.overall_score)}
                    </span>
                  </div>
                  <div style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--color-text-muted)' }}>
                    {detail.supported_sentences}/{detail.answer_sentences} sentences supported
                  </div>
                </div>

                {/* Metrics */}
                <MetricRow
                  label="Faithfulness (Grounding)"
                  value={detail.faithfulness}
                  description="Fraction of answer sentences supported by retrieved context"
                />
                <MetricRow
                  label="Answer Relevance"
                  value={detail.answer_relevance}
                  description="Semantic similarity between the question and the answer"
                />
                <MetricRow
                  label="Context Precision"
                  value={detail.context_precision}
                  description="Fraction of retrieved chunks that contributed to the answer"
                />
                {detail.context_recall != null && (
                  <MetricRow
                    label="Context Recall"
                    value={detail.context_recall}
                    description="Coverage of ground-truth information by retrieved context"
                  />
                )}
                {detail.answer_similarity != null && (
                  <MetricRow
                    label="Answer Similarity"
                    value={detail.answer_similarity}
                    description="Semantic similarity between generated answer and ground truth"
                  />
                )}

                {/* Per-chunk contribution table */}
                {detail.chunk_details && detail.chunk_details.length > 0 && (
                  <div style={{ marginTop: 20 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-muted)',
                      textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
                      Chunk Contribution
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ color: 'var(--color-text-muted)', textAlign: 'left' }}>
                            <th style={{ padding: '4px 8px' }}>Chunk</th>
                            <th style={{ padding: '4px 8px' }}>Citation</th>
                            <th style={{ padding: '4px 8px' }}>Supported</th>
                            <th style={{ padding: '4px 8px' }}>Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {detail.chunk_details.map((c, i) => (
                            <tr key={i} style={{
                              borderTop: '1px solid var(--color-divider)',
                              background: c.contributed
                                ? 'oklch(from var(--color-success) l c h / 0.06)'
                                : 'transparent',
                            }}>
                              <td style={{ padding: '5px 8px', fontFamily: 'monospace',
                                color: 'var(--color-text-muted)', fontSize: 11 }}>
                                {c.chunk_id}
                              </td>
                              <td style={{ padding: '5px 8px', color: 'var(--color-primary)',
                                fontWeight: 600 }}>{c.citation || '—'}</td>
                              <td style={{ padding: '5px 8px' }}>
                                <span style={{
                                  display: 'inline-flex', alignItems: 'center', gap: 4,
                                  color: c.contributed ? '#10b981' : 'var(--color-text-faint)',
                                }}>
                                  {c.contributed ? '✓' : '–'} {c.sentences_supported} sent.
                                </span>
                              </td>
                              <td style={{ padding: '5px 8px',
                                fontVariantNumeric: 'tabular-nums',
                                color: c.score > 0 ? 'var(--color-text)' : 'var(--color-text-faint)' }}>
                                {pct(c.score)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 0',
                color: 'var(--color-text-muted)', fontSize: 13 }}>
                Send a message to see RAGAS evaluation results.
              </div>
            )
          )}

          {/* HISTORY TAB */}
          {activeTab === 'history' && (
            history.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ color: 'var(--color-text-muted)', textAlign: 'left' }}>
                      <th style={{ padding: '4px 8px' }}>#</th>
                      <th style={{ padding: '4px 8px' }}>Question</th>
                      <th style={{ padding: '4px 8px' }}>Faith.</th>
                      <th style={{ padding: '4px 8px' }}>Relev.</th>
                      <th style={{ padding: '4px 8px' }}>Prec.</th>
                      <th style={{ padding: '4px 8px' }}>Grade</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((h, i) => (
                      <tr key={i} style={{ borderTop: '1px solid var(--color-divider)' }}
                        onClick={() => { setDetail(h); setActiveTab('detail'); }}
                        className="ragas-row"
                      >
                        <td style={{ padding: '5px 8px', color: 'var(--color-text-muted)' }}>
                          {i + 1}
                        </td>
                        <td style={{ padding: '5px 8px', maxWidth: 160,
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                          color: 'var(--color-text)', cursor: 'pointer' }}>
                          {h.question}
                        </td>
                        <td style={{ padding: '5px 8px', fontVariantNumeric: 'tabular-nums',
                          color: h.faithfulness >= 0.7 ? '#10b981' : '#f59e0b' }}>
                          {pct(h.faithfulness)}
                        </td>
                        <td style={{ padding: '5px 8px', fontVariantNumeric: 'tabular-nums' }}>
                          {pct(h.answer_relevance)}
                        </td>
                        <td style={{ padding: '5px 8px', fontVariantNumeric: 'tabular-nums' }}>
                          {pct(h.context_precision)}
                        </td>
                        <td style={{ padding: '5px 8px' }}>
                          <span style={{
                            padding: '2px 7px', borderRadius: 99, fontSize: 11,
                            fontWeight: 600,
                            background: `${gradeColor(h.grade)}22`,
                            color: gradeColor(h.grade),
                          }}>
                            {h.grade}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 0',
                color: 'var(--color-text-muted)', fontSize: 13 }}>
                No evaluation history yet.
              </div>
            )
          )}
        </div>
      </div>
    </>
  );
}
