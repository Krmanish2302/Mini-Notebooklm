/**
 * RagasBadge.jsx  —  Inline grounding score badge shown below each chat message.
 *
 * Shown as small muted text right after the response completes.
 * Clicking the badge triggers the parent to open the RAGAS panel.
 *
 * Props
 * -----
 *   ragas       : { faithfulness, overall_score, grade, answer_relevance, context_precision }
 *   onOpenPanel : () => void
 */
import React from 'react';

function scoreColor(v) {
  if (v == null) return 'var(--color-text-faint)';
  if (v >= 0.75) return '#10b981';
  if (v >= 0.55) return '#f59e0b';
  return '#ef4444';
}

export default function RagasBadge({ ragas, onOpenPanel }) {
  if (!ragas) return null;

  const { faithfulness, overall_score, grade } = ragas;
  const color = scoreColor(faithfulness);

  return (
    <button
      onClick={onOpenPanel}
      title="Click to view full RAGAS evaluation"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        marginTop: 6,
        padding: '2px 9px 2px 7px',
        borderRadius: 99,
        border: `1px solid ${color}44`,
        background: `${color}0f`,
        fontSize: 11,
        color: 'var(--color-text-muted)',
        cursor: 'pointer',
        fontFamily: 'inherit',
        lineHeight: 1.5,
        transition: 'background 180ms, border-color 180ms',
      }}
    >
      {/* dot */}
      <span style={{
        width: 6, height: 6, borderRadius: '50%',
        background: color, flexShrink: 0,
      }} />
      {/* grounding score */}
      <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 600, color }}>
        {Math.round(faithfulness * 100)}%
      </span>
      <span style={{ color: 'var(--color-text-faint)', fontSize: 10 }}>grounded</span>
      {/* separator */}
      <span style={{ color: 'var(--color-divider' }}>·</span>
      {/* grade */}
      <span style={{ color: 'var(--color-text-muted)', fontSize: 10 }}>{grade}</span>
      {/* expand icon */}
      <span style={{ color: 'var(--color-text-faint)', fontSize: 10, marginLeft: 1 }}>↗</span>
    </button>
  );
}
