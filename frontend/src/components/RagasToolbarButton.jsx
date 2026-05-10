/**
 * RagasToolbarButton.jsx  —  "RAGAS" button in the chat toolbar.
 *
 * Shows a subtle pill button.  If a session summary is available it shows
 * the session-average faithfulness next to the label.
 *
 * Props
 * -----
 *   onClick      : () => void
 *   sessionAvg   : number | null   — avg faithfulness this session (0-1)
 */
import React from 'react';

export default function RagasToolbarButton({ onClick, sessionAvg }) {
  const hasData = sessionAvg != null;
  const color   = !hasData ? 'var(--color-text-muted)'
                : sessionAvg >= 0.75 ? '#10b981'
                : sessionAvg >= 0.55 ? '#f59e0b'
                : '#ef4444';
  return (
    <button
      onClick={onClick}
      title="Open RAGAS evaluation dashboard"
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 5,
        padding: '4px 11px',
        borderRadius: 99,
        border: `1px solid var(--color-border)`,
        background: 'var(--color-surface-offset)',
        fontSize: 12, fontWeight: 600,
        color: 'var(--color-text-muted)',
        cursor: 'pointer',
        fontFamily: 'inherit',
        transition: 'background 180ms, border-color 180ms',
      }}
    >
      <span style={{
        width: 7, height: 7, borderRadius: '50%',
        background: color,
      }} />
      RAGAS
      {hasData && (
        <span style={{
          fontVariantNumeric: 'tabular-nums',
          color,
          fontWeight: 700,
          fontSize: 11,
        }}>
          {Math.round(sessionAvg * 100)}%
        </span>
      )}
    </button>
  );
}
