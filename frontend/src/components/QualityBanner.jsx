import React from 'react';
import { CheckCircle, AlertTriangle, AlertOctagon, ChevronRight, FlaskConical, Lock } from 'lucide-react';

export default function QualityBanner({ quality, onProceedToGrading, grading }) {
  if (!quality) return null;

  const isAccepted = quality.status === 'accepted';
  const isBorderline = quality.status === 'borderline';
  const isRejected = quality.status === 'rejected';
  const canGrade = !isRejected;

  const badgeStyle = isAccepted
    ? { border: 'rgba(16, 185, 129, 0.4)', bg: 'rgba(16, 185, 129, 0.15)', text: '#10b981', label: 'Verified Gradeable' }
    : isBorderline
    ? { border: 'rgba(245, 158, 11, 0.4)', bg: 'rgba(245, 158, 11, 0.15)', text: '#f59e0b', label: 'Borderline Quality' }
    : { border: 'rgba(239, 68, 68, 0.4)', bg: 'rgba(239, 68, 68, 0.15)', text: '#ef4444', label: 'Recapture Required' };

  return (
    <div
      className="glass-panel"
      style={{
        padding: '18px 20px',
        marginBottom: 20,
        border: `1px solid ${badgeStyle.border}`,
        background: badgeStyle.bg,
      }}
    >
      {/* Step label */}
      <div
        style={{
          fontSize: 10,
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.12em',
          color: '#64748b',
          marginBottom: 10,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <span
          style={{
            background: 'rgba(6, 182, 212, 0.2)',
            border: '1px solid rgba(6, 182, 212, 0.4)',
            color: '#06b6d4',
            borderRadius: 4,
            padding: '1px 7px',
          }}
        >
          Step 1 of 2
        </span>
        Image Quality Gate
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
        {/* Left: Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {isAccepted ? (
            <CheckCircle size={20} color="#10b981" />
          ) : isBorderline ? (
            <AlertTriangle size={20} color="#f59e0b" />
          ) : (
            <AlertOctagon size={20} color="#ef4444" />
          )}
          <div>
            <span style={{ fontWeight: 700, fontSize: 14, color: badgeStyle.text, marginRight: 8 }}>
              {badgeStyle.label}
            </span>
            <span style={{ fontSize: 12, color: '#cbd5e1' }}>
              Quality Index: <strong>{quality.score}</strong> / 1.00
            </span>
          </div>
        </div>

        {/* Right: Metrics */}
        <div style={{ display: 'flex', gap: 10, fontSize: 11, color: '#94a3b8', flexWrap: 'wrap' }}>
          <span>Focus: <strong style={{ color: '#fff' }}>{quality.metrics?.focus_var}</strong></span>
          <span>Illum CV: <strong style={{ color: '#fff' }}>{quality.metrics?.illumination_cv}</strong></span>
          <span>Contrast: <strong style={{ color: '#fff' }}>{quality.metrics?.contrast_std}</strong></span>
          <span>FOV: <strong style={{ color: '#fff' }}>{quality.metrics?.fov_area_ratio}</strong></span>
        </div>
      </div>

      {quality.reasons && quality.reasons.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 12, color: '#fca5a5' }}>
          <strong>Flagged:</strong> {quality.reasons.join(', ')}
        </div>
      )}

      {/* Proceed to Grading CTA — only shown when grading has not yet run */}
      {onProceedToGrading && !grading && (
        <div style={{ marginTop: 14 }}>
          {canGrade ? (
            <button
              onClick={onProceedToGrading}
              className="btn-primary"
              style={{
                width: '100%',
                justifyContent: 'center',
                background: isAccepted
                  ? 'linear-gradient(135deg, rgba(6,182,212,0.25), rgba(16,185,129,0.25))'
                  : 'linear-gradient(135deg, rgba(245,158,11,0.25), rgba(239,100,30,0.25))',
                border: isAccepted ? '1px solid rgba(6,182,212,0.6)' : '1px solid rgba(245,158,11,0.6)',
                boxShadow: isAccepted
                  ? '0 0 18px rgba(6,182,212,0.25)'
                  : '0 0 18px rgba(245,158,11,0.2)',
                animation: 'pulse-cta 2.2s ease-in-out infinite',
              }}
            >
              <FlaskConical size={16} />
              <span>
                {isAccepted
                  ? 'Run Full Clinical Benchmark Grading →'
                  : 'Run Grading (Borderline Quality — Proceed with Caution) →'}
              </span>
            </button>
          ) : (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '12px 16px',
                borderRadius: 10,
                background: 'rgba(239, 68, 68, 0.08)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                color: '#fca5a5',
                fontSize: 13,
                fontWeight: 600,
              }}
            >
              <Lock size={15} color="#ef4444" />
              Benchmark grading blocked — image quality too low. Please recapture or upload a clearer fundus image.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
