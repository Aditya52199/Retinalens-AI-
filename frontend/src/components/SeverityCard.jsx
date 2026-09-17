import React from 'react';
import { AlertCircle, Clock, HeartPulse, ChevronRight } from 'lucide-react';

export default function SeverityCard({ grading }) {
  if (!grading || grading.severity_level === null || grading.severity_level === undefined) {
    return null;
  }

  const level = grading.severity_level;
  const levelClass = `level-${level}`;

  const ICDR_LABELS = {
    0: { label: 'No apparent retinopathy', short: 'No DR', color: '#10b981' },
    1: { label: 'Mild non-proliferative DR', short: 'Mild NPDR', color: '#0ea5e9' },
    2: { label: 'Moderate non-proliferative DR', short: 'Moderate NPDR', color: '#f59e0b' },
    3: { label: 'Severe non-proliferative DR', short: 'Severe NPDR', color: '#ef4444' },
    4: { label: 'Proliferative diabetic retinopathy', short: 'PDR', color: '#a855f7' },
  };

  const info = ICDR_LABELS[level] || { label: grading.severity_label, short: grading.icdr_short, color: '#fff' };
  const riskVal = Math.min(Math.max(grading.referable_probability || 0, 0), 1);
  const cutoff = grading.referable_threshold || 0.5;

  return (
    <div className="glass-panel" style={{ padding: 24, marginBottom: 20 }}>
      {/* Severity Banner */}
      <div className={`severity-banner ${levelClass}`}>
        <div>
          <div className="severity-badge-text" style={{ color: info.color }}>
            ICDR Level {level}: {info.short}
          </div>
          <div style={{ fontSize: 13, color: '#e2e8f0', marginTop: 4 }}>
            {grading.severity_label || info.label}
          </div>
        </div>

        <span className={`referable-tag ${grading.referable_dr ? 'referable-yes' : 'referable-no'}`}>
          {grading.referable_dr ? 'REFERABLE DR (Level 2+)' : 'NON-REFERABLE'}
        </span>
      </div>

      {/* Referable Risk Bar */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
          <span style={{ color: '#94a3b8', fontWeight: 600 }}>Referable Progression Risk:</span>
          <span style={{ fontWeight: 800, color: grading.referable_dr ? '#ef4444' : '#10b981' }}>
            {(riskVal * 100).toFixed(1)}% (Cutoff: {(cutoff * 100).toFixed(1)}%)
          </span>
        </div>
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{
              width: `${riskVal * 100}%`,
              background: grading.referable_dr
                ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
                : 'linear-gradient(90deg, #0ea5e9, #10b981)',
            }}
          />
        </div>
      </div>

      {/* Clinical Findings Box */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: 10,
          padding: 14,
          marginBottom: 18,
          fontSize: 13,
        }}
      >
        <div style={{ color: '#06b6d4', fontWeight: 700, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
          <AlertCircle size={14} />
          <span>Hallmark Diagnostic Criteria:</span>
        </div>
        <div style={{ color: '#cbd5e1', fontStyle: 'italic', lineHeight: 1.4 }}>
          "{grading.findings}"
        </div>
      </div>

      {/* Action and Timeline */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20, fontSize: 12 }}>
        <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: 10, borderRadius: 8, border: '1px solid rgba(255, 255, 255, 0.05)' }}>
          <div style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <Clock size={12} color="#0ea5e9" />
            <span style={{ textTransform: 'uppercase', fontWeight: 700 }}>Referral Urgency</span>
          </div>
          <div style={{ fontWeight: 600, color: '#f1f5f9' }}>{grading.referral_urgency}</div>
        </div>

        <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: 10, borderRadius: 8, border: '1px solid rgba(255, 255, 255, 0.05)' }}>
          <div style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <HeartPulse size={12} color="#10b981" />
            <span style={{ textTransform: 'uppercase', fontWeight: 700 }}>Clinical Action</span>
          </div>
          <div style={{ fontWeight: 500, color: '#f1f5f9', fontSize: 11 }}>{grading.clinical_action}</div>
        </div>
      </div>

      {/* 5-Class Probability Distribution */}
      {grading.probabilities && (
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 8 }}>
            ICDR Class Probability Distribution:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {Object.entries(grading.probabilities).map(([cls, prob]) => {
              const numCls = parseInt(cls, 10);
              const pVal = typeof prob === 'number' ? prob : parseFloat(prob);
              const isTop = numCls === level;
              const clr = ICDR_LABELS[numCls]?.color || '#06b6d4';
              return (
                <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12 }}>
                  <span style={{ width: 90, color: isTop ? '#fff' : '#94a3b8', fontWeight: isTop ? 700 : 500 }}>
                    Level {numCls} ({ICDR_LABELS[numCls]?.short}):
                  </span>
                  <div style={{ flex: 1, height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ width: `${pVal * 100}%`, height: '100%', background: clr, borderRadius: 4 }} />
                  </div>
                  <span style={{ width: 45, textAlign: 'right', fontWeight: 600, color: isTop ? clr : '#94a3b8' }}>
                    {(pVal * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
