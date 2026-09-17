import React from 'react';
import { Gauge, ShieldAlert, Sparkles, HelpCircle } from 'lucide-react';

export default function UncertaintyCard({ calibration }) {
  if (!calibration) return null;

  const getTierClass = (tier) => {
    if (!tier) return 'rel-moderate';
    if (tier.includes('High')) return 'rel-high';
    if (tier.includes('Moderate')) return 'rel-moderate';
    return 'rel-review';
  };

  const tierClass = getTierClass(calibration.reliability_tier);
  const confPct = ((calibration.calibrated_confidence || 0) * 100).toFixed(1);
  const marginPct = ((calibration.confidence_margin || 0) * 100).toFixed(1);
  const entropyVal = (calibration.normalized_entropy || 0).toFixed(2);

  return (
    <div className="glass-panel" style={{ padding: 24, marginBottom: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Gauge size={18} color="#06b6d4" />
          <h3 style={{ fontSize: 16, fontWeight: 700 }}>Confidence & Uncertainty Analysis</h3>
        </div>

        <span className={`reliability-pill ${tierClass}`}>
          <Sparkles size={12} />
          <span>{calibration.reliability_tier}</span>
        </span>
      </div>

      <div className="metric-grid-3">
        <div className="stat-tile">
          <span className="stat-label">Calibrated Confidence</span>
          <span className="stat-value" style={{ color: '#06b6d4' }}>{confPct}%</span>
        </div>
        <div className="stat-tile">
          <span className="stat-label">Shannon Entropy (0-1)</span>
          <span className="stat-value" style={{ color: '#a855f7' }}>{entropyVal}</span>
        </div>
        <div className="stat-tile">
          <span className="stat-label">Confidence Margin</span>
          <span className="stat-value" style={{ color: '#10b981' }}>{marginPct}%</span>
        </div>
      </div>

      <div
        style={{
          background: 'rgba(6, 182, 212, 0.08)',
          border: '1px solid rgba(6, 182, 212, 0.25)',
          borderRadius: 8,
          padding: '12px 16px',
          fontSize: 12,
          lineHeight: 1.4,
          color: '#e2e8f0',
        }}
      >
        <strong style={{ color: '#06b6d4' }}>Decision Protocol: </strong>
        {calibration.clinical_recommendation}
      </div>
    </div>
  );
}
