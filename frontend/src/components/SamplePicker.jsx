import React from 'react';
import { Layers, Sparkles } from 'lucide-react';

export default function SamplePicker({ samples, selectedSample, onSelectSample, loading }) {
  const getLevelColor = (level) => {
    switch (level) {
      case 0: return { bg: 'rgba(16, 185, 129, 0.2)', color: '#10b981', label: 'No DR' };
      case 1: return { bg: 'rgba(14, 165, 233, 0.2)', color: '#0ea5e9', label: 'Mild' };
      case 2: return { bg: 'rgba(245, 158, 11, 0.2)', color: '#f59e0b', label: 'Moderate' };
      case 3: return { bg: 'rgba(239, 68, 68, 0.2)', color: '#ef4444', label: 'Severe' };
      case 4: return { bg: 'rgba(168, 85, 247, 0.2)', color: '#a855f7', label: 'PDR' };
      default: return { bg: 'rgba(255, 255, 255, 0.1)', color: '#94a3b8', label: 'Test' };
    }
  };

  return (
    <div className="sample-bar glass-panel">
      <div className="sample-bar-title">
        <Sparkles size={16} color="#06b6d4" />
        <span>Clinical Benchmark Samples (1-Click Test):</span>
      </div>

      <div className="sample-chip-list">
        {samples.map((sample) => {
          const pill = getLevelColor(sample.expected_grade);
          const isSelected = selectedSample === sample.id;
          return (
            <button
              key={sample.id}
              className={`sample-chip ${isSelected ? 'active' : ''}`}
              onClick={() => onSelectSample(sample.id)}
              disabled={loading}
              title={`Load ${sample.name}`}
            >
              <span
                className="chip-level-pill"
                style={{ background: pill.bg, color: pill.color }}
              >
                {pill.label}
              </span>
              <span>{sample.name.split(':')[1] || sample.name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
