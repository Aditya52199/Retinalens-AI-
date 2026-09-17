import React, { useState } from 'react';
import { ClipboardCheck, AlertTriangle, Eye, Shield, Check, X, ChevronDown, ChevronUp } from 'lucide-react';

export default function ClinicalCriteriaCard({ criteria }) {
  const [showTable, setShowTable] = useState(true);

  if (!criteria) return null;

  const is421Met = criteria.meets_4_2_1_rule;
  const dmeThreat = criteria.dme_threat_level || 'None';
  const isDmeHigh = dmeThreat.includes('Center-involving') || dmeThreat.includes('High');
  const isDmeMod = dmeThreat.includes('Moderate');
  const isNvDetected = criteria.neovascularization_status && criteria.neovascularization_status.includes('detected');

  return (
    <div className="glass-panel" style={{ padding: 24, marginBottom: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <ClipboardCheck size={18} color="#06b6d4" />
          <h3 style={{ fontSize: 16, fontWeight: 700 }}>Clinical Criteria Correlation</h3>
        </div>

        <button
          className="btn-secondary"
          style={{ padding: '4px 10px', fontSize: 11 }}
          onClick={() => setShowTable(!showTable)}
        >
          <span>4-Quadrant Table</span>
          {showTable ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>
      </div>

      {/* 3 Clinical Pillar Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginBottom: 18 }}>
        {/* 4-2-1 Rule Card */}
        <div
          style={{
            background: is421Met ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
            border: `1px solid ${is421Met ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
            borderRadius: 10,
            padding: 14,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontSize: 11, textTransform: 'uppercase', fontWeight: 700, color: '#94a3b8' }}>
              Severe NPDR 4-2-1 Rule
            </span>
            {is421Met ? <AlertTriangle size={14} color="#ef4444" /> : <Check size={14} color="#10b981" />}
          </div>
          <div style={{ fontSize: 13, fontWeight: 700, color: is421Met ? '#ef4444' : '#10b981', marginBottom: 6 }}>
            {is421Met ? 'Criteria Met (Severe NPDR)' : 'Negative (Not Severe)'}
          </div>
          <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.4 }}>
            <div>• Rule 4 (Hemorrhages): <strong>{criteria.rule_4_hemorrhage_quadrants}/4 quads</strong></div>
            <div>• Rule 2 (Venous beading): <strong>{criteria.rule_2_venous_beading_quadrants} quads</strong></div>
            <div>• Rule 1 (IRMA): <strong>{criteria.rule_1_irma_quadrants} quad(s)</strong></div>
          </div>
        </div>

        {/* DME Risk Card */}
        <div
          style={{
            background: isDmeHigh ? 'rgba(239, 68, 68, 0.1)' : isDmeMod ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)',
            border: `1px solid ${isDmeHigh ? 'rgba(239, 68, 68, 0.3)' : isDmeMod ? 'rgba(245, 158, 11, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
            borderRadius: 10,
            padding: 14,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontSize: 11, textTransform: 'uppercase', fontWeight: 700, color: '#94a3b8' }}>
              Macular Edema (DME)
            </span>
            <Eye size={14} color={isDmeHigh ? '#ef4444' : isDmeMod ? '#f59e0b' : '#10b981'} />
          </div>
          <div style={{ fontSize: 13, fontWeight: 700, color: isDmeHigh ? '#ef4444' : isDmeMod ? '#f59e0b' : '#10b981', marginBottom: 6 }}>
            {dmeThreat}
          </div>
          <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.4 }}>
            <div>Foveal Distance: <strong>{criteria.dme_min_distance_disc_diameters ? `${criteria.dme_min_distance_disc_diameters} DD` : 'None detected'}</strong></div>
            <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 2 }}>Center-involving threshold: &lt;1.0 DD</div>
          </div>
        </div>

        {/* Neovascularization PDR Card */}
        <div
          style={{
            background: isNvDetected ? 'rgba(168, 85, 247, 0.15)' : 'rgba(16, 185, 129, 0.1)',
            border: `1px solid ${isNvDetected ? 'rgba(168, 85, 247, 0.4)' : 'rgba(16, 185, 129, 0.3)'}`,
            borderRadius: 10,
            padding: 14,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontSize: 11, textTransform: 'uppercase', fontWeight: 700, color: '#94a3b8' }}>
              Neovascularization (PDR)
            </span>
            <Shield size={14} color={isNvDetected ? '#a855f7' : '#10b981'} />
          </div>
          <div style={{ fontSize: 13, fontWeight: 700, color: isNvDetected ? '#a855f7' : '#10b981', marginBottom: 6 }}>
            {isNvDetected ? 'Proliferative Vessels Flagged' : 'Negative (No NVD/NVE)'}
          </div>
          <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.4 }}>
            <div>Score: <strong>{(criteria.neovascularization_score || 0).toFixed(3)}</strong></div>
            <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{criteria.neovascularization_status}</div>
          </div>
        </div>
      </div>

      {/* 4-Quadrant Lesion Breakdown Table */}
      {showTable && criteria.quadrants && (
        <div style={{ overflowX: 'auto' }}>
          <table className="quadrant-table">
            <thead>
              <tr>
                <th>Quadrant</th>
                <th>Intraretinal Hemorrhages</th>
                <th>Severe Hem (≥20)</th>
                <th>Microaneurysms</th>
                <th>Exudate Ratio</th>
                <th>Vessel Density</th>
                <th>Venous Beading</th>
                <th>IRMA</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(criteria.quadrants).map(([code, q]) => (
                <tr key={code}>
                  <td style={{ fontWeight: 700, color: '#06b6d4' }}>{code} ({q.quadrant_name})</td>
                  <td>{q.hemorrhage_count}</td>
                  <td>
                    {q.severe_hemorrhage_flag ? (
                      <span style={{ color: '#ef4444', fontWeight: 700 }}>Yes (≥20)</span>
                    ) : (
                      <span style={{ color: '#94a3b8' }}>No</span>
                    )}
                  </td>
                  <td>{q.microaneurysm_count}</td>
                  <td>{(q.exudate_area_ratio * 100).toFixed(2)}%</td>
                  <td>{(q.vessel_density * 100).toFixed(2)}%</td>
                  <td>
                    {q.venous_beading_flag ? (
                      <span style={{ color: '#f59e0b', fontWeight: 700 }}>Present</span>
                    ) : (
                      <span style={{ color: '#94a3b8' }}>Normal</span>
                    )}
                  </td>
                  <td>
                    {q.irma_flag ? (
                      <span style={{ color: '#ef4444', fontWeight: 700 }}>Present</span>
                    ) : (
                      <span style={{ color: '#94a3b8' }}>Normal</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
