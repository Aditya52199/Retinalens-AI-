import React, { useState } from 'react';
import { Eye, Sliders, Layers, Maximize2, Sparkles, MapPin } from 'lucide-react';

export default function InteractiveViewer({
  images,
  attention,
  activeColormap,
  onColormapChange,
  alpha,
  onAlphaChange,
  onReanalyzeWithSettings,
  loading,
}) {
  const [activeLayer, setActiveLayer] = useState('attention_overlay');

  if (!images) {
    return (
      <div className="glass-panel" style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>
        <Eye size={36} style={{ marginBottom: 12, opacity: 0.5 }} />
        <div>Upload or select a retinal scan to view visual attention & diagnostic layers.</div>
      </div>
    );
  }

  const getActiveImageSrc = () => {
    switch (activeLayer) {
      case 'attention_overlay':
        return images.attention_overlay || images.enhanced;
      case 'attention_heatmap':
        return images.attention_heatmap || images.enhanced;
      case 'segmentation_overlay':
        return images.segmentation_overlay || images.enhanced;
      case 'composite':
        return images.composite || images.enhanced;
      case 'enhanced':
        return images.enhanced;
      case 'original':
      default:
        return images.original || images.enhanced;
    }
  };

  const coveragePct = ((attention?.coverage_ratio || 0) * 100).toFixed(1);
  const peakCount = attention?.peak_count || 0;

  return (
    <div className="glass-panel" style={{ padding: 24, marginBottom: 20 }}>
      {/* Header & Layer Selector */}
      <div className="viewer-header" style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Layers size={18} color="#06b6d4" />
          <h3 style={{ fontSize: 16, fontWeight: 700 }}>Visual Evidence & Diagnostic Layers</h3>
        </div>

        <div className="layer-tabs">
          <button
            className={`layer-tab-btn ${activeLayer === 'attention_overlay' ? 'active' : ''}`}
            onClick={() => setActiveLayer('attention_overlay')}
          >
            Grad-CAM Overlay
          </button>
          <button
            className={`layer-tab-btn ${activeLayer === 'segmentation_overlay' ? 'active' : ''}`}
            onClick={() => setActiveLayer('segmentation_overlay')}
          >
            Lesion Segmentation
          </button>
          <button
            className={`layer-tab-btn ${activeLayer === 'composite' ? 'active' : ''}`}
            onClick={() => setActiveLayer('composite')}
          >
            4-Panel Composite
          </button>
          <button
            className={`layer-tab-btn ${activeLayer === 'enhanced' ? 'active' : ''}`}
            onClick={() => setActiveLayer('enhanced')}
          >
            Enhanced
          </button>
          <button
            className={`layer-tab-btn ${activeLayer === 'original' ? 'active' : ''}`}
            onClick={() => setActiveLayer('original')}
          >
            Original
          </button>
        </div>
      </div>

      {/* Main Canvas Display */}
      <div className="viewer-canvas-box" style={{ marginBottom: 16 }}>
        <img
          src={getActiveImageSrc()}
          alt="Retinal Diagnostic Layer"
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        />

        {loading && <div className="scanner-overlay" />}
      </div>

      {/* Interactive Controls Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 14 }}>
        {/* Colormap Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
          <span style={{ color: '#94a3b8', fontWeight: 600 }}>Palette:</span>
          {['turbo', 'jet', 'viridis'].map((cm) => (
            <button
              key={cm}
              className={`btn-secondary ${activeColormap === cm ? 'active' : ''}`}
              style={{
                padding: '4px 10px',
                fontSize: 11,
                textTransform: 'capitalize',
                borderColor: activeColormap === cm ? '#06b6d4' : 'rgba(255,255,255,0.1)',
                background: activeColormap === cm ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
              }}
              onClick={() => onColormapChange(cm)}
              disabled={loading}
            >
              {cm}
            </button>
          ))}
        </div>

        {/* Blending Alpha Slider */}
        <div className="slider-group">
          <Sliders size={14} color="#06b6d4" />
          <span>Heatmap Alpha: <strong>{Math.round(alpha * 100)}%</strong></span>
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={alpha}
            onChange={(e) => onAlphaChange(parseFloat(e.target.value))}
            className="slider-input"
            disabled={loading}
          />
        </div>

        {/* Attention Metrics Badges */}
        <div style={{ display: 'flex', gap: 12, fontSize: 12 }}>
          <span style={{ color: '#94a3b8' }}>
            Focus Coverage: <strong style={{ color: '#06b6d4' }}>{coveragePct}%</strong>
          </span>
          <span style={{ color: '#94a3b8' }}>
            Focal Clusters: <strong style={{ color: '#a855f7' }}>{peakCount} peaks</strong>
          </span>
        </div>
      </div>
    </div>
  );
}
