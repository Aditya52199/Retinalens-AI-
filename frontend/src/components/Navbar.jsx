import React from 'react';
import { Eye, Activity, Cpu, ShieldCheck } from 'lucide-react';

export default function Navbar({ modelType, onModelChange, systemHealth }) {
  return (
    <header className="navbar glass-panel">
      <div className="brand-group">
        <div className="brand-icon-box">
          <Eye size={26} strokeWidth={2.2} />
        </div>
        <div>
          <div className="brand-title">RetinaLens AI</div>
          <div className="brand-tagline">Autonomous Diabetic Retinopathy Diagnostic & Explainability Workstation</div>
        </div>
      </div>

      <div className="nav-controls">
        <div className="model-toggle-group">
          <button
            className={`model-toggle-btn ${modelType === 'torch' ? 'active' : ''}`}
            onClick={() => onModelChange('torch')}
            title="PyTorch Deep CNN with Grad-CAM++ Attention"
          >
            <Activity size={14} />
            <span>PyTorch CNN (Grad-CAM)</span>
          </button>
          <button
            className={`model-toggle-btn ${modelType === 'sklearn' ? 'active' : ''}`}
            onClick={() => onModelChange('sklearn')}
            title="Scikit-Learn Baseline with Spatial Feature-Attribution"
          >
            <Cpu size={14} />
            <span>Scikit-Learn (RF)</span>
          </button>
        </div>

        <div className="status-badge">
          <span className="status-dot"></span>
          <span>
            {systemHealth?.status === 'online' ? (
              systemHealth.torch_device === 'cuda' ? 'GPU Accelerated (CUDA)' : 'System Ready (CPU)'
            ) : (
              'Connecting...'
            )}
          </span>
        </div>
      </div>
    </header>
  );
}
