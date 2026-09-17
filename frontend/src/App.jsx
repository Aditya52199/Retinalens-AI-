import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import SamplePicker from './components/SamplePicker';
import Dropzone from './components/Dropzone';
import QualityBanner from './components/QualityBanner';
import SeverityCard from './components/SeverityCard';
import UncertaintyCard from './components/UncertaintyCard';
import InteractiveViewer from './components/InteractiveViewer';
import ClinicalCriteriaCard from './components/ClinicalCriteriaCard';
import ReportModal from './components/ReportModal';
import { AlertCircle, FileText, FlaskConical, RotateCcw } from 'lucide-react';

// Step states: 'idle' | 'testing' | 'tested' | 'grading' | 'graded'
export default function App() {
  const [modelType, setModelType] = useState('torch');
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [activeFileName, setActiveFileName] = useState('');

  const [activeColormap, setActiveColormap] = useState('turbo');
  const [alpha, setAlpha] = useState(0.55);
  const [temperature, setTemperature] = useState(1.30);

  // Two-step state
  const [step, setStep] = useState('idle'); // 'idle' | 'testing' | 'tested' | 'grading' | 'graded'
  const [qualityResult, setQualityResult] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [showReportModal, setShowReportModal] = useState(false);

  // Fetch initial health and sample cases
  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => setSystemHealth(data))
      .catch((err) => console.error('Health check failed:', err));

    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => {
        setSamples(data);
        // Pre-load Case 3 (Moderate NPDR - Referable) for instant demonstration
        if (data.length > 2) {
          handleSelectSample(data[2].id, data);
        } else if (data.length > 0) {
          handleSelectSample(data[0].id, data);
        }
      })
      .catch((err) => console.error('Failed to load samples:', err));
  }, []);

  // ─── PHASE 1: Quality Test ───────────────────────────────────────────────
  const runQualityTest = async ({ file = null, sampleId = null }) => {
    setStep('testing');
    setError(null);
    setQualityResult(null);
    setResult(null);

    try {
      const formData = new FormData();
      if (file) {
        formData.append('file', file);
      } else if (sampleId) {
        formData.append('sample_id', sampleId);
      } else {
        setStep('idle');
        return;
      }

      const res = await fetch('/api/test-quality', { method: 'POST', body: formData });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      setQualityResult(data);
      // Update preview from server's base64 if we don't have one yet
      if (!previewUrl && data.preview) {
        setPreviewUrl(`data:image/jpeg;base64,${data.preview}`);
      }
      setStep('tested');
    } catch (err) {
      console.error('Quality test error:', err);
      setError(err.message || 'Quality test failed. Please try again.');
      setStep('idle');
    }
  };

  // ─── PHASE 2: Full Benchmark Grading ─────────────────────────────────────
  const runGrading = async ({
    file = null,
    sampleId = null,
    model = modelType,
    colormap = activeColormap,
    alphaVal = alpha,
  } = {}) => {
    setStep('grading');
    setError(null);

    try {
      const formData = new FormData();
      const srcFile = file || uploadedFile;
      const srcSample = sampleId || selectedSample;

      if (srcFile) {
        formData.append('file', srcFile);
      } else if (srcSample) {
        formData.append('sample_id', srcSample);
      } else {
        setStep('tested');
        return;
      }

      formData.append('model_type', model);
      formData.append('colormap', colormap);
      formData.append('alpha', alphaVal.toString());
      formData.append('temperature', temperature.toString());

      const res = await fetch('/api/grade', { method: 'POST', body: formData });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      if (data.images?.original) {
        setPreviewUrl(`data:image/jpeg;base64,${data.images.original}`);
      }
      setStep('graded');
    } catch (err) {
      console.error('Grading error:', err);
      setError(err.message || 'Benchmark grading failed. Please try again.');
      setStep('tested');
    }
  };

  // ─── Event Handlers ───────────────────────────────────────────────────────
  const handleSelectSample = (sampleId, sampleList = samples) => {
    setSelectedSample(sampleId);
    setUploadedFile(null);
    setResult(null);
    setQualityResult(null);
    setStep('idle');
    const s = sampleList.find((x) => x.id === sampleId);
    if (s) setActiveFileName(s.name);
    runQualityTest({ sampleId });
  };

  const handleFileSelected = (file) => {
    setUploadedFile(file);
    setSelectedSample(null);
    setResult(null);
    setQualityResult(null);
    setStep('idle');
    setActiveFileName(file.name);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    runQualityTest({ file });
  };

  const handleModelChange = (newModel) => {
    setModelType(newModel);
    if (step === 'graded') {
      runGrading({ model: newModel });
    }
  };

  const handleColormapChange = (newColormap) => {
    setActiveColormap(newColormap);
    if (step === 'graded') {
      runGrading({ colormap: newColormap });
    }
  };

  const handleAlphaChange = (newAlpha) => {
    setAlpha(newAlpha);
    if (step === 'graded') {
      runGrading({ alphaVal: newAlpha });
    }
  };

  const handleReset = () => {
    setStep('idle');
    setQualityResult(null);
    setResult(null);
    setError(null);
    setUploadedFile(null);
    setSelectedSample(null);
    setPreviewUrl(null);
    setActiveFileName('');
  };

  const isTesting = step === 'testing';
  const isGrading = step === 'grading';
  const loading = isTesting || isGrading;

  return (
    <div className="app-shell">
      {/* Navbar with Brand and Model Toggle */}
      <Navbar
        modelType={modelType}
        onModelChange={handleModelChange}
        systemHealth={systemHealth}
      />

      {/* 1-Click Clinical Benchmark Samples */}
      <SamplePicker
        samples={samples}
        selectedSample={selectedSample}
        onSelectSample={handleSelectSample}
        loading={loading}
      />

      {error && (
        <div
          className="glass-panel"
          style={{
            padding: '16px 20px',
            marginBottom: 20,
            border: '1px solid rgba(239, 68, 68, 0.4)',
            background: 'rgba(239, 68, 68, 0.15)',
            color: '#fca5a5',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <AlertCircle size={20} color="#ef4444" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Dashboard Grid */}
      <div className="dashboard-grid">
        {/* Left Column */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>

          {/* Dropzone — always visible */}
          <div style={{ marginBottom: 20, position: 'relative' }}>
            <Dropzone
              onFileSelected={handleFileSelected}
              previewUrl={previewUrl}
              loading={loading}
              activeFileName={activeFileName}
            />

            {/* Step progress indicator */}
            {(step !== 'idle') && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0,
                  marginTop: 10,
                  fontSize: 11,
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                }}
              >
                {/* Step 1 */}
                <div
                  style={{
                    flex: 1,
                    padding: '6px 10px',
                    textAlign: 'center',
                    borderRadius: '8px 0 0 8px',
                    background:
                      step === 'testing'
                        ? 'rgba(6,182,212,0.2)'
                        : 'rgba(6,182,212,0.35)',
                    border: '1px solid rgba(6,182,212,0.5)',
                    color: '#06b6d4',
                    borderRight: 'none',
                  }}
                >
                  {step === 'testing' ? '⏳ Testing Quality…' : '✅ Quality Tested'}
                </div>
                {/* Arrow */}
                <div
                  style={{
                    width: 0,
                    height: 0,
                    borderTop: '16px solid transparent',
                    borderBottom: '16px solid transparent',
                    borderLeft: '10px solid rgba(6,182,212,0.5)',
                    zIndex: 1,
                  }}
                />
                {/* Step 2 */}
                <div
                  style={{
                    flex: 1,
                    padding: '6px 10px',
                    textAlign: 'center',
                    borderRadius: '0 8px 8px 0',
                    background:
                      step === 'grading'
                        ? 'rgba(168,85,247,0.2)'
                        : step === 'graded'
                        ? 'rgba(16,185,129,0.3)'
                        : 'rgba(255,255,255,0.04)',
                    border: '1px solid',
                    borderColor:
                      step === 'grading'
                        ? 'rgba(168,85,247,0.5)'
                        : step === 'graded'
                        ? 'rgba(16,185,129,0.5)'
                        : 'rgba(255,255,255,0.1)',
                    color:
                      step === 'grading'
                        ? '#a855f7'
                        : step === 'graded'
                        ? '#10b981'
                        : '#475569',
                    borderLeft: 'none',
                  }}
                >
                  {step === 'grading'
                    ? '⏳ Grading Benchmarks…'
                    : step === 'graded'
                    ? '✅ Benchmark Complete'
                    : '◦ Awaiting Grading'}
                </div>
              </div>
            )}
          </div>

          {/* Phase 1 result: Quality Banner */}
          {qualityResult && (
            <>
              <QualityBanner
                quality={qualityResult.quality}
                grading={result?.grading}
                onProceedToGrading={runGrading}
              />
            </>
          )}

          {/* Phase 2 results */}
          {result && (
            <>
              {/* ICDR Severity Grading & Referable Risk */}
              <SeverityCard grading={result.grading} />

              {/* Confidence Calibration & Uncertainty Meter */}
              <UncertaintyCard calibration={result.calibration} />

              {/* Report Export Button */}
              <div style={{ marginTop: 8, display: 'flex', gap: 10 }}>
                <button
                  className="btn-primary"
                  style={{ flex: 1, justifyContent: 'center' }}
                  onClick={() => setShowReportModal(true)}
                >
                  <FileText size={16} />
                  <span>View & Download Clinical Report (HTML)</span>
                </button>
                <button
                  className="btn-secondary"
                  style={{ padding: '10px 14px', borderRadius: 10 }}
                  title="Retest with new image"
                  onClick={handleReset}
                >
                  <RotateCcw size={15} />
                </button>
              </div>
            </>
          )}
        </div>

        {/* Right Column: Visual Evidence */}
        <div>
          {result && (
            <>
              {/* Multi-Layer Visual Evidence & Grad-CAM Heatmap Viewer */}
              <InteractiveViewer
                images={result.images}
                attention={result.attention}
                activeColormap={activeColormap}
                onColormapChange={handleColormapChange}
                alpha={alpha}
                onAlphaChange={handleAlphaChange}
                loading={isGrading}
              />

              {/* 4-2-1 Rule, DME Macular Threat & Quadrant Breakdown */}
              <ClinicalCriteriaCard criteria={result.criteria} />
            </>
          )}

          {/* Placeholder when tested but not yet graded */}
          {step === 'tested' && !result && (
            <div
              className="glass-panel"
              style={{
                padding: 32,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 14,
                minHeight: 220,
                border: '1px dashed rgba(6,182,212,0.3)',
                textAlign: 'center',
              }}
            >
              <FlaskConical size={36} color="rgba(6,182,212,0.5)" />
              <div style={{ color: '#64748b', fontSize: 14, lineHeight: 1.5 }}>
                <strong style={{ color: '#94a3b8', display: 'block', marginBottom: 4 }}>
                  Awaiting Benchmark Grading
                </strong>
                Grad-CAM heatmaps, ICDR severity analysis, clinical criteria, and the composite
                evidence panel will appear here after you run the benchmark.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Standalone HTML Report Preview Modal */}
      {showReportModal && result?.html_report && (
        <ReportModal
          htmlReport={result.html_report}
          caseName={result.case_name}
          onClose={() => setShowReportModal(false)}
        />
      )}
    </div>
  );
}
