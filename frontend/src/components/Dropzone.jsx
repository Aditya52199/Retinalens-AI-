import React, { useRef } from 'react';
import { UploadCloud, Image as ImageIcon, CheckCircle2 } from 'lucide-react';

export default function Dropzone({ onFileSelected, previewUrl, loading, activeFileName }) {
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelected(e.target.files[0]);
    }
  };

  return (
    <div
      className={`dropzone glass-panel ${loading ? 'loading' : ''}`}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current && fileInputRef.current.click()}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleChange}
        accept="image/png,image/jpeg,image/tiff,image/bmp"
        style={{ display: 'none' }}
      />

      {loading && <div className="scanner-overlay" />}

      {previewUrl ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
          <div
            style={{
              width: 140,
              height: 140,
              borderRadius: 14,
              overflow: 'hidden',
              border: '2px solid rgba(6, 182, 212, 0.4)',
              boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
            }}
          >
            <img
              src={previewUrl}
              alt="Uploaded Fundus"
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#06b6d4', fontSize: 13, fontWeight: 600 }}>
            <CheckCircle2 size={16} />
            <span>{activeFileName || 'Fundus Image Loaded'}</span>
          </div>
          <span style={{ fontSize: 11, color: '#94a3b8' }}>Click or drop to replace</span>
        </div>
      ) : (
        <>
          <div className="drop-icon-box">
            <UploadCloud size={28} />
          </div>
          <div className="drop-title">Upload Retinal Fundus Scan</div>
          <div className="drop-subtitle">
            Drag & drop high-resolution fundus image (.png, .jpg, .tif) or click to browse
          </div>
        </>
      )}
    </div>
  );
}
