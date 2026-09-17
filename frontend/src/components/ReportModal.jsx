import React from 'react';
import { X, Download, Printer, FileText } from 'lucide-react';

export default function ReportModal({ htmlReport, caseName, onClose }) {
  if (!htmlReport) return null;

  const handleDownload = () => {
    const blob = new Blob([htmlReport], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${caseName || 'dr_screening'}_clinical_report.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handlePrint = () => {
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(htmlReport);
      printWindow.document.close();
      printWindow.focus();
      printWindow.print();
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <FileText size={22} color="#06b6d4" />
            <h2 style={{ fontSize: 18, fontWeight: 700 }}>Clinical Screening Report Preview</h2>
          </div>

          <div style={{ display: 'flex', gap: 10 }}>
            <button className="btn-secondary" onClick={handlePrint}>
              <Printer size={15} />
              <span>Print / PDF</span>
            </button>
            <button className="btn-primary" onClick={handleDownload}>
              <Download size={15} />
              <span>Download HTML</span>
            </button>
            <button
              className="btn-secondary"
              onClick={onClose}
              style={{ padding: '8px 12px', color: '#94a3b8' }}
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Iframe preview of the standalone HTML */}
        <div
          style={{
            background: '#ffffff',
            borderRadius: 10,
            overflow: 'hidden',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            height: '70vh',
          }}
        >
          <iframe
            srcDoc={htmlReport}
            title="Clinical Report Preview"
            style={{ width: '100%', height: '100%', border: 'none' }}
          />
        </div>
      </div>
    </div>
  );
}
