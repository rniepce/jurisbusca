import React, { useState } from 'react';
import { FaFileLines, FaChevronDown, FaChevronUp, FaExpand, FaXmark } from 'react-icons/fa6';
import './OcrPreview.css';

const PREVIEW_CHARS = 250;

const OcrPreview = ({ filename, text, engine, charCount }) => {
    const [expanded, setExpanded] = useState(false);
    const [modalOpen, setModalOpen] = useState(false);

    const isLong = text && text.length > PREVIEW_CHARS;
    const previewText = isLong ? text.slice(0, PREVIEW_CHARS) + '…' : text;
    const displayCharCount = charCount || (text ? text.length : 0);

    const engineLabels = {
        marker: 'Marker (PDF→MD)',
        mistral_doc_ai: 'Mistral DocAI',
    };

    return (
        <>
            <div className="ocr-preview-card">
                <div className="ocr-preview-header">
                    <div className="ocr-preview-icon">
                        <FaFileLines size={18} />
                    </div>
                    <div className="ocr-preview-meta">
                        <span className="ocr-preview-filename">{filename}</span>
                        <span className="ocr-preview-badges">
                            <span className="ocr-badge engine">{engineLabels[engine] || engine}</span>
                            <span className="ocr-badge chars">{displayCharCount.toLocaleString()} chars</span>
                        </span>
                    </div>
                    <button
                        className="ocr-expand-full-btn"
                        onClick={() => setModalOpen(true)}
                        title="Expandir texto completo"
                    >
                        <FaExpand size={12} />
                    </button>
                </div>

                <div className={`ocr-preview-body ${expanded ? 'expanded' : ''}`}>
                    <pre className="ocr-preview-text">
                        {expanded ? text : previewText}
                    </pre>
                </div>

                {isLong && (
                    <button
                        className="ocr-toggle-btn"
                        onClick={() => setExpanded(!expanded)}
                    >
                        {expanded ? (
                            <>
                                <FaChevronUp size={10} />
                                <span>Recolher</span>
                            </>
                        ) : (
                            <>
                                <FaChevronDown size={10} />
                                <span>Clique para expandir e ler o texto</span>
                            </>
                        )}
                    </button>
                )}
            </div>

            {/* Full-Screen Modal */}
            {modalOpen && (
                <div className="ocr-modal-overlay" onClick={() => setModalOpen(false)}>
                    <div className="ocr-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="ocr-modal-header">
                            <div className="ocr-modal-title">
                                <FaFileLines size={16} />
                                <span>{filename}</span>
                                <span className="ocr-badge engine">{engineLabels[engine] || engine}</span>
                            </div>
                            <button
                                className="ocr-modal-close"
                                onClick={() => setModalOpen(false)}
                            >
                                <FaXmark size={16} />
                            </button>
                        </div>
                        <pre className="ocr-modal-text">{text}</pre>
                    </div>
                </div>
            )}
        </>
    );
};

export default OcrPreview;
