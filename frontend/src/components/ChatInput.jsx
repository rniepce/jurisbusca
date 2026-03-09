import React, { useState, useRef, useEffect } from 'react';
import {
    FaPaperclip, FaBook, FaSlash,
    FaArrowRotateRight, FaChevronDown, FaCheck,
    FaXmark, FaFile, FaPalette, FaDatabase,
    FaBullseye, FaMicrochip
} from 'react-icons/fa6';
import { IoSend } from 'react-icons/io5';
import { uploadTemplates, clearTemplates } from '../services/api';
import './ChatInput.css';

const LLM_OPTIONS = [
    { id: 'gpt52', name: 'GPT-5.2', color: '#4285F4', deployment: 'gpt-5.2-chat' },
    { id: 'gemini', name: 'Gemini 3.1 Pro', color: '#34A853', deployment: 'gemini-3.1-pro' },
    { id: 'claude', name: 'Claude Sonnet 4.6', color: '#D97706', deployment: 'claude-sonnet-4-6' },
];

const ACCEPTED_TYPES = '.pdf,.docx,.txt';

const OCR_ENGINES = [
    { id: 'mistral_doc_ai', label: 'Mistral DocAI' },
    { id: 'marker', label: 'Marker (PDF→MD)' },
    { id: 'none', label: 'Sem OCR' },
];

const ChatInput = ({ onSend, onXray, onFilesUploaded, onStyleReport, onModelChange, onOpenModelManager, onJurisprudenceToggle, isLoading = false, ocrProcessing = false, hasContext = false, ragStatus = null, onRagStatusChange, activeAgent = null, jurisEnabled = false }) => {
    const [message, setMessage] = useState('');
    const [selectedLlm, setSelectedLlm] = useState(LLM_OPTIONS[0]);
    const [llmDropdownOpen, setLlmDropdownOpen] = useState(false);
    const [files, setFiles] = useState([]);
    const [templateFiles, setTemplateFiles] = useState([]);
    const [ocrEngine, setOcrEngine] = useState(OCR_ENGINES[0].id);
    const [compressEnabled, setCompressEnabled] = useState(false);
    const [ragEnabled, setRagEnabled] = useState(false);
    const [ragAvailable, setRagAvailable] = useState(false);
    const [indexing, setIndexing] = useState(false);
    const [indexSuccess, setIndexSuccess] = useState(false);
    const llmDropdownRef = useRef(null);
    const fileInputRef = useRef(null);
    const templateInputRef = useRef(null);

    // Close dropdowns when clicking outside
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (llmDropdownRef.current && !llmDropdownRef.current.contains(e.target)) {
                setLlmDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Notify parent when LLM changes (parent derives engine from activeAgent)
    useEffect(() => {
        if (onModelChange && activeAgent) {
            const combinedModel = {
                id: activeAgent.engineId || 'v0',
                name: activeAgent.name,
                color: activeAgent.color,
                llm: selectedLlm.deployment,
            };
            onModelChange(combinedModel);
        }
    }, [selectedLlm, activeAgent]);

    const handleSend = () => {
        if (isLoading || ocrProcessing) return;
        if (message.trim() || files.length > 0 || hasContext) {
            // Build model from active agent + selected LLM
            const engineId = activeAgent?.engineId || 'v0';
            const combinedModel = {
                id: engineId,
                name: activeAgent?.name || 'Gabinete 1.0',
                color: activeAgent?.color || '#10B981',
                llm: selectedLlm.deployment,
            };
            if (onSend) onSend(message, combinedModel, files, ocrEngine, templateFiles, ragEnabled);
            setMessage('');
            setFiles([]);
        }
    };

    const handleXray = () => {
        if (isLoading || files.length < 2) return;
        if (onXray) onXray(files);
        setFiles([]);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleSelectLlm = (llm) => {
        setSelectedLlm(llm);
        setLlmDropdownOpen(false);
    };

    const handleFileClick = () => {
        fileInputRef.current?.click();
    };

    const handleTemplateClick = () => {
        templateInputRef.current?.click();
    };

    const handleFileChange = (e) => {
        const selected = Array.from(e.target.files);
        if (selected.length > 0) {
            setFiles((prev) => [...prev, ...selected]);
            // Trigger OCR immediately - callback returns rag_available info
            if (onFilesUploaded) {
                onFilesUploaded(selected, ocrEngine, compressEnabled).then((results) => {
                    if (results && results.some((r) => r.rag_available)) {
                        setRagAvailable(true);
                    }
                }).catch(() => { });
            }
        }
        e.target.value = '';
    };

    const handleTemplateChange = async (e) => {
        const selected = Array.from(e.target.files);
        if (selected.length > 0) {
            setTemplateFiles((prev) => [...prev, ...selected]);
            // Auto-index in RAG immediately
            setIndexing(true);
            try {
                const result = await uploadTemplates(selected);
                if (onRagStatusChange) {
                    onRagStatusChange({
                        indexed_chunks: result.indexed_chunks,
                        has_dossier: result.has_dossier,
                    });
                }
                setIndexing(false);
                setIndexSuccess(true);
                setTimeout(() => {
                    setIndexSuccess(false);
                    setTemplateFiles([]);
                }, 2500);
            } catch (err) {
                console.error('Auto-indexing failed:', err);
                setIndexing(false);
                alert(`Erro ao indexar modelos: ${err.message}`);
            }
        }
        e.target.value = '';
    };

    const removeFile = (index) => {
        setFiles((prev) => prev.filter((_, i) => i !== index));
    };

    const _removeTemplate = (index) => {
        setTemplateFiles((prev) => prev.filter((_, i) => i !== index));
    };

    const formatSize = (bytes) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    return (
        <div className="chat-footer">
            {/* Toolbar */}
            <div className="chat-toolbar">
                <div className="toolbar-left">
                    <button className="toolbar-btn" aria-label="Anexar processo" onClick={handleFileClick}>
                        <FaPaperclip />
                    </button>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept={ACCEPTED_TYPES}
                        multiple
                        onChange={handleFileChange}
                        style={{ display: 'none' }}
                    />
                    <button className="toolbar-btn" aria-label="Modelos de decisão" onClick={() => onOpenModelManager && onOpenModelManager()}>
                        <FaBook />
                    </button>
                    <input
                        ref={templateInputRef}
                        type="file"
                        accept={ACCEPTED_TYPES}
                        multiple
                        onChange={handleTemplateChange}
                        style={{ display: 'none' }}
                    />
                    <button className="toolbar-btn" aria-label="Prompts"><FaSlash /></button>
                </div>
                <div className="toolbar-right">
                    <button className="toolbar-btn" aria-label="Recarregar"><FaArrowRotateRight /></button>
                </div>
            </div>

            {/* File Chips */}
            {files.length > 0 && (
                <div className="file-chips">
                    {files.map((file, idx) => (
                        <div key={`proc-${file.name}-${idx}`} className="file-chip">
                            <FaFile size={12} className="file-chip-icon" />
                            <span className="file-chip-name">{file.name}</span>
                            <span className="file-chip-size">{formatSize(file.size)}</span>
                            <button
                                className="file-chip-remove"
                                onClick={() => removeFile(idx)}
                                aria-label={`Remover ${file.name}`}
                            >
                                <FaXmark size={10} />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Template Files — compact pill + persistent indexing */}
            {(templateFiles.length > 0 || (ragStatus && ragStatus.indexed_chunks > 0)) && (
                <div className="template-bar">
                    {/* Show persisted RAG status */}
                    {ragStatus && ragStatus.indexed_chunks > 0 && templateFiles.length === 0 && (
                        <div className="template-pill rag-active">
                            <FaDatabase size={12} />
                            <span>📚 {ragStatus.indexed_chunks} chunks indexados {ragStatus.has_dossier ? '+ estilo' : ''}</span>
                            <button
                                className="template-pill-clear"
                                onClick={async () => {
                                    try {
                                        await clearTemplates();
                                        if (onRagStatusChange) onRagStatusChange({ indexed_chunks: 0, has_dossier: false });
                                    } catch (err) {
                                        console.warn('Failed to clear templates:', err);
                                    }
                                }}
                                aria-label="Limpar modelos"
                                title="Limpar modelos indexados"
                            >
                                <FaXmark size={10} />
                            </button>
                        </div>
                    )}

                    {/* Show newly selected templates — auto-indexing in progress */}
                    {templateFiles.length > 0 && (
                        <>
                            <div className={`template-pill ${indexing ? 'indexing-active' : ''} ${indexSuccess ? 'rag-active' : ''}`}>
                                {indexing ? (
                                    <><FaDatabase size={12} /><span>Indexando {templateFiles.length} modelo{templateFiles.length > 1 ? 's' : ''}...</span></>
                                ) : indexSuccess ? (
                                    <><FaCheck size={12} /><span>Modelos indexados!</span></>
                                ) : (
                                    <>
                                        <FaBook size={12} />
                                        <span>{templateFiles.length} modelo{templateFiles.length > 1 ? 's' : ''} selecionado{templateFiles.length > 1 ? 's' : ''}</span>
                                        <button
                                            className="template-pill-clear"
                                            onClick={() => setTemplateFiles([])}
                                            aria-label="Remover modelos"
                                        >
                                            <FaXmark size={10} />
                                        </button>
                                    </>
                                )}
                            </div>
                            <button
                                className="style-report-btn"
                                onClick={() => onStyleReport && onStyleReport(templateFiles)}
                                disabled={isLoading || indexing}
                            >
                                <FaPalette size={12} />
                                <span>Relatório de Estilo</span>
                            </button>
                        </>
                    )}
                </div>
            )}

            {/* Input Area */}
            <div className="chat-input-box">
                <textarea
                    className="chat-textarea"
                    placeholder={ocrProcessing ? '⏳ Aguardando OCR finalizar...' : isLoading ? 'Processando...' : 'Insira o seu prompt aqui. @ para modelos, / para prompts'}
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={3}
                    disabled={isLoading || ocrProcessing}
                />
                {files.length >= 2 && (
                    <button
                        className="xray-btn"
                        onClick={handleXray}
                        disabled={isLoading}
                        aria-label="Raio-X"
                        title="Analisar carteira (Raio-X)"
                    >
                        ⚡ Raio-X
                    </button>
                )}
                <button
                    className={`send-btn ${(message.trim() || files.length > 0 || hasContext) && !isLoading && !ocrProcessing ? 'active' : ''}`}
                    onClick={handleSend}
                    disabled={isLoading || ocrProcessing}
                    aria-label={ocrProcessing ? 'Aguardando OCR...' : 'Enviar'}
                    title={ocrProcessing ? 'Aguarde o OCR finalizar antes de enviar' : 'Enviar mensagem'}
                >
                    <IoSend size={14} />
                </button>
            </div>

            {/* OCR Engine Selector + LLM Selector + Compress Toggle */}
            <div className="ocr-bar">
                {/* LLM Selector */}
                <div className="llm-selector-wrapper" ref={llmDropdownRef}>
                    <button
                        className={`llm-selector-btn ${llmDropdownOpen ? 'open' : ''}`}
                        onClick={() => setLlmDropdownOpen(!llmDropdownOpen)}
                        title="Selecionar LLM"
                    >
                        <FaMicrochip size={10} />
                        <span
                            className="llm-dot"
                            style={{ background: selectedLlm.color }}
                        />
                        <span className="llm-name">{selectedLlm.name}</span>
                        <FaChevronDown size={8} className={`llm-chevron ${llmDropdownOpen ? 'rotated' : ''}`} />
                    </button>
                    {llmDropdownOpen && (
                        <div className="llm-dropdown">
                            <div className="llm-dropdown-header">Selecionar LLM</div>
                            {LLM_OPTIONS.map((llm) => (
                                <button
                                    key={llm.id}
                                    className={`llm-dropdown-item ${selectedLlm.id === llm.id ? 'selected' : ''}`}
                                    onClick={() => handleSelectLlm(llm)}
                                >
                                    <span className="llm-dot" style={{ background: llm.color }} />
                                    <span>{llm.name}</span>
                                    {selectedLlm.id === llm.id && <FaCheck size={10} className="llm-check" />}
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <span className="ocr-separator" />

                <span className="ocr-label">OCR:</span>
                {OCR_ENGINES.map((engine) => (
                    <button
                        key={engine.id}
                        className={`ocr-option ${ocrEngine === engine.id ? 'active' : ''}`}
                        onClick={() => setOcrEngine(engine.id)}
                    >
                        {engine.label}
                    </button>
                ))}
                <span className="ocr-separator" />
                <label className="compress-toggle" title="Comprimir PDF para otimizar análise (reduz imagens, preserva texto)">
                    <input
                        type="checkbox"
                        checked={compressEnabled}
                        onChange={(e) => setCompressEnabled(e.target.checked)}
                    />
                    <span className="compress-label">📦 Comprimir</span>
                </label>
                {ragAvailable && (
                    <>
                        <span className="ocr-separator" />
                        <label className={`compress-toggle rag-toggle ${ragEnabled ? 'rag-active' : ''}`} title="RAG: enviar apenas trechos relevantes ao LLM (economiza tokens e melhora precisão)">
                            <input
                                type="checkbox"
                                checked={ragEnabled}
                                onChange={(e) => setRagEnabled(e.target.checked)}
                            />
                            <span className="compress-label">🎯 RAG Processo</span>
                        </label>
                    </>
                )}
            </div>
        </div>
    );
};

export default ChatInput;

