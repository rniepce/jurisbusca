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

const ENGINE_VERSIONS = [
    { id: 'v0', name: 'Gabinete V0', color: '#10B981' },
    { id: 'v0.5', name: 'V0.5 + Jurisprud.', color: '#8B5CF6' },
    { id: 'v1', name: 'Gabinete V1', color: '#4285F4' },
    { id: 'v2', name: 'Gabinete V2 (Agêntico)', color: '#D97706' },
];

const LLM_OPTIONS = [
    { id: 'gpt52', name: 'GPT-5.2', color: '#4285F4', deployment: 'gpt-5.2-chat' },
    { id: 'gemini', name: 'Gemini 3.1 Pro', color: '#34A853', deployment: 'gemini-3.1-pro' },
    { id: 'claude', name: 'Claude Sonnet 4.6', color: '#D97706', deployment: 'claude-sonnet-4-6' },
];

const ACCEPTED_TYPES = '.pdf,.docx,.txt';

const OCR_ENGINES = [
    { id: 'none', label: 'Sem OCR' },
    { id: 'gpt4o_mini', label: 'GPT-4o mini' },
    { id: 'paddle', label: 'PaddleOCR' },
    { id: 'deepseek', label: 'DeepSeek-OCR' },
    { id: 'mistral_doc_ai', label: 'Mistral DocAI' },
];

const ChatInput = ({ onSend, onXray, onFilesUploaded, onStyleReport, onModelChange, isLoading = false, ocrProcessing = false, hasContext = false, ragStatus = null, onRagStatusChange }) => {
    const [message, setMessage] = useState('');
    const [selectedModel, setSelectedModel] = useState(ENGINE_VERSIONS[0]);
    const [selectedLlm, setSelectedLlm] = useState(LLM_OPTIONS[0]);
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [llmDropdownOpen, setLlmDropdownOpen] = useState(false);
    const [files, setFiles] = useState([]);
    const [templateFiles, setTemplateFiles] = useState([]);
    const [ocrEngine, setOcrEngine] = useState(OCR_ENGINES[0].id);
    const [compressEnabled, setCompressEnabled] = useState(false);
    const [ragEnabled, setRagEnabled] = useState(false);
    const [ragAvailable, setRagAvailable] = useState(false);
    const [indexing, setIndexing] = useState(false);
    const [indexSuccess, setIndexSuccess] = useState(false);
    const dropdownRef = useRef(null);
    const llmDropdownRef = useRef(null);
    const fileInputRef = useRef(null);
    const templateInputRef = useRef(null);

    // Close dropdowns when clicking outside
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
                setDropdownOpen(false);
            }
            if (llmDropdownRef.current && !llmDropdownRef.current.contains(e.target)) {
                setLlmDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleSend = () => {
        if (isLoading) return;
        if (message.trim() || files.length > 0 || hasContext) {
            // Pass both engine version and LLM choice
            const combinedModel = { ...selectedModel, llm: selectedLlm.deployment };
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

    const handleSelectModel = (model) => {
        setSelectedModel(model);
        const combinedModel = { ...model, llm: selectedLlm.deployment };
        if (onModelChange) {
            onModelChange(combinedModel);
        }
        setDropdownOpen(false);
    };

    const handleSelectLlm = (llm) => {
        setSelectedLlm(llm);
        const combinedModel = { ...selectedModel, llm: llm.deployment };
        if (onModelChange) {
            onModelChange(combinedModel);
        }
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

    const handleTemplateChange = (e) => {
        const selected = Array.from(e.target.files);
        if (selected.length > 0) {
            setTemplateFiles((prev) => [...prev, ...selected]);
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
                    <div className="model-selector-wrapper" ref={dropdownRef}>
                        <button
                            className={`toolbar-btn model-selector ${dropdownOpen ? 'open' : ''}`}
                            onClick={() => setDropdownOpen(!dropdownOpen)}
                            aria-label="Selecionar modelo"
                            aria-expanded={dropdownOpen}
                        >
                            <span
                                className="model-dot"
                                style={{ background: selectedModel.color }}
                            />
                            <span className="model-label">{selectedModel.name}</span>
                            <FaChevronDown size={10} className={`chevron ${dropdownOpen ? 'rotated' : ''}`} />
                        </button>

                        {dropdownOpen && (
                            <div className="model-dropdown">
                                <div className="dropdown-header">Selecionar Versão do Gabinete</div>
                                {ENGINE_VERSIONS.map((model) => (
                                    <button
                                        key={model.id}
                                        className={`dropdown-item ${selectedModel.id === model.id ? 'selected' : ''}`}
                                        onClick={() => handleSelectModel(model)}
                                    >
                                        <span
                                            className="model-dot"
                                            style={{ background: model.color }}
                                        />
                                        <span className="dropdown-item-name">{model.name}</span>
                                        {selectedModel.id === model.id && (
                                            <FaCheck size={12} className="dropdown-check" />
                                        )}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

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
                    <button className="toolbar-btn" aria-label="Modelos de decisão" onClick={handleTemplateClick}>
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

                    {/* Show newly selected (un-indexed) templates */}
                    {templateFiles.length > 0 && (
                        <>
                            <div className="template-pill">
                                <FaBook size={12} />
                                <span>{templateFiles.length} modelo{templateFiles.length > 1 ? 's' : ''} selecionado{templateFiles.length > 1 ? 's' : ''}</span>
                                <button
                                    className="template-pill-clear"
                                    onClick={() => setTemplateFiles([])}
                                    aria-label="Remover modelos"
                                >
                                    <FaXmark size={10} />
                                </button>
                            </div>
                            <button
                                className={`style-report-btn ${indexSuccess ? 'index-success' : ''} ${indexing ? 'indexing-active' : ''}`}
                                onClick={async () => {
                                    if (indexing || indexSuccess) return;
                                    setIndexing(true);
                                    try {
                                        const result = await uploadTemplates(templateFiles);
                                        if (onRagStatusChange) {
                                            onRagStatusChange({
                                                indexed_chunks: result.indexed_chunks,
                                                has_dossier: result.has_dossier,
                                            });
                                        }
                                        // Show success state
                                        setIndexing(false);
                                        setIndexSuccess(true);
                                        setTimeout(() => {
                                            setIndexSuccess(false);
                                            setTemplateFiles([]);
                                        }, 2500);
                                    } catch (err) {
                                        console.error('Indexing failed:', err);
                                        setIndexing(false);
                                        alert(`Erro ao indexar modelos: ${err.message}`);
                                    }
                                }}
                                disabled={isLoading || indexing || indexSuccess}
                            >
                                {indexSuccess ? (
                                    <><FaCheck size={12} /><span>Modelos indexados!</span></>
                                ) : (
                                    <><FaDatabase size={12} /><span>{indexing ? 'Indexando...' : 'Indexar no RAG'}</span></>
                                )}
                            </button>
                            <button
                                className="style-report-btn"
                                onClick={() => onStyleReport && onStyleReport(templateFiles)}
                                disabled={isLoading}
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
                    placeholder={isLoading ? 'Processando...' : ocrProcessing ? 'Executando OCR...' : 'Insira o seu prompt aqui. @ para modelos, / para prompts'}
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
                    className={`send-btn ${(message.trim() || files.length > 0 || hasContext) && !isLoading ? 'active' : ''}`}
                    onClick={handleSend}
                    disabled={isLoading}
                    aria-label="Enviar"
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

