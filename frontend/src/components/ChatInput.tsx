import React, { useState, useRef, useEffect } from 'react';
import {
    FaPaperclip, FaBook, FaSlash,
    FaArrowRotateRight, FaChevronDown, FaCheck,
    FaXmark, FaFile, FaDatabase,
    FaBullseye, FaTableColumns
} from 'react-icons/fa6';
import { IoSend } from 'react-icons/io5';
import { getSlmStatus } from '../services/api';
import './ChatInput.css';



const LLM_OPTIONS = [
    { id: 'gpt52', name: 'GPT-5.2', color: '#4285F4', deployment: 'gpt-5.2-chat' },
    { id: 'gemini', name: 'Gemini 3.1 Pro', color: '#34A853', deployment: 'gemini-3.1-pro' },
    { id: 'claude', name: 'Claude Sonnet 4.6', color: '#D97706', deployment: 'claude-sonnet-4-6' },
    { id: 'local', name: 'SLMs Locais', color: '#F97316', deployment: 'local-slm' },
];

const ACCEPTED_TYPES = '.pdf,.docx,.txt';

const OCR_ENGINES = [
    { id: 'mistral_doc_ai', label: 'Mistral DocAI' },
    { id: 'marker', label: 'Marker (PDF→MD)' },
    { id: 'none', label: 'Sem OCR' },
    { id: 'gpt4o_mini', label: 'GPT-4o mini' },
    { id: 'paddle', label: 'PaddleOCR' },
    { id: 'deepseek', label: 'DeepSeek-OCR' },
];

const ChatInput = ({ onSend, onXray, onFilesUploaded, onStyleReport: _onStyleReport, onModelChange, onOpenModelManager, onJurisprudenceToggle: _onJurisprudenceToggle, isLoading = false, ocrProcessing = false, hasContext = false, ragStatus = null, onRagStatusChange: _onRagStatusChange, activeAgent: _activeAgent = null, jurisEnabled: _jurisEnabled = false, canvasOpen = false, canvasSelection = null, onCanvasToggle, chatTextareaRef }: any) => {
    const [message, setMessage] = useState('');
    const [selectedModel, setSelectedModel] = useState(LLM_OPTIONS[0]);
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [showOverflow, setShowOverflow] = useState(false);
    const overflowRef = useRef(null);

    const [files, setFiles] = useState([]);
    const [ocrEngine, setOcrEngine] = useState(OCR_ENGINES[0].id);
    const [compressEnabled, setCompressEnabled] = useState(false);
    const [ragEnabled, setRagEnabled] = useState(false);
    const [ragAvailable, setRagAvailable] = useState(false);
    const [_indexing, _setIndexing] = useState(false);
    const [_indexSuccess, _setIndexSuccess] = useState(false);
    const [slmStatus, setSlmStatus] = useState({ available: false, mode: 'none' });
    const dropdownRef = useRef(null);

    const fileInputRef = useRef(null);

    // Close dropdowns when clicking outside
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
                setDropdownOpen(false);
            }
            if (overflowRef.current && !overflowRef.current.contains(e.target)) {
                setShowOverflow(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Poll SLM server status every 30s
    useEffect(() => {
        const checkSlm = () => getSlmStatus().then(setSlmStatus).catch(() => { });
        checkSlm();
        const interval = setInterval(checkSlm, 30000);
        return () => clearInterval(interval);
    }, []);

    const handleSend = () => {
        if (isLoading) return;
        if (message.trim() || files.length > 0 || hasContext) {
            // Pass both engine version and LLM choice
            const combinedModel = { id: selectedModel.id, name: selectedModel.name, color: selectedModel.color, llm: selectedModel.deployment };
            if (onSend) onSend(message, combinedModel, files, ocrEngine, [], false);
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
        const combinedModel = { id: model.id, name: model.name, color: model.color, llm: model.deployment };
        if (onModelChange) {
            onModelChange(combinedModel);
        }
        setDropdownOpen(false);
    };



    const handleFileClick = () => {
        fileInputRef.current?.click();
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



    const removeFile = (index) => {
        setFiles((prev) => prev.filter((_, i) => i !== index));
    };

    // eslint-disable-next-line no-unused-vars
    const _removeTemplate = (_index) => { /* placeholder — templateFiles state not yet used */ };

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
                            aria-label="Selecionar LLM"
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
                                <div className="dropdown-header">Selecionar LLM</div>
                                {LLM_OPTIONS.map((model) => (
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
                                        {model.id === 'local' && (
                                            <span
                                                className={`slm-status-dot ${slmStatus.available ? 'online' : 'offline'}`}
                                                title={slmStatus.available
                                                    ? `SLM Conectado (${slmStatus.mode})`
                                                    : 'SLM Desconectado'}
                                            />
                                        )}
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

                    {/* Secondary toolbar buttons — hidden on mobile, shown in overflow menu */}
                    <div className="toolbar-secondary-group">
                        <button className="toolbar-btn" aria-label="Modelos de decisão" onClick={() => onOpenModelManager && onOpenModelManager()}>
                            <FaBook />
                        </button>
                        <button
                            className={`toolbar-btn canvas-toggle-btn ${canvasOpen ? 'canvas-active' : ''}`}
                            aria-label={canvasOpen ? 'Fechar Canvas' : 'Abrir Canvas'}
                            title={canvasOpen ? 'Fechar modo Canvas' : 'Abrir modo Canvas (editor de minuta)'}
                            onClick={() => onCanvasToggle && onCanvasToggle()}
                        >
                            <FaTableColumns />
                            {canvasOpen && <span className="canvas-indicator">Canvas</span>}
                        </button>
                    </div>

                    {/* Overflow menu button — visible on mobile only */}
                    <div className="toolbar-overflow-wrapper" ref={overflowRef}>
                        <button
                            className="toolbar-btn toolbar-overflow-btn"
                            aria-label="Mais opções"
                            aria-expanded={showOverflow}
                            onClick={() => setShowOverflow((v) => !v)}
                        >
                            ⋯
                        </button>
                        {showOverflow && (
                            <div className="toolbar-overflow-menu">
                                <button
                                    className="toolbar-btn"
                                    aria-label="Modelos de decisão"
                                    onClick={() => { onOpenModelManager && onOpenModelManager(); setShowOverflow(false); }}
                                >
                                    <FaBook /> <span style={{ marginLeft: 6, fontSize: 12 }}>Modelos</span>
                                </button>
                                <button
                                    className="toolbar-btn"
                                    aria-label="Prompts"
                                    onClick={() => setShowOverflow(false)}
                                >
                                    <FaSlash /> <span style={{ marginLeft: 6, fontSize: 12 }}>Prompts</span>
                                </button>
                                <button
                                    className={`toolbar-btn canvas-toggle-btn ${canvasOpen ? 'canvas-active' : ''}`}
                                    aria-label={canvasOpen ? 'Fechar Canvas' : 'Abrir Canvas'}
                                    onClick={() => { onCanvasToggle && onCanvasToggle(); setShowOverflow(false); }}
                                >
                                    <FaTableColumns /> <span style={{ marginLeft: 6, fontSize: 12 }}>Canvas</span>
                                </button>
                            </div>
                        )}
                    </div>
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

            {/* RAG status indicator (managed via ModelManagerPanel) */}
            {ragStatus && ragStatus.indexed_chunks > 0 && (
                <div className="template-bar">
                    <div className="template-pill rag-active" onClick={() => onOpenModelManager && onOpenModelManager()} style={{ cursor: 'pointer' }}>
                        <FaDatabase size={12} />
                        <span>📚 {ragStatus.indexed_chunks} chunks indexados {ragStatus.has_dossier ? '+ estilo' : ''}</span>
                    </div>
                </div>
            )}

            {/* Canvas context badge */}
            {canvasOpen && (
                <div className="canvas-context-badge">
                    <span className="canvas-badge-icon">📄</span>
                    <span className="canvas-badge-text">
                        {canvasSelection
                            ? `Refinando trecho: "${canvasSelection.slice(0, 50)}${canvasSelection.length > 50 ? '...' : ''}"`
                            : 'Canvas ativo — suas mensagens editarão o documento'}
                    </span>
                </div>
            )}

            {/* Input Area */}
            <div className="chat-input-box">
                <textarea
                    ref={chatTextareaRef}
                    className="chat-textarea"
                    placeholder={isLoading ? 'Processando...' : ocrProcessing ? 'Executando OCR...' : canvasOpen ? 'Digite uma instrução para editar o documento no Canvas...' : 'Insira o seu prompt aqui. @ para modelos, / para prompts'}
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

            <div className="ocr-bar">

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

