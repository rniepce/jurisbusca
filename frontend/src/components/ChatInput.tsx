import { useState, useRef, useEffect } from 'react';
import {
    FaPaperclip, FaChevronDown, FaCheck,
    FaXmark, FaFile, FaDatabase, FaTableColumns,
    FaRobot, FaScaleBalanced, FaBrain,
} from 'react-icons/fa6';
import { IoSend } from 'react-icons/io5';
import { getSlmStatus } from '../services/api';
import './ChatInput.css';

const LLM_OPTIONS = [
    { id: 'deepseek', name: 'DeepSeek V4 Pro', color: '#0891B2', deployment: 'DeepSeek-V4-Pro' },
    { id: 'gpt53', name: 'GPT-5.3', color: '#4285F4', deployment: 'gpt-5.3-chat' },
    { id: 'gemini', name: 'Gemini 3.1 Pro', color: '#34A853', deployment: 'gemini-3.1-pro' },
    { id: 'claude', name: 'Claude Sonnet 4.6', color: '#D97706', deployment: 'claude-sonnet-4-6' },
    { id: 'local', name: 'SLMs Locais', color: '#F97316', deployment: 'local-slm' },
];

const ACCEPTED_TYPES = '.pdf,.docx,.txt';

/** Engine padrão de OCR — sem mais escolha pelo usuário; backend usa Mistral DocAI. */
const DEFAULT_OCR_ENGINE = 'mistral_doc_ai';
const NO_OCR_ENGINE = 'none';
const DEFAULT_COMPRESS = true;

interface PrefillSignal {
    text: string;
    key: number; // changes to retrigger
}

const ChatInput = ({
    onSend,
    onXray,
    onFilesUploaded,
    onStyleReport: _onStyleReport,
    onModelChange,
    onOpenModelManager,
    onJurisprudenceToggle: _onJurisprudenceToggle,
    isLoading = false,
    ocrProcessing = false,
    hasContext = false,
    ragStatus = null,
    onRagStatusChange: _onRagStatusChange,
    activeAgent = null,
    jurisEnabled: _jurisEnabled = false,
    canvasOpen = false,
    canvasSelection = null,
    onCanvasToggle,
    chatTextareaRef,
    prefill,
}: any) => {
    const [message, setMessage] = useState('');
    const [selectedModel, setSelectedModel] = useState(LLM_OPTIONS[0]);
    const [dropdownOpen, setDropdownOpen] = useState(false);

    const [files, setFiles] = useState<File[]>([]);
    const [skipOcr, setSkipOcr] = useState(false);
    const [slmStatus, setSlmStatus] = useState({ available: false, mode: 'none' });
    const [jurisEnabled, setJurisEnabled] = useState(() => {
        try { return localStorage.getItem('juris_enabled') !== 'false'; } catch { return true; }
    });
    const [reasoningEnabled, setReasoningEnabled] = useState(() => {
        try { return localStorage.getItem('reasoning_enabled') === 'true'; } catch { return false; }
    });

    const toggleJuris = () => setJurisEnabled((v) => {
        const next = !v;
        try { localStorage.setItem('juris_enabled', String(next)); } catch { /* noop */ }
        return next;
    });

    const toggleReasoning = () => setReasoningEnabled((v) => {
        const next = !v;
        try { localStorage.setItem('reasoning_enabled', String(next)); } catch { /* noop */ }
        return next;
    });

    const reasoningSupported = ['deepseek', 'gpt53', 'claude'].includes(selectedModel.id);
    const dropdownRef = useRef<HTMLDivElement | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);

    // External prefill — when a quick-action chip is clicked, set the textarea text and focus it.
    useEffect(() => {
        if (prefill && typeof prefill.text === 'string') {
            setMessage(prefill.text);
            // Defer focus to next tick so the textarea has the new value mounted
            setTimeout(() => {
                const ta = chatTextareaRef?.current;
                if (ta) {
                    ta.focus();
                    // Move caret to end
                    const len = prefill.text.length;
                    ta.setSelectionRange(len, len);
                }
            }, 0);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [prefill?.key]);

    // Close model dropdown on outside click
    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
                setDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Poll SLM status only when the user is interacting with the LLM picker.
    // Avoids unnecessary 502s in the console when the model picked isn't local.
    useEffect(() => {
        const needsPolling = selectedModel.id === 'local' || dropdownOpen;
        if (!needsPolling) return;
        const checkSlm = () => getSlmStatus().then(setSlmStatus).catch(() => { });
        checkSlm();
        const interval = setInterval(checkSlm, 30000);
        return () => clearInterval(interval);
    }, [selectedModel.id, dropdownOpen]);

    const handleSend = () => {
        if (isLoading) return;
        if (message.trim() || files.length > 0 || hasContext) {
            const combinedModel = { id: selectedModel.id, name: selectedModel.name, color: selectedModel.color, llm: selectedModel.deployment };
            if (onSend) onSend(message, combinedModel, files, DEFAULT_OCR_ENGINE, [], false, {}, jurisEnabled, reasoningEnabled);
            setMessage('');
            setFiles([]);
        }
    };

    const handleXray = () => {
        if (isLoading || files.length < 2) return;
        if (onXray) onXray(files);
        setFiles([]);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleSelectModel = (model: typeof LLM_OPTIONS[number]) => {
        setSelectedModel(model);
        const combinedModel = { id: model.id, name: model.name, color: model.color, llm: model.deployment };
        if (onModelChange) onModelChange(combinedModel);
        setDropdownOpen(false);
    };

    const handleFileClick = () => fileInputRef.current?.click();

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selected: File[] = Array.from(e.target.files || []);
        if (selected.length > 0) {
            setFiles((prev) => [...prev, ...selected]);
            if (onFilesUploaded) {
                const engine = skipOcr ? NO_OCR_ENGINE : DEFAULT_OCR_ENGINE;
                onFilesUploaded(selected, engine, DEFAULT_COMPRESS).catch(() => { });
            }
        }
        e.target.value = '';
    };

    const removeFile = (index: number) => setFiles((prev) => prev.filter((_, i) => i !== index));

    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const canSend = (message.trim().length > 0 || files.length > 0 || hasContext) && !isLoading;

    return (
        <div className="chat-footer">
            {/* RAG indicator (only shown when templates indexed) */}
            {ragStatus && ragStatus.indexed_chunks > 0 && (
                <div className="template-bar">
                    <button
                        className="template-pill rag-active"
                        onClick={() => onOpenModelManager && onOpenModelManager()}
                        type="button"
                    >
                        <FaDatabase size={12} />
                        <span>{ragStatus.indexed_chunks} chunks indexados{ragStatus.has_dossier ? ' + estilo' : ''}</span>
                    </button>
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

            {/* === Input Card === */}
            <div className="input-card">
                <textarea
                    ref={chatTextareaRef}
                    className="input-card-textarea"
                    placeholder={
                        isLoading ? 'Processando...' :
                        ocrProcessing ? 'Executando OCR...' :
                        canvasOpen ? 'Digite uma instrução para editar o documento no Canvas...' :
                        'Insira o seu prompt aqui'
                    }
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={3}
                    disabled={isLoading || ocrProcessing}
                />

                <button
                    className={`input-card-send ${canSend ? 'active' : ''}`}
                    onClick={handleSend}
                    disabled={!canSend}
                    aria-label="Enviar"
                    title="Enviar (Enter)"
                >
                    <IoSend size={15} />
                </button>

                {/* Slot row — visible context controls */}
                <div className="input-card-slots">
                    {/* LLM model picker */}
                    <div className="slot-wrapper" ref={dropdownRef}>
                        <button
                            type="button"
                            className={`slot-btn slot-model ${dropdownOpen ? 'open' : ''}`}
                            onClick={() => setDropdownOpen((v) => !v)}
                            aria-haspopup="listbox"
                            aria-expanded={dropdownOpen}
                            title="Trocar modelo"
                        >
                            <span className="slot-dot" style={{ background: selectedModel.color }} />
                            <span className="slot-label">{selectedModel.name}</span>
                            <FaChevronDown size={9} className={`slot-chevron ${dropdownOpen ? 'rotated' : ''}`} />
                        </button>

                        {dropdownOpen && (
                            <div className="slot-dropdown" role="listbox">
                                <div className="slot-dropdown-header">Modelo de IA</div>
                                {LLM_OPTIONS.map((model) => (
                                    <button
                                        key={model.id}
                                        type="button"
                                        className={`slot-dropdown-item ${selectedModel.id === model.id ? 'selected' : ''}`}
                                        onClick={() => handleSelectModel(model)}
                                    >
                                        <span className="slot-dot" style={{ background: model.color }} />
                                        <span className="slot-dropdown-name">{model.name}</span>
                                        {model.id === 'local' && (
                                            <span
                                                className={`slm-status-dot ${slmStatus.available ? 'online' : 'offline'}`}
                                                title={slmStatus.available ? `SLM Conectado (${slmStatus.mode})` : 'SLM Desconectado'}
                                            />
                                        )}
                                        {selectedModel.id === model.id && <FaCheck size={11} />}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Anexar processo */}
                    <button
                        type="button"
                        className={`slot-btn ${files.length > 0 ? 'has-content' : ''}`}
                        onClick={handleFileClick}
                        title="Anexar processo — envie um PDF ou DOCX para análise"
                        disabled={ocrProcessing}
                    >
                        <FaPaperclip size={13} />
                        <span className="slot-label">
                            {files.length === 0
                                ? 'Anexar'
                                : files.length === 1
                                    ? '1 arquivo'
                                    : `${files.length} arquivos`}
                        </span>
                    </button>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept={ACCEPTED_TYPES}
                        multiple
                        onChange={handleFileChange}
                        style={{ display: 'none' }}
                    />

                    {/* Agente ativo */}
                    {activeAgent && (
                        <div
                            className="slot-btn slot-agent active"
                            title={`Agente ativo: ${activeAgent.name}`}
                        >
                            <FaRobot size={13} style={{ color: activeAgent.color || 'var(--primary-color)' }} />
                            <span className="slot-label">{activeAgent.name}</span>
                        </div>
                    )}

                    {/* Skip OCR toggle (atalho pra PDFs nativos quando o OCR está lento) */}
                    <button
                        type="button"
                        className={`slot-btn ${skipOcr ? 'active' : ''}`}
                        onClick={() => setSkipOcr((v) => !v)}
                        title={skipOcr
                            ? 'OCR desativado — usando apenas extração de texto nativo (rápido para PDFs digitais).'
                            : 'OCR ativado (Mistral Doc AI). Clique para desativar e usar só extração nativa (mais rápido em PDFs já digitais).'}
                    >
                        <span aria-hidden="true">{skipOcr ? '🚫' : '🔍'}</span>
                        <span className="slot-label">{skipOcr ? 'OCR off' : 'OCR'}</span>
                    </button>

                    {/* Jurisprudência toggle */}
                    <button
                        type="button"
                        className={`slot-btn ${jurisEnabled ? 'active' : ''}`}
                        onClick={toggleJuris}
                        title={jurisEnabled ? 'Pesquisa de jurisprudência ativa' : 'Pesquisa de jurisprudência desativada'}
                    >
                        <FaScaleBalanced size={12} />
                        <span className="slot-label">Juris</span>
                    </button>

                    {/* Raciocínio estendido */}
                    <button
                        type="button"
                        className={`slot-btn ${reasoningEnabled ? 'active' : ''}`}
                        onClick={toggleReasoning}
                        disabled={!reasoningSupported}
                        title={
                            !reasoningSupported
                                ? 'Raciocínio estendido não disponível para este modelo'
                                : reasoningEnabled
                                    ? 'Raciocínio estendido ativo'
                                    : 'Ativar raciocínio estendido'
                        }
                    >
                        <FaBrain size={12} />
                        <span className="slot-label">Raciocínio</span>
                    </button>

                    {/* Canvas toggle */}
                    <button
                        type="button"
                        className={`slot-btn ${canvasOpen ? 'active' : ''}`}
                        onClick={() => onCanvasToggle && onCanvasToggle()}
                        title={canvasOpen ? 'Fechar Canvas (editor de minuta)' : 'Abrir Canvas (editor de minuta)'}
                    >
                        <FaTableColumns size={12} />
                        <span className="slot-label">Canvas</span>
                    </button>

                    {/* X-Ray (only when 2+ files attached) */}
                    {files.length >= 2 && (
                        <button
                            type="button"
                            className="slot-btn slot-xray"
                            onClick={handleXray}
                            disabled={isLoading}
                            title="Raio-X — classificar e agrupar os processos anexados"
                        >
                            ⚡ <span className="slot-label">Raio-X</span>
                        </button>
                    )}
                </div>

                {/* File chips — only when files attached */}
                {files.length > 0 && (
                    <div className="input-card-files">
                        {files.map((file, idx) => (
                            <div key={`f-${file.name}-${idx}`} className="file-chip">
                                <FaFile size={11} className="file-chip-icon" />
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
            </div>
        </div>
    );
};

export default ChatInput;
