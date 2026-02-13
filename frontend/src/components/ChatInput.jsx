import React, { useState, useRef, useEffect } from 'react';
import {
    FaPaperclip, FaBook, FaSlash,
    FaArrowRotateRight, FaChevronDown, FaCheck,
    FaXmark, FaFile
} from 'react-icons/fa6';
import { IoSend } from 'react-icons/io5';
import './ChatInput.css';

const LLM_MODELS = [
    { id: 'gemini', name: 'Gemini 2.5 Pro', color: '#4285F4' },
    { id: 'gpt', name: 'GPT-5', color: '#10A37F' },
    { id: 'claude', name: 'Claude 4.5 Sonnet', color: '#D97706' },
    { id: 'deepseek', name: 'DeepSeek-R1', color: '#6366F1' },
];

const ACCEPTED_TYPES = '.pdf,.docx,.txt';

const OCR_ENGINES = [
    { id: 'gemini_flash', label: 'Gemini Flash' },
    { id: 'paddle', label: 'PaddleOCR' },
    { id: 'deepseek', label: 'DeepSeek-OCR' },
];

const ChatInput = ({ onSend }) => {
    const [message, setMessage] = useState('');
    const [selectedModel, setSelectedModel] = useState(LLM_MODELS[0]);
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [files, setFiles] = useState([]);
    const [templateFiles, setTemplateFiles] = useState([]);
    const [ocrEngine, setOcrEngine] = useState(OCR_ENGINES[0].id);
    const dropdownRef = useRef(null);
    const fileInputRef = useRef(null);
    const templateInputRef = useRef(null);

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
                setDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleSend = () => {
        if (message.trim() || files.length > 0) {
            if (onSend) onSend(message, selectedModel, files, ocrEngine, templateFiles);
            setMessage('');
            setFiles([]);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleSelectModel = (model) => {
        setSelectedModel(model);
        setDropdownOpen(false);
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

    const removeTemplate = (index) => {
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
                                <div className="dropdown-header">Selecionar Modelo</div>
                                {LLM_MODELS.map((model) => (
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
            {(files.length > 0 || templateFiles.length > 0) && (
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
                    {templateFiles.map((file, idx) => (
                        <div key={`tpl-${file.name}-${idx}`} className="file-chip template-chip">
                            <FaBook size={12} className="file-chip-icon" />
                            <span className="file-chip-tag">Modelo</span>
                            <span className="file-chip-name">{file.name}</span>
                            <span className="file-chip-size">{formatSize(file.size)}</span>
                            <button
                                className="file-chip-remove"
                                onClick={() => removeTemplate(idx)}
                                aria-label={`Remover modelo ${file.name}`}
                            >
                                <FaXmark size={10} />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Input Area */}
            <div className="chat-input-box">
                <textarea
                    className="chat-textarea"
                    placeholder="Insira o seu prompt aqui. @ para modelos, / para prompts"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={3}
                />
                <button
                    className={`send-btn ${(message.trim() || files.length > 0) ? 'active' : ''}`}
                    onClick={handleSend}
                    aria-label="Enviar"
                >
                    <IoSend size={14} />
                </button>
            </div>

            {/* OCR Engine Selector */}
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
            </div>
        </div>
    );
};

export default ChatInput;

