import React, { useState, useEffect, useRef } from 'react';
import { FaBrain, FaXmark, FaCheck } from 'react-icons/fa6';
import { getMemory, saveMemory } from '../services/api';
import './MemoryPanel.css';

const MAX_CHARS = 2000;

const MemoryPanel = ({ isOpen, onClose }) => {
    const [content, setContent] = useState('');
    const [enabled, setEnabled] = useState(true);
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [loading, setLoading] = useState(true);
    const [dirty, setDirty] = useState(false);
    const textareaRef = useRef(null);

    // Load memory on open
    useEffect(() => {
        if (isOpen) {
            setLoading(true);
            setSaved(false);
            setDirty(false);
            getMemory().then((data) => {
                setContent(data.content || '');
                setEnabled(data.enabled !== false);
                setLoading(false);
                // Focus textarea after loading
                setTimeout(() => textareaRef.current?.focus(), 100);
            });
        }
    }, [isOpen]);

    const handleContentChange = (e) => {
        const val = e.target.value;
        if (val.length <= MAX_CHARS) {
            setContent(val);
            setDirty(true);
            setSaved(false);
        }
    };

    const handleToggle = () => {
        setEnabled((prev) => !prev);
        setDirty(true);
        setSaved(false);
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            await saveMemory(content, enabled);
            setSaved(true);
            setDirty(false);
            // Auto-close after short delay
            setTimeout(() => {
                setSaved(false);
                onClose();
            }, 1200);
        } catch (err) {
            console.error('Failed to save memory:', err);
        } finally {
            setSaving(false);
        }
    };

    const handleOverlayClick = (e) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    // Handle Escape key
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape' && isOpen) onClose();
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    const charPercent = content.length / MAX_CHARS;
    const charClass = charPercent >= 1 ? 'at-limit' : charPercent >= 0.85 ? 'near-limit' : '';

    return (
        <div className="memory-overlay" onClick={handleOverlayClick} role="dialog" aria-modal="true" aria-label="Painel de Memória">
            <div className="memory-panel" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="memory-header">
                    <div className="memory-header-left">
                        <span className="memory-header-icon">🧠</span>
                        <h2>Memória</h2>
                    </div>
                    <button className="memory-close-btn" onClick={onClose} aria-label="Fechar">
                        <FaXmark size={16} />
                    </button>
                </div>

                {/* Body */}
                <div className="memory-body">
                    <p className="memory-description">
                        Escreva informações que o assistente deve considerar em <strong>todas</strong> as conversas.
                        Suas preferências, cargo, vara, estilo de redação, etc.
                    </p>

                    {loading ? (
                        <div style={{ textAlign: 'center', padding: '40px 0', color: '#999' }}>
                            Carregando...
                        </div>
                    ) : (
                        <>
                            <textarea
                                ref={textareaRef}
                                className="memory-textarea"
                                value={content}
                                onChange={handleContentChange}
                                placeholder={"Ex:\n• Sou juiz da 1ª Vara Cível de Belo Horizonte\n• Prefiro decisões concisas e objetivas\n• Use sempre o art. 487, I do CPC ao julgar procedente\n• Evite latinismos desnecessários"}
                                maxLength={MAX_CHARS}
                                spellCheck={false}
                                id="memory-textarea"
                            />

                            <div className="memory-meta">
                                <span className={`memory-char-counter ${charClass}`}>
                                    {content.length}/{MAX_CHARS}
                                </span>

                                <div className="memory-toggle" onClick={handleToggle}>
                                    <span className="memory-toggle-label">
                                        {enabled ? 'Ativa' : 'Desativada'}
                                    </span>
                                    <div
                                        className={`memory-toggle-switch ${enabled ? 'active' : ''}`}
                                        role="switch"
                                        aria-checked={enabled}
                                        id="memory-toggle"
                                    />
                                </div>
                            </div>
                        </>
                    )}
                </div>

                {/* Footer */}
                <div className="memory-footer">
                    <button className="memory-cancel-btn" onClick={onClose}>
                        Cancelar
                    </button>
                    <button
                        className={`memory-save-btn ${saved ? 'saved' : ''}`}
                        onClick={handleSave}
                        disabled={saving || !dirty}
                        id="memory-save-btn"
                    >
                        {saved ? (
                            <>
                                <FaCheck size={12} /> Salvo!
                            </>
                        ) : saving ? (
                            'Salvando...'
                        ) : (
                            'Salvar'
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default MemoryPanel;
