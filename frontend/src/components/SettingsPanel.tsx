import React, { useState, useEffect, useRef } from 'react';
import { FaXmark, FaCheck, FaTrash, FaPencil } from 'react-icons/fa6';
import {
    getMemory, saveMemory,
    getAutoMemories, updateAutoMemory, deleteAutoMemory,
    type AutoMemory,
} from '../services/api';
import './SettingsPanel.css';

const MAX_CHARS = 2000;

type Tab = 'memory' | 'preferences';
type ResponseStyle = 'default' | 'short' | 'long' | 'custom';

const SettingsPanel = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
    const [activeTab, setActiveTab] = useState<Tab>('memory');

    // ── Memory tab state ──────────────────────────────────────────────────────
    const [content, setContent] = useState('');
    const [enabled, setEnabled] = useState(true);
    const [autoMemories, setAutoMemories] = useState<AutoMemory[]>([]);
    const [editingId, setEditingId] = useState<number | null>(null);
    const [editingText, setEditingText] = useState('');
    const [loadingMemory, setLoadingMemory] = useState(true);

    // ── Preferences tab state ─────────────────────────────────────────────────
    const [responseStyle, setResponseStyle] = useState<ResponseStyle>('default');
    const [responseStyleText, setResponseStyleText] = useState('');

    // ── Save state ────────────────────────────────────────────────────────────
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [dirty, setDirty] = useState(false);

    const textareaRef = useRef<HTMLTextAreaElement | null>(null);

    useEffect(() => {
        if (!isOpen) return;
        setLoadingMemory(true);
        setSaved(false);
        setDirty(false);
        Promise.all([getMemory(), getAutoMemories()]).then(([mem, autoMem]) => {
            setContent(mem.content || '');
            setEnabled(mem.enabled !== false);
            setResponseStyle((mem as any).response_style || 'default');
            setResponseStyleText((mem as any).response_style_text || '');
            setAutoMemories(autoMem.memories || []);
            setLoadingMemory(false);
            setTimeout(() => textareaRef.current?.focus(), 100);
        });
    }, [isOpen]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && isOpen) onClose();
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, onClose]);

    const handleSave = async () => {
        setSaving(true);
        try {
            await saveMemory(content, enabled, responseStyle, responseStyleText);
            setSaved(true);
            setDirty(false);
            setTimeout(() => { setSaved(false); onClose(); }, 1200);
        } catch (err) {
            console.error('Failed to save settings:', err);
        } finally {
            setSaving(false);
        }
    };

    const handleToggleAutoMemory = async (mem: AutoMemory) => {
        const next = !mem.enabled;
        setAutoMemories((prev) => prev.map((m) => m.id === mem.id ? { ...m, enabled: next } : m));
        await updateAutoMemory(mem.id, { enabled: next }).catch(console.error);
    };

    const handleDeleteAutoMemory = async (id: number) => {
        setAutoMemories((prev) => prev.filter((m) => m.id !== id));
        await deleteAutoMemory(id).catch(console.error);
    };

    const handleEditAutoMemory = async (id: number) => {
        await updateAutoMemory(id, { content: editingText }).catch(console.error);
        setAutoMemories((prev) => prev.map((m) => m.id === id ? { ...m, content: editingText } : m));
        setEditingId(null);
    };

    const mark = () => { setDirty(true); setSaved(false); };
    const charPercent = content.length / MAX_CHARS;
    const charClass = charPercent >= 1 ? 'at-limit' : charPercent >= 0.85 ? 'near-limit' : '';

    if (!isOpen) return null;

    return (
        <div
            className="settings-overlay"
            onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
            role="dialog"
            aria-modal="true"
            aria-label="Configurações"
        >
            <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="settings-header">
                    <h2>Configurações</h2>
                    <button className="settings-close-btn" onClick={onClose} aria-label="Fechar">
                        <FaXmark size={16} />
                    </button>
                </div>

                {/* Tabs */}
                <div className="settings-tabs">
                    <button
                        className={`settings-tab ${activeTab === 'memory' ? 'active' : ''}`}
                        onClick={() => setActiveTab('memory')}
                    >
                        🧠 Memória
                    </button>
                    <button
                        className={`settings-tab ${activeTab === 'preferences' ? 'active' : ''}`}
                        onClick={() => setActiveTab('preferences')}
                    >
                        ⚙️ Preferências
                    </button>
                </div>

                {/* Body */}
                <div className="settings-body">
                    {loadingMemory ? (
                        <div className="settings-loading">Carregando...</div>
                    ) : activeTab === 'memory' ? (
                        <>
                            {/* Manual memory */}
                            <div className="settings-section-label">Contexto Permanente</div>
                            <p className="settings-description">
                                Informações que o assistente deve considerar em <strong>todas</strong> as conversas.
                                Seu cargo, vara, estilo de redação, etc.
                            </p>
                            <textarea
                                ref={textareaRef}
                                className="settings-textarea"
                                value={content}
                                onChange={(e) => { if (e.target.value.length <= MAX_CHARS) { setContent(e.target.value); mark(); } }}
                                placeholder={"Ex:\n• Sou juiz da 1ª Vara Cível de Belo Horizonte\n• Prefiro decisões concisas e objetivas\n• Use sempre o art. 487, I do CPC ao julgar procedente"}
                                maxLength={MAX_CHARS}
                                spellCheck={false}
                            />
                            <div className="settings-meta">
                                <span className={`settings-char-counter ${charClass}`}>{content.length}/{MAX_CHARS}</span>
                                <div className="settings-toggle" onClick={() => { setEnabled((v) => !v); mark(); }}>
                                    <span className="settings-toggle-label">{enabled ? 'Ativa' : 'Desativada'}</span>
                                    <div className={`settings-toggle-switch ${enabled ? 'active' : ''}`} role="switch" aria-checked={enabled} />
                                </div>
                            </div>

                            {/* Auto-memories */}
                            <div className="settings-section-label" style={{ marginTop: 24 }}>
                                Memórias Extraídas de Conversas
                            </div>
                            {autoMemories.length === 0 ? (
                                <p className="settings-empty">
                                    Nenhuma memória extraída ainda. Continue conversando e o assistente aprenderá suas preferências automaticamente.
                                </p>
                            ) : (
                                <div className="auto-memories-list">
                                    {autoMemories.map((mem) => (
                                        <div key={mem.id} className={`auto-memory-card ${mem.enabled ? '' : 'disabled'}`}>
                                            {editingId === mem.id ? (
                                                <div className="auto-memory-edit">
                                                    <textarea
                                                        className="auto-memory-edit-input"
                                                        value={editingText}
                                                        onChange={(e) => setEditingText(e.target.value)}
                                                        rows={2}
                                                        autoFocus
                                                    />
                                                    <div className="auto-memory-edit-actions">
                                                        <button className="btn-save-inline" onClick={() => handleEditAutoMemory(mem.id)}>Salvar</button>
                                                        <button className="btn-cancel-inline" onClick={() => setEditingId(null)}>Cancelar</button>
                                                    </div>
                                                </div>
                                            ) : (
                                                <>
                                                    <span className="auto-memory-text">{mem.content}</span>
                                                    <div className="auto-memory-actions">
                                                        <div
                                                            className={`mini-toggle ${mem.enabled ? 'active' : ''}`}
                                                            onClick={() => handleToggleAutoMemory(mem)}
                                                            title={mem.enabled ? 'Desativar' : 'Ativar'}
                                                        />
                                                        <button
                                                            className="icon-btn"
                                                            title="Editar"
                                                            onClick={() => { setEditingId(mem.id); setEditingText(mem.content); }}
                                                        >
                                                            <FaPencil size={11} />
                                                        </button>
                                                        <button
                                                            className="icon-btn danger"
                                                            title="Apagar"
                                                            onClick={() => handleDeleteAutoMemory(mem.id)}
                                                        >
                                                            <FaTrash size={11} />
                                                        </button>
                                                    </div>
                                                </>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </>
                    ) : (
                        /* Preferences tab */
                        <>
                            <div className="settings-section-label">Estilo de Resposta</div>
                            <p className="settings-description">
                                Defina o comprimento e o formato padrão das respostas do assistente.
                            </p>
                            <div className="response-style-options">
                                {([
                                    { value: 'default', label: 'Padrão', desc: 'O assistente decide o comprimento conforme a pergunta' },
                                    { value: 'short', label: 'Curta', desc: 'Respostas concisas, máximo 3 parágrafos' },
                                    { value: 'long', label: 'Longa', desc: 'Respostas detalhadas com análise aprofundada' },
                                    { value: 'custom', label: 'Personalizada', desc: 'Defina suas próprias instruções abaixo' },
                                ] as { value: ResponseStyle; label: string; desc: string }[]).map((opt) => (
                                    <label key={opt.value} className={`style-option ${responseStyle === opt.value ? 'selected' : ''}`}>
                                        <input
                                            type="radio"
                                            name="response_style"
                                            value={opt.value}
                                            checked={responseStyle === opt.value}
                                            onChange={() => { setResponseStyle(opt.value); mark(); }}
                                        />
                                        <div className="style-option-content">
                                            <span className="style-option-label">{opt.label}</span>
                                            <span className="style-option-desc">{opt.desc}</span>
                                        </div>
                                    </label>
                                ))}
                            </div>

                            {responseStyle === 'custom' && (
                                <textarea
                                    className="settings-textarea"
                                    style={{ marginTop: 12, minHeight: 100 }}
                                    value={responseStyleText}
                                    onChange={(e) => { setResponseStyleText(e.target.value); mark(); }}
                                    placeholder="Ex: Responda sempre em tópicos numerados, com citações legais completas e sem linguagem informal."
                                />
                            )}
                        </>
                    )}
                </div>

                {/* Footer */}
                <div className="settings-footer">
                    <button className="settings-cancel-btn" onClick={onClose}>Cancelar</button>
                    <button
                        className={`settings-save-btn ${saved ? 'saved' : ''}`}
                        onClick={handleSave}
                        disabled={saving || !dirty}
                    >
                        {saved ? <><FaCheck size={12} /> Salvo!</> : saving ? 'Salvando...' : 'Salvar'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SettingsPanel;
