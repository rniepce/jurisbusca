import React, { useState } from 'react';
import { FaXmark, FaWandMagicSparkles } from 'react-icons/fa6';
import './CreateAgentDialog.css';

const COLORS = [
    '#10B981', '#4285F4', '#D97706', '#EF4444',
    '#8B5CF6', '#EC4899', '#14B8A6', '#F59E0B',
    '#6366F1', '#0EA5E9',
];

const CreateAgentDialog = ({ isOpen, onClose, onConfirm, initialPrompt = '' }) => {
    const [name, setName] = useState('');
    const [prompt, setPrompt] = useState(initialPrompt);
    const [color, setColor] = useState(COLORS[0]);

    if (!isOpen) return null;

    const handleConfirm = () => {
        if (!name.trim() || !prompt.trim()) return;
        onConfirm({ name: name.trim(), prompt: prompt.trim(), color });
        setName('');
        setPrompt('');
        setColor(COLORS[0]);
    };

    return (
        <div className="create-agent-overlay" onClick={onClose}>
            <div className="create-agent-dialog" onClick={(e) => e.stopPropagation()}>
                <div className="create-agent-header">
                    <h2><FaWandMagicSparkles /> Criar Agente</h2>
                    <button className="create-agent-close" onClick={onClose}>
                        <FaXmark />
                    </button>
                </div>

                <div className="create-agent-body">
                    <div className="create-agent-field">
                        <label>Nome do Agente</label>
                        <input
                            type="text"
                            placeholder="Ex: Analista de Contratos"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            autoFocus
                            id="create-agent-name"
                        />
                    </div>

                    <div className="create-agent-field">
                        <label>Cor</label>
                        <div className="create-agent-colors">
                            {COLORS.map((c) => (
                                <button
                                    key={c}
                                    className={`color-option ${color === c ? 'selected' : ''}`}
                                    style={{ background: c }}
                                    onClick={() => setColor(c)}
                                    aria-label={`Cor ${c}`}
                                />
                            ))}
                        </div>
                    </div>

                    <div className="create-agent-field">
                        <label>Prompt do Agente</label>
                        <textarea
                            placeholder="Cole ou escreva o prompt do agente..."
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            id="create-agent-prompt"
                        />
                    </div>
                </div>

                <div className="create-agent-footer">
                    <button className="btn-cancel" onClick={onClose}>Cancelar</button>
                    <button
                        className="btn-confirm"
                        onClick={handleConfirm}
                        disabled={!name.trim() || !prompt.trim()}
                        id="btn-confirm-agent"
                    >
                        Confirmar
                    </button>
                </div>
            </div>
        </div>
    );
};

export default CreateAgentDialog;
