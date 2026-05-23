import { useState } from 'react';
import { FaXmark, FaWandMagicSparkles, FaPaperclip, FaFile } from 'react-icons/fa6';
import { createAgentSchema, firstError, type CreateAgentInput } from '../validation/schemas';
import './CreateAgentDialog.css';

const COLORS = [
    '#10B981', '#4285F4', '#D97706', '#EF4444',
    '#8B5CF6', '#EC4899', '#14B8A6', '#F59E0B',
    '#6366F1', '#0EA5E9',
];

interface Props {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: (input: CreateAgentInput) => void | Promise<void>;
    initialPrompt?: string;
}

const CreateAgentDialog = ({ isOpen, onClose, onConfirm, initialPrompt = '' }: Props) => {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [prompt, setPrompt] = useState(initialPrompt);
    const [color, setColor] = useState(COLORS[0]);
    const [files, setFiles] = useState<File[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [confirming, setConfirming] = useState(false);

    if (!isOpen) return null;

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selected = Array.from(e.target.files || []);
        setFiles((prev) => [...prev, ...selected]);
        e.target.value = '';
    };

    const removeFile = (i: number) => setFiles((prev) => prev.filter((_, idx) => idx !== i));

    const formatSize = (b: number) => b < 1024 * 1024 ? `${(b / 1024).toFixed(0)} KB` : `${(b / (1024 * 1024)).toFixed(1)} MB`;

    const handleConfirm = async () => {
        const parsed = createAgentSchema.safeParse({ name, description, prompt, color });
        if (!parsed.success) {
            setError(firstError(parsed.error));
            return;
        }
        setError(null);
        setConfirming(true);
        try {
            await onConfirm({ ...parsed.data, files });
            setName('');
            setDescription('');
            setPrompt('');
            setColor(COLORS[0]);
            setFiles([]);
        } finally {
            setConfirming(false);
        }
    };

    return (
        <div className="create-agent-overlay" onClick={onClose}>
            <div className="create-agent-dialog" onClick={(e) => e.stopPropagation()}>
                <div className="create-agent-header">
                    <h2><FaWandMagicSparkles /> Criar Agente</h2>
                    <button className="create-agent-close" onClick={onClose}><FaXmark /></button>
                </div>

                <div className="create-agent-body">
                    {error && (
                        <div
                            role="alert"
                            style={{
                                background: 'color-mix(in srgb, var(--danger-color) 10%, transparent)',
                                color: 'var(--danger-color)',
                                border: '1px solid var(--danger-color)',
                                borderRadius: 'var(--radius-sm)',
                                padding: '8px 12px',
                                marginBottom: '12px',
                                fontSize: 'var(--text-sm)',
                            }}
                        >
                            {error}
                        </div>
                    )}

                    <div className="create-agent-field">
                        <label>Nome do Agente</label>
                        <input
                            type="text"
                            placeholder="Ex: Analista de Contratos"
                            value={name}
                            onChange={(e) => { setName(e.target.value); if (error) setError(null); }}
                            autoFocus
                            id="create-agent-name"
                        />
                    </div>

                    <div className="create-agent-field">
                        <label>Descrição <span style={{ fontWeight: 400, color: '#888' }}>(opcional)</span></label>
                        <input
                            type="text"
                            placeholder="Ex: Especialista em contratos administrativos"
                            value={description}
                            maxLength={200}
                            onChange={(e) => setDescription(e.target.value)}
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
                        <label>Instruções do Agente</label>
                        <textarea
                            placeholder="Descreva como o agente deve se comportar, em qual área é especialista, qual tom deve usar, etc. (mínimo 20 caracteres)"
                            value={prompt}
                            onChange={(e) => { setPrompt(e.target.value); if (error) setError(null); }}
                            id="create-agent-prompt"
                        />
                    </div>

                    <div className="create-agent-field">
                        <label>
                            Arquivos de Conhecimento <span style={{ fontWeight: 400, color: '#888' }}>(opcional)</span>
                        </label>
                        <p className="create-agent-field-hint">
                            O agente terá acesso ao conteúdo desses arquivos em todas as conversas. Aceita PDF, TXT e DOCX.
                        </p>
                        <label className="agent-file-btn">
                            <FaPaperclip size={13} />
                            <span>Adicionar arquivos</span>
                            <input
                                type="file"
                                accept=".pdf,.txt,.docx"
                                multiple
                                onChange={handleFileChange}
                                style={{ display: 'none' }}
                            />
                        </label>
                        {files.length > 0 && (
                            <div className="agent-file-list">
                                {files.map((f, i) => (
                                    <div key={i} className="agent-file-chip">
                                        <FaFile size={11} />
                                        <span>{f.name}</span>
                                        <span className="agent-file-size">{formatSize(f.size)}</span>
                                        <button className="agent-file-remove" onClick={() => removeFile(i)}>
                                            <FaXmark size={10} />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                <div className="create-agent-footer">
                    <button className="btn-cancel" onClick={onClose}>Cancelar</button>
                    <button
                        className="btn-confirm"
                        onClick={handleConfirm}
                        disabled={!name.trim() || !prompt.trim() || confirming}
                        id="btn-confirm-agent"
                    >
                        {confirming ? 'Criando...' : 'Criar Agente'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default CreateAgentDialog;
