import { useRef, useState } from 'react';
import { FaXmark, FaFileLines, FaPaperclip, FaSpinner, FaCircleCheck } from 'react-icons/fa6';
import { extractFlowFiles } from '../../services/api';

const MODELS = [
    { value: 'gpt-5.3-chat', label: 'GPT-5.3 Chat (Azure)' },
    { value: 'gpt-5.4-mini', label: 'GPT-5.4 Mini (Azure)' },
    { value: 'claude-sonnet-4-6', label: 'Claude Sonnet 4.6' },
    { value: 'claude-sonnet-4-5', label: 'Claude Sonnet 4.5' },
    { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
];

interface Props {
    node: { id: string; type: string; data: Record<string, string> };
    onChange: (nodeId: string, data: Record<string, string>) => void;
    onClose: () => void;
}

export default function NodeConfigPanel({ node, onChange, onClose }: Props) {
    const { type, data } = node;
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [uploading, setUploading] = useState(false);
    const [uploadError, setUploadError] = useState('');

    const set = (key: string, value: string) => onChange(node.id, { ...data, [key]: value });

    const knowledgeFiles = data.knowledge_files
        ? data.knowledge_files.split('|').filter(Boolean)
        : [];
    const knowledgeChars = data.knowledge ? data.knowledge.length : 0;

    const handleFiles = async (files: FileList | null) => {
        if (!files || files.length === 0) return;
        setUploading(true);
        setUploadError('');
        try {
            const fileArr = Array.from(files);
            const result = await extractFlowFiles(fileArr);
            const newKnowledge = (data.knowledge ? data.knowledge + '\n\n' : '') + result.text;
            const newFiles = [...knowledgeFiles, ...fileArr.map(f => f.name)].join('|');
            onChange(node.id, {
                ...data,
                knowledge: newKnowledge,
                knowledge_files: newFiles,
            });
        } catch (err) {
            setUploadError(String(err));
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const removeFile = (fileName: string) => {
        const newFiles = knowledgeFiles.filter(f => f !== fileName);
        const newKnowledgeBlocks = (data.knowledge || '')
            .split(/(?=--- )/)
            .filter(block => {
                const m = block.match(/^---\s+([^\n]+?)\s+---/);
                return !m || m[1] !== fileName;
            });
        onChange(node.id, {
            ...data,
            knowledge_files: newFiles.join('|'),
            knowledge: newKnowledgeBlocks.join('').trim(),
        });
    };

    const clearAllFiles = () => {
        onChange(node.id, { ...data, knowledge: '', knowledge_files: '' });
    };

    return (
        <div className="flow-config-panel">
            <div className="flow-config-header">
                <span className="flow-config-title">
                    {type === 'agent' ? 'Configurar Agente' : type === 'router' ? 'Configurar Roteador' : 'Nó'}
                </span>
                <button onClick={onClose} className="flow-config-close" title="Fechar">
                    <FaXmark size={16} />
                </button>
            </div>

            <div className="flow-config-field">
                <label className="flow-config-label">Nome</label>
                <input
                    className="flow-input"
                    value={data.label || ''}
                    onChange={e => set('label', e.target.value)}
                    placeholder="Nome do nó"
                />
            </div>

            {type === 'agent' && (
                <>
                    <div className="flow-config-field">
                        <label className="flow-config-label">Modelo</label>
                        <select
                            className="flow-select"
                            value={data.model || 'gpt-5.3-chat'}
                            onChange={e => set('model', e.target.value)}
                        >
                            {MODELS.map(m => (
                                <option key={m.value} value={m.value}>{m.label}</option>
                            ))}
                        </select>
                    </div>

                    <div className="flow-config-field">
                        <label className="flow-config-label">Prompt do Sistema</label>
                        <textarea
                            className="flow-textarea"
                            value={data.prompt || ''}
                            onChange={e => set('prompt', e.target.value)}
                            placeholder="Você é um agente especializado em..."
                        />
                    </div>

                    <div className="flow-config-field">
                        <label className="flow-config-label">Base de Conhecimento</label>
                        <input
                            ref={fileInputRef}
                            type="file"
                            multiple
                            style={{ display: 'none' }}
                            onChange={e => handleFiles(e.target.files)}
                            accept=".pdf,.txt,.docx,.md,.html"
                        />
                        <div
                            className={`flow-knowledge-dropzone ${uploading ? 'uploading' : ''}`}
                            onClick={() => !uploading && fileInputRef.current?.click()}
                            onDragOver={e => e.preventDefault()}
                            onDrop={e => {
                                e.preventDefault();
                                if (!uploading) handleFiles(e.dataTransfer.files);
                            }}
                        >
                            {uploading ? (
                                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                                    <FaSpinner size={12} className="fa-spin" /> Extraindo texto...
                                </span>
                            ) : (
                                <span>
                                    <FaPaperclip size={11} /> Clique ou arraste arquivos (PDF, DOCX, TXT)
                                </span>
                            )}
                        </div>

                        {uploadError && (
                            <span style={{ color: 'var(--danger-color)', fontSize: 11, marginTop: 4 }}>
                                {uploadError}
                            </span>
                        )}

                        {knowledgeFiles.length > 0 && (
                            <>
                                <div className="flow-knowledge-list">
                                    {knowledgeFiles.map((f, i) => (
                                        <div key={i} className="flow-knowledge-file">
                                            <FaFileLines size={12} className="flow-knowledge-file-icon" />
                                            <span className="flow-knowledge-file-name" title={f}>{f}</span>
                                            <button
                                                className="flow-knowledge-file-remove"
                                                onClick={() => removeFile(f)}
                                                title="Remover"
                                            >
                                                <FaXmark size={11} />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                                <div className="flow-knowledge-chars">
                                    <FaCircleCheck size={10} /> {knowledgeChars.toLocaleString('pt-BR')} caracteres indexados
                                </div>
                                <button
                                    className="flow-btn flow-btn-danger"
                                    onClick={clearAllFiles}
                                    style={{ marginTop: 6, alignSelf: 'flex-start' }}
                                >
                                    Limpar todos
                                </button>
                            </>
                        )}

                        <span className="flow-config-help">
                            Os arquivos serão injetados como contexto no prompt do agente.
                        </span>
                    </div>
                </>
            )}

            {type === 'router' && (
                <div className="flow-config-field">
                    <label className="flow-config-label">Condição (Python)</label>
                    <input
                        className="flow-input mono"
                        value={data.condition || ''}
                        onChange={e => set('condition', e.target.value)}
                        placeholder="len(resultado_triagem) > 100"
                    />
                    <span className="flow-config-help">
                        Use variáveis de saída dos agentes anteriores. Resultado verdadeiro segue pela saída superior, falso pela inferior.
                    </span>
                </div>
            )}
        </div>
    );
}
