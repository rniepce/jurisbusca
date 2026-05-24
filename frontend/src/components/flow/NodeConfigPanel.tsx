import { useRef, useState } from 'react';
import { FaXmark, FaFileLines, FaPaperclip, FaSpinner, FaCircleCheck } from 'react-icons/fa6';
import { extractFlowFiles } from '../../services/api';
import PromptEditor from './PromptEditor';

const MODELS = [
    { value: 'gpt-5.3-chat', label: 'GPT-5.3 Chat (Azure)' },
    { value: 'gpt-5.4-mini', label: 'GPT-5.4 Mini (Azure)' },
    { value: 'claude-sonnet-4-6', label: 'Claude Sonnet 4.6' },
    { value: 'claude-sonnet-4-5', label: 'Claude Sonnet 4.5' },
    { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
];

const NODE_TITLES: Record<string, string> = {
    agent: 'Configurar Agente',
    router: 'Configurar Roteador',
    switch: 'Configurar Classificador (Switch)',
    hil: 'Configurar Aprovação Humana',
    docx: 'Configurar Gerador DOCX',
    juris: 'Configurar Pesquisa de Jurisprudência',
    modelo: 'Configurar Busca de Modelo',
    estilo: 'Configurar Aplicação de Estilo',
};

interface Props {
    node: { id: string; type: string; data: Record<string, string> };
    onChange: (nodeId: string, data: Record<string, string>) => void;
    onClose: () => void;
    availableLabels?: string[];
}

export default function NodeConfigPanel({ node, onChange, onClose, availableLabels = [] }: Props) {
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
            onChange(node.id, { ...data, knowledge: newKnowledge, knowledge_files: newFiles });
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

    const clearAllFiles = () => onChange(node.id, { ...data, knowledge: '', knowledge_files: '' });

    return (
        <div className="flow-config-panel">
            <div className="flow-config-header">
                <span className="flow-config-title">{NODE_TITLES[type] || 'Nó'}</span>
                <button onClick={onClose} className="flow-config-close" title="Fechar"><FaXmark size={16} /></button>
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

            {/* ─── AGENT ───────────────────────────────────────────── */}
            {type === 'agent' && (
                <>
                    <div className="flow-config-field">
                        <label className="flow-config-label">Modelo</label>
                        <select
                            className="flow-select"
                            value={data.model || 'gpt-5.3-chat'}
                            onChange={e => set('model', e.target.value)}
                        >
                            {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                        </select>
                    </div>

                    <div className="flow-config-field">
                        <label className="flow-config-label">Prompt do Sistema</label>
                        <PromptEditor
                            value={data.prompt || ''}
                            onChange={(v) => set('prompt', v)}
                            placeholder="Você é um agente especializado em..."
                            availableLabels={availableLabels}
                        />
                    </div>

                    <FileUploadField
                        files={knowledgeFiles}
                        chars={knowledgeChars}
                        uploading={uploading}
                        error={uploadError}
                        onPick={() => fileInputRef.current?.click()}
                        onDrop={handleFiles}
                        onRemove={removeFile}
                        onClear={clearAllFiles}
                    />
                    <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        style={{ display: 'none' }}
                        onChange={e => handleFiles(e.target.files)}
                        accept=".pdf,.txt,.docx,.md,.html"
                    />
                </>
            )}

            {/* ─── ROUTER ──────────────────────────────────────────── */}
            {type === 'router' && (
                <div className="flow-config-field">
                    <label className="flow-config-label">Condição (Python)</label>
                    <input
                        className="flow-input mono"
                        value={data.condition || ''}
                        onChange={e => set('condition', e.target.value)}
                        placeholder='len(triagem) > 100'
                    />
                    <span className="flow-config-help">
                        Variáveis disponíveis: {availableLabels.join(', ') || '(nenhuma)'}. Avalia true/false e segue a saída correspondente.
                    </span>
                </div>
            )}

            {/* ─── SWITCH (multi-saída) ────────────────────────────── */}
            {type === 'switch' && (
                <>
                    <div className="flow-config-field">
                        <label className="flow-config-label">Categorias (uma por linha)</label>
                        <textarea
                            className="flow-textarea mono"
                            style={{ minHeight: 80 }}
                            value={(data.categories || '').split('|').join('\n')}
                            onChange={e => set('categories', e.target.value.split('\n').map(s => s.trim()).filter(Boolean).join('|'))}
                            placeholder={'Civil\nPenal\nTributário'}
                        />
                        <span className="flow-config-help">
                            Cada categoria vira uma saída do nó. O LLM escolhe a que melhor classifica a entrada.
                        </span>
                    </div>
                    <div className="flow-config-field">
                        <label className="flow-config-label">Modelo classificador</label>
                        <select className="flow-select" value={data.model || 'gpt-5.4-mini'} onChange={e => set('model', e.target.value)}>
                            {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                        </select>
                    </div>
                </>
            )}

            {/* ─── HIL ─────────────────────────────────────────────── */}
            {type === 'hil' && (
                <div className="flow-config-field">
                    <label className="flow-config-label">Mensagem para o usuário</label>
                    <textarea
                        className="flow-textarea"
                        value={data.question || ''}
                        onChange={e => set('question', e.target.value)}
                        placeholder="Por favor, revise o rascunho antes de continuar."
                        style={{ minHeight: 80 }}
                    />
                    <span className="flow-config-help">
                        Quando rodando no chat, o fluxo pausa aqui e o usuário verá o rascunho parcial com botões Aprovar / Editar / Rejeitar.
                    </span>
                </div>
            )}

            {/* ─── DOCX ────────────────────────────────────────────── */}
            {type === 'docx' && (
                <div className="flow-config-field">
                    <label className="flow-config-label">Nome do arquivo</label>
                    <input
                        className="flow-input mono"
                        value={data.filename || ''}
                        onChange={e => set('filename', e.target.value)}
                        placeholder="minuta.docx"
                    />
                    <span className="flow-config-help">
                        Converte a saída do nó anterior (em markdown) para um arquivo .docx baixável.
                    </span>
                </div>
            )}

            {/* ─── JURIS / MODELO ──────────────────────────────────── */}
            {(type === 'juris' || type === 'modelo') && (
                <>
                    <div className="flow-config-field">
                        <label className="flow-config-label">Termo de busca (opcional)</label>
                        <PromptEditor
                            value={data.query || ''}
                            onChange={(v) => set('query', v)}
                            placeholder={type === 'juris' ? 'Ex: prescrição da pretensão punitiva' : 'Ex: sentença de improcedência por prescrição'}
                            availableLabels={availableLabels}
                            minHeight={60}
                        />
                        <span className="flow-config-help">
                            Se vazio, usa a saída do nó anterior automaticamente. Pode usar <code>@variável</code>.
                        </span>
                    </div>
                    <div className="flow-config-field">
                        <label className="flow-config-label">Quantos resultados</label>
                        <input
                            className="flow-input mono"
                            type="number"
                            min="1" max="20"
                            value={data.top_k || (type === 'juris' ? '5' : '3')}
                            onChange={e => set('top_k', e.target.value)}
                        />
                    </div>
                </>
            )}

            {/* ─── ESTILO ──────────────────────────────────────────── */}
            {type === 'estilo' && (
                <div className="flow-config-help">
                    Aplica o seu <strong>style dossier</strong> no texto produzido pelo nó anterior.
                    Configure seu estilo em <strong>Gestor de Modelos</strong> antes de usar.
                </div>
            )}
        </div>
    );
}

interface FileUploadProps {
    files: string[];
    chars: number;
    uploading: boolean;
    error: string;
    onPick: () => void;
    onDrop: (files: FileList | null) => void;
    onRemove: (name: string) => void;
    onClear: () => void;
}

function FileUploadField({ files, chars, uploading, error, onPick, onDrop, onRemove, onClear }: FileUploadProps) {
    return (
        <div className="flow-config-field">
            <label className="flow-config-label">Base de Conhecimento</label>
            <div
                className={`flow-knowledge-dropzone ${uploading ? 'uploading' : ''}`}
                onClick={() => !uploading && onPick()}
                onDragOver={e => e.preventDefault()}
                onDrop={e => { e.preventDefault(); if (!uploading) onDrop(e.dataTransfer.files); }}
            >
                {uploading ? (
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        <FaSpinner size={12} /> Extraindo texto...
                    </span>
                ) : (
                    <span><FaPaperclip size={11} /> Clique ou arraste arquivos (PDF, DOCX, TXT)</span>
                )}
            </div>
            {error && <span style={{ color: 'var(--danger-color)', fontSize: 11, marginTop: 4 }}>{error}</span>}
            {files.length > 0 && (
                <>
                    <div className="flow-knowledge-list">
                        {files.map((f, i) => (
                            <div key={i} className="flow-knowledge-file">
                                <FaFileLines size={12} className="flow-knowledge-file-icon" />
                                <span className="flow-knowledge-file-name" title={f}>{f}</span>
                                <button className="flow-knowledge-file-remove" onClick={() => onRemove(f)} title="Remover">
                                    <FaXmark size={11} />
                                </button>
                            </div>
                        ))}
                    </div>
                    <div className="flow-knowledge-chars">
                        <FaCircleCheck size={10} /> {chars.toLocaleString('pt-BR')} caracteres indexados
                    </div>
                    <button className="flow-btn flow-btn-danger" onClick={onClear} style={{ marginTop: 6, alignSelf: 'flex-start' }}>
                        Limpar todos
                    </button>
                </>
            )}
            <span className="flow-config-help">Os arquivos são injetados como contexto no prompt.</span>
        </div>
    );
}
