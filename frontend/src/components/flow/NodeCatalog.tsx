import { FaRobot, FaCodeBranch, FaShuffle, FaUserClock, FaFileWord, FaScaleBalanced, FaCubes, FaWandMagicSparkles, FaXmark } from 'react-icons/fa6';

interface CatalogItem {
    type: string;
    accent: string;
    icon: React.ReactNode;
    title: string;
    desc: string;
    defaults: Record<string, string>;
}

export const NODE_CATALOG: CatalogItem[] = [
    {
        type: 'agent',
        accent: 'agent',
        icon: <FaRobot size={20} />,
        title: 'Agente LLM',
        desc: 'Chama um modelo (GPT, Claude, Gemini) com um prompt customizado.',
        defaults: { label: 'Agente', model: 'gpt-5.3-chat', prompt: '', knowledge: '', knowledge_files: '' },
    },
    {
        type: 'router',
        accent: 'router',
        icon: <FaCodeBranch size={20} />,
        title: 'Roteador (true/false)',
        desc: 'Bifurca o fluxo em duas saídas com base numa condição Python simples.',
        defaults: { label: 'Roteador', condition: '' },
    },
    {
        type: 'switch',
        accent: 'switch',
        icon: <FaShuffle size={20} />,
        title: 'Switch (classificador)',
        desc: 'LLM classifica a entrada em N categorias e segue a saída correspondente.',
        defaults: { label: 'Classificador', categories: 'Civil|Penal|Tributário', model: 'gpt-5.4-mini' },
    },
    {
        type: 'hil',
        accent: 'hil',
        icon: <FaUserClock size={20} />,
        title: 'Aprovação Humana',
        desc: 'Pausa o fluxo, mostra o rascunho no chat com botões Aprovar / Editar / Rejeitar.',
        defaults: { label: 'Aprovar rascunho', question: 'Por favor, revise antes de continuar.' },
    },
    {
        type: 'docx',
        accent: 'docx',
        icon: <FaFileWord size={20} />,
        title: 'Gerar DOCX',
        desc: 'Converte o markdown do nó anterior em um arquivo .docx para download.',
        defaults: { label: 'Gerar DOCX', filename: 'minuta.docx' },
    },
    {
        type: 'juris',
        accent: 'juris',
        icon: <FaScaleBalanced size={20} />,
        title: 'Pesquisar Jurisprudência',
        desc: 'Consulta o banco de acordãos do TJMG e injeta resultados no contexto.',
        defaults: { label: 'Pesquisar Jurisprudência', query: '', top_k: '5' },
    },
    {
        type: 'modelo',
        accent: 'modelo',
        icon: <FaCubes size={20} />,
        title: 'Buscar Modelo',
        desc: 'Busca semântica nos seus templates de minuta e devolve os mais relevantes.',
        defaults: { label: 'Buscar Modelo', query: '', top_k: '3' },
    },
    {
        type: 'estilo',
        accent: 'estilo',
        icon: <FaWandMagicSparkles size={20} />,
        title: 'Estilo do Juiz',
        desc: 'Reescreve o texto aplicando o style dossier salvo nas suas configurações.',
        defaults: { label: 'Aplicar estilo' },
    },
];

interface Props {
    open: boolean;
    onClose: () => void;
    onAdd: (item: CatalogItem) => void;
}

export default function NodeCatalog({ open, onClose, onAdd }: Props) {
    if (!open) return null;

    return (
        <div className="flow-catalog-overlay" onClick={onClose}>
            <div className="flow-catalog-panel" onClick={e => e.stopPropagation()}>
                <div className="flow-catalog-header">
                    <span className="flow-catalog-title">Nós Prontos</span>
                    <button className="flow-config-close" onClick={onClose}><FaXmark size={16} /></button>
                </div>
                <p className="flow-catalog-help">
                    Clique para adicionar ao fluxo. Depois é só conectar e configurar.
                </p>
                <div className="flow-catalog-grid">
                    {NODE_CATALOG.map(item => (
                        <button
                            key={item.type}
                            type="button"
                            className={`flow-catalog-card accent-${item.accent}`}
                            onClick={() => { onAdd(item); onClose(); }}
                        >
                            <span className="flow-catalog-icon">{item.icon}</span>
                            <div className="flow-catalog-text">
                                <span className="flow-catalog-card-title">{item.title}</span>
                                <span className="flow-catalog-card-desc">{item.desc}</span>
                            </div>
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}
