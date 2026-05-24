import { FaGavel, FaScaleBalanced, FaBolt, FaXmark } from 'react-icons/fa6';
import type { FlowConfig } from '../../services/api';

interface Template {
    id: string;
    icon: React.ReactNode;
    name: string;
    desc: string;
    color: string;
    config: FlowConfig;
}

// helper to construct flow templates concisely
const makeStart = (id: string, x = 60, y = 240) => ({ id, type: 'start', position: { x, y }, data: {} });
const makeEnd = (id: string, x = 1100, y = 240) => ({ id, type: 'end', position: { x, y }, data: {} });
const makeAgent = (id: string, label: string, model: string, prompt: string, x: number, y: number) => ({
    id, type: 'agent', position: { x, y },
    data: { label, model, prompt, knowledge: '', knowledge_files: '', output_var: id },
});
const makeJuris = (id: string, label: string, query: string, x: number, y: number) => ({
    id, type: 'juris', position: { x, y }, data: { label, query, top_k: '5', output_var: id },
});
const makeSwitch = (id: string, label: string, categories: string, x: number, y: number) => ({
    id, type: 'switch', position: { x, y },
    data: { label, categories, model: 'gpt-5.4-mini', output_var: id },
});
const edge = (id: string, source: string, target: string, label = '', sourceHandle = '') => ({
    id, source, target, label, sourceHandle,
});

export const FLOW_TEMPLATES: Template[] = [
    {
        id: 'triagem-minuta-revisao',
        icon: <FaGavel size={22} />,
        name: 'Triagem + Minuta + Revisão',
        desc: 'Pipeline V2 reimaginado: triagem do processo → minuta automática → auditoria de conformidade.',
        color: '#3b82f6',
        config: {
            nodes: [
                makeStart('start'),
                makeAgent('triagem', 'Triagem', 'gpt-5.3-chat',
                    'Você é um agente de triagem judicial. Identifique o tipo de processo, as partes, o pedido principal e os pontos controvertidos. Retorne em formato estruturado e conciso.',
                    300, 240),
                makeAgent('minuta', 'Redigir Minuta', 'claude-sonnet-4-6',
                    'Você é um magistrado experiente. Com base na triagem abaixo: {{Triagem}}\n\nRedija uma minuta de decisão completa, com fundamentação clara, citação de dispositivos legais e dispositivo final.',
                    580, 240),
                makeAgent('revisao', 'Revisão / QA', 'gpt-5.3-chat',
                    'Você é um revisor sênior. Audite a minuta abaixo: {{Redigir Minuta}}\n\nIdentifique inconsistências factuais, erros de citação legal e problemas de conformidade. Retorne dashboard com checkboxes (✅/⚠️/❌) por critério.',
                    860, 240),
                makeEnd('end'),
            ],
            edges: [
                edge('e1', 'start', 'triagem'),
                edge('e2', 'triagem', 'minuta'),
                edge('e3', 'minuta', 'revisao'),
                edge('e4', 'revisao', 'end'),
            ],
        },
    },
    {
        id: 'analise-recurso',
        icon: <FaScaleBalanced size={22} />,
        name: 'Análise de Recurso',
        desc: 'Classifica tipo de recurso → pesquisa jurisprudência relevante → produz análise comparativa.',
        color: '#8b5cf6',
        config: {
            nodes: [
                makeStart('start'),
                makeSwitch('tipo', 'Tipo de Recurso', 'Apelação|Agravo|Embargos|Recurso Especial', 280, 240),
                makeJuris('juris', 'Pesquisar Jurisprudência', '{{Tipo de Recurso}}', 600, 240),
                makeAgent('analise', 'Análise Comparativa', 'claude-sonnet-4-6',
                    'Você é um assessor de gabinete. Com base no recurso classificado como {{Tipo de Recurso}} e nos acordãos pesquisados {{Pesquisar Jurisprudência}}, redija parecer comparando a tese do recorrente com o entendimento dominante do TJMG. Indique se há divergência e como decidir.',
                    900, 240),
                makeEnd('end', 1240),
            ],
            edges: [
                edge('e1', 'start', 'tipo'),
                edge('e2a', 'tipo', 'juris', '', 'Apelação'),
                edge('e2b', 'tipo', 'juris', '', 'Agravo'),
                edge('e2c', 'tipo', 'juris', '', 'Embargos'),
                edge('e2d', 'tipo', 'juris', '', 'Recurso Especial'),
                edge('e3', 'juris', 'analise'),
                edge('e4', 'analise', 'end'),
            ],
        },
    },
    {
        id: 'despacho-rapido',
        icon: <FaBolt size={22} />,
        name: 'Despacho Rápido',
        desc: 'Um agente único para despachos curtos: lê o processo e produz despacho de mero expediente.',
        color: '#f59e0b',
        config: {
            nodes: [
                makeStart('start'),
                makeAgent('despacho', 'Redigir Despacho', 'gpt-5.4-mini',
                    'Você é um magistrado. Leia o processo abaixo e produza um despacho de mero expediente curto (3-5 linhas) determinando a próxima providência. Seja objetivo e formal.',
                    400, 240),
                makeEnd('end', 780),
            ],
            edges: [
                edge('e1', 'start', 'despacho'),
                edge('e2', 'despacho', 'end'),
            ],
        },
    },
];

interface Props {
    open: boolean;
    onClose: () => void;
    onPick: (template: Template) => void;
}

export default function FlowTemplates({ open, onClose, onPick }: Props) {
    if (!open) return null;
    return (
        <div className="flow-modal-backdrop" onClick={onClose}>
            <div className="flow-modal flow-templates-modal" onClick={e => e.stopPropagation()}>
                <div className="flow-modal-header">
                    <span className="flow-modal-title">Templates de Fluxo</span>
                    <button className="flow-config-close" onClick={onClose}><FaXmark size={16} /></button>
                </div>
                <p className="flow-config-help" style={{ marginTop: -6 }}>
                    Escolha um ponto de partida. Você pode customizar tudo depois.
                </p>
                <div className="flow-templates-grid">
                    {FLOW_TEMPLATES.map(t => (
                        <button
                            key={t.id}
                            className="flow-template-card"
                            style={{ borderLeftColor: t.color }}
                            onClick={() => { onPick(t); onClose(); }}
                        >
                            <span className="flow-template-icon" style={{ background: `${t.color}22`, color: t.color }}>
                                {t.icon}
                            </span>
                            <div className="flow-template-text">
                                <span className="flow-template-name">{t.name}</span>
                                <span className="flow-template-desc">{t.desc}</span>
                            </div>
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}
