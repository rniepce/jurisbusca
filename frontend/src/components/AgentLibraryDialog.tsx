import React, { useState } from 'react';
import {
    FaXmark, FaBookOpen, FaRobot, FaDiagramProject,
    FaWandMagicSparkles, FaGavel, FaMagnifyingGlass,
    FaBrain, FaScaleBalanced, FaSitemap, FaFileLines,
    FaArrowRight, FaPlus,
} from 'react-icons/fa6';
import './AgentLibraryDialog.css';

type Tab = 'agents' | 'flows';

interface AgentTemplate {
    id: string;
    name: string;
    description: string;
    icon: React.ReactNode;
    color: string;
    tags: string[];
}

interface FlowTemplate {
    id: string;
    name: string;
    description: string;
    steps: string[];
    color: string;
    tags: string[];
}

const AGENT_TEMPLATES: AgentTemplate[] = [
    {
        id: 'redator',
        name: 'Redator Jurídico',
        description: 'Especialista em redigir decisões, acórdãos e peças processuais com linguagem jurídica precisa e fundamentação adequada.',
        icon: <FaScaleBalanced />,
        color: '#3b82f6',
        tags: ['Redação', 'Decisões'],
    },
    {
        id: 'jurisprudencia',
        name: 'Pesquisador de Jurisprudência',
        description: 'Busca e analisa precedentes relevantes, identificando teses aplicáveis ao caso concreto em bases jurídicas.',
        icon: <FaMagnifyingGlass />,
        color: '#8B5CF6',
        tags: ['Pesquisa', 'Jurisprudência'],
    },
    {
        id: 'revisor',
        name: 'Revisor de Documentos',
        description: 'Revisa minutas para identificar inconsistências, erros formais e lacunas de fundamentação antes da publicação.',
        icon: <FaFileLines />,
        color: '#10b981',
        tags: ['Revisão', 'Qualidade'],
    },
    {
        id: 'audiencias',
        name: 'Assistente de Audiências',
        description: 'Apoio em tempo real durante sessões e audiências — resume alegações, sinaliza pontos controvertidos e sugere perguntas.',
        icon: <FaGavel />,
        color: '#f59e0b',
        tags: ['Audiências', 'Tempo real'],
    },
    {
        id: 'sintetizador',
        name: 'Sintetizador de Processos',
        description: 'Resume processos extensos em pontos-chave: partes, pedidos, provas e histórico processual de forma estruturada.',
        icon: <FaBrain />,
        color: '#ec4899',
        tags: ['Resumo', 'Análise'],
    },
    {
        id: 'instrucao',
        name: 'Instrutor de Processo',
        description: 'Analisa a fase instrutória e identifica quais provas faltam, quais já foram produzidas e o que pode ser deferido ou indeferido.',
        icon: <FaWandMagicSparkles />,
        color: '#06b6d4',
        tags: ['Instrução', 'Provas'],
    },
];

const FLOW_TEMPLATES: FlowTemplate[] = [
    {
        id: 'analise-completa',
        name: 'Análise Completa de Processo',
        description: 'Pipeline end-to-end que parte da leitura do processo até a minuta final de decisão.',
        steps: ['Triagem', 'Extração de Fatos', 'Pesquisa Jurídica', 'Redação', 'Revisão'],
        color: '#3b82f6',
        tags: ['Completo', 'Decisão'],
    },
    {
        id: 'triagem',
        name: 'Triagem Automática',
        description: 'Classifica automaticamente processos por urgência, matéria e tipo de provimento mais adequado.',
        steps: ['Leitura', 'Classificação', 'Priorização'],
        color: '#f59e0b',
        tags: ['Triagem', 'Automação'],
    },
    {
        id: 'revisao-camadas',
        name: 'Revisão em Múltiplas Camadas',
        description: 'Passa a minuta por três etapas de revisão: jurídica, formal e de coerência interna.',
        steps: ['Rascunho', 'Revisão Jurídica', 'Revisão Formal', 'Coerência'],
        color: '#8B5CF6',
        tags: ['Revisão', 'Qualidade'],
    },
    {
        id: 'pesquisa-sintese',
        name: 'Pesquisa e Síntese',
        description: 'Busca jurisprudência relevante em múltiplas fontes e consolida em um memorando sintético.',
        steps: ['Query', 'Busca STF/STJ', 'Filtragem', 'Síntese'],
        color: '#10b981',
        tags: ['Pesquisa', 'Jurisprudência'],
    },
    {
        id: 'voto',
        name: 'Elaboração de Voto',
        description: 'Fluxo completo para elaboração de votos em câmaras e turmas, da análise ao relatório final.',
        steps: ['Relatório', 'Fundamentação', 'Dispositivo', 'Revisão'],
        color: '#ec4899',
        tags: ['Voto', 'Câmara'],
    },
];

interface Props {
    onClose: () => void;
    onUseAgent?: (agent: AgentTemplate) => void;
    onUseFlow?: (flow: FlowTemplate) => void;
}

const AgentLibraryDialog: React.FC<Props> = ({ onClose, onUseAgent, onUseFlow }) => {
    const [tab, setTab] = useState<Tab>('agents');
    const [search, setSearch] = useState('');

    const filteredAgents = AGENT_TEMPLATES.filter(
        (a) =>
            a.name.toLowerCase().includes(search.toLowerCase()) ||
            a.description.toLowerCase().includes(search.toLowerCase()) ||
            a.tags.some((t) => t.toLowerCase().includes(search.toLowerCase()))
    );

    const filteredFlows = FLOW_TEMPLATES.filter(
        (f) =>
            f.name.toLowerCase().includes(search.toLowerCase()) ||
            f.description.toLowerCase().includes(search.toLowerCase()) ||
            f.tags.some((t) => t.toLowerCase().includes(search.toLowerCase()))
    );

    return (
        <div className="lib-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label="Biblioteca">
            <div className="lib-dialog" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="lib-header">
                    <div className="lib-header-left">
                        <span className="lib-header-icon">
                            <FaBookOpen size={16} />
                        </span>
                        <div>
                            <h2 className="lib-title">Biblioteca</h2>
                            <p className="lib-subtitle">Agentes e fluxos prontos para usar</p>
                        </div>
                    </div>
                    <button className="lib-close" onClick={onClose} aria-label="Fechar">
                        <FaXmark size={16} />
                    </button>
                </div>

                {/* Tab + Search bar */}
                <div className="lib-toolbar">
                    <div className="lib-tabs">
                        <button
                            className={`lib-tab ${tab === 'agents' ? 'active' : ''}`}
                            onClick={() => setTab('agents')}
                        >
                            <FaRobot size={13} />
                            Agentes
                            <span className="lib-tab-count">{AGENT_TEMPLATES.length}</span>
                        </button>
                        <button
                            className={`lib-tab ${tab === 'flows' ? 'active' : ''}`}
                            onClick={() => setTab('flows')}
                        >
                            <FaDiagramProject size={13} />
                            Fluxos
                            <span className="lib-tab-count">{FLOW_TEMPLATES.length}</span>
                        </button>
                    </div>

                    <div className="lib-search-wrap">
                        <FaMagnifyingGlass size={12} className="lib-search-icon" />
                        <input
                            className="lib-search"
                            type="search"
                            placeholder="Buscar..."
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            autoFocus
                        />
                    </div>
                </div>

                {/* Content */}
                <div className="lib-body">
                    {tab === 'agents' && (
                        <div className="lib-grid">
                            {filteredAgents.length === 0 ? (
                                <p className="lib-empty">Nenhum agente encontrado.</p>
                            ) : (
                                filteredAgents.map((agent) => (
                                    <div key={agent.id} className="lib-card">
                                        <div className="lib-card-icon" style={{ background: `${agent.color}18`, color: agent.color }}>
                                            {agent.icon}
                                        </div>
                                        <div className="lib-card-body">
                                            <div className="lib-card-name">{agent.name}</div>
                                            <div className="lib-card-desc">{agent.description}</div>
                                            <div className="lib-card-tags">
                                                {agent.tags.map((t) => (
                                                    <span key={t} className="lib-tag" style={{ '--tag-color': agent.color } as any}>{t}</span>
                                                ))}
                                            </div>
                                        </div>
                                        <button
                                            className="lib-use-btn"
                                            style={{ '--btn-color': agent.color } as any}
                                            onClick={() => onUseAgent?.(agent)}
                                            title="Usar este agente"
                                        >
                                            <FaPlus size={11} />
                                            Usar
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    )}

                    {tab === 'flows' && (
                        <div className="lib-grid">
                            {filteredFlows.length === 0 ? (
                                <p className="lib-empty">Nenhum fluxo encontrado.</p>
                            ) : (
                                filteredFlows.map((flow) => (
                                    <div key={flow.id} className="lib-card lib-card-flow">
                                        <div className="lib-card-icon" style={{ background: `${flow.color}18`, color: flow.color }}>
                                            <FaSitemap size={16} />
                                        </div>
                                        <div className="lib-card-body">
                                            <div className="lib-card-name">{flow.name}</div>
                                            <div className="lib-card-desc">{flow.description}</div>
                                            <div className="lib-flow-steps">
                                                {flow.steps.map((step, i) => (
                                                    <React.Fragment key={step}>
                                                        <span className="lib-flow-step">{step}</span>
                                                        {i < flow.steps.length - 1 && (
                                                            <FaArrowRight size={9} className="lib-flow-arrow" />
                                                        )}
                                                    </React.Fragment>
                                                ))}
                                            </div>
                                            <div className="lib-card-tags">
                                                {flow.tags.map((t) => (
                                                    <span key={t} className="lib-tag" style={{ '--tag-color': flow.color } as any}>{t}</span>
                                                ))}
                                            </div>
                                        </div>
                                        <button
                                            className="lib-use-btn"
                                            style={{ '--btn-color': flow.color } as any}
                                            onClick={() => onUseFlow?.(flow)}
                                            title="Usar este fluxo"
                                        >
                                            <FaPlus size={11} />
                                            Usar
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default AgentLibraryDialog;
