import React, { useState } from 'react';
import {
    FaXmark, FaBookOpen, FaRobot, FaDiagramProject,
    FaMagnifyingGlass, FaSitemap, FaArrowRight, FaPlus, FaCircleCheck, FaSpinner,
} from 'react-icons/fa6';
import './AgentLibraryDialog.css';
import {
    AGENT_TEMPLATES, FLOW_TEMPLATES,
    type AgentTemplate, type FlowTemplate,
} from './agentLibraryData';

type Tab = 'agents' | 'flows';

interface Props {
    onClose: () => void;
    onUseAgent?: (agent: AgentTemplate) => Promise<void> | void;
    onUseFlow?: (flow: FlowTemplate) => Promise<void> | void;
}

const AgentLibraryDialog: React.FC<Props> = ({ onClose, onUseAgent, onUseFlow }) => {
    const [tab, setTab] = useState<Tab>('agents');
    const [search, setSearch] = useState('');
    const [busyId, setBusyId] = useState<string | null>(null);
    const [doneId, setDoneId] = useState<string | null>(null);

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

    const handleUseAgent = async (agent: AgentTemplate) => {
        if (!onUseAgent) return;
        setBusyId(agent.id);
        try {
            await onUseAgent(agent);
            setDoneId(agent.id);
            setTimeout(() => setDoneId(null), 1500);
        } catch (err) {
            alert(`Erro ao adicionar: ${err}`);
        } finally {
            setBusyId(null);
        }
    };

    const handleUseFlow = async (flow: FlowTemplate) => {
        if (!onUseFlow) return;
        setBusyId(flow.id);
        try {
            await onUseFlow(flow);
            setDoneId(flow.id);
            setTimeout(() => setDoneId(null), 1500);
        } catch (err) {
            alert(`Erro ao adicionar: ${err}`);
        } finally {
            setBusyId(null);
        }
    };

    const renderButton = (id: string, color: string, onClick: () => void, label = 'Usar') => {
        const isBusy = busyId === id;
        const isDone = doneId === id;
        return (
            <button
                className="lib-use-btn"
                style={{ '--btn-color': color } as React.CSSProperties}
                onClick={onClick}
                title="Adicionar à sua sidebar"
                disabled={isBusy}
            >
                {isBusy ? (<><FaSpinner size={11} /> Adicionando...</>)
                    : isDone ? (<><FaCircleCheck size={11} /> Adicionado!</>)
                    : (<><FaPlus size={11} /> {label}</>)}
            </button>
        );
    };

    return (
        <div className="lib-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label="Biblioteca">
            <div className="lib-dialog" onClick={(e) => e.stopPropagation()}>
                <div className="lib-header">
                    <div className="lib-header-left">
                        <span className="lib-header-icon"><FaBookOpen size={16} /></span>
                        <div>
                            <h2 className="lib-title">Biblioteca</h2>
                            <p className="lib-subtitle">Agentes e fluxos prontos para usar</p>
                        </div>
                    </div>
                    <button className="lib-close" onClick={onClose} aria-label="Fechar"><FaXmark size={16} /></button>
                </div>

                <div className="lib-toolbar">
                    <div className="lib-tabs">
                        <button className={`lib-tab ${tab === 'agents' ? 'active' : ''}`} onClick={() => setTab('agents')}>
                            <FaRobot size={13} /> Agentes
                            <span className="lib-tab-count">{AGENT_TEMPLATES.length}</span>
                        </button>
                        <button className={`lib-tab ${tab === 'flows' ? 'active' : ''}`} onClick={() => setTab('flows')}>
                            <FaDiagramProject size={13} /> Fluxos
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
                                                    <span key={t} className="lib-tag" style={{ '--tag-color': agent.color } as React.CSSProperties}>{t}</span>
                                                ))}
                                            </div>
                                        </div>
                                        {renderButton(agent.id, agent.color, () => handleUseAgent(agent))}
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
                                                    <span key={t} className="lib-tag" style={{ '--tag-color': flow.color } as React.CSSProperties}>{t}</span>
                                                ))}
                                            </div>
                                        </div>
                                        {renderButton(flow.id, flow.color, () => handleUseFlow(flow))}
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
