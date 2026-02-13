import React, { useState } from 'react';
import {
    FaPlus, FaGear, FaCircleUser,
    FaChevronDown, FaChevronRight,
    FaScaleBalanced, FaFileLines, FaMagnifyingGlass,
    FaBookOpen, FaPenNib
} from 'react-icons/fa6';
import './Sidebar.css';

const agents = [
    { id: 'ementador', icon: <FaFileLines />, name: 'Ementador', desc: 'Gera ementas a partir de decisões', color: '#2563eb' },
    { id: 'resumidor', icon: <FaBookOpen />, name: 'Resumidor', desc: 'Resumos consolidados de processos', color: '#7c3aed' },
    { id: 'analisador', icon: <FaMagnifyingGlass />, name: 'Analisador de Peças', desc: 'Analisa peças processuais', color: '#059669' },
    { id: 'consultor', icon: <FaScaleBalanced />, name: 'Consultor Jurisprudencial', desc: 'Consulta jurisprudência', color: '#d97706' },
    { id: 'redator', icon: <FaPenNib />, name: 'Redator de Minutas', desc: 'Auxilia na redação de minutas', color: '#dc2626' },
];

const Sidebar = ({ isOpen, onToggle, history = [] }) => {
    const [agentsOpen, setAgentsOpen] = useState(true);
    const [activeAgent, setActiveAgent] = useState(null);

    return (
        <nav className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
            {/* New Chat Button */}
            <div className="sidebar-top">
                <button className="new-chat-btn" id="new-chat-btn">
                    <FaPlus size={14} />
                    <span>Novo Chat</span>
                </button>
            </div>

            {/* Agents Section */}
            <div className="sidebar-section">
                <button
                    className="section-toggle"
                    onClick={() => setAgentsOpen(!agentsOpen)}
                    aria-expanded={agentsOpen}
                >
                    <span className="section-toggle-icon">
                        {agentsOpen ? <FaChevronDown size={10} /> : <FaChevronRight size={10} />}
                    </span>
                    <span className="section-label">Agentes Jurídicos</span>
                </button>

                <div className={`agents-list ${agentsOpen ? 'expanded' : 'collapsed'}`}>
                    {agents.map((agent) => (
                        <button
                            key={agent.id}
                            className={`agent-card ${activeAgent === agent.id ? 'active' : ''}`}
                            onClick={() => setActiveAgent(agent.id)}
                            id={`agent-${agent.id}`}
                        >
                            <span className="agent-icon" style={{ color: agent.color }}>
                                {agent.icon}
                            </span>
                            <div className="agent-info">
                                <span className="agent-name">{agent.name}</span>
                                <span className="agent-desc">{agent.desc}</span>
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Divider */}
            <div className="sidebar-divider" />

            {/* History Section — only shows when there are conversations */}
            <div className="sidebar-history">
                {history.length > 0 ? (
                    <>
                        <span className="section-label-static">Histórico</span>
                        {history.map((group, gi) => (
                            <div key={gi} className="history-group">
                                <span className="history-group-label">{group.label}</span>
                                {group.items.map((item, ii) => (
                                    <button key={ii} className="history-item" id={`history-${gi}-${ii}`}>
                                        {item}
                                    </button>
                                ))}
                            </div>
                        ))}
                    </>
                ) : (
                    <div className="history-empty">
                        <span className="section-label-static">Histórico</span>
                        <p className="history-empty-text">Suas conversas aparecerão aqui</p>
                    </div>
                )}
            </div>

            {/* Bottom Icons */}
            <div className="sidebar-bottom">
                <div className="sidebar-divider" />
                <div className="sidebar-bottom-icons">
                    <button className="sidebar-icon-btn" aria-label="Configurações" id="btn-settings">
                        <FaGear />
                    </button>
                    <button className="sidebar-icon-btn" aria-label="Perfil" id="btn-profile">
                        <FaCircleUser />
                    </button>
                </div>
            </div>
        </nav>
    );
};

export default Sidebar;
