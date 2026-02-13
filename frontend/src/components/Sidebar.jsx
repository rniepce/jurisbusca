import React, { useState } from 'react';
import {
    FaPlus, FaGear, FaCircleUser,
    FaChevronDown, FaChevronRight,
    FaScaleBalanced, FaFileLines, FaMagnifyingGlass,
    FaBookOpen, FaPenNib
} from 'react-icons/fa6';
import './Sidebar.css';
import agentDefinitions from '../config/agents';

// Map string icon names to React components
const iconMap = {
    FaScaleBalanced: <FaScaleBalanced />,
    FaFileLines: <FaFileLines />,
    FaMagnifyingGlass: <FaMagnifyingGlass />,
    FaBookOpen: <FaBookOpen />,
    FaPenNib: <FaPenNib />,
};

const Sidebar = ({ isOpen, onToggle, history = [], activeAgent, onAgentSelect, onNewChat }) => {
    const [agentsOpen, setAgentsOpen] = useState(true);

    const handleAgentClick = (agent) => {
        if (onAgentSelect) onAgentSelect(agent);
    };

    const handleNewChat = () => {
        if (onNewChat) onNewChat();
    };

    return (
        <nav className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
            {/* New Chat Button */}
            <div className="sidebar-top">
                <button className="new-chat-btn" id="new-chat-btn" onClick={handleNewChat}>
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
                    {agentDefinitions.map((agent) => (
                        <button
                            key={agent.id}
                            className={`agent-card ${activeAgent?.id === agent.id ? 'active' : ''}`}
                            onClick={() => handleAgentClick(agent)}
                            id={`agent-${agent.id}`}
                        >
                            <span className="agent-icon" style={{ color: agent.color }}>
                                {iconMap[agent.icon] || <FaScaleBalanced />}
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
