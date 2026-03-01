import React, { useState } from 'react';
import {
    FaPlus, FaGear, FaCircleUser,
    FaChevronDown, FaChevronRight,
    FaScaleBalanced, FaFileLines, FaMagnifyingGlass,
    FaBookOpen, FaPenNib, FaComments, FaClipboardCheck,
    FaWandMagicSparkles, FaTrash, FaShareNodes, FaEllipsisVertical, FaRobot
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
    FaClipboardCheck: <FaClipboardCheck />,
};

const Sidebar = ({
    isOpen,
    history = [],
    activeAgent,
    onAgentSelect,
    onNewChat,
    onLoadChat,
    onCloseMobile,
    customAgents = [],
    onCreateAgent,
    onDeleteAgent,
    onShareAgent,
}) => {
    const [agentsOpen, setAgentsOpen] = useState(true);
    const [customAgentsOpen, setCustomAgentsOpen] = useState(true);
    const [menuOpenId, setMenuOpenId] = useState(null);

    const handleAgentClick = (agent) => {
        if (onAgentSelect) onAgentSelect(agent);
        if (onCloseMobile) onCloseMobile();
    };

    const handleNewChat = () => {
        if (onNewChat) onNewChat();
        if (onCloseMobile) onCloseMobile();
    };

    const toggleMenu = (e, agentId) => {
        e.stopPropagation();
        setMenuOpenId(menuOpenId === agentId ? null : agentId);
    };

    const handleDelete = (e, agentId) => {
        e.stopPropagation();
        setMenuOpenId(null);
        if (onDeleteAgent) onDeleteAgent(agentId);
    };

    const handleShare = (e, agent) => {
        e.stopPropagation();
        setMenuOpenId(null);
        if (onShareAgent) onShareAgent(agent);
    };

    // Close menu when clicking outside
    React.useEffect(() => {
        const close = () => setMenuOpenId(null);
        if (menuOpenId) {
            document.addEventListener('click', close);
            return () => document.removeEventListener('click', close);
        }
    }, [menuOpenId]);

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

            {/* Custom Agents Section */}
            <div className="sidebar-section">
                <button
                    className="section-toggle"
                    onClick={() => setCustomAgentsOpen(!customAgentsOpen)}
                    aria-expanded={customAgentsOpen}
                >
                    <span className="section-toggle-icon">
                        {customAgentsOpen ? <FaChevronDown size={10} /> : <FaChevronRight size={10} />}
                    </span>
                    <span className="section-label">Meus Agentes</span>
                </button>

                <div className={`agents-list ${customAgentsOpen ? 'expanded' : 'collapsed'}`}>
                    {customAgents.map((agent) => (
                        <div key={agent.id} className="custom-agent-wrapper">
                            <button
                                className={`agent-card ${activeAgent?.id === agent.id ? 'active' : ''}`}
                                onClick={() => handleAgentClick(agent)}
                                id={`agent-custom-${agent.id}`}
                            >
                                <span className="agent-icon" style={{ color: agent.color || '#8B5CF6' }}>
                                    <FaRobot />
                                </span>
                                <div className="agent-info">
                                    <span className="agent-name">{agent.name}</span>
                                    <span className="agent-desc">Agente personalizado</span>
                                </div>
                            </button>
                            <button
                                className="agent-menu-btn"
                                onClick={(e) => toggleMenu(e, agent.id)}
                                aria-label="Opções"
                            >
                                <FaEllipsisVertical size={12} />
                            </button>
                            {menuOpenId === agent.id && (
                                <div className="agent-context-menu">
                                    <button
                                        className="context-menu-item"
                                        onClick={(e) => handleShare(e, agent)}
                                    >
                                        <FaShareNodes size={12} /> Compartilhar
                                    </button>
                                    <button
                                        className="context-menu-item danger"
                                        onClick={(e) => handleDelete(e, agent.id)}
                                    >
                                        <FaTrash size={12} /> Apagar
                                    </button>
                                </div>
                            )}
                        </div>
                    ))}

                    {/* Create Agent Button */}
                    <button
                        className="create-agent-sidebar-btn"
                        onClick={onCreateAgent}
                        id="btn-create-agent"
                    >
                        <FaWandMagicSparkles size={13} />
                        <span>Criar Agente</span>
                    </button>
                </div>
            </div>

            {/* Divider */}
            <div className="sidebar-divider" />

            {/* History Section */}
            <div className="sidebar-history">
                <span className="section-label-static">Histórico</span>
                {history.length > 0 ? (
                    <>
                        {history.map((group, gi) => (
                            <div key={gi} className="history-group">
                                <span className="history-group-label">{group.label}</span>
                                {group.items.map((item) => (
                                    <button
                                        key={item.id}
                                        className="history-item"
                                        onClick={() => {
                                            if (onLoadChat) onLoadChat(item.id);
                                            if (onCloseMobile) onCloseMobile();
                                        }}
                                        title={item.title}
                                    >
                                        <FaComments size={12} className="history-item-icon" />
                                        <span className="history-item-title">{item.title}</span>
                                    </button>
                                ))}
                            </div>
                        ))}
                    </>
                ) : (
                    <p className="history-empty-text">Suas conversas aparecerão aqui</p>
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
