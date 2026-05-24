import React, { useState } from 'react';
import {
    FaPlus,
    FaChevronDown, FaChevronRight,
    FaScaleBalanced,
    FaComments,
    FaWandMagicSparkles, FaTrash, FaShareNodes, FaEllipsisVertical, FaRobot,
    FaGavel, FaLayerGroup, FaBook, FaDiagramProject, FaSitemap,
} from 'react-icons/fa6';
import { useNavigate, useLocation } from 'react-router-dom';
import UserMenu from './UserMenu';
import logoSvg from '../assets/logo.svg';
import './Sidebar.css';


const Sidebar = ({
    isOpen,
    onToggle,
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
    onOpenMemory,
    onOpenSustentacao,
}: any) => {
    const navigate = useNavigate();
    const location = useLocation();
    const [customAgentsOpen, setCustomAgentsOpen] = useState(true);
    const [menuOpenId, setMenuOpenId] = useState(null);
    const [searchQuery, setSearchQuery] = useState('');

    const goTo = (path: string) => {
        navigate(path);
        if (onCloseMobile) onCloseMobile();
    };

    const navItems = [
        { path: '/lote', label: 'Análise em lote', icon: <FaLayerGroup size={13} /> },
        { path: '/jurisprudencia', label: 'Jurisprudência', icon: <FaScaleBalanced size={13} /> },
        { path: '/modelos', label: 'Gestor de Modelos', icon: <FaBook size={13} /> },
        { path: '/engenheiro-prompts', label: 'Engenheiro de Prompts', icon: <FaWandMagicSparkles size={13} /> },
        { path: '/flows', label: 'Construtor de Fluxos', icon: <FaDiagramProject size={13} /> },
    ];

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
            {/* Brand — Assistente TJMG (sempre visível; texto some quando colapsada) */}
            <button
                type="button"
                className="sidebar-brand"
                onClick={handleNewChat}
                title="Assistente TJMG — clique para novo chat"
                aria-label="Assistente TJMG"
            >
                <img src={logoSvg} alt="" className="sidebar-brand-logo" />
                <span className="sidebar-brand-text">Assistente TJMG</span>
            </button>

            {/* New Chat Button */}
            <div className="sidebar-top">
                <button className="new-chat-btn" id="new-chat-btn" onClick={handleNewChat}>
                    <FaPlus size={14} />
                    <span>Novo Chat</span>
                </button>
                <button
                    className="sustentacao-btn"
                    id="btn-sustentacao"
                    onClick={() => {
                        if (onOpenSustentacao) onOpenSustentacao();
                        if (onCloseMobile) onCloseMobile();
                    }}
                    title="Painel de apoio para sessão / audiência"
                >
                    <FaGavel size={13} />
                    <span>Sessão / Audiência</span>
                </button>
            </div>

            {/* Navegação direta para painéis principais */}
            <div className="sidebar-nav">
                {navItems.map((item) => {
                    const active = location.pathname === item.path;
                    return (
                        <button
                            key={item.path}
                            className={`sidebar-nav-link ${active ? 'active' : ''}`}
                            onClick={() => goTo(item.path)}
                            title={item.label}
                        >
                            <span className="sidebar-nav-icon" aria-hidden="true">{item.icon}</span>
                            <span>{item.label}</span>
                        </button>
                    );
                })}
            </div>

            {/* Agents Section */}
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
                    {customAgents.length === 0 && (
                        <p className="agents-empty-text">
                            Nenhum agente criado ainda. Clique em + para criar seu primeiro agente.
                        </p>
                    )}
                    {customAgents.map((agent) => {
                        const isOrch = (agent as any).type === 'orquestrador';
                        return (
                        <div key={agent.id} className="custom-agent-wrapper">
                            <button
                                className={`agent-card ${activeAgent?.id === agent.id ? 'active' : ''} ${isOrch ? 'orquestrador' : ''}`}
                                onClick={() => handleAgentClick(agent)}
                                id={`agent-custom-${agent.id}`}
                            >
                                <span className="agent-icon" style={{ color: agent.color || '#8B5CF6' }}>
                                    {isOrch ? <FaSitemap /> : <FaRobot />}
                                </span>
                                <div className="agent-info">
                                    <div className="agent-name-row">
                                        <span className="agent-name">{agent.name}</span>
                                        {isOrch && <span className="agent-badge agent-badge-orquestrador">Orquestrador</span>}
                                    </div>
                                    <span className="agent-desc">{(agent as any).description || (isOrch ? 'Fluxo orquestrador' : 'Agente personalizado')}</span>
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
                        );
                    })}

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

                {/* Search input */}
                {history.length > 0 && (
                    <input
                        type="search"
                        className="sidebar-search"
                        placeholder="Buscar conversa..."
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                        aria-label="Buscar conversa no histórico"
                    />
                )}

                {history.length > 0 ? (
                    (() => {
                        // Filter all items across groups
                        const filteredGroups = history.map((group) => ({
                            ...group,
                            items: group.items.filter(
                                (item) => !searchQuery || item.title?.toLowerCase().includes(searchQuery.toLowerCase())
                            ),
                        })).filter((g) => g.items.length > 0);

                        if (filteredGroups.length === 0) {
                            return <p className="history-empty-text history-no-results">Nenhuma conversa encontrada</p>;
                        }

                        return filteredGroups.map((group, gi) => (
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
                        ));
                    })()
                ) : (
                    <p className="history-empty-text">Suas conversas aparecerão aqui</p>
                )}
            </div>

            {/* User menu (Configurações + Sair) */}
            <div className="sidebar-bottom">
                <div className="sidebar-divider" />
                <div className="sidebar-bottom-user">
                    <UserMenu onOpenMemory={onOpenMemory} />
                </div>
            </div>
        </nav>
    );
};

export default Sidebar;
