import React from 'react';
import { FaRegSquarePlus, FaRegFolder, FaGear, FaCircleUser } from 'react-icons/fa6';
import './Sidebar.css';

const historyData = [
    { label: 'Hoje', items: ['Resumo de processo'] },
    { label: 'Ontem', items: ['Resumo de processo', 'Resumo de processo'] },
    { label: '7 dias', items: ['Resumo de processo', 'Resumo de processo', 'Resumo de processo', 'Resumo de processo'] },
    { label: '30 dias', items: ['Resumo de processo', 'Resumo de processo', 'Resumo de processo'] },
];

const Sidebar = ({ isOpen, onToggle }) => {
    return (
        <nav className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
            {/* New Chat Button */}
            <div className="sidebar-top">
                <button className="new-chat-btn">
                    <FaRegSquarePlus size={16} />
                    <span>Novo chat</span>
                </button>
            </div>

            {/* History */}
            <div className="sidebar-history">
                <span className="history-label-main">HISTÓRICO</span>
                {historyData.map((group, gi) => (
                    <div key={gi} className="history-group">
                        <span className="history-group-label">{group.label}</span>
                        {group.items.map((item, ii) => (
                            <button key={ii} className="history-item">
                                {item}
                            </button>
                        ))}
                    </div>
                ))}
                <a href="#" className="history-see-all">Ver histórico completo...</a>
            </div>

            {/* Bottom Icons */}
            <div className="sidebar-bottom">
                <div className="sidebar-divider" />
                <div className="sidebar-bottom-icons">
                    <button className="sidebar-icon-btn" aria-label="Pastas"><FaRegFolder /></button>
                    <button className="sidebar-icon-btn" aria-label="Configurações"><FaGear /></button>
                    <button className="sidebar-icon-btn" aria-label="Perfil"><FaCircleUser /></button>
                </div>
            </div>
        </nav>
    );
};

export default Sidebar;
