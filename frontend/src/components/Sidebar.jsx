import React from 'react';
import { FaMinus, FaXmark, FaFileLines, FaLayerGroup, FaCopy } from 'react-icons/fa6';
import ActionCard from './ActionCard';
import ChatInput from './ChatInput';
import './Sidebar.css';

const Sidebar = () => {
    const handleAction = (action) => {
        console.log('Action triggered:', action);
    };

    const handleSendPrompt = (prompt) => {
        console.log('Sending prompt:', prompt);
    };

    return (
        <aside className="assistente-sidebar">
            {/* Header */}
            <header className="header">
                <div className="logo-area">
                    <img
                        src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR0Kqa5yJ0q2i9o52w_y6vjA44c20q2u9u9-w&s"
                        alt="Logo TJMG"
                        className="logo-img"
                    />
                    <span className="title">Assistente TJMG</span>
                </div>
                <div className="window-controls">
                    <button className="control-btn minimize" aria-label="Minimizar"><FaMinus /></button>
                    <button className="control-btn close" aria-label="Fechar"><FaXmark /></button>
                </div>
            </header>

            {/* Main Content */}
            <main className="content">
                <div className="welcome-section">
                    <h1 className="welcome-title">Bem-vindo,</h1>
                    <h2 className="welcome-subtitle">O que você deseja fazer?</h2>
                </div>

                <div className="action-buttons">
                    <ActionCard
                        icon={<FaFileLines />}
                        text="Gerar ementa"
                        onClick={() => handleAction('Gerar ementa')}
                    />
                    <ActionCard
                        icon={<FaLayerGroup />}
                        text="Gerar resumo consolidado"
                        onClick={() => handleAction('Gerar resumo consolidado')}
                    />
                    <ActionCard
                        icon={<FaCopy />}
                        text="Gerar resumo de peças individualizadas"
                        onClick={() => handleAction('Gerar resumo de peças')}
                    />
                </div>

                <div className="history-link">
                    <a href="#">Ver histórico completo...</a>
                </div>
            </main>

            {/* Footer */}
            <footer className="footer">
                <ChatInput onSend={handleSendPrompt} />
            </footer>
        </aside>
    );
};

export default Sidebar;
