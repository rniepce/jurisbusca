import React from 'react';
import './WelcomeContent.css';

const actions = [
    'Gerar ementa',
    'Gerar resumo consolidado',
    'Gerar resumo de peças individualizadas',
];

const WelcomeContent = () => {
    const handleAction = (action) => {
        console.log('Action:', action);
    };

    return (
        <div className="welcome-container">
            <h1 className="welcome-title">Bem-vindo,</h1>
            <h2 className="welcome-subtitle">O que você deseja fazer?</h2>

            <div className="action-list">
                {actions.map((action, i) => (
                    <button key={i} className="action-btn" onClick={() => handleAction(action)}>
                        {action}
                    </button>
                ))}
            </div>
        </div>
    );
};

export default WelcomeContent;
