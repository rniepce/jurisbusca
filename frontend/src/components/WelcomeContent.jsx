import React from 'react';
import './WelcomeContent.css';

const WelcomeContent = () => {
    return (
        <div className="welcome-container">
            <h1 className="welcome-title">Bem-vindo ao Jurisbusca</h1>
            <p className="welcome-hint">
                Selecione um agente na barra lateral e envie um processo para começar.
            </p>
        </div>
    );
};

export default WelcomeContent;
