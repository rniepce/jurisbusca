import React from 'react';
import { FaScaleBalanced } from 'react-icons/fa6';
import './WelcomeContent.css';

const WelcomeContent = ({ onOpenJurisprudencia }) => {
    return (
        <div className="welcome-container">
            <h1 className="welcome-title">Bem-vindo ao Assistente TJMG</h1>
            <p className="welcome-hint">
                Selecione um agente na barra lateral e envie um processo para começar.
            </p>

            <div className="welcome-actions">
                <button
                    className="welcome-action-card"
                    onClick={onOpenJurisprudencia}
                    id="btn-open-jurisprudencia"
                >
                    <FaScaleBalanced size={22} className="welcome-action-icon" />
                    <div className="welcome-action-text">
                        <span className="welcome-action-title">Pesquisar Jurisprudência</span>
                        <span className="welcome-action-desc">Busque acórdãos do TJMG por palavras-chave</span>
                    </div>
                </button>
            </div>
        </div>
    );
};

export default WelcomeContent;
