import React from 'react';
import { FaScaleBalanced, FaGavel, FaPaperclip, FaRobot } from 'react-icons/fa6';
import './WelcomeContent.css';

interface Action {
    id: string;
    icon: React.ReactNode;
    title: string;
    desc: string;
    onClick?: () => void;
    accent?: 'blue' | 'amber' | 'purple' | 'green';
}

interface Props {
    onOpenJurisprudencia?: () => void;
    onOpenSustentacao?: () => void;
    onAttachFile?: () => void;
    onOpenAgents?: () => void;
}

const WelcomeContent: React.FC<Props> = ({
    onOpenJurisprudencia,
    onOpenSustentacao,
    onAttachFile,
    onOpenAgents,
}) => {
    const actions: Action[] = [
        {
            id: 'sustentacao',
            icon: <FaGavel size={22} />,
            title: 'Sustentação Oral',
            desc: 'Apoio para audiência ou sessão',
            onClick: onOpenSustentacao,
            accent: 'amber',
        },
        {
            id: 'jurisprudencia',
            icon: <FaScaleBalanced size={22} />,
            title: 'Pesquisar Jurisprudência',
            desc: 'Acórdãos do TJMG com IA',
            onClick: onOpenJurisprudencia,
            accent: 'blue',
        },
        {
            id: 'analisar',
            icon: <FaPaperclip size={22} />,
            title: 'Analisar processo',
            desc: 'Anexe um PDF para começar a conversa',
            onClick: onAttachFile,
            accent: 'purple',
        },
        {
            id: 'agentes',
            icon: <FaRobot size={22} />,
            title: 'Agentes Jurídicos',
            desc: 'Gabinete 2.0, Revisor QA e mais',
            onClick: onOpenAgents,
            accent: 'green',
        },
    ];

    return (
        <div className="welcome-container">
            <h1 className="welcome-title">Bem-vindo ao Assistente TJMG</h1>
            <p className="welcome-hint">
                Por onde quer começar?
            </p>

            <div className="welcome-actions" role="list">
                {actions.map((a) => (
                    <button
                        key={a.id}
                        type="button"
                        role="listitem"
                        className={`welcome-action-card accent-${a.accent || 'blue'}`}
                        onClick={a.onClick}
                        disabled={!a.onClick}
                        id={`btn-welcome-${a.id}`}
                    >
                        <span className="welcome-action-icon" aria-hidden="true">{a.icon}</span>
                        <div className="welcome-action-text">
                            <span className="welcome-action-title">{a.title}</span>
                            <span className="welcome-action-desc">{a.desc}</span>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
};

export default WelcomeContent;
