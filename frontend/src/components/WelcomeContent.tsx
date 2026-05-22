import React from 'react';
import { FaScaleBalanced, FaGavel, FaPaperclip, FaRobot } from 'react-icons/fa6';
import { useAuth } from './AuthContext';
import QuickActionChips from './QuickActionChips';
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
    onQuickAction?: (prompt: string) => void;
}

function getGreeting(): string {
    const hour = new Date().getHours();
    if (hour < 5) return 'Boa madrugada';
    if (hour < 12) return 'Bom dia';
    if (hour < 18) return 'Boa tarde';
    return 'Boa noite';
}

function getFirstName(fullName?: string | null, email?: string | null): string {
    if (fullName) return fullName.trim().split(/\s+/)[0];
    if (email) return email.split('@')[0];
    return '';
}

const WelcomeContent: React.FC<Props> = ({
    onOpenJurisprudencia,
    onOpenSustentacao,
    onAttachFile,
    onOpenAgents,
    onQuickAction,
}) => {
    const { user } = useAuth();
    const firstName = getFirstName(
        user?.user_metadata?.full_name as string | undefined,
        user?.email
    );
    const greeting = `${getGreeting()}${firstName ? `, ${firstName}` : ''}`;

    const actions: Action[] = [
        {
            id: 'sustentacao',
            icon: <FaGavel size={22} />,
            title: 'Sessão',
            desc: 'Apoio para audiência ou sessão de julgamento',
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
            <span className="welcome-greeting">{greeting}</span>
            <h1 className="welcome-title">Como posso te ajudar hoje?</h1>

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

            {onQuickAction && <QuickActionChips onAction={onQuickAction} />}
        </div>
    );
};

export default WelcomeContent;
