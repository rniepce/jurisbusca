import { FaPenNib, FaScaleBalanced, FaClipboardCheck, FaListUl, FaGavel } from 'react-icons/fa6';
import './QuickActionChips.css';

export interface QuickAction {
    id: string;
    label: string;
    icon: React.ReactNode;
    /** Prompt text that fills the textarea when clicked (user can still edit and send). */
    prompt: string;
}

const DEFAULT_ACTIONS: QuickAction[] = [
    {
        id: 'minuta',
        label: 'Elaborar minuta',
        icon: <FaPenNib size={13} />,
        prompt:
            'Elabore uma minuta de decisão sobre o processo anexado, seguindo o estilo dos modelos indexados. Estruture com: relatório, fundamentação (com citação de lei e jurisprudência aplicável) e dispositivo.',
    },
    {
        id: 'analise',
        label: 'Análise jurídica',
        icon: <FaScaleBalanced size={13} />,
        prompt:
            'Faça uma análise jurídica completa do processo anexado: pedidos, fundamentos do autor, defesa, provas produzidas, e indique os pontos controvertidos a serem decididos.',
    },
    {
        id: 'revisar',
        label: 'Revisar minuta',
        icon: <FaClipboardCheck size={13} />,
        prompt:
            'Revise a minuta acima quanto a: conformidade fática com o processo, fundamentação jurídica adequada, citações corretas, coerência interna e clareza redacional. Aponte ajustes objetivos.',
    },
    {
        id: 'resumir',
        label: 'Resumir processo',
        icon: <FaListUl size={13} />,
        prompt:
            'Faça um resumo executivo do processo anexado em até 10 linhas: partes, pedidos, principais fatos, decisões já tomadas e o que está pendente de julgamento.',
    },
    {
        id: 'sessao',
        label: 'Preparar sessão',
        icon: <FaGavel size={13} />,
        prompt:
            'Prepare um briefing de sessão sobre o processo anexado: contextualização breve, ponto controvertido principal, voto sugerido com fundamentação resumida e possíveis pontos de questionamento.',
    },
];

interface Props {
    onAction: (prompt: string) => void;
    actions?: QuickAction[];
}

/**
 * Horizontal pill row of quick prompt suggestions. Clicking a chip fills the
 * chat textarea via the `onAction` callback (the user can still edit before sending).
 */
export default function QuickActionChips({ onAction, actions = DEFAULT_ACTIONS }: Props) {
    return (
        <div className="quick-actions" role="list" aria-label="Sugestões rápidas">
            {actions.map((a) => (
                <button
                    key={a.id}
                    type="button"
                    role="listitem"
                    className="quick-action-chip"
                    onClick={() => onAction(a.prompt)}
                    title={`Preencher: "${a.prompt.slice(0, 60)}…"`}
                >
                    <span className="quick-action-icon" aria-hidden="true">{a.icon}</span>
                    <span>{a.label}</span>
                </button>
            ))}
        </div>
    );
}
