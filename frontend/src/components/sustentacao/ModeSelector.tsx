import React from 'react';
import { FaGavel, FaScaleBalanced, FaBookOpen, FaPlay } from 'react-icons/fa6';
import type { TipoAto, Modo } from '../../services/api';

interface Props {
    tipoAto: TipoAto;
    modo: Modo;
    onSelect: (tipo: TipoAto, modo: Modo) => void;
}

interface Combo {
    tipoAto: TipoAto;
    modo: Modo;
    title: string;
    desc: string;
    instancia: string;
    icon: React.ReactNode;
    accent: 'amber' | 'blue' | 'green' | 'purple';
}

const COMBOS: Combo[] = [
    {
        tipoAto: 'sustentacao',
        modo: 'preparacao',
        title: 'Sustentação · Preparação',
        instancia: '2ª instância',
        desc: 'Estudo do recurso antes da sessão de julgamento',
        icon: <><FaGavel /><FaBookOpen /></>,
        accent: 'amber',
    },
    {
        tipoAto: 'sustentacao',
        modo: 'realizacao',
        title: 'Sustentação · Realização',
        instancia: '2ª instância',
        desc: 'Apoio ao desembargador durante a sustentação oral',
        icon: <><FaGavel /><FaPlay /></>,
        accent: 'amber',
    },
    {
        tipoAto: 'audiencia',
        modo: 'preparacao',
        title: 'Audiência · Preparação',
        instancia: '1ª instância',
        desc: 'Plano de audiência: pontos controvertidos, testemunhas, quesitos',
        icon: <><FaScaleBalanced /><FaBookOpen /></>,
        accent: 'blue',
    },
    {
        tipoAto: 'audiencia',
        modo: 'realizacao',
        title: 'Audiência · Realização',
        instancia: '1ª instância',
        desc: 'Registro ao vivo: depoentes, perguntas, anotações',
        icon: <><FaScaleBalanced /><FaPlay /></>,
        accent: 'blue',
    },
];

const ModeSelector: React.FC<Props> = ({ tipoAto, modo, onSelect }) => {
    return (
        <div className="sust-mode-grid" role="radiogroup" aria-label="Tipo de ato e modo">
            {COMBOS.map((c) => {
                const active = c.tipoAto === tipoAto && c.modo === modo;
                return (
                    <button
                        key={`${c.tipoAto}-${c.modo}`}
                        type="button"
                        role="radio"
                        aria-checked={active}
                        className={`sust-mode-card accent-${c.accent} ${active ? 'active' : ''}`}
                        onClick={() => onSelect(c.tipoAto, c.modo)}
                    >
                        <div className="sust-mode-card-icon" aria-hidden="true">{c.icon}</div>
                        <div className="sust-mode-card-body">
                            <span className="sust-mode-card-tag">{c.instancia}</span>
                            <span className="sust-mode-card-title">{c.title}</span>
                            <span className="sust-mode-card-desc">{c.desc}</span>
                        </div>
                    </button>
                );
            })}
        </div>
    );
};

export default ModeSelector;
