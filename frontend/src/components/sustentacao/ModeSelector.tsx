import React from 'react';
import { FaGavel, FaScaleBalanced, FaBookOpen, FaPlay } from 'react-icons/fa6';
import type { TipoAto, Modo } from '../../services/api';

interface Props {
    tipoAto: TipoAto;
    modo: Modo;
    onTipoChange: (t: TipoAto) => void;
    onModoChange: (m: Modo) => void;
}

const ModeSelector: React.FC<Props> = ({ tipoAto, modo, onTipoChange, onModoChange }) => {
    return (
        <div className="sust-mode-selector">
            <div className="sust-mode-group">
                <span className="sust-mode-label">Tipo de ato</span>
                <div className="sust-mode-buttons">
                    <button
                        className={`sust-mode-btn ${tipoAto === 'sustentacao' ? 'active' : ''}`}
                        onClick={() => onTipoChange('sustentacao')}
                    >
                        <FaGavel /> Sustentação Oral <small>(2ª inst.)</small>
                    </button>
                    <button
                        className={`sust-mode-btn ${tipoAto === 'audiencia' ? 'active' : ''}`}
                        onClick={() => onTipoChange('audiencia')}
                    >
                        <FaScaleBalanced /> Audiência <small>(1ª inst.)</small>
                    </button>
                </div>
            </div>
            <div className="sust-mode-group">
                <span className="sust-mode-label">Modo</span>
                <div className="sust-mode-buttons">
                    <button
                        className={`sust-mode-btn ${modo === 'preparacao' ? 'active' : ''}`}
                        onClick={() => onModoChange('preparacao')}
                    >
                        <FaBookOpen /> Preparação
                    </button>
                    <button
                        className={`sust-mode-btn ${modo === 'realizacao' ? 'active' : ''}`}
                        onClick={() => onModoChange('realizacao')}
                    >
                        <FaPlay /> Realização
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ModeSelector;
