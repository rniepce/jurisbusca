import React, { useState } from 'react';
import { FaMagnifyingGlass, FaBookOpen } from 'react-icons/fa6';
import FutureFeatureAlert from './FutureFeatureAlert';

interface Props {
    onOpenJurisprudencia?: () => void;
}

const ResearchActions: React.FC<Props> = ({ onOpenJurisprudencia }) => {
    const [showFutureAlert, setShowFutureAlert] = useState(false);

    return (
        <section className="sust-card">
            <h3 className="sust-card-title"><FaMagnifyingGlass /> Pesquisa externa</h3>
            <div className="sust-research-actions">
                <button
                    className="sust-btn-primary"
                    onClick={onOpenJurisprudencia}
                    disabled={!onOpenJurisprudencia}
                    title="Buscar acórdãos no banco de jurisprudência do TJMG"
                >
                    <FaMagnifyingGlass /> Consultar jurisprudência
                </button>
                <button
                    className="sust-btn-secondary-light"
                    onClick={() => setShowFutureAlert(true)}
                    title="Verificar precedentes vinculantes (STF, STJ, IRDR)"
                >
                    <FaBookOpen /> Verificar precedentes
                </button>
            </div>
            {showFutureAlert && (
                <FutureFeatureAlert
                    feature="Verificação de precedentes"
                    description="Identificação automática de precedentes vinculantes (STF, STJ, IRDR, IAC) — disponível em versão futura."
                    onDismiss={() => setShowFutureAlert(false)}
                />
            )}
        </section>
    );
};

export default ResearchActions;
