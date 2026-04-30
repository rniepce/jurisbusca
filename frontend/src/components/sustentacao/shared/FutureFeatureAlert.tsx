import React, { useState } from 'react';
import { FaCircleInfo, FaXmark } from 'react-icons/fa6';

interface Props {
    feature: string;
    description?: string;
    onDismiss?: () => void;
}

const FutureFeatureAlert: React.FC<Props> = ({ feature, description, onDismiss }) => {
    const [open, setOpen] = useState(true);
    if (!open) return null;
    return (
        <div className="sust-future-alert">
            <FaCircleInfo className="sust-future-alert-icon" />
            <div className="sust-future-alert-body">
                <strong>{feature}</strong>
                <span>{description || 'Funcionalidade prevista para versão futura.'}</span>
            </div>
            <button
                className="sust-future-alert-close"
                onClick={() => { setOpen(false); onDismiss?.(); }}
                aria-label="Fechar"
            >
                <FaXmark />
            </button>
        </div>
    );
};

export default FutureFeatureAlert;
