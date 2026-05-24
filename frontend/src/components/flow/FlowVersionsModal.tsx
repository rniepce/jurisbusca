import { useEffect, useState } from 'react';
import { FaXmark, FaRotateLeft, FaClockRotateLeft } from 'react-icons/fa6';
import { listFlowVersions, restoreFlowVersion, type FlowVersion } from '../../services/api';

interface Props {
    open: boolean;
    flowId: string | null;
    onClose: () => void;
    onRestored: () => void; // chamado após restaurar, para o builder recarregar o fluxo
}

export default function FlowVersionsModal({ open, flowId, onClose, onRestored }: Props) {
    const [versions, setVersions] = useState<FlowVersion[]>([]);
    const [loading, setLoading] = useState(false);
    const [restoring, setRestoring] = useState<number | null>(null);

    useEffect(() => {
        if (!open || !flowId) return;
        setLoading(true);
        listFlowVersions(flowId)
            .then(setVersions)
            .catch(() => setVersions([]))
            .finally(() => setLoading(false));
    }, [open, flowId]);

    const handleRestore = async (versionNum: number) => {
        if (!flowId) return;
        if (!confirm(`Restaurar a versão ${versionNum}? A versão atual será salva como nova versão antes da troca.`)) return;
        setRestoring(versionNum);
        try {
            await restoreFlowVersion(flowId, versionNum);
            onRestored();
            onClose();
        } catch (err) {
            alert(`Erro ao restaurar: ${err}`);
        } finally {
            setRestoring(null);
        }
    };

    if (!open) return null;

    return (
        <div className="flow-modal-backdrop" onClick={onClose}>
            <div className="flow-modal flow-versions-modal" onClick={e => e.stopPropagation()}>
                <div className="flow-modal-header">
                    <span className="flow-modal-title">
                        <FaClockRotateLeft size={14} style={{ marginRight: 8 }} />
                        Histórico de Versões
                    </span>
                    <button onClick={onClose} className="flow-config-close"><FaXmark size={16} /></button>
                </div>

                {!flowId && (
                    <p className="flow-list-empty">Salve o fluxo primeiro para começar a registrar versões.</p>
                )}

                {flowId && loading && (
                    <p className="flow-list-empty">Carregando versões...</p>
                )}

                {flowId && !loading && versions.length === 0 && (
                    <p className="flow-list-empty">
                        Nenhuma versão antiga ainda. Cada vez que você salvar, a versão anterior será preservada aqui.
                    </p>
                )}

                {flowId && versions.length > 0 && (
                    <div className="flow-versions-list">
                        {versions.map(v => (
                            <div key={v.version_num} className="flow-version-item">
                                <span className="flow-version-num">v{v.version_num}</span>
                                <div className="flow-version-info">
                                    <span className="flow-version-name">{v.name}</span>
                                    <span className="flow-version-date">
                                        {new Date(v.created_at + 'Z').toLocaleString('pt-BR')}
                                    </span>
                                    {v.description && (
                                        <span className="flow-version-desc">{v.description}</span>
                                    )}
                                </div>
                                <button
                                    className="flow-btn flow-btn-primary"
                                    onClick={() => handleRestore(v.version_num)}
                                    disabled={restoring !== null}
                                >
                                    <FaRotateLeft size={11} />
                                    {restoring === v.version_num ? 'Restaurando...' : 'Restaurar'}
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                <span className="flow-config-help" style={{ marginTop: 4 }}>
                    💡 Restaurar uma versão antiga preserva a versão atual em uma nova entrada — você nunca perde nada.
                </span>
            </div>
        </div>
    );
}
