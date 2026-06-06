import { useEffect, useState } from 'react';
import {
    FaXmark, FaClockRotateLeft, FaTrash, FaArrowLeft, FaFileCsv, FaTrophy,
} from 'react-icons/fa6';
import {
    listArenaComparisons, getArenaComparison, deleteArenaComparison,
    exportArenaComparisonsCsv,
    type ArenaComparisonSummary, type ArenaComparisonDetail,
} from '../services/api';
import { formatMarkdown } from '../utils/markdown';
import './ArenaPanel.css';

interface Props {
    open: boolean;
    onClose: () => void;
}

function voteLabel(c: { vote: string; model_a: string; model_b: string }): string {
    switch (c.vote) {
        case 'A': return `${c.model_a} (A)`;
        case 'B': return `${c.model_b} (B)`;
        case 'tie': return 'Empate';
        case 'both_bad': return 'Ambas ruins';
        default: return '— sem voto —';
    }
}
function fmtLatency(ms: number): string {
    if (!ms) return '—';
    return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)}s`;
}
function fmtCost(usd: number): string {
    return usd ? `$${usd.toFixed(4)}` : '—';
}

export default function ArenaHistoryModal({ open, onClose }: Props) {
    const [rows, setRows] = useState<ArenaComparisonSummary[]>([]);
    const [loading, setLoading] = useState(false);
    const [selected, setSelected] = useState<ArenaComparisonDetail | null>(null);
    const [loadingDetail, setLoadingDetail] = useState(false);

    useEffect(() => {
        if (!open) return;
        setSelected(null);
        setLoading(true);
        listArenaComparisons()
            .then(setRows)
            .catch(() => setRows([]))
            .finally(() => setLoading(false));
    }, [open]);

    const openRow = async (id: string) => {
        setLoadingDetail(true);
        try {
            setSelected(await getArenaComparison(id));
        } catch (err) {
            alert(`Erro ao carregar comparação: ${err}`);
        } finally {
            setLoadingDetail(false);
        }
    };

    const removeRow = async (id: string) => {
        if (!confirm('Apagar este registro de comparação?')) return;
        try {
            await deleteArenaComparison(id);
            setRows((rs) => rs.filter((r) => r.id !== id));
            if (selected?.id === id) setSelected(null);
        } catch (err) {
            alert(`Erro ao apagar: ${err}`);
        }
    };

    if (!open) return null;

    return (
        <div className="arena-modal-backdrop" onClick={onClose}>
            <div className="arena-modal" onClick={(e) => e.stopPropagation()}>
                <div className="arena-modal-header">
                    <span className="arena-modal-title">
                        {selected ? (
                            <>
                                <button className="arena-ghost-btn" onClick={() => setSelected(null)} title="Voltar">
                                    <FaArrowLeft size={13} />
                                </button>
                                Comparação
                            </>
                        ) : (
                            <><FaClockRotateLeft size={14} /> Histórico de comparações</>
                        )}
                    </span>
                    <div className="arena-header-actions">
                        {!selected && rows.length > 0 && (
                            <button className="arena-ghost-btn" onClick={() => void exportArenaComparisonsCsv()} title="Exportar CSV">
                                <FaFileCsv size={13} /> CSV
                            </button>
                        )}
                        <button className="arena-ghost-btn" onClick={onClose} title="Fechar">
                            <FaXmark size={15} />
                        </button>
                    </div>
                </div>

                {!selected && loading && <p className="arena-modal-empty">Carregando…</p>}
                {!selected && !loading && rows.length === 0 && (
                    <p className="arena-modal-empty">Nenhuma comparação registrada ainda.</p>
                )}

                {!selected && rows.length > 0 && (
                    <div className="arena-hist-list">
                        {rows.map((r) => (
                            <div key={r.id} className="arena-hist-item" onClick={() => openRow(r.id)}>
                                <div className="arena-hist-main">
                                    <span className="arena-hist-models">{r.model_a} <em>×</em> {r.model_b}</span>
                                    <span className="arena-hist-date">
                                        {new Date(r.created_at + 'Z').toLocaleString('pt-BR')}
                                    </span>
                                </div>
                                <div className="arena-hist-meta">
                                    <span className={`arena-hist-vote ${r.vote || 'none'}`}>
                                        {r.vote && <FaTrophy size={10} />} {voteLabel(r)}
                                    </span>
                                    <span title="Custo total">💰 {fmtCost(r.cost_usd_a + r.cost_usd_b)}</span>
                                    <button
                                        className="arena-hist-del"
                                        onClick={(e) => { e.stopPropagation(); void removeRow(r.id); }}
                                        title="Apagar"
                                    >
                                        <FaTrash size={10} />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {selected && loadingDetail && <p className="arena-modal-empty">Carregando detalhes…</p>}
                {selected && !loadingDetail && <ArenaDetailView c={selected} />}
            </div>
        </div>
    );
}

function ArenaDetailView({ c }: { c: ArenaComparisonDetail }) {
    return (
        <div className="arena-detail">
            <div className="arena-detail-summary">
                <div><strong>Data:</strong> {new Date(c.created_at + 'Z').toLocaleString('pt-BR')}</div>
                <div><strong>Voto:</strong> {voteLabel(c)}</div>
                {c.agent_name && <div><strong>Agente:</strong> {c.agent_name}</div>}
                {c.justification && <div className="arena-detail-just"><strong>Justificativa:</strong> {c.justification}</div>}
            </div>

            <div className="arena-detail-block">
                <h4>Prompt</h4>
                <pre className="arena-detail-pre">{c.prompt || '—'}</pre>
            </div>
            {c.uploaded_text && (
                <details className="arena-detail-block">
                    <summary>Documento anexado ({c.uploaded_text.length.toLocaleString('pt-BR')} caracteres)</summary>
                    <pre className="arena-detail-pre">{c.uploaded_text}</pre>
                </details>
            )}

            <div className="arena-cols">
                <div className="arena-col">
                    <div className="arena-col-head">
                        <span className="arena-col-title">{c.model_a} (A)</span>
                        <span className="arena-col-status done">
                            ⏱ {fmtLatency(c.latency_ms_a)} · 🔤 {c.input_tokens_a}/{c.output_tokens_a} · 💰 {fmtCost(c.cost_usd_a)}
                        </span>
                    </div>
                    <div className="arena-col-body markdown"
                        dangerouslySetInnerHTML={{ __html: formatMarkdown(c.response_a || '') }} />
                </div>
                <div className="arena-col">
                    <div className="arena-col-head">
                        <span className="arena-col-title">{c.model_b} (B)</span>
                        <span className="arena-col-status done">
                            ⏱ {fmtLatency(c.latency_ms_b)} · 🔤 {c.input_tokens_b}/{c.output_tokens_b} · 💰 {fmtCost(c.cost_usd_b)}
                        </span>
                    </div>
                    <div className="arena-col-body markdown"
                        dangerouslySetInnerHTML={{ __html: formatMarkdown(c.response_b || '') }} />
                </div>
            </div>
        </div>
    );
}
