import { useEffect, useState } from 'react';
import {
    FaXmark, FaChartLine, FaTrash, FaCircleCheck, FaCircleXmark,
    FaCirclePause, FaSpinner, FaArrowLeft,
} from 'react-icons/fa6';
import {
    listFlowRuns, getFlowRun, deleteFlowRun,
    type FlowRunSummary, type FlowRunDetail,
} from '../../services/api';

interface Props {
    open: boolean;
    flowId: string | null;
    onClose: () => void;
}

function fmtDuration(ms: number): string {
    if (!ms) return '—';
    if (ms < 1000) return `${ms} ms`;
    const s = ms / 1000;
    if (s < 60) return `${s.toFixed(1)}s`;
    const m = Math.floor(s / 60);
    const rem = (s - m * 60).toFixed(0);
    return `${m}m ${rem}s`;
}

function fmtCost(usd: number): string {
    if (!usd) return '—';
    if (usd < 0.01) return `$${usd.toFixed(4)}`;
    return `$${usd.toFixed(4)}`;
}

function fmtTokens(n: number): string {
    if (!n) return '—';
    if (n < 1000) return String(n);
    if (n < 1000_000) return `${(n / 1000).toFixed(1)}k`;
    return `${(n / 1000_000).toFixed(2)}M`;
}

function StatusBadge({ status }: { status: FlowRunSummary['status'] }) {
    const map: Record<string, { icon: React.ReactNode; label: string; cls: string }> = {
        completed:        { icon: <FaCircleCheck size={11} />,  label: 'OK',        cls: 'ok' },
        error:            { icon: <FaCircleXmark size={11} />,  label: 'Erro',      cls: 'err' },
        running:          { icon: <FaSpinner size={11} />,      label: 'Rodando',   cls: 'run' },
        awaiting_human:   { icon: <FaCirclePause size={11} />,  label: 'Aguardando', cls: 'wait' },
    };
    const m = map[status] || map.completed;
    return (
        <span className={`flow-run-status-badge ${m.cls}`}>
            {m.icon} {m.label}
        </span>
    );
}

export default function FlowRunsModal({ open, flowId, onClose }: Props) {
    const [runs, setRuns] = useState<FlowRunSummary[]>([]);
    const [loading, setLoading] = useState(false);
    const [selected, setSelected] = useState<FlowRunDetail | null>(null);
    const [loadingDetail, setLoadingDetail] = useState(false);

    useEffect(() => {
        if (!open) return;
        setSelected(null);
        setLoading(true);
        listFlowRuns(flowId || undefined)
            .then(setRuns)
            .catch(() => setRuns([]))
            .finally(() => setLoading(false));
    }, [open, flowId]);

    const openRun = async (runId: string) => {
        setLoadingDetail(true);
        try {
            setSelected(await getFlowRun(runId));
        } catch (err) {
            alert(`Erro ao carregar execução: ${err}`);
        } finally {
            setLoadingDetail(false);
        }
    };

    const removeRun = async (runId: string) => {
        if (!confirm('Apagar este registro de execução?')) return;
        try {
            await deleteFlowRun(runId);
            setRuns(rs => rs.filter(r => r.id !== runId));
            if (selected?.id === runId) setSelected(null);
        } catch (err) {
            alert(`Erro ao apagar: ${err}`);
        }
    };

    if (!open) return null;

    return (
        <div className="flow-modal-backdrop" onClick={onClose}>
            <div className="flow-modal flow-runs-modal" onClick={e => e.stopPropagation()}>
                <div className="flow-modal-header">
                    <span className="flow-modal-title">
                        {selected ? (
                            <>
                                <button
                                    className="flow-config-close"
                                    onClick={() => setSelected(null)}
                                    title="Voltar à lista"
                                    style={{ marginRight: 8 }}
                                >
                                    <FaArrowLeft size={14} />
                                </button>
                                Execução — {selected.flow_name}
                            </>
                        ) : (
                            <>
                                <FaChartLine size={14} style={{ marginRight: 8 }} />
                                Execuções {flowId ? 'deste fluxo' : '(todas)'}
                            </>
                        )}
                    </span>
                    <button onClick={onClose} className="flow-config-close"><FaXmark size={16} /></button>
                </div>

                {!selected && loading && (
                    <p className="flow-list-empty">Carregando execuções...</p>
                )}

                {!selected && !loading && runs.length === 0 && (
                    <p className="flow-list-empty">
                        Nenhuma execução registrada ainda. Rode o fluxo (ou um preview) para começar a popular o histórico.
                    </p>
                )}

                {!selected && runs.length > 0 && (
                    <div className="flow-runs-list">
                        {runs.map(r => (
                            <div key={r.id} className="flow-run-item" onClick={() => openRun(r.id)}>
                                <div className="flow-run-item-main">
                                    <StatusBadge status={r.status} />
                                    <span className="flow-run-item-name">
                                        {r.flow_name}
                                        {r.is_preview ? <em style={{ opacity: 0.7, marginLeft: 6 }}>(preview)</em> : null}
                                    </span>
                                    <span className="flow-run-item-date">
                                        {new Date(r.started_at + 'Z').toLocaleString('pt-BR')}
                                    </span>
                                </div>
                                <div className="flow-run-item-stats">
                                    <span title="Duração">⏱ {fmtDuration(r.duration_ms)}</span>
                                    <span title="Tokens (entrada + saída)">
                                        🔤 {fmtTokens(r.total_input_tokens)} / {fmtTokens(r.total_output_tokens)}
                                    </span>
                                    <span title="Custo estimado em USD">💰 {fmtCost(r.total_cost_usd)}</span>
                                    <button
                                        className="flow-run-delete"
                                        onClick={e => { e.stopPropagation(); void removeRun(r.id); }}
                                        title="Apagar"
                                    >
                                        <FaTrash size={10} />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {selected && loadingDetail && (
                    <p className="flow-list-empty">Carregando detalhes...</p>
                )}

                {selected && !loadingDetail && (
                    <FlowRunDetailView run={selected} />
                )}
            </div>
        </div>
    );
}

function FlowRunDetailView({ run }: { run: FlowRunDetail }) {
    const nodeEvents = run.events.filter(e =>
        e.event_type === 'node_start' || e.event_type === 'node_done' || e.event_type === 'node_error'
    );

    // Agrupa por node_id para mostrar uma linha por nó com métricas
    const nodes: Record<string, {
        node_id: string;
        label: string;
        status: 'done' | 'error' | 'started';
        duration_ms: number;
        model: string;
        in_tokens: number;
        out_tokens: number;
        cost_usd: number;
        branch?: string;
        error?: string;
        output?: string;
    }> = {};

    for (const e of nodeEvents) {
        const id = e.node_id;
        if (!id) continue;
        if (!nodes[id]) {
            nodes[id] = {
                node_id: id, label: e.label || id, status: 'started',
                duration_ms: 0, model: '', in_tokens: 0, out_tokens: 0, cost_usd: 0,
            };
        }
        const d = e.data as Record<string, unknown>;
        if (e.event_type === 'node_done') {
            nodes[id].status = 'done';
            nodes[id].duration_ms = Number(d.duration_ms || 0);
            nodes[id].model = String(d.model || '');
            nodes[id].in_tokens = Number(d.input_tokens || 0);
            nodes[id].out_tokens = Number(d.output_tokens || 0);
            nodes[id].cost_usd = Number(d.cost_usd || 0);
            nodes[id].branch = d.branch ? String(d.branch) : undefined;
            nodes[id].output = d.output ? String(d.output) : undefined;
        } else if (e.event_type === 'node_error') {
            nodes[id].status = 'error';
            nodes[id].duration_ms = Number(d.duration_ms || 0);
            nodes[id].error = String(d.error || '');
        }
    }

    return (
        <div className="flow-run-detail">
            {/* Resumo */}
            <div className="flow-run-detail-summary">
                <div><strong>Status:</strong> <StatusBadge status={run.status} /></div>
                <div><strong>Iniciado:</strong> {new Date(run.started_at + 'Z').toLocaleString('pt-BR')}</div>
                <div><strong>Duração:</strong> {fmtDuration(run.duration_ms)}</div>
                <div><strong>Tokens in/out:</strong> {fmtTokens(run.total_input_tokens)} / {fmtTokens(run.total_output_tokens)}</div>
                <div><strong>Custo:</strong> {fmtCost(run.total_cost_usd)}</div>
                {run.error && <div className="flow-run-detail-error"><strong>Erro:</strong> {run.error}</div>}
            </div>

            {/* Tabela por nó */}
            <div className="flow-run-detail-section">
                <h4>Trace por nó</h4>
                <div className="flow-run-trace-table">
                    <div className="flow-run-trace-row header">
                        <span>Nó</span>
                        <span>Modelo</span>
                        <span>Tempo</span>
                        <span>Tokens (in/out)</span>
                        <span>Custo</span>
                        <span>Resultado</span>
                    </div>
                    {Object.values(nodes).map(n => (
                        <div key={n.node_id} className={`flow-run-trace-row ${n.status}`}>
                            <span title={n.node_id}>{n.label}</span>
                            <span>{n.model || '—'}</span>
                            <span>{fmtDuration(n.duration_ms)}</span>
                            <span>{fmtTokens(n.in_tokens)} / {fmtTokens(n.out_tokens)}</span>
                            <span>{fmtCost(n.cost_usd)}</span>
                            <span className="flow-run-trace-result">
                                {n.error ? `✗ ${n.error}`
                                    : n.branch ? `→ ${n.branch}`
                                    : n.status === 'done' ? '✓ OK'
                                    : '…'}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Timeline cru */}
            <details className="flow-run-detail-section">
                <summary>Timeline detalhada ({run.events.length} eventos)</summary>
                <div className="flow-run-events-list">
                    {run.events.map((e, i) => (
                        <div key={i} className={`flow-run-event-row ${e.event_type}`}>
                            <span className="flow-run-event-seq">#{e.seq}</span>
                            <span className="flow-run-event-type">{e.event_type}</span>
                            <span className="flow-run-event-label">{e.label || e.node_id || ''}</span>
                        </div>
                    ))}
                </div>
            </details>

            {run.final_output && (
                <details className="flow-run-detail-section">
                    <summary>Saída final</summary>
                    <pre className="flow-run-output-pre">{run.final_output}</pre>
                </details>
            )}
        </div>
    );
}
