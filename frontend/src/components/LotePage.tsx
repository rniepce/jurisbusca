import { useState, useRef, useCallback, useEffect } from 'react';
import { FaLayerGroup, FaUpload, FaRotateLeft } from 'react-icons/fa6';
import { FiFileText, FiCheck, FiAlertTriangle, FiCopy, FiX } from 'react-icons/fi';
import XRayDashboard from './XRayDashboard';
import { uploadBatchXray, analyzeCluster } from '../services/api';
import { useUIStore } from '../store';
import { formatMarkdown } from '../utils/markdown';
import type { XrayReport, XrayCluster, BatchResult } from '../types/chat';
import './LotePage.css';

/* ── Types & constants ─────────────────────────────────────────────── */
type Phase = 'idle' | 'uploading' | 'xray' | 'processing' | 'results';

const UPLOAD_PHASES = [
    { icon: '📤', title: 'Enviando processos...', sub: 'Upload e leitura dos arquivos' },
    { icon: '🔍', title: 'Classificando documentos...', sub: 'Extraindo texto e tipo processual' },
    { icon: '🧩', title: 'Agrupando por matéria...', sub: 'Clusterizando por similaridade temática' },
    { icon: '📊', title: 'Gerando Raio-X da Carteira...', sub: 'Compilando estatísticas e grupos' },
];

const PROCESS_PHASES = [
    { icon: '📄', title: 'Lendo cada processo...', sub: 'Extraindo e interpretando o conteúdo' },
    { icon: '⚖️', title: 'Aplicando raciocínio jurídico...', sub: 'Identificando fundamentos e pedidos' },
    { icon: '✍️', title: 'Redigindo minutas...', sub: 'Gerando decisões individualizadas' },
];

const ACCEPTED = '.pdf,.docx,.txt';

/* ── Animated phase indicator ─────────────────────────────────────── */
function PhaseAnimation({
    phases,
    phaseIdx,
    progress,
}: {
    phases: typeof UPLOAD_PHASES;
    phaseIdx: number;
    progress?: string;
}) {
    const cur = phases[phaseIdx];
    return (
        <div className="lote-anim">
            <div className="lote-anim-spinner" aria-hidden="true" />
            <div className="lote-anim-icon">{cur.icon}</div>
            <div className="lote-anim-title">{cur.title}</div>
            <div className="lote-anim-sub">{progress || cur.sub}</div>
            <div className="lote-anim-steps" role="list" aria-label="Progresso">
                {phases.map((p, i) => (
                    <div
                        key={i}
                        role="listitem"
                        className={`lote-anim-step ${i === phaseIdx ? 'active' : i < phaseIdx ? 'done' : ''}`}
                    >
                        <div className="lote-anim-step-dot" />
                        <span className="lote-anim-step-label">{p.title}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ── Main component ──────────────────────────────────────────────────── */
interface Props {
    onClose: () => void;
    onPilotMode: (cluster: XrayCluster) => void;
}

export default function LotePage({ onClose, onPilotMode }: Props) {
    const [phase, setPhase] = useState<Phase>('idle');
    const [phaseIdx, setPhaseIdx] = useState(0);
    const [progress, setProgress] = useState('');
    const [error, setError] = useState('');
    const [dragging, setDragging] = useState(false);
    const [fileCount, setFileCount] = useState(0);
    const [clusterName, setClusterName] = useState('');
    const [xrayReport, setXrayReport] = useState<XrayReport | null>(null);
    const [xrayCache, setXrayCache] = useState<Record<string, string>>({});
    const [results, setResults] = useState<BatchResult[]>([]);
    const [selectedIdx, setSelectedIdx] = useState(0);
    const [copied, setCopied] = useState(false);

    const inputRef = useRef<HTMLInputElement>(null);
    const { selectedModel } = useUIStore();

    /* Rotate animation phases while loading */
    useEffect(() => {
        if (phase !== 'uploading' && phase !== 'processing') return;
        const phases = phase === 'uploading' ? UPLOAD_PHASES : PROCESS_PHASES;
        const id = setInterval(() => setPhaseIdx((i) => (i + 1) % phases.length), 3500);
        return () => clearInterval(id);
    }, [phase]);

    const enterPhase = useCallback((p: Phase) => {
        setPhaseIdx(0);
        setPhase(p);
    }, []);

    /* ── Upload handler ── */
    const handleFiles = useCallback(
        async (files: File[]) => {
            if (files.length < 2) {
                setError('Selecione pelo menos 2 processos para análise em lote.');
                return;
            }
            setError('');
            setFileCount(files.length);
            enterPhase('uploading');

            try {
                const result = await uploadBatchXray(files, (msg) => setProgress(msg));
                setXrayReport(result.report);
                setXrayCache(result.text_cache || {});
                setProgress('');
                setPhase('xray');
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Erro ao processar arquivos.');
                setPhase('idle');
            }
        },
        [enterPhase]
    );

    /* ── Cluster analysis ── */
    const handleClusterAction = useCallback(
        async (cluster: XrayCluster) => {
            const procs = (cluster.arquivos || [])
                .filter((f) => xrayCache[f])
                .map((f) => ({ filename: f, text: xrayCache[f] }));

            if (!procs.length) return;

            setClusterName(cluster.nome);
            enterPhase('processing');

            try {
                const result = await analyzeCluster(procs, '', selectedModel.id, selectedModel.llm ?? null);
                setResults(result.results);
                setSelectedIdx(0);
                setPhase('results');
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Erro ao analisar grupo.');
                setPhase('xray');
            }
        },
        [xrayCache, selectedModel, enterPhase]
    );

    /* ── Pilot mode — delegate to global handler, then go home ── */
    const handlePilotMode = useCallback(
        (cluster: XrayCluster) => {
            onPilotMode(cluster);
            onClose();
        },
        [onPilotMode, onClose]
    );

    /* ── Reset to upload state ── */
    const handleReset = useCallback(() => {
        setPhase('idle');
        setXrayReport(null);
        setXrayCache({});
        setResults([]);
        setSelectedIdx(0);
        setError('');
        setProgress('');
    }, []);

    /* ── Copy result to clipboard ── */
    const handleCopy = useCallback(() => {
        const res = results[selectedIdx];
        if (!res) return;
        const text = typeof res.response === 'string' ? res.response : JSON.stringify(res.response);
        navigator.clipboard.writeText(text).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        });
    }, [results, selectedIdx]);

    /* ── Drag-and-drop ── */
    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setDragging(true);
    };
    const handleDragLeave = () => setDragging(false);
    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragging(false);
        handleFiles(Array.from(e.dataTransfer.files));
    };

    /* ═══════════════════════════════════════════════════════
       PHASE: IDLE
    ═══════════════════════════════════════════════════════ */
    if (phase === 'idle') {
        return (
            <div className="lote-page">
                <header className="lote-page-header">
                    <div className="lote-page-title-group">
                        <FaLayerGroup size={20} />
                        <h2 className="lote-page-title">Análise em Lote</h2>
                    </div>
                    <button className="lote-page-close" onClick={onClose} aria-label="Fechar">
                        <FiX size={18} />
                    </button>
                </header>

                <div className="lote-page-body">
                    <p className="lote-page-lead">
                        Anexe <strong>2 ou mais processos</strong> em PDF, DOCX ou TXT para classificar
                        automaticamente por matéria e processar cada grupo em paralelo.
                    </p>

                    <div
                        className={`lote-dropzone${dragging ? ' dragging' : ''}`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={() => inputRef.current?.click()}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
                        aria-label="Área de upload de processos"
                    >
                        <FaUpload size={38} className="lote-dropzone-icon" />
                        <div className="lote-dropzone-title">
                            {dragging ? 'Solte os arquivos aqui' : 'Clique ou arraste os arquivos'}
                        </div>
                        <div className="lote-dropzone-hint">PDF · DOCX · TXT — mínimo 2 arquivos</div>
                        <button
                            type="button"
                            className="lote-upload-btn"
                            onClick={(e) => {
                                e.stopPropagation();
                                inputRef.current?.click();
                            }}
                        >
                            Escolher arquivos
                        </button>
                        <input
                            ref={inputRef}
                            type="file"
                            accept={ACCEPTED}
                            multiple
                            onChange={(e) => {
                                const files = Array.from(e.target.files || []);
                                e.target.value = '';
                                handleFiles(files);
                            }}
                            style={{ display: 'none' }}
                        />
                    </div>

                    {error && (
                        <div className="lote-error" role="alert">
                            <FiAlertTriangle size={15} /> {error}
                        </div>
                    )}

                    <div className="lote-how">
                        <div className="lote-how-title">Como funciona</div>
                        <div className="lote-how-steps">
                            {[
                                { n: '1', label: 'Envie 2 ou mais processos em PDF ou DOCX', icon: '📎' },
                                { n: '2', label: 'O Raio-X classifica e agrupa por matéria', icon: '🧩' },
                                { n: '3', label: 'Processe grupos em paralelo ou via piloto', icon: '⚡' },
                            ].map((s) => (
                                <div key={s.n} className="lote-how-step">
                                    <div className="lote-how-step-num">{s.n}</div>
                                    <span className="lote-how-step-icon">{s.icon}</span>
                                    <span className="lote-how-step-label">{s.label}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    /* ═══════════════════════════════════════════════════════
       PHASE: UPLOADING
    ═══════════════════════════════════════════════════════ */
    if (phase === 'uploading') {
        return (
            <div className="lote-page lote-page--centered">
                <div className="lote-loading-wrap">
                    <div className="lote-loading-badge">
                        <FaLayerGroup size={14} />
                        {fileCount} arquivo{fileCount !== 1 ? 's' : ''} enviado{fileCount !== 1 ? 's' : ''}
                    </div>
                    <PhaseAnimation phases={UPLOAD_PHASES} phaseIdx={phaseIdx} progress={progress} />
                    <div className="lote-progress-bar" aria-label="Carregando">
                        <div className="lote-progress-bar-fill" />
                    </div>
                </div>
            </div>
        );
    }

    /* ═══════════════════════════════════════════════════════
       PHASE: XRAY
    ═══════════════════════════════════════════════════════ */
    if (phase === 'xray' && xrayReport) {
        return (
            <div className="lote-page lote-page--xray">
                <div className="lote-xray-topbar">
                    <button className="lote-ghost-btn" onClick={handleReset} title="Fazer nova análise">
                        <FaRotateLeft size={13} /> Nova análise
                    </button>
                </div>
                <XRayDashboard
                    report={xrayReport}
                    onClose={handleReset}
                    onClusterAction={handleClusterAction}
                    onClusterPilotAction={handlePilotMode}
                />
            </div>
        );
    }

    /* ═══════════════════════════════════════════════════════
       PHASE: PROCESSING
    ═══════════════════════════════════════════════════════ */
    if (phase === 'processing') {
        return (
            <div className="lote-page lote-page--centered">
                <div className="lote-loading-wrap">
                    <div className="lote-loading-badge lote-loading-badge--cluster">
                        🧩 Grupo: <strong>{clusterName}</strong>
                    </div>
                    <PhaseAnimation phases={PROCESS_PHASES} phaseIdx={phaseIdx} />
                    <div className="lote-progress-bar" aria-label="Processando">
                        <div className="lote-progress-bar-fill" />
                    </div>
                </div>
            </div>
        );
    }

    /* ═══════════════════════════════════════════════════════
       PHASE: RESULTS
    ═══════════════════════════════════════════════════════ */
    if (phase === 'results') {
        const okCount = results.filter((r) => r.status === 'ok').length;
        const errCount = results.length - okCount;
        const totalAlerts = results.reduce((s, r) => s + (r.alerts_count ?? 0), 0);
        const selected = results[selectedIdx];

        return (
            <div className="lote-page lote-page--results">
                {/* Header bar */}
                <div className="lote-results-bar">
                    <div className="lote-results-bar-left">
                        <FaLayerGroup size={15} />
                        <span className="lote-results-bar-title">{clusterName}</span>
                        <div className="lote-results-stats">
                            <span className="lote-stat ok">
                                <FiCheck size={11} /> {okCount} concluído{okCount !== 1 ? 's' : ''}
                            </span>
                            {errCount > 0 && (
                                <span className="lote-stat err">
                                    <FiAlertTriangle size={11} /> {errCount} com erro
                                </span>
                            )}
                            {totalAlerts > 0 && (
                                <span className="lote-stat warn">
                                    ⚠ {totalAlerts} alerta{totalAlerts !== 1 ? 's' : ''}
                                </span>
                            )}
                        </div>
                    </div>
                    <div className="lote-results-bar-right">
                        <button
                            className="lote-ghost-btn"
                            onClick={() => setPhase('xray')}
                            title="Voltar ao Raio-X"
                        >
                            <FaRotateLeft size={13} /> Raio-X
                        </button>
                        <button className="lote-ghost-btn" onClick={handleReset} title="Nova análise">
                            Nova análise
                        </button>
                        <button className="lote-ghost-btn lote-ghost-btn--close" onClick={onClose}>
                            <FiX size={16} />
                        </button>
                    </div>
                </div>

                {/* Two-panel layout */}
                <div className="lote-results-layout">
                    {/* ── Left: file list ── */}
                    <div className="lote-file-list" role="list" aria-label="Lista de processos">
                        <div className="lote-file-list-label">
                            {results.length} processo{results.length !== 1 ? 's' : ''}
                        </div>
                        {results.map((r, i) => {
                            const name = (r.filename || `Processo ${i + 1}`).replace(/\.[^.]+$/, '');
                            const isErr = r.status === 'error';
                            const isSelected = i === selectedIdx;
                            const alerts = r.alerts_count ?? 0;

                            return (
                                <button
                                    key={i}
                                    role="listitem"
                                    className={`lote-file-card${isSelected ? ' selected' : ''}${isErr ? ' error' : ''}`}
                                    onClick={() => setSelectedIdx(i)}
                                    title={r.filename}
                                >
                                    <div className="lote-file-card-icon">
                                        {isErr ? (
                                            <FiAlertTriangle size={14} />
                                        ) : (
                                            <FiFileText size={14} />
                                        )}
                                    </div>
                                    <div className="lote-file-card-info">
                                        <span className="lote-file-card-name">
                                            {name.length > 26 ? name.slice(0, 26) + '…' : name}
                                        </span>
                                        <span className="lote-file-card-meta">
                                            {isErr ? 'Erro na análise' : r.model || 'Concluído'}
                                        </span>
                                    </div>
                                    {alerts > 0 && (
                                        <span className="lote-alerts-pill" title={`${alerts} alerta(s) de diferença`}>
                                            ⚠ {alerts}
                                        </span>
                                    )}
                                    {isSelected && <div className="lote-file-card-bar" />}
                                </button>
                            );
                        })}
                    </div>

                    {/* ── Right: result viewer ── */}
                    <div className="lote-viewer">
                        {selected ? (
                            <>
                                <div className="lote-viewer-toolbar">
                                    <span className="lote-viewer-filename">
                                        {selected.status === 'error' ? '⚠️' : '📄'} {selected.filename}
                                    </span>
                                    <button
                                        className="lote-viewer-copy-btn"
                                        onClick={handleCopy}
                                        title="Copiar para área de transferência"
                                    >
                                        {copied ? (
                                            <>
                                                <FiCheck size={13} /> Copiado!
                                            </>
                                        ) : (
                                            <>
                                                <FiCopy size={13} /> Copiar texto
                                            </>
                                        )}
                                    </button>
                                </div>
                                <div
                                    className="lote-viewer-content"
                                    dangerouslySetInnerHTML={{ __html: formatMarkdown(selected.response) }}
                                />
                            </>
                        ) : (
                            <div className="lote-viewer-empty">
                                <FiFileText size={32} />
                                <p>Selecione um processo na lista para visualizar o resultado</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    }

    return null;
}
