import { useState } from 'react';
import {
    FaFlask, FaXmark, FaCheck, FaCircleExclamation,
    FaSpinner, FaCopy, FaClipboardCheck, FaBookOpen,
} from 'react-icons/fa6';
import { formatMarkdown } from '../utils/markdown';
import type { DeepResearchState } from '../hooks/useDeepResearch';
import './DeepResearchPanel.css';

interface Props {
    state: DeepResearchState;
    onClose: () => void;
    onSendToCanvas?: (dossier: string) => void;
}

type View =
    | { kind: 'report' }
    | { kind: 'summary' }
    | { kind: 'section'; id: string }
    | { kind: 'question'; index: number };

/**
 * Full-page panel that mirrors the live progress of a deep research run.
 * Sidebar shows: full report > executive summary > each generated section >
 * each investigation question. Each can be selected to view independently.
 */
export default function DeepResearchPanel({ state, onClose, onSendToCanvas }: Props) {
    const [view, setView] = useState<View>({ kind: 'report' });
    const [copied, setCopied] = useState(false);

    const getCopyTarget = (): string => {
        if (view.kind === 'report') return state.dossier || '';
        if (view.kind === 'summary') return state.executiveSummary || '';
        if (view.kind === 'section') return state.sections.find((s) => s.id === view.id)?.content || '';
        if (view.kind === 'question') return state.questions[view.index]?.answer || '';
        return '';
    };

    const handleCopy = async () => {
        const target = getCopyTarget();
        if (!target) return;
        try {
            await navigator.clipboard.writeText(target);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch { /* noop */ }
    };

    const doneQ = state.questions.filter((q) => q.status === 'done').length;
    const errQ = state.questions.filter((q) => q.status === 'error').length;
    const totalQ = state.questions.length;
    const doneS = state.sections.filter((s) => s.status === 'done').length;
    const totalS = state.sections.length;

    // Compose subtitle based on phase
    let subtitle = 'Aguardando início...';
    if (state.error) {
        subtitle = 'Falha — veja detalhes abaixo';
    } else if (state.dossier) {
        const wordCount = state.dossier.split(/\s+/).filter(Boolean).length;
        subtitle = `Relatório pronto · ${state.chunksIndexed} trechos · ${doneQ}/${totalQ} pontos · ${doneS}/${totalS} seções · ~${wordCount.toLocaleString('pt-BR')} palavras`;
    } else if (state.active) {
        subtitle = state.phase || 'Processando...';
    }

    const totalProgress = totalQ + totalS;
    const doneProgress = doneQ + errQ + doneS;
    const progressPct = totalProgress > 0 ? (doneProgress / totalProgress) * 100 : 0;

    const renderIcon = (status: 'pending' | 'running' | 'done' | 'error') => {
        if (status === 'done') return <FaCheck size={11} className="dr-q-icon ok" />;
        if (status === 'error') return <FaCircleExclamation size={11} className="dr-q-icon err" />;
        if (status === 'running') return <FaSpinner size={11} className="dr-q-icon dr-spin" />;
        return <span className="dr-q-icon dr-q-pending">•</span>;
    };

    return (
        <div className="dr-panel">
            <header className="dr-header">
                <div className="dr-header-left">
                    <FaFlask size={20} />
                    <div>
                        <h2 className="dr-title">
                            Deep Research · <span className="dr-badge-high">high effort</span>
                        </h2>
                        <p className="dr-subtitle">{subtitle}</p>
                    </div>
                </div>
                <button className="dr-close" onClick={onClose} aria-label="Fechar">
                    <FaXmark size={18} />
                </button>
            </header>

            {state.error && (
                <div className="dr-error">
                    <FaCircleExclamation size={16} />
                    <span>{state.error}</span>
                </div>
            )}

            {state.active && (
                <div className="dr-progress-strip">
                    <FaSpinner className="dr-spin" size={14} />
                    <span>{state.phase}</span>
                    {totalProgress > 0 && (
                        <div className="dr-progress-bar">
                            <div
                                className="dr-progress-fill"
                                style={{ width: `${progressPct}%` }}
                            />
                        </div>
                    )}
                </div>
            )}

            <div className="dr-body">
                <aside className="dr-sidebar">
                    {/* Full report */}
                    {state.dossier && (
                        <button
                            type="button"
                            className={`dr-side-item dr-side-dossier ${view.kind === 'report' ? 'selected' : ''}`}
                            onClick={() => setView({ kind: 'report' })}
                        >
                            <FaBookOpen size={12} /> Relatório completo
                        </button>
                    )}

                    {/* Executive summary */}
                    {state.executiveSummary && (
                        <button
                            type="button"
                            className={`dr-side-item ${view.kind === 'summary' ? 'selected' : ''}`}
                            onClick={() => setView({ kind: 'summary' })}
                        >
                            📌 Resumo executivo
                        </button>
                    )}

                    {/* Sections being built */}
                    {state.sections.length > 0 && (
                        <div className="dr-side-section">
                            Seções ({doneS}/{totalS || 12})
                        </div>
                    )}
                    {state.sections.map((s) => {
                        const isSel = view.kind === 'section' && view.id === s.id;
                        return (
                            <button
                                key={s.id}
                                type="button"
                                className={`dr-side-item ${isSel ? 'selected' : ''} dr-side-q-${s.status}`}
                                onClick={() => setView({ kind: 'section', id: s.id })}
                                disabled={s.status === 'pending'}
                                title={s.title}
                            >
                                {renderIcon(s.status)}
                                <span className="dr-q-text">{s.title}</span>
                            </button>
                        );
                    })}

                    {/* Investigation questions */}
                    {state.questions.length > 0 && (
                        <div className="dr-side-section">
                            Perguntas ({doneQ}/{totalQ})
                        </div>
                    )}
                    {state.questions.map((q, i) => {
                        const isSel = view.kind === 'question' && view.index === i;
                        return (
                            <button
                                key={`q-${i}`}
                                type="button"
                                className={`dr-side-item ${isSel ? 'selected' : ''} dr-side-q-${q.status}`}
                                onClick={() => setView({ kind: 'question', index: i })}
                                disabled={q.status === 'pending'}
                                title={q.question}
                            >
                                {renderIcon(q.status)}
                                <span className="dr-q-text">{q.question}</span>
                            </button>
                        );
                    })}
                </aside>

                <main className="dr-viewer">
                    {view.kind === 'report' && state.dossier ? (
                        <>
                            <div className="dr-viewer-toolbar">
                                <button className="dr-toolbar-btn" onClick={handleCopy}>
                                    {copied ? <FaClipboardCheck size={13} /> : <FaCopy size={13} />}
                                    {copied ? ' Copiado' : ' Copiar relatório'}
                                </button>
                                {onSendToCanvas && (
                                    <button
                                        className="dr-toolbar-btn dr-toolbar-primary"
                                        onClick={() => onSendToCanvas(state.dossier!)}
                                    >
                                        📝 Abrir no Canvas
                                    </button>
                                )}
                            </div>
                            <div
                                className="dr-markdown"
                                dangerouslySetInnerHTML={{ __html: formatMarkdown(state.dossier) }}
                            />
                        </>
                    ) : view.kind === 'summary' && state.executiveSummary ? (
                        <>
                            <div className="dr-viewer-toolbar">
                                <button className="dr-toolbar-btn" onClick={handleCopy}>
                                    {copied ? <FaClipboardCheck size={13} /> : <FaCopy size={13} />}
                                    {copied ? ' Copiado' : ' Copiar resumo'}
                                </button>
                            </div>
                            <div
                                className="dr-markdown"
                                dangerouslySetInnerHTML={{ __html: formatMarkdown(state.executiveSummary) }}
                            />
                        </>
                    ) : view.kind === 'section' ? (
                        (() => {
                            const s = state.sections.find((x) => x.id === view.id);
                            if (!s) return <div className="dr-empty">Seção não encontrada.</div>;
                            if (s.status === 'done' && s.content) {
                                return (
                                    <>
                                        <div className="dr-viewer-toolbar">
                                            <button className="dr-toolbar-btn" onClick={handleCopy}>
                                                {copied ? <FaClipboardCheck size={13} /> : <FaCopy size={13} />}
                                                {copied ? ' Copiado' : ' Copiar seção'}
                                            </button>
                                        </div>
                                        <div
                                            className="dr-markdown"
                                            dangerouslySetInnerHTML={{ __html: formatMarkdown(s.content) }}
                                        />
                                    </>
                                );
                            }
                            if (s.status === 'running') {
                                return (
                                    <div className="dr-loading">
                                        <FaSpinner className="dr-spin" size={20} />
                                        <span>Redigindo "{s.title}"...</span>
                                    </div>
                                );
                            }
                            if (s.status === 'error') {
                                return (
                                    <div className="dr-error">
                                        <FaCircleExclamation size={14} />
                                        <span>{s.error || 'Erro desconhecido'}</span>
                                    </div>
                                );
                            }
                            return <div className="dr-loading">Aguardando...</div>;
                        })()
                    ) : view.kind === 'question' && state.questions[view.index] ? (
                        <>
                            <h3 className="dr-q-title">{state.questions[view.index].question}</h3>
                            {state.questions[view.index].status === 'done' ? (
                                <>
                                    <div className="dr-viewer-toolbar">
                                        <button className="dr-toolbar-btn" onClick={handleCopy}>
                                            {copied ? <FaClipboardCheck size={13} /> : <FaCopy size={13} />}
                                            {copied ? ' Copiado' : ' Copiar resposta'}
                                        </button>
                                    </div>
                                    <div
                                        className="dr-markdown"
                                        dangerouslySetInnerHTML={{
                                            __html: formatMarkdown(state.questions[view.index].answer || ''),
                                        }}
                                    />
                                </>
                            ) : state.questions[view.index].status === 'running' ? (
                                <div className="dr-loading">
                                    <FaSpinner className="dr-spin" size={20} />
                                    <span>Pesquisando nos trechos do processo...</span>
                                </div>
                            ) : state.questions[view.index].status === 'error' ? (
                                <div className="dr-error">
                                    <FaCircleExclamation size={14} />
                                    <span>{state.questions[view.index].error || 'Erro desconhecido'}</span>
                                </div>
                            ) : (
                                <div className="dr-loading">Aguardando...</div>
                            )}
                        </>
                    ) : (
                        <div className="dr-empty">
                            <FaFlask size={36} className="dr-empty-icon" />
                            <p>
                                O relatório aparecerá aqui assim que as perguntas forem investigadas e as seções,
                                redigidas. Pode levar 10-15 minutos para processos grandes.
                            </p>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}
