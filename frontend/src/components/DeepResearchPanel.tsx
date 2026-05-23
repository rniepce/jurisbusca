import { useState } from 'react';
import {
    FaFlask, FaXmark, FaCheck, FaCircleExclamation,
    FaSpinner, FaCopy, FaClipboardCheck,
} from 'react-icons/fa6';
import { formatMarkdown } from '../utils/markdown';
import type { DeepResearchState } from '../hooks/useDeepResearch';
import './DeepResearchPanel.css';

interface Props {
    state: DeepResearchState;
    onClose: () => void;
    onSendToCanvas?: (dossier: string) => void;
}

/**
 * Full-page panel that mirrors the live progress of a deep research run and
 * renders the final dossier in a two-pane layout: question list on the left,
 * markdown viewer on the right.
 */
export default function DeepResearchPanel({ state, onClose, onSendToCanvas }: Props) {
    const [view, setView] = useState<'dossier' | number>('dossier');
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        const target =
            view === 'dossier'
                ? state.dossier || ''
                : state.questions[view as number]?.answer || '';
        if (!target) return;
        try {
            await navigator.clipboard.writeText(target);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch { /* noop */ }
    };

    const doneCount = state.questions.filter((q) => q.status === 'done').length;
    const errCount = state.questions.filter((q) => q.status === 'error').length;
    const totalQ = state.questions.length;

    return (
        <div className="dr-panel">
            <header className="dr-header">
                <div className="dr-header-left">
                    <FaFlask size={20} />
                    <div>
                        <h2 className="dr-title">Deep Research</h2>
                        <p className="dr-subtitle">
                            {state.active
                                ? state.phase || 'Processando...'
                                : state.dossier
                                    ? `Dossiê pronto · ${state.chunksIndexed} trechos indexados · ${doneCount}/${totalQ} pontos investigados`
                                    : state.error
                                        ? 'Falha — veja detalhes abaixo'
                                        : 'Aguardando início...'}
                        </p>
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

            {/* Active progress strip */}
            {state.active && (
                <div className="dr-progress-strip">
                    <FaSpinner className="dr-spin" size={14} />
                    <span>{state.phase}</span>
                    {totalQ > 0 && (
                        <div className="dr-progress-bar">
                            <div
                                className="dr-progress-fill"
                                style={{ width: `${((doneCount + errCount) / totalQ) * 100}%` }}
                            />
                        </div>
                    )}
                </div>
            )}

            {/* Two-pane body: question list + viewer */}
            <div className="dr-body">
                <aside className="dr-sidebar">
                    {state.dossier && (
                        <button
                            type="button"
                            className={`dr-side-item dr-side-dossier ${view === 'dossier' ? 'selected' : ''}`}
                            onClick={() => setView('dossier')}
                        >
                            📋 Dossiê consolidado
                        </button>
                    )}
                    {state.questions.length > 0 && (
                        <div className="dr-side-section">
                            Perguntas ({doneCount}/{totalQ})
                        </div>
                    )}
                    {state.questions.map((q, i) => {
                        const isSel = view === i;
                        const icon =
                            q.status === 'done' ? <FaCheck size={11} className="dr-q-icon ok" /> :
                                q.status === 'error' ? <FaCircleExclamation size={11} className="dr-q-icon err" /> :
                                    q.status === 'running' ? <FaSpinner size={11} className="dr-q-icon dr-spin" /> :
                                        <span className="dr-q-icon dr-q-pending">•</span>;
                        return (
                            <button
                                key={i}
                                type="button"
                                className={`dr-side-item ${isSel ? 'selected' : ''} dr-side-q-${q.status}`}
                                onClick={() => setView(i)}
                                disabled={q.status === 'pending'}
                                title={q.question}
                            >
                                {icon}
                                <span className="dr-q-text">{q.question}</span>
                            </button>
                        );
                    })}
                </aside>

                <main className="dr-viewer">
                    {view === 'dossier' && state.dossier ? (
                        <>
                            <div className="dr-viewer-toolbar">
                                <button className="dr-toolbar-btn" onClick={handleCopy}>
                                    {copied ? <FaClipboardCheck size={13} /> : <FaCopy size={13} />}
                                    {copied ? ' Copiado' : ' Copiar dossiê'}
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
                    ) : typeof view === 'number' && state.questions[view] ? (
                        <>
                            <h3 className="dr-q-title">{state.questions[view].question}</h3>
                            {state.questions[view].status === 'done' ? (
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
                                            __html: formatMarkdown(state.questions[view].answer || ''),
                                        }}
                                    />
                                </>
                            ) : state.questions[view].status === 'running' ? (
                                <div className="dr-loading">
                                    <FaSpinner className="dr-spin" size={20} />
                                    <span>Pesquisando nos trechos do processo...</span>
                                </div>
                            ) : state.questions[view].status === 'error' ? (
                                <div className="dr-error">
                                    <FaCircleExclamation size={14} />
                                    <span>{state.questions[view].error || 'Erro desconhecido'}</span>
                                </div>
                            ) : (
                                <div className="dr-loading">Aguardando...</div>
                            )}
                        </>
                    ) : (
                        <div className="dr-empty">
                            <FaFlask size={36} className="dr-empty-icon" />
                            <p>O dossiê aparecerá aqui assim que as perguntas forem investigadas.</p>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}
