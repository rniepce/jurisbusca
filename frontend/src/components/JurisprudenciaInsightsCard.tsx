import React, { useState, useEffect } from 'react';
import { FaBookOpen, FaChevronDown, FaChevronUp, FaPlus, FaSpinner, FaXmark, FaFileLines } from 'react-icons/fa6';
import { getJurisprudenciaDoc } from '../services/api';
import './JurisprudenciaInsightsCard.css';

/**
 * Minicard showing jurisprudence research results (V0.5).
 * White background, clean design. Click to expand LLM summary per theme.
 * User can include individual results into the minuta.
 * Click on ementa to expand full text; click "Ver voto" to open full vote modal.
 */

/** Format date from YYYY-MM-DD (or similar) to DD/MM/YYYY */
function fmtDate(d) {
    if (!d) return '?';
    const m = String(d).match(/(\d{4})-(\d{2})-(\d{2})/);
    if (m) return `${m[3]}/${m[2]}/${m[1]}`;
    return d;
}

/** Modal dialog showing full vote text */
function VotoDialog({ doc, onClose, isLoading }) {
    // Close on Escape key
    useEffect(() => {
        const handler = (e) => { if (e.key === 'Escape') onClose(); };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [onClose]);

    return (
        <div className="juris-voto-overlay" onClick={onClose}>
            <div className="juris-voto-dialog" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="juris-voto-header">
                    <div className="juris-voto-header-info">
                        {doc ? (
                            <>
                                <span className="juris-voto-tipo">{doc.tipo_recurso || 'Acórdão'}</span>
                                <span className="juris-voto-proc">{doc.numero_processo || '?'}</span>
                                <span className="juris-voto-date">{fmtDate(doc.data_publicacao)}</span>
                                {doc.relator && <span className="juris-voto-relator">Rel. {doc.relator}</span>}
                                {doc.comarca && <span className="juris-voto-comarca">{doc.comarca}</span>}
                            </>
                        ) : (
                            <span className="juris-voto-proc">Carregando...</span>
                        )}
                    </div>
                    <button className="juris-voto-close" onClick={onClose} title="Fechar">
                        <FaXmark size={16} />
                    </button>
                </div>

                {/* Body */}
                <div className="juris-voto-body">
                    {isLoading ? (
                        <div className="juris-voto-loading">
                            <FaSpinner size={18} className="juris-spin" />
                            <span>Carregando voto completo...</span>
                        </div>
                    ) : doc ? (
                        <>
                            {doc.ementa && (
                                <div className="juris-voto-section">
                                    <h4>📋 Ementa</h4>
                                    <div className="juris-voto-ementa-text">{doc.ementa}</div>
                                </div>
                            )}
                            {doc.texto_completo && (
                                <div className="juris-voto-section">
                                    <h4>📄 Voto Completo</h4>
                                    <div className="juris-voto-full-text">
                                        {doc.texto_completo.split('\n').map((line, i) => (
                                            <p key={i}>{line || '\u00A0'}</p>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {!doc.texto_completo && (
                                <div className="juris-voto-section">
                                    <p style={{ color: '#999', fontStyle: 'italic' }}>
                                        Texto completo não disponível para este acórdão.
                                    </p>
                                </div>
                            )}
                        </>
                    ) : (
                        <div className="juris-voto-section">
                            <p style={{ color: '#c00' }}>Erro ao carregar o acórdão.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

const EMENTA_TRUNCATE = 250;

const JurisprudenciaInsightsCard = ({ data, isLoading, progress, onImport }) => {
    const [expanded, setExpanded] = useState(false);
    const [expandedThemes, setExpandedThemes] = useState({});
    const [includedItems, setIncludedItems] = useState(new Set());
    const [expandedEmentas, setExpandedEmentas] = useState(new Set());
    const [votoDoc, setVotoDoc] = useState(null);
    const [votoLoading, setVotoLoading] = useState(false);
    const [showVoto, setShowVoto] = useState(false);

    const toggleTheme = (index) => {
        setExpandedThemes((prev) => ({ ...prev, [index]: !prev[index] }));
    };

    const toggleEmenta = (key) => {
        setExpandedEmentas((prev) => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    const openVoto = async (docId) => {
        setShowVoto(true);
        setVotoLoading(true);
        setVotoDoc(null);
        try {
            const doc = await getJurisprudenciaDoc(docId);
            setVotoDoc(doc);
        } catch (err) {
            console.error('Erro ao carregar voto:', err);
            setVotoDoc(null);
        } finally {
            setVotoLoading(false);
        }
    };

    const closeVoto = () => {
        setShowVoto(false);
        setVotoDoc(null);
    };

    const handleInclude = (theme, result) => {
        const key = `${theme}-${result.numero_processo}`;
        if (includedItems.has(key)) return;
        setIncludedItems((prev) => new Set([...prev, key]));
        if (onImport) {
            // Import a single result
            onImport({
                research: [{
                    theme,
                    summary: `**${result.tipo_recurso || 'Acórdão'}** — ${result.numero_processo || '?'} (${fmtDate(result.data_publicacao)})\n\n${(result.ementa || '').slice(0, 500)}`,
                    results: [result],
                    total: 1,
                }],
                total_themes: 1,
            });
        }
    };

    // Loading state — compact
    if (isLoading && !data) {
        return (
            <div className="juris-minicard">
                <div className="juris-minicard-header" onClick={() => setExpanded(!expanded)}>
                    <FaBookOpen size={13} className="juris-minicard-icon" />
                    <span className="juris-minicard-title">Pesquisa na Jurisprudência</span>
                    <span className="juris-minicard-status">
                        <FaSpinner size={10} className="juris-spin" />
                        <span>Buscando...</span>
                    </span>
                </div>
                {expanded && (
                    <div className="juris-minicard-body">
                        <div className="juris-loading-bar" />
                        <p className="juris-loading-text">{progress || 'Extraindo temas jurídicos...'}</p>
                    </div>
                )}
            </div>
        );
    }

    if (!data || !data.research || data.research.length === 0) return null;

    const totalAcordaos = data.research.reduce((sum, r) => sum + (r.results?.length || 0), 0);

    return (
        <div className="juris-minicard">
            <div className="juris-minicard-header" onClick={() => setExpanded(!expanded)}>
                <FaBookOpen size={13} className="juris-minicard-icon" />
                <span className="juris-minicard-title">Pesquisa na Jurisprudência</span>
                <span className="juris-minicard-count">{totalAcordaos} acórdão{totalAcordaos !== 1 ? 's' : ''} · {data.total_themes} tema{data.total_themes !== 1 ? 's' : ''}</span>
                {expanded ? <FaChevronUp size={10} className="juris-minicard-chevron" /> : <FaChevronDown size={10} className="juris-minicard-chevron" />}
            </div>

            {expanded && (
                <div className="juris-minicard-body">
                    {data.research.map((item, idx) => (
                        <div key={idx} className="juris-theme">
                            <div className="juris-theme-head" onClick={() => toggleTheme(idx)}>
                                <span className="juris-theme-num">{idx + 1}.</span>
                                <span className="juris-theme-label">{item.theme}</span>
                                {expandedThemes[idx] ? <FaChevronUp size={9} /> : <FaChevronDown size={9} />}
                            </div>

                            {expandedThemes[idx] && (
                                <div className="juris-theme-content">
                                    {/* LLM Summary */}
                                    <div className="juris-summary">
                                        {item.summary.split('\n').filter(Boolean).map((p, i) => (
                                            <p key={i}>{p}</p>
                                        ))}
                                    </div>

                                    {/* Individual results with "include" button */}
                                    {item.results && item.results.length > 0 && (
                                        <div className="juris-results">
                                            {item.results.map((r, rIdx) => {
                                                const key = `${item.theme}-${r.numero_processo}`;
                                                const ementaKey = `${idx}-${rIdx}`;
                                                const isIncluded = includedItems.has(key);
                                                const ementaText = r.ementa || '';
                                                const isLong = ementaText.length > EMENTA_TRUNCATE;
                                                const isEmentaExpanded = expandedEmentas.has(ementaKey);

                                                return (
                                                    <div key={rIdx} className={`juris-result ${isIncluded ? 'included' : ''}`}>
                                                        <div className="juris-result-top">
                                                            <span className="juris-result-tipo">{r.tipo_recurso || 'Acórdão'}</span>
                                                            <span className="juris-result-proc">{r.numero_processo || '?'}</span>
                                                            <span className="juris-result-date">{fmtDate(r.data_publicacao)}</span>
                                                            <button
                                                                className={`juris-include-btn ${isIncluded ? 'done' : ''}`}
                                                                onClick={(e) => { e.stopPropagation(); handleInclude(item.theme, r); }}
                                                                title={isIncluded ? 'Incluído na minuta' : 'Incluir na minuta'}
                                                                disabled={isIncluded}
                                                            >
                                                                {isIncluded ? '✓ Incluído' : <><FaPlus size={9} /> Incluir</>}
                                                            </button>
                                                        </div>

                                                        {/* Ementa — expandable */}
                                                        <p className="juris-result-ementa">
                                                            {isLong && !isEmentaExpanded
                                                                ? ementaText.slice(0, EMENTA_TRUNCATE) + '…'
                                                                : ementaText}
                                                        </p>
                                                        {isLong && (
                                                            <button
                                                                className="juris-ementa-toggle"
                                                                onClick={(e) => { e.stopPropagation(); toggleEmenta(ementaKey); }}
                                                            >
                                                                {isEmentaExpanded ? 'Ver menos ▲' : 'Ver mais ▼'}
                                                            </button>
                                                        )}

                                                        {/* View full vote button */}
                                                        {r.id && (
                                                            <button
                                                                className="juris-voto-btn"
                                                                onClick={(e) => { e.stopPropagation(); openVoto(r.id); }}
                                                            >
                                                                <FaFileLines size={10} />
                                                                Ver voto completo
                                                            </button>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {/* Full vote modal */}
            {showVoto && (
                <VotoDialog doc={votoDoc} onClose={closeVoto} isLoading={votoLoading} />
            )}
        </div>
    );
};

export default JurisprudenciaInsightsCard;
