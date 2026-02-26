import React, { useState, useEffect } from 'react';
import { FaBookOpen, FaChevronDown, FaChevronUp, FaPlus, FaSpinner } from 'react-icons/fa6';
import './JurisprudenciaInsightsCard.css';

/**
 * Minicard showing jurisprudence research results (V0.5).
 * White background, clean design. Click to expand LLM summary per theme.
 * User can include individual results into the minuta.
 */

/** Format date from YYYY-MM-DD (or similar) to DD/MM/YYYY */
function fmtDate(d) {
    if (!d) return '?';
    const m = String(d).match(/(\d{4})-(\d{2})-(\d{2})/);
    if (m) return `${m[3]}/${m[2]}/${m[1]}`;
    return d;
}

const JurisprudenciaInsightsCard = ({ data, isLoading, progress, onImport }) => {
    const [expanded, setExpanded] = useState(false);
    const [expandedThemes, setExpandedThemes] = useState({});
    const [includedItems, setIncludedItems] = useState(new Set());

    const toggleTheme = (index) => {
        setExpandedThemes((prev) => ({ ...prev, [index]: !prev[index] }));
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
                                                const isIncluded = includedItems.has(key);
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
                                                        <p className="juris-result-ementa">
                                                            {(r.ementa || '').slice(0, 250)}
                                                            {(r.ementa || '').length > 250 ? '…' : ''}
                                                        </p>
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
        </div>
    );
};

export default JurisprudenciaInsightsCard;
