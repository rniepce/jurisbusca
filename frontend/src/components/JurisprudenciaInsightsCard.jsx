import React, { useState, useEffect } from 'react';
import { FaBookOpen, FaChevronDown, FaChevronUp, FaArrowRight, FaSpinner, FaGavel } from 'react-icons/fa6';
import './JurisprudenciaInsightsCard.css';

/**
 * Collapsible card showing jurisprudence research results.
 * Used by the V0.5 engine — appears after the main analysis response.
 */
const JurisprudenciaInsightsCard = ({ data, isLoading, progress, onImport }) => {
    const [expanded, setExpanded] = useState(true);
    const [expandedThemes, setExpandedThemes] = useState({});

    // Auto-expand when data arrives
    useEffect(() => {
        if (data && data.research && data.research.length > 0) {
            setExpanded(true);
        }
    }, [data]);

    const toggleTheme = (index) => {
        setExpandedThemes((prev) => ({ ...prev, [index]: !prev[index] }));
    };

    // Loading state
    if (isLoading && !data) {
        return (
            <div className="juris-insights-card loading">
                <div className="juris-insights-header" onClick={() => setExpanded(!expanded)}>
                    <div className="juris-insights-title">
                        <FaBookOpen className="juris-icon pulse" />
                        <span>📚 Agente Pesquisador — Buscando Jurisprudência...</span>
                    </div>
                    <div className="juris-insights-badge loading-badge">
                        <FaSpinner className="spin" size={12} />
                        <span>Pesquisando</span>
                    </div>
                </div>
                {expanded && (
                    <div className="juris-insights-body">
                        <div className="juris-loading-indicator">
                            <div className="juris-loading-bar" />
                            <span className="juris-loading-text">{progress || 'Extraindo temas jurídicos do processo...'}</span>
                        </div>
                    </div>
                )}
            </div>
        );
    }

    // No data
    if (!data || !data.research || data.research.length === 0) {
        return null;
    }

    const totalAcordaos = data.research.reduce((sum, r) => sum + (r.results?.length || 0), 0);

    return (
        <div className="juris-insights-card">
            <div className="juris-insights-header" onClick={() => setExpanded(!expanded)}>
                <div className="juris-insights-title">
                    <FaBookOpen className="juris-icon" />
                    <span>📚 Jurisprudência Sugerida para este Caso</span>
                </div>
                <div className="juris-insights-meta">
                    <span className="juris-insights-badge">
                        <FaGavel size={11} />
                        <span>{totalAcordaos} acórdão{totalAcordaos !== 1 ? 's' : ''}</span>
                    </span>
                    <span className="juris-insights-badge theme-badge">
                        {data.total_themes} tema{data.total_themes !== 1 ? 's' : ''}
                    </span>
                    {expanded ? <FaChevronUp size={12} /> : <FaChevronDown size={12} />}
                </div>
            </div>

            {expanded && (
                <div className="juris-insights-body">
                    {data.research.map((item, idx) => (
                        <div key={idx} className="juris-theme-block">
                            <div
                                className="juris-theme-header"
                                onClick={() => toggleTheme(idx)}
                            >
                                <span className="juris-theme-number">{idx + 1}</span>
                                <span className="juris-theme-title">{item.theme}</span>
                                <span className="juris-theme-count">
                                    {item.results?.length || 0} resultado{(item.results?.length || 0) !== 1 ? 's' : ''}
                                </span>
                                {expandedThemes[idx] ? <FaChevronUp size={10} /> : <FaChevronDown size={10} />}
                            </div>

                            {/* Summary always visible */}
                            <div className="juris-theme-summary">
                                {formatSummary(item.summary)}
                            </div>

                            {/* Detailed results (collapsible) */}
                            {expandedThemes[idx] && item.results && item.results.length > 0 && (
                                <div className="juris-theme-results">
                                    {item.results.map((r, rIdx) => (
                                        <div key={rIdx} className="juris-result-item">
                                            <div className="juris-result-header">
                                                <span className="juris-result-tipo">{r.tipo_recurso || 'Acórdão'}</span>
                                                <span className="juris-result-processo">{r.numero_processo || '?'}</span>
                                                <span className="juris-result-data">{r.data_publicacao || '?'}</span>
                                                {r.similarity && (
                                                    <span className="juris-result-sim">{Math.round(r.similarity * 100)}%</span>
                                                )}
                                            </div>
                                            <div className="juris-result-ementa">
                                                {(r.ementa || '').slice(0, 300)}
                                                {(r.ementa || '').length > 300 ? '...' : ''}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}

                    {/* Import button */}
                    {onImport && (
                        <button className="juris-import-btn" onClick={() => onImport(data)}>
                            <FaArrowRight size={12} />
                            <span>Importar Insights para o Chat</span>
                        </button>
                    )}
                </div>
            )}
        </div>
    );
};

/**
 * Simple markdown-like formatter for summaries.
 */
function formatSummary(text) {
    if (!text) return null;
    // Split into paragraphs and render
    return text.split('\n').filter(Boolean).map((para, i) => (
        <p key={i} className="juris-summary-para">{para}</p>
    ));
}

export default JurisprudenciaInsightsCard;
