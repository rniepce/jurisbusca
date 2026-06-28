import React, { useState, useCallback, useEffect, useRef } from 'react';
import { FaMagnifyingGlass, FaChevronLeft, FaChevronRight, FaXmark, FaScaleBalanced, FaCalendarDays, FaLocationDot, FaArrowLeft, FaCopy, FaCheck, FaBrain, FaChevronDown, FaChevronUp, FaFileLines } from 'react-icons/fa6';
import { askJurisprudencia, getJurisprudenciaDoc, getJurisprudenciaStats } from '../services/api';
import { formatMarkdown, sanitizeSnippet } from '../utils/markdown';
import './JurisprudenciaPanel.css';

const JurisprudenciaPanel = ({ onClose }) => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<any>(null);
    const [summary, setSummary] = useState('');
    const [showResults, setShowResults] = useState(false);
    const [loading, setLoading] = useState(false);
    const [loadingPhase, setLoadingPhase] = useState('');

    const [error, setError] = useState('');
    const [anoInicio, setAnoInicio] = useState(2020);
    const [anoFim, setAnoFim] = useState(2026);
    const [stats, setStats] = useState<any>(null);
    const [selectedDoc, setSelectedDoc] = useState<any>(null);
    const [loadingDoc, setLoadingDoc] = useState(false);
    const [copied, setCopied] = useState(false);
    const [expandedSnippets, setExpandedSnippets] = useState<Set<string>>(new Set());
    const searchInputRef = useRef<HTMLInputElement | null>(null);

    // Load stats on mount
    useEffect(() => {
        getJurisprudenciaStats()
            .then(setStats)
            .catch(() => { });
    }, []);

    // Focus search input on mount
    useEffect(() => {
        if (searchInputRef.current) {
            searchInputRef.current.focus();
        }
    }, []);

    const handleSearch = useCallback(async () => {
        if (!query.trim()) return;

        setLoading(true);
        setError('');
        setSummary('');
        setResults(null);
        setShowResults(false);
        setLoadingPhase('Buscando acórdãos relevantes...');

        try {
            setTimeout(() => setLoadingPhase('Analisando jurisprudência com IA...'), 2000);
            const data = await askJurisprudencia(query, {
                anoInicio,
                anoFim,
            });
            setSummary(data.summary || '');
            setResults(data);
        } catch (err: any) {
            setError(err?.message || 'Erro na busca.');
        } finally {
            setLoading(false);
            setLoadingPhase('');
        }
    }, [query, anoInicio, anoFim]);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    };

    const handleDocClick = async (docId) => {
        setLoadingDoc(true);
        try {
            const doc = await getJurisprudenciaDoc(docId);
            setSelectedDoc(doc);
        } catch (err: any) {
            setError(err?.message || 'Erro ao carregar documento.');
        } finally {
            setLoadingDoc(false);
        }
    };

    const handleCopy = async () => {
        if (!selectedDoc) return;
        try {
            await navigator.clipboard.writeText(selectedDoc.texto_completo);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            // Fallback
            const ta = document.createElement('textarea');
            ta.value = selectedDoc.texto_completo;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    const formatProcesso = (num) => {
        if (!num || num.length < 15) return num;
        // Format: NNNNNNN-NN.NNNN.N.NN.NNNN
        return num.replace(
            /^(\d{7})(\d{2})(\d{4})(\d)(\d{2})(\d{4})$/,
            '$1-$2.$3.$4.$5.$6'
        );
    };

    // Document viewer
    if (selectedDoc) {
        return (
            <div className="jurisprudencia-panel">
                <div className="jurisprudencia-doc-view">
                    <div className="jurisprudencia-doc-header">
                        <button
                            className="jurisprudencia-back-btn"
                            onClick={() => setSelectedDoc(null)}
                        >
                            <FaArrowLeft size={14} />
                            <span>Voltar aos resultados</span>
                        </button>
                        <button
                            className={`jurisprudencia-copy-btn ${copied ? 'copied' : ''}`}
                            onClick={handleCopy}
                            title="Copiar texto completo"
                        >
                            {copied ? <><FaCheck size={13} /> Copiado</> : <><FaCopy size={13} /> Copiar</>}
                        </button>
                    </div>

                    <div className="jurisprudencia-doc-meta">
                        <div className="jurisprudencia-doc-meta-row">
                            <span className="meta-badge tipo">{selectedDoc.tipo_recurso || 'Acórdão'}</span>
                            {selectedDoc.data_publicacao && (
                                <span className="meta-badge date">
                                    <FaCalendarDays size={11} /> {selectedDoc.data_publicacao}
                                </span>
                            )}
                            {selectedDoc.comarca && (
                                <span className="meta-badge comarca">
                                    <FaLocationDot size={11} /> {selectedDoc.comarca}
                                </span>
                            )}
                        </div>
                        <div className="jurisprudencia-doc-processo">
                            Processo: {formatProcesso(selectedDoc.numero_processo)}
                        </div>
                        {selectedDoc.relator && (
                            <div className="jurisprudencia-doc-relator">
                                Relator(a): {selectedDoc.relator}
                            </div>
                        )}
                    </div>

                    <div className="jurisprudencia-doc-content">
                        <pre>{selectedDoc.texto_completo}</pre>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="jurisprudencia-panel">
            {/* Header */}
            <div className="jurisprudencia-header">
                <div className="jurisprudencia-header-content">
                    <div className="jurisprudencia-title-row">
                        <FaScaleBalanced size={22} className="jurisprudencia-title-icon" />
                        <div>
                            <h2>Pesquisa de Jurisprudência</h2>
                            <p className="jurisprudencia-subtitle">
                                {stats
                                    ? `${stats.total?.toLocaleString('pt-BR')} acórdãos do TJMG (${stats.ano_min}–${stats.ano_max})`
                                    : 'Base de acórdãos do TJMG'}
                            </p>
                        </div>
                    </div>
                    {onClose && (
                        <button className="jurisprudencia-close-btn" onClick={onClose} title="Fechar">
                            <FaXmark size={18} />
                        </button>
                    )}
                </div>
            </div>

            {/* Search bar */}
            <div className="jurisprudencia-search-section">
                <div className="jurisprudencia-search-bar">
                    <FaMagnifyingGlass size={16} className="jurisprudencia-search-icon" />
                    <input
                        ref={searchInputRef}
                        type="text"
                        placeholder='Descreva o que procura, ex: "indenização por erro médico", "guarda de menor"...'
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        className="jurisprudencia-search-input"
                        id="jurisprudencia-search-input"
                    />
                    <button
                        className="jurisprudencia-search-btn"
                        onClick={() => handleSearch()}
                        disabled={loading || !query.trim()}
                        id="jurisprudencia-search-btn"
                    >
                        {loading ? 'Analisando...' : 'Pesquisar com IA'}
                    </button>
                </div>

                {/* Filters */}
                <div className="jurisprudencia-filters">
                    <label className="jurisprudencia-filter">
                        <span>De:</span>
                        <select
                            value={anoInicio}
                            onChange={(e) => setAnoInicio(Number(e.target.value))}
                        >
                            {Array.from({ length: 7 }, (_, i) => 2020 + i).map((y) => (
                                <option key={y} value={y}>{y}</option>
                            ))}
                        </select>
                    </label>
                    <label className="jurisprudencia-filter">
                        <span>Até:</span>
                        <select
                            value={anoFim}
                            onChange={(e) => setAnoFim(Number(e.target.value))}
                        >
                            {Array.from({ length: 7 }, (_, i) => 2020 + i).map((y) => (
                                <option key={y} value={y}>{y}</option>
                            ))}
                        </select>
                    </label>
                </div>
            </div>

            {/* Error */}
            {error && (
                <div className="jurisprudencia-error">
                    ⚠️ {error}
                </div>
            )}

            {/* Results */}
            <div className="jurisprudencia-results">
                {loading && (
                    <div className="jurisprudencia-loading">
                        <div className="jurisprudencia-spinner" />
                        <span>{loadingPhase || 'Analisando...'}</span>
                    </div>
                )}

                {/* LLM Summary */}
                {summary && !loading && (
                    <div className="jurisprudencia-summary-section">
                        <div className="jurisprudencia-summary-header">
                            <FaBrain size={16} className="jurisprudencia-summary-icon" />
                            <span>Análise da IA</span>
                            {results?.mode && (
                                <span className="jurisprudencia-mode-badge semantic"><FaBrain size={10} /> Semântica</span>
                            )}
                        </div>
                        <div className="jurisprudencia-summary-content" dangerouslySetInnerHTML={{ __html: formatMarkdown(summary) }} />
                    </div>
                )}

                {/* Expandable Results */}
                {results && !loading && results.results?.length > 0 && (
                    <>
                        <button
                            className="jurisprudencia-toggle-results"
                            onClick={() => setShowResults(!showResults)}
                        >
                            {showResults ? <FaChevronUp size={12} /> : <FaChevronDown size={12} />}
                            {showResults ? 'Ocultar' : 'Ver'} {results.results.length} acórdãos encontrados
                            {results.results[0]?.similarity != null && (
                                <span className="jurisprudencia-toggle-hint">
                                    (até {Math.round((results.results[0]?.similarity || 0) * 100)}% relevante)
                                </span>
                            )}
                        </button>

                        {showResults && (
                            <div className="jurisprudencia-list">
                                {results.results.map((item) => {
                                    const snippetKey = `panel-${item.id}`;
                                    const isSnippetExpanded = expandedSnippets.has(snippetKey);
                                    const ementaText = item.ementa || '';
                                    const isLong = ementaText.length > 300;
                                    return (
                                        <div
                                            key={item.id}
                                            className="jurisprudencia-card"
                                            id={`juris-result-${item.id}`}
                                        >
                                            <div className="jurisprudencia-card-top">
                                                <span className="jurisprudencia-card-tipo">
                                                    {item.tipo_recurso || 'Acórdão'}
                                                </span>
                                                <div className="jurisprudencia-card-top-right">
                                                    {item.similarity != null && item.similarity > 0 && (
                                                        <span className={`jurisprudencia-card-similarity ${item.similarity >= 0.6 ? 'high' : item.similarity >= 0.4 ? 'mid' : 'low'}`}>
                                                            {Math.round(item.similarity * 100)}% relevante
                                                        </span>
                                                    )}
                                                    <span className="jurisprudencia-card-date">
                                                        {item.data_publicacao}
                                                    </span>
                                                </div>
                                            </div>
                                            <div className="jurisprudencia-card-processo">
                                                {formatProcesso(item.numero_processo)}
                                            </div>
                                            <div className={`jurisprudencia-card-snippet ${isSnippetExpanded ? 'expanded' : ''}`}>
                                                {item.snippet && !isSnippetExpanded
                                                    ? <span dangerouslySetInnerHTML={{ __html: sanitizeSnippet(item.snippet) }} />
                                                    : isSnippetExpanded
                                                        ? ementaText
                                                        : ementaText.slice(0, 300) + (isLong ? '…' : '')}
                                            </div>
                                            {isLong && (
                                                <button
                                                    className="jurisprudencia-ementa-toggle"
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        setExpandedSnippets(prev => {
                                                            const next = new Set(prev);
                                                            if (next.has(snippetKey)) next.delete(snippetKey);
                                                            else next.add(snippetKey);
                                                            return next;
                                                        });
                                                    }}
                                                >
                                                    {isSnippetExpanded ? 'Ver menos ▲' : 'Ver ementa completa ▼'}
                                                </button>
                                            )}
                                            <div className="jurisprudencia-card-bottom">
                                                {item.comarca && (
                                                    <span className="jurisprudencia-card-comarca">
                                                        <FaLocationDot size={10} /> {item.comarca}
                                                    </span>
                                                )}
                                                {item.relator && (
                                                    <span className="jurisprudencia-card-relator">
                                                        Rel. {item.relator}
                                                    </span>
                                                )}
                                                <button
                                                    className="jurisprudencia-voto-btn"
                                                    onClick={(e) => { e.stopPropagation(); handleDocClick(item.id); }}
                                                >
                                                    <FaFileLines size={11} /> Ver voto completo
                                                </button>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </>
                )}

                {/* Empty state: summary exists but no results */}
                {results && !loading && (!results.results || results.results.length === 0) && !summary && (
                    <div className="jurisprudencia-empty">
                        <FaMagnifyingGlass size={40} />
                        <p>Nenhum acórdão encontrado para "<strong>{results.query}</strong>"</p>
                        <p className="jurisprudencia-empty-hint">
                            Tente termos diferentes ou ajuste os filtros de ano.
                        </p>
                    </div>
                )}

                {/* Empty state (no search yet) */}
                {!results && !loading && !error && (
                    <div className="jurisprudencia-welcome">
                        <FaBrain size={48} className="jurisprudencia-welcome-icon" />
                        <h3>Pesquisa Inteligente de Jurisprudência</h3>
                        <p>
                            Descreva em linguagem natural o que procura.
                            A busca usa IA semântica para encontrar acórdãos por <strong>significado</strong>, não apenas palavras-chave.
                        </p>
                        <div className="jurisprudencia-tips">
                            <span className="jurisprudencia-tip" onClick={() => { setQuery('indenização por acidente de trânsito com dano moral'); }}>acidente de trânsito</span>
                            <span className="jurisprudencia-tip" onClick={() => { setQuery('erro médico em hospital com responsabilidade civil'); }}>erro médico</span>
                            <span className="jurisprudencia-tip" onClick={() => { setQuery('guarda compartilhada de menor e alienação parental'); }}>guarda de menor</span>
                            <span className="jurisprudencia-tip" onClick={() => { setQuery('rescisão de contrato de aluguel por inadimplência'); }}>rescisão de aluguel</span>
                            <span className="jurisprudencia-tip" onClick={() => { setQuery('prescrição em cobrança de dívida bancária'); }}>prescrição bancária</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Loading overlay for doc */}
            {loadingDoc && (
                <div className="jurisprudencia-doc-loading-overlay">
                    <div className="jurisprudencia-spinner" />
                    <span>Carregando acórdão...</span>
                </div>
            )}
        </div>
    );
};

// Simple markdown to HTML converter

export default JurisprudenciaPanel;
