import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
    FaMagnifyingGlass, FaXmark, FaBook, FaPlus, FaTrash,
    FaBrain, FaChevronDown, FaChevronUp, FaFile, FaSpinner,
    FaCheck, FaScaleBalanced, FaArrowLeft, FaTriangleExclamation,
    FaCircleCheck, FaCircleQuestion, FaChevronRight, FaLocationDot,
    FaCalendarDays
} from 'react-icons/fa6';
import {
    listTemplates, deleteTemplate, askTemplates, uploadTemplates,
    extractThemes, verifyTheme, getJurisprudenciaDoc
} from '../services/api';
import './ModelManagerPanel.css';

const ModelManagerPanel = ({ onClose, onRagStatusChange }) => {
    // ── Model list state ──
    const [templates, setTemplates] = useState([]);
    const [loading, setLoading] = useState(true);
    const [query, setQuery] = useState('');
    const [searchResults, setSearchResults] = useState(null);
    const [summary, setSummary] = useState('');
    const [searching, setSearching] = useState(false);
    const [searchPhase, setSearchPhase] = useState('');
    const [showResults, setShowResults] = useState(false);
    const [error, setError] = useState('');
    const [uploading, setUploading] = useState(false);
    const [deleting, setDeleting] = useState(null);
    const [confirmDelete, setConfirmDelete] = useState(null);
    const fileInputRef = useRef(null);
    const searchInputRef = useRef(null);

    // ── Verification state ──
    const [verifyMode, setVerifyMode] = useState(false);
    const [themes, setThemes] = useState([]);
    const [themesLoading, setThemesLoading] = useState(false);
    const [themesPhase, setThemesPhase] = useState('');
    const [themePage, setThemePage] = useState(0);
    const THEMES_PER_PAGE = 10;

    // Per-theme verification
    const [verifyResults, setVerifyResults] = useState({}); // { themeTitle: result }
    const [verifying, setVerifying] = useState(null); // themeTitle being verified
    const [expandedTheme, setExpandedTheme] = useState(null);
    const [expandedAlert, setExpandedAlert] = useState(null);

    // Document viewer
    const [selectedDoc, setSelectedDoc] = useState(null);
    const [loadingDoc, setLoadingDoc] = useState(false);

    useEffect(() => { loadTemplates(); }, []);
    useEffect(() => { if (searchInputRef.current && !verifyMode) searchInputRef.current.focus(); }, [verifyMode]);

    const loadTemplates = async () => {
        setLoading(true);
        try {
            const data = await listTemplates();
            setTemplates(data.templates || []);
        } catch {
            setError('Erro ao carregar modelos.');
        } finally {
            setLoading(false);
        }
    };

    // ── Search handlers ──
    const handleSearch = useCallback(async () => {
        if (!query.trim()) return;
        setSearching(true); setError(''); setSummary(''); setSearchResults(null); setShowResults(false);
        setSearchPhase('Buscando nos modelos...');
        try {
            setTimeout(() => setSearchPhase('Analisando com IA...'), 2000);
            const data = await askTemplates(query);
            setSummary(data.summary || '');
            setSearchResults(data);
        } catch (err) {
            setError(err.message || 'Erro na busca.');
        } finally {
            setSearching(false); setSearchPhase('');
        }
    }, [query]);

    const handleKeyDown = (e) => { if (e.key === 'Enter') handleSearch(); };
    const handleUploadClick = () => { fileInputRef.current?.click(); };

    const handleFileChange = async (e) => {
        const selected = Array.from(e.target.files);
        if (selected.length === 0) return;
        e.target.value = '';
        setUploading(true); setError('');
        try {
            const result = await uploadTemplates(selected as File[]);
            if (onRagStatusChange) {
                onRagStatusChange({ indexed_chunks: result.indexed_chunks, has_dossier: result.has_dossier });
            }
            await loadTemplates();
        } catch (err) {
            setError(`Erro ao indexar: ${err.message}`);
        } finally {
            setUploading(false);
        }
    };

    const handleDelete = async (filename) => {
        if (confirmDelete !== filename) { setConfirmDelete(filename); return; }
        setConfirmDelete(null); setDeleting(filename); setError('');
        try {
            await deleteTemplate(filename);
            await loadTemplates();
            if (onRagStatusChange) {
                const updated = await listTemplates();
                const totalChunks = (updated.templates || []).reduce((acc, t) => acc + t.chunk_count, 0);
                onRagStatusChange({ indexed_chunks: totalChunks, has_dossier: totalChunks > 0 });
            }
        } catch (err) {
            setError(`Erro ao remover: ${err.message}`);
        } finally {
            setDeleting(null);
        }
    };

    // ── Verification handlers ──
    const handleStartVerification = async () => {
        setVerifyMode(true);
        setThemesLoading(true);
        setThemes([]);
        setVerifyResults({});
        setExpandedTheme(null);
        setExpandedAlert(null);
        setThemePage(0);
        setThemesPhase('Lendo modelos de decisão...');
        setError('');

        try {
            setTimeout(() => setThemesPhase('Identificando temas jurídicos...'), 2500);
            const data = await extractThemes();
            setThemes(data.themes || []);
        } catch (err) {
            setError(err.message || 'Erro ao extrair temas.');
        } finally {
            setThemesLoading(false);
            setThemesPhase('');
        }
    };

    const handleVerifyTheme = async (theme) => {
        if (verifying) return;
        setVerifying(theme.title);
        setExpandedTheme(theme.title);
        setExpandedAlert(null);
        setError('');

        try {
            const result = await verifyTheme(theme.title);
            setVerifyResults(prev => ({ ...prev, [theme.title]: result }));
        } catch (err) {
            setVerifyResults(prev => ({
                ...prev,
                [theme.title]: { status: 'error', theme: theme.title, error: err.message }
            }));
        } finally {
            setVerifying(null);
        }
    };

    const handleDocClick = async (docId) => {
        if (!docId) return;
        setLoadingDoc(true);
        try {
            const doc = await getJurisprudenciaDoc(docId);
            setSelectedDoc(doc);
        } catch {
            setError('Erro ao carregar acórdão.');
        } finally {
            setLoadingDoc(false);
        }
    };

    // ── Helpers ──
    const formatSize = (chars) => {
        if (chars < 1000) return `${chars} chars`;
        if (chars < 1000000) return `${(chars / 1000).toFixed(0)}K chars`;
        return `${(chars / 1000000).toFixed(1)}M chars`;
    };

    const formatDate = (isoDate) => {
        if (!isoDate) return '';
        try {
            return new Date(isoDate).toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit', year: '2-digit' });
        } catch { return ''; }
    };

    const formatProcesso = (num) => {
        if (!num || num.length < 15) return num;
        return num.replace(/^(\d{7})(\d{2})(\d{4})(\d)(\d{2})(\d{4})$/, '$1-$2.$3.$4.$5.$6');
    };

    // Pagination
    const pagedThemes = themes.slice(themePage * THEMES_PER_PAGE, (themePage + 1) * THEMES_PER_PAGE);
    const totalPages = Math.ceil(themes.length / THEMES_PER_PAGE);

    // ═══════════════════════ Document Viewer ═══════════════════════
    if (selectedDoc) {
        return (
            <div className="model-manager-panel">
                <div className="model-manager-header">
                    <div className="model-manager-header-content">
                        <div className="model-manager-title-row">
                            <button className="mm-back-btn" onClick={() => setSelectedDoc(null)}>
                                <FaArrowLeft size={14} />
                            </button>
                            <div>
                                <h2>{selectedDoc.tipo_recurso || 'Acórdão'}</h2>
                                <p className="model-manager-subtitle">
                                    {formatProcesso(selectedDoc.numero_processo)}
                                </p>
                            </div>
                        </div>
                        {onClose && (
                            <button className="model-manager-close-btn" onClick={onClose}><FaXmark size={18} /></button>
                        )}
                    </div>
                </div>
                <div className="model-manager-content">
                    <div className="mm-doc-meta">
                        {selectedDoc.data_publicacao && (
                            <span className="model-manager-badge date"><FaCalendarDays size={10} /> {selectedDoc.data_publicacao}</span>
                        )}
                        {selectedDoc.comarca && (
                            <span className="model-manager-badge chunks"><FaLocationDot size={10} /> {selectedDoc.comarca}</span>
                        )}
                        {selectedDoc.relator && (
                            <span className="model-manager-badge size">Rel. {selectedDoc.relator}</span>
                        )}
                    </div>
                    <pre className="mm-doc-text">{selectedDoc.texto_completo}</pre>
                </div>
            </div>
        );
    }

    // ═══════════════════════ Verify Mode ═══════════════════════
    if (verifyMode) {
        return (
            <div className="model-manager-panel">
                {/* Header */}
                <div className="model-manager-header verify-header">
                    <div className="model-manager-header-content">
                        <div className="model-manager-title-row">
                            <button className="mm-back-btn" onClick={() => setVerifyMode(false)}>
                                <FaArrowLeft size={14} />
                            </button>
                            <div>
                                <h2>Verificar Jurisprudência</h2>
                                <p className="model-manager-subtitle">
                                    {themesLoading
                                        ? 'Analisando modelos...'
                                        : `${themes.length} tema${themes.length !== 1 ? 's' : ''} identificado${themes.length !== 1 ? 's' : ''}`
                                    }
                                </p>
                            </div>
                        </div>
                        {onClose && (
                            <button className="model-manager-close-btn" onClick={onClose}><FaXmark size={18} /></button>
                        )}
                    </div>
                </div>

                {error && <div className="model-manager-error">⚠️ {error}</div>}

                <div className="model-manager-content">
                    {/* Loading themes */}
                    {themesLoading && (
                        <div className="model-manager-loading">
                            <div className="model-manager-spinner" />
                            <span>{themesPhase || 'Analisando...'}</span>
                        </div>
                    )}

                    {/* Theme cards */}
                    {!themesLoading && themes.length > 0 && (
                        <>
                            <div className="mm-themes-grid">
                                {pagedThemes.map((theme) => {
                                    const result = verifyResults[theme.title];
                                    const isExpanded = expandedTheme === theme.title;
                                    const isVerifying = verifying === theme.title;

                                    return (
                                        <div key={theme.id} className={`mm-theme-card ${isExpanded ? 'expanded' : ''}`}>
                                            {/* Card header — clickable to verify */}
                                            <button
                                                className="mm-theme-card-header"
                                                onClick={() => {
                                                    if (!result && !isVerifying) {
                                                        handleVerifyTheme(theme);
                                                    } else {
                                                        setExpandedTheme(isExpanded ? null : theme.title);
                                                    }
                                                }}
                                                disabled={isVerifying}
                                            >
                                                <div className="mm-theme-info">
                                                    <span className="mm-theme-title">{theme.title}</span>
                                                    <span className="mm-theme-desc">{theme.description}</span>
                                                </div>
                                                <div className="mm-theme-status">
                                                    {isVerifying ? (
                                                        <span className="mm-status-badge loading"><FaSpinner size={12} className="spin" /> Verificando</span>
                                                    ) : result ? (
                                                        result.status === 'divergent' ? (
                                                            <span className="mm-status-badge divergent"><FaTriangleExclamation size={12} /> Divergente</span>
                                                        ) : result.status === 'aligned' ? (
                                                            <span className="mm-status-badge aligned"><FaCircleCheck size={12} /> Alinhado</span>
                                                        ) : (
                                                            <span className="mm-status-badge nodata"><FaCircleQuestion size={12} /> Sem dados</span>
                                                        )
                                                    ) : (
                                                        <span className="mm-status-badge pending"><FaChevronRight size={12} /> Verificar</span>
                                                    )}
                                                </div>
                                            </button>

                                            {/* Expanded result */}
                                            {isExpanded && result && (
                                                <div className="mm-theme-result">
                                                    {/* Majority understanding */}
                                                    {result.majority_understanding && (
                                                        <div className="mm-result-section">
                                                            <div className="mm-result-section-title">
                                                                <FaScaleBalanced size={13} /> Entendimento Majoritário do TJMG
                                                            </div>
                                                            <div className="mm-result-text" dangerouslySetInnerHTML={{ __html: formatMarkdown(result.majority_understanding) }} />
                                                        </div>
                                                    )}

                                                    {/* Model approach */}
                                                    {result.model_approach && (
                                                        <div className="mm-result-section">
                                                            <div className="mm-result-section-title">
                                                                <FaBook size={13} /> Abordagem dos Seus Modelos
                                                            </div>
                                                            <div className="mm-result-text" dangerouslySetInnerHTML={{ __html: formatMarkdown(result.model_approach) }} />
                                                        </div>
                                                    )}

                                                    {/* Divergence alert */}
                                                    {result.alert && (
                                                        <div className="mm-alert-section">
                                                            <button
                                                                className="mm-alert-header"
                                                                onClick={() => setExpandedAlert(expandedAlert === theme.title ? null : theme.title)}
                                                            >
                                                                <div className="mm-alert-title">
                                                                    <FaTriangleExclamation size={14} />
                                                                    <span>{result.alert.title}</span>
                                                                </div>
                                                                {expandedAlert === theme.title ? <FaChevronUp size={12} /> : <FaChevronDown size={12} />}
                                                            </button>

                                                            {expandedAlert === theme.title && (
                                                                <div className="mm-alert-body">
                                                                    {/* Detailed comparison */}
                                                                    {result.comparison && (
                                                                        <div className="mm-comparison">
                                                                            <div className="mm-comparison-title"><FaBrain size={13} /> Análise Comparativa</div>
                                                                            <div className="mm-result-text" dangerouslySetInnerHTML={{ __html: formatMarkdown(result.comparison) }} />
                                                                        </div>
                                                                    )}

                                                                    {result.alert.detail && (
                                                                        <div className="mm-alert-detail" dangerouslySetInnerHTML={{ __html: formatMarkdown(result.alert.detail) }} />
                                                                    )}

                                                                    {/* Acórdãos */}
                                                                    {result.acordaos?.length > 0 && (
                                                                        <div className="mm-acordaos">
                                                                            <div className="mm-acordaos-title">Acórdãos Consultados</div>
                                                                            {result.acordaos.map((ac, idx) => (
                                                                                <button
                                                                                    key={idx}
                                                                                    className="mm-acordao-card"
                                                                                    onClick={() => handleDocClick(ac.id)}
                                                                                >
                                                                                    <div className="mm-acordao-top">
                                                                                        <span className="mm-acordao-tipo">{ac.tipo_recurso}</span>
                                                                                        {ac.similarity > 0 && (
                                                                                            <span className={`mm-acordao-sim ${ac.similarity >= 0.6 ? 'high' : ac.similarity >= 0.4 ? 'mid' : 'low'}`}>
                                                                                                {Math.round(ac.similarity * 100)}%
                                                                                            </span>
                                                                                        )}
                                                                                        <span className="mm-acordao-date">{ac.data_publicacao}</span>
                                                                                    </div>
                                                                                    <div className="mm-acordao-processo">{formatProcesso(ac.numero_processo)}</div>
                                                                                    <div className="mm-acordao-ementa">{ac.ementa?.slice(0, 200)}...</div>
                                                                                    {ac.comarca && (
                                                                                        <div className="mm-acordao-bottom">
                                                                                            <FaLocationDot size={10} /> {ac.comarca}
                                                                                            {ac.relator && <span> · Rel. {ac.relator}</span>}
                                                                                        </div>
                                                                                    )}
                                                                                </button>
                                                                            ))}
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            )}
                                                        </div>
                                                    )}

                                                    {/* Acórdãos when aligned (no alert) */}
                                                    {!result.alert && result.acordaos?.length > 0 && (
                                                        <div className="mm-acordaos">
                                                            <div className="mm-acordaos-title">Acórdãos Consultados</div>
                                                            {result.acordaos.map((ac, idx) => (
                                                                <button
                                                                    key={idx}
                                                                    className="mm-acordao-card"
                                                                    onClick={() => handleDocClick(ac.id)}
                                                                >
                                                                    <div className="mm-acordao-top">
                                                                        <span className="mm-acordao-tipo">{ac.tipo_recurso}</span>
                                                                        {ac.similarity > 0 && (
                                                                            <span className={`mm-acordao-sim ${ac.similarity >= 0.6 ? 'high' : 'mid'}`}>
                                                                                {Math.round(ac.similarity * 100)}%
                                                                            </span>
                                                                        )}
                                                                        <span className="mm-acordao-date">{ac.data_publicacao}</span>
                                                                    </div>
                                                                    <div className="mm-acordao-processo">{formatProcesso(ac.numero_processo)}</div>
                                                                    <div className="mm-acordao-ementa">{ac.ementa?.slice(0, 200)}...</div>
                                                                </button>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>

                            {/* Pagination */}
                            {totalPages > 1 && (
                                <div className="mm-pagination">
                                    <button
                                        disabled={themePage === 0}
                                        onClick={() => setThemePage(p => p - 1)}
                                        className="mm-page-btn"
                                    >
                                        ← Anterior
                                    </button>
                                    <span className="mm-page-info">{themePage + 1} / {totalPages}</span>
                                    <button
                                        disabled={themePage >= totalPages - 1}
                                        onClick={() => setThemePage(p => p + 1)}
                                        className="mm-page-btn"
                                    >
                                        Próximo →
                                    </button>
                                </div>
                            )}
                        </>
                    )}

                    {/* Empty themes */}
                    {!themesLoading && themes.length === 0 && !error && (
                        <div className="model-manager-empty">
                            <FaScaleBalanced size={40} className="model-manager-empty-icon" />
                            <p>Nenhum tema identificado</p>
                            <p className="model-manager-empty-hint">
                                Verifique se há modelos indexados e tente novamente.
                            </p>
                        </div>
                    )}
                </div>

                {/* Doc loading overlay */}
                {loadingDoc && (
                    <div className="mm-doc-loading">
                        <div className="model-manager-spinner" />
                        <span>Carregando acórdão...</span>
                    </div>
                )}
            </div>
        );
    }

    // ═══════════════════════ Default: Model Manager View ═══════════════════════
    return (
        <div className="model-manager-panel">
            {/* Header */}
            <div className="model-manager-header">
                <div className="model-manager-header-content">
                    <div className="model-manager-title-row">
                        <FaBook size={22} className="model-manager-title-icon" />
                        <div>
                            <h2>Modelos de Decisão</h2>
                            <p className="model-manager-subtitle">
                                {loading
                                    ? 'Carregando...'
                                    : `${templates.length} modelo${templates.length !== 1 ? 's' : ''} indexado${templates.length !== 1 ? 's' : ''}`
                                }
                            </p>
                        </div>
                    </div>
                    <div className="model-manager-header-actions">
                        <button
                            className="model-manager-verify-btn"
                            onClick={handleStartVerification}
                            disabled={templates.length === 0 || loading}
                            title="Verificar se os modelos estão alinhados com a jurisprudência do TJMG"
                        >
                            <FaScaleBalanced size={13} /> Verificar Jurisprudência
                        </button>
                        <button
                            className="model-manager-add-btn"
                            onClick={handleUploadClick}
                            disabled={uploading}
                            title="Adicionar modelos"
                        >
                            {uploading ? (
                                <><FaSpinner size={13} className="spin" /> Indexando...</>
                            ) : (
                                <><FaPlus size={13} /> Adicionar Modelo</>
                            )}
                        </button>
                        {onClose && (
                            <button className="model-manager-close-btn" onClick={onClose} title="Fechar">
                                <FaXmark size={18} />
                            </button>
                        )}
                    </div>
                </div>
            </div>
            <input ref={fileInputRef} type="file" accept=".pdf,.docx,.txt" multiple onChange={handleFileChange} style={{ display: 'none' }} />

            {/* Search bar */}
            <div className="model-manager-search-section">
                <div className="model-manager-search-bar">
                    <FaMagnifyingGlass size={16} className="model-manager-search-icon" />
                    <input
                        ref={searchInputRef}
                        type="text"
                        placeholder='Buscar nos modelos, ex: "dano moral", "tutela antecipada"...'
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        className="model-manager-search-input"
                        id="model-manager-search-input"
                    />
                    <button className="model-manager-search-btn" onClick={handleSearch} disabled={searching || !query.trim()} id="model-manager-search-btn">
                        {searching ? 'Analisando...' : 'Pesquisar com IA'}
                    </button>
                </div>
            </div>

            {error && <div className="model-manager-error">⚠️ {error}</div>}

            <div className="model-manager-content">
                {searching && (
                    <div className="model-manager-loading">
                        <div className="model-manager-spinner" />
                        <span>{searchPhase || 'Analisando...'}</span>
                    </div>
                )}

                {summary && !searching && (
                    <div className="model-manager-summary-section">
                        <div className="model-manager-summary-header">
                            <FaBrain size={16} className="model-manager-summary-icon" />
                            <span>Análise da IA</span>
                        </div>
                        <div className="model-manager-summary-content" dangerouslySetInnerHTML={{ __html: formatMarkdown(summary) }} />
                    </div>
                )}

                {searchResults && !searching && searchResults.results?.length > 0 && (
                    <button className="model-manager-toggle-results" onClick={() => setShowResults(!showResults)}>
                        {showResults ? <FaChevronUp size={12} /> : <FaChevronDown size={12} />}
                        {showResults ? 'Ocultar' : 'Ver'} {searchResults.results.length} trechos encontrados
                    </button>
                )}

                {showResults && searchResults?.results && (
                    <div className="model-manager-search-results">
                        {searchResults.results.map((item, idx) => (
                            <div key={idx} className="model-manager-result-card">
                                <div className="model-manager-result-source">
                                    <FaFile size={11} />
                                    <span>{item.source}</span>
                                    <span className="model-manager-result-size">{formatSize(item.full_length)}</span>
                                </div>
                                <div className="model-manager-result-text">
                                    {item.text.slice(0, 400)}{item.text.length > 400 && '...'}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                <div className="model-manager-list-header"><span>Modelos Indexados</span></div>

                {loading ? (
                    <div className="model-manager-loading">
                        <div className="model-manager-spinner" />
                        <span>Carregando modelos...</span>
                    </div>
                ) : templates.length === 0 ? (
                    <div className="model-manager-empty">
                        <FaBook size={40} className="model-manager-empty-icon" />
                        <p>Nenhum modelo indexado</p>
                        <p className="model-manager-empty-hint">
                            Clique em <strong>"Adicionar"</strong> para subir modelos de decisão/sentença.
                        </p>
                    </div>
                ) : (
                    <div className="model-manager-list">
                        {templates.map((tpl) => (
                            <div key={tpl.filename} className={`model-manager-card ${deleting === tpl.filename ? 'deleting' : ''}`}>
                                <div className="model-manager-card-info">
                                    <div className="model-manager-card-name">
                                        <FaFile size={13} className="model-manager-card-icon" />
                                        <span>{tpl.filename}</span>
                                    </div>
                                    <div className="model-manager-card-meta">
                                        <span className="model-manager-badge chunks">{tpl.chunk_count} chunk{tpl.chunk_count !== 1 ? 's' : ''}</span>
                                        <span className="model-manager-badge size">{formatSize(tpl.total_chars)}</span>
                                        {tpl.upload_date && <span className="model-manager-badge date">{formatDate(tpl.upload_date)}</span>}
                                    </div>
                                </div>
                                <button
                                    className={`model-manager-delete-btn ${confirmDelete === tpl.filename ? 'confirm' : ''}`}
                                    onClick={() => handleDelete(tpl.filename)}
                                    disabled={deleting === tpl.filename}
                                    title={confirmDelete === tpl.filename ? 'Confirmar remoção' : 'Remover modelo'}
                                >
                                    {deleting === tpl.filename ? (
                                        <FaSpinner size={12} className="spin" />
                                    ) : confirmDelete === tpl.filename ? (
                                        <><FaCheck size={12} /> Confirmar</>
                                    ) : (
                                        <FaTrash size={12} />
                                    )}
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

function formatMarkdown(text) {
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/^### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^## (.+)$/gm, '<h3>$1</h3>')
        .replace(/^# (.+)$/gm, '<h2>$1</h2>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
        .replace(/<\/ul>\s*<ul>/g, '')
        .replace(/\n\n/g, '<br/><br/>')
        .replace(/\n/g, '<br/>');
}

export default ModelManagerPanel;
