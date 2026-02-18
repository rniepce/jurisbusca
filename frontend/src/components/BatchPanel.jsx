import React from 'react';
import { FiFileText, FiCheck, FiAlertTriangle, FiX } from 'react-icons/fi';
import './BatchPanel.css';

/**
 * Right-side panel that appears during batch analysis.
 * Shows process cards that the user can click to view in the main chat area.
 */
export default function BatchPanel({ results, selectedIndex, onSelect, onClose }) {
    if (!results || results.length === 0) return null;

    const okCount = results.filter(r => r.status === 'ok').length;

    return (
        <div className="batch-panel">
            <div className="batch-panel-header">
                <div className="batch-panel-title">
                    <FiFileText size={16} />
                    <span>Processos ({results.length})</span>
                </div>
                <button className="batch-panel-close" onClick={onClose} title="Fechar painel">
                    <FiX size={16} />
                </button>
            </div>

            <div className="batch-panel-summary">
                <span className="batch-summary-ok">
                    <FiCheck size={12} /> {okCount} concluídos
                </span>
                {results.length - okCount > 0 && (
                    <span className="batch-summary-err">
                        <FiAlertTriangle size={12} /> {results.length - okCount} com erro
                    </span>
                )}
            </div>

            <div className="batch-panel-list">
                {results.map((res, i) => {
                    const isError = res.status === 'error';
                    const isSelected = i === selectedIndex;
                    const shortName = (res.filename || `Processo ${i + 1}`).replace(/\.[^.]+$/, '');

                    return (
                        <button
                            key={i}
                            className={`batch-card ${isSelected ? 'selected' : ''} ${isError ? 'error' : ''}`}
                            onClick={() => onSelect(i)}
                        >
                            <div className="batch-card-icon">
                                {isError ? <FiAlertTriangle size={14} /> : <FiFileText size={14} />}
                            </div>
                            <div className="batch-card-info">
                                <span className="batch-card-name" title={res.filename}>
                                    {shortName.length > 22 ? shortName.slice(0, 22) + '…' : shortName}
                                </span>
                                <span className="batch-card-meta">
                                    {isError ? 'Erro' : res.model || 'Concluído'}
                                </span>
                            </div>
                            {isSelected && <div className="batch-card-indicator" />}
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
