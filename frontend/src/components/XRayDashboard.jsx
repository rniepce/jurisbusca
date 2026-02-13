import React, { useMemo } from 'react';
import { FiFileText, FiAlertTriangle, FiTrendingUp, FiUsers } from 'react-icons/fi';
import './XRayDashboard.css';

/* ── SVG Pie Chart (zero dependencies) ─────────────────────────────── */
const PALETTE = [
    '#6366f1', '#8b5cf6', '#a855f7', '#d946ef',
    '#ec4899', '#f43f5e', '#f97316', '#eab308',
    '#22c55e', '#14b8a6', '#06b6d4', '#3b82f6',
];

function PieChart({ slices }) {
    if (!slices || slices.length === 0) return null;
    const total = slices.reduce((s, d) => s + d.value, 0);
    if (total === 0) return null;

    let cumAngle = 0;
    const paths = slices.map((slice, i) => {
        const frac = slice.value / total;
        const startAngle = cumAngle;
        const endAngle = cumAngle + frac * 2 * Math.PI;
        cumAngle = endAngle;

        const largeArc = frac > 0.5 ? 1 : 0;
        const x1 = 100 + 90 * Math.cos(startAngle);
        const y1 = 100 + 90 * Math.sin(startAngle);
        const x2 = 100 + 90 * Math.cos(endAngle - 0.001);
        const y2 = 100 + 90 * Math.sin(endAngle - 0.001);

        const d = [
            `M 100 100`,
            `L ${x1} ${y1}`,
            `A 90 90 0 ${largeArc} 1 ${x2} ${y2}`,
            `Z`,
        ].join(' ');

        return (
            <path key={i} d={d} fill={PALETTE[i % PALETTE.length]} opacity={0.9}>
                <title>{slice.label}: {slice.value}</title>
            </path>
        );
    });

    return (
        <svg viewBox="0 0 200 200" className="xray-pie">
            {paths}
            {/* Center hole for donut effect */}
            <circle cx="100" cy="100" r="50" fill="var(--xray-bg, #1a1a2e)" />
            <text x="100" y="96" textAnchor="middle" fill="#fff" fontSize="22" fontWeight="700">
                {total}
            </text>
            <text x="100" y="114" textAnchor="middle" fill="rgba(255,255,255,0.6)" fontSize="10">
                processos
            </text>
        </svg>
    );
}

/* ── Legend ─────────────────────────────────────────────────────────── */
function PieLegend({ slices }) {
    return (
        <div className="xray-legend">
            {slices.map((s, i) => (
                <div key={i} className="xray-legend-item">
                    <span
                        className="xray-legend-dot"
                        style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
                    />
                    <span className="xray-legend-label">{s.label}</span>
                    <span className="xray-legend-value">{s.value}</span>
                </div>
            ))}
        </div>
    );
}

/* ── Cluster Card ──────────────────────────────────────────────────── */
function ClusterCard({ cluster, index, onActionClick }) {
    const color = PALETTE[index % PALETTE.length];
    return (
        <div className="xray-cluster-card" style={{ borderLeftColor: color }}>
            <div className="xray-cluster-header">
                <span className="xray-cluster-badge" style={{ background: color }}>
                    {cluster.quantidade}
                </span>
                <h4 className="xray-cluster-name">{cluster.nome}</h4>
            </div>
            <p className="xray-cluster-desc">{cluster.descricao_fato}</p>
            {cluster.sugestao_minuta && (
                <p className="xray-cluster-hint">
                    💡 {cluster.sugestao_minuta}
                </p>
            )}
            <div className="xray-cluster-files">
                {(cluster.arquivos || []).map((fname, j) => (
                    <span key={j} className="xray-file-chip">
                        <FiFileText size={12} /> {fname}
                    </span>
                ))}
            </div>
            {onActionClick && (
                <button
                    className="xray-cluster-action"
                    style={{ background: color }}
                    onClick={() => onActionClick(cluster)}
                >
                    ⚡ Processar Grupo
                </button>
            )}
        </div>
    );
}

/* ── Stat Card ─────────────────────────────────────────────────────── */
function StatCard({ icon: Icon, label, value }) {
    return (
        <div className="xray-stat-card">
            <Icon className="xray-stat-icon" size={20} />
            <div>
                <div className="xray-stat-value">{value}</div>
                <div className="xray-stat-label">{label}</div>
            </div>
        </div>
    );
}

/* ── Main Dashboard ────────────────────────────────────────────────── */
export default function XRayDashboard({ report, onClose, onClusterAction }) {
    const clusters = report?.clusters || [];
    const temas = report?.temas_predominantes || [];
    const alertas = report?.alertas_globais || [];
    const stats = report?.estatisticas || {};

    const slices = useMemo(
        () => clusters.map((c) => ({ label: c.nome, value: c.quantidade })),
        [clusters]
    );

    return (
        <div className="xray-dashboard">
            {/* Header */}
            <div className="xray-header">
                <div className="xray-header-left">
                    <h2 className="xray-title">📊 Raio-X da Carteira</h2>
                    <p className="xray-subtitle">
                        {report?.total_processos || 0} processos analisados · {clusters.length} grupos identificados
                    </p>
                </div>
                {onClose && (
                    <button className="xray-close-btn" onClick={onClose}>✕</button>
                )}
            </div>

            {/* Stat Cards */}
            <div className="xray-stats-row">
                <StatCard
                    icon={FiFileText}
                    label="Processos"
                    value={report?.total_processos || 0}
                />
                <StatCard
                    icon={FiUsers}
                    label="Clusters"
                    value={clusters.length}
                />
                <StatCard
                    icon={FiTrendingUp}
                    label="Réu Frequente"
                    value={stats.reu_frequente || '—'}
                />
                {alertas.length > 0 && (
                    <StatCard
                        icon={FiAlertTriangle}
                        label="Alertas"
                        value={alertas.length}
                    />
                )}
            </div>

            {/* Temas */}
            {temas.length > 0 && (
                <div className="xray-temas">
                    {temas.map((t, i) => (
                        <span key={i} className="xray-tema-tag">{t}</span>
                    ))}
                </div>
            )}

            {/* Alertas */}
            {alertas.length > 0 && (
                <div className="xray-alertas">
                    {alertas.map((a, i) => (
                        <div key={i} className="xray-alerta-item">
                            <FiAlertTriangle size={14} /> {a}
                        </div>
                    ))}
                </div>
            )}

            {/* Chart + Legend */}
            <div className="xray-chart-section">
                <PieChart slices={slices} />
                <PieLegend slices={slices} />
            </div>

            {/* Cluster Cards */}
            <div className="xray-clusters-grid">
                {clusters.map((cluster, i) => (
                    <ClusterCard
                        key={cluster.id || i}
                        cluster={cluster}
                        index={i}
                        onActionClick={onClusterAction}
                    />
                ))}
            </div>
        </div>
    );
}
