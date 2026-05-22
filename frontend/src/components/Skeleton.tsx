import type { CSSProperties } from 'react';
import './Skeleton.css';

interface SkeletonProps {
    width?: number | string;
    height?: number | string;
    radius?: number | string;
    className?: string;
    style?: CSSProperties;
}

/** Single shimmer line. Compose into more complex layouts via PanelSkeleton. */
export function Skeleton({ width = '100%', height = 14, radius = 6, className, style }: SkeletonProps) {
    return (
        <div
            className={`skel ${className || ''}`}
            style={{ width, height, borderRadius: radius, ...style }}
            aria-hidden="true"
        />
    );
}

/** Reusable loading state for a full lazy panel (header + body lines). */
export function PanelSkeleton({ lines = 6 }: { lines?: number }) {
    return (
        <div className="skel-panel" role="status" aria-busy="true" aria-label="Carregando...">
            <div className="skel-panel-header">
                <Skeleton width={36} height={36} radius={10} />
                <div className="skel-panel-header-text">
                    <Skeleton width="40%" height={16} />
                    <Skeleton width="65%" height={12} />
                </div>
            </div>
            <div className="skel-panel-body">
                {Array.from({ length: lines }).map((_, i) => (
                    <Skeleton
                        key={i}
                        width={`${85 - (i % 3) * 15}%`}
                        height={12}
                        style={{ marginBottom: 10 }}
                    />
                ))}
            </div>
        </div>
    );
}

/** Compact spinner-replacement for inline loading inside a card. */
export function InlineSkeleton({ rows = 3 }: { rows?: number }) {
    return (
        <div className="skel-inline" role="status" aria-busy="true" aria-label="Carregando...">
            {Array.from({ length: rows }).map((_, i) => (
                <Skeleton key={i} width={`${90 - i * 10}%`} height={10} style={{ marginBottom: 8 }} />
            ))}
        </div>
    );
}
