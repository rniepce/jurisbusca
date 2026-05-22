import { Suspense, type ReactNode } from 'react';
import ErrorBoundary from './ErrorBoundary';
import { PanelSkeleton } from './Skeleton';

interface Props {
    label: string;
    onReset?: () => void;
    fallback?: ReactNode;
    /** Number of skeleton lines in the default fallback. */
    skeletonLines?: number;
    children: ReactNode;
}

/**
 * Wraps a lazy-loaded panel with Suspense + ErrorBoundary so a crash in one
 * panel cannot break the rest of the app. Falls back to a shimmering skeleton.
 */
export default function LazyPanel({ label, onReset, fallback, skeletonLines, children }: Props) {
    return (
        <ErrorBoundary label={label} onReset={onReset}>
            <Suspense fallback={fallback ?? <PanelSkeleton lines={skeletonLines} />}>
                {children}
            </Suspense>
        </ErrorBoundary>
    );
}
