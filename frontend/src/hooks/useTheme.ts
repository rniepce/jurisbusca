import { useCallback, useEffect, useState } from 'react';

export type ThemeMode = 'light' | 'dark' | 'system';
type ResolvedTheme = 'light' | 'dark';

const STORAGE_KEY = 'theme-mode';

function getSystemTheme(): ResolvedTheme {
    if (typeof window === 'undefined' || !window.matchMedia) return 'light';
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function resolve(mode: ThemeMode): ResolvedTheme {
    return mode === 'system' ? getSystemTheme() : mode;
}

function readStoredMode(): ThemeMode {
    if (typeof window === 'undefined') return 'system';
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark' || stored === 'system') return stored;
    return 'system';
}

function applyTheme(theme: ResolvedTheme) {
    if (typeof document === 'undefined') return;
    document.documentElement.setAttribute('data-theme', theme);
}

/**
 * Theme hook — persists user preference and reacts to system changes when in 'system' mode.
 *
 * Returns:
 * - `mode`: the user's selected mode ('light' | 'dark' | 'system')
 * - `resolved`: the actually-applied theme ('light' | 'dark')
 * - `setMode`: change the mode
 * - `toggle`: cycle light → dark → system → light
 */
export function useTheme() {
    const [mode, setModeState] = useState<ThemeMode>(() => readStoredMode());
    const [resolved, setResolved] = useState<ResolvedTheme>(() => resolve(readStoredMode()));

    // Apply the resolved theme to <html>
    useEffect(() => {
        const next = resolve(mode);
        setResolved(next);
        applyTheme(next);
    }, [mode]);

    // Listen for system theme changes when in 'system' mode
    useEffect(() => {
        if (mode !== 'system' || typeof window === 'undefined' || !window.matchMedia) return;
        const mq = window.matchMedia('(prefers-color-scheme: dark)');
        const handler = (e: MediaQueryListEvent) => {
            const next: ResolvedTheme = e.matches ? 'dark' : 'light';
            setResolved(next);
            applyTheme(next);
        };
        mq.addEventListener('change', handler);
        return () => mq.removeEventListener('change', handler);
    }, [mode]);

    const setMode = useCallback((m: ThemeMode) => {
        setModeState(m);
        if (typeof window !== 'undefined') localStorage.setItem(STORAGE_KEY, m);
    }, []);

    const toggle = useCallback(() => {
        const next: ThemeMode = mode === 'light' ? 'dark' : mode === 'dark' ? 'system' : 'light';
        setMode(next);
    }, [mode, setMode]);

    return { mode, resolved, setMode, toggle };
}
