import { useState, useEffect, useRef } from 'react';

/**
 * Estado React com persistência automática em localStorage.
 * O valor é hidratado uma vez ao montar; mudanças subsequentes são sincronizadas.
 */
export function useLocalState<T>(key: string, initial: T): [T, React.Dispatch<React.SetStateAction<T>>] {
    const [value, setValue] = useState<T>(() => {
        try {
            const stored = localStorage.getItem(key);
            return stored ? (JSON.parse(stored) as T) : initial;
        } catch {
            return initial;
        }
    });

    const firstRender = useRef(true);
    useEffect(() => {
        if (firstRender.current) {
            firstRender.current = false;
            return;
        }
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch {
            // Quota cheia ou storage indisponível: ignora silenciosamente.
        }
    }, [key, value]);

    return [value, setValue];
}

/** Persistência de Set<number> (checklists) — converte para array no storage. */
export function useLocalNumberSet(key: string, initial: Set<number> = new Set()): [Set<number>, React.Dispatch<React.SetStateAction<Set<number>>>] {
    const [arr, setArr] = useState<number[]>(() => {
        try {
            const stored = localStorage.getItem(key);
            return stored ? (JSON.parse(stored) as number[]) : Array.from(initial);
        } catch {
            return Array.from(initial);
        }
    });

    const firstRender = useRef(true);
    useEffect(() => {
        if (firstRender.current) {
            firstRender.current = false;
            return;
        }
        try {
            localStorage.setItem(key, JSON.stringify(arr));
        } catch {}
    }, [key, arr]);

    const setSetState: React.Dispatch<React.SetStateAction<Set<number>>> = (action) => {
        setArr((prev) => {
            const prevSet = new Set(prev);
            const next = typeof action === 'function' ? (action as (s: Set<number>) => Set<number>)(prevSet) : action;
            return Array.from(next);
        });
    };

    return [new Set(arr), setSetState];
}
