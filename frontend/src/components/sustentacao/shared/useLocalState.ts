import { useState, useEffect, useRef, useMemo } from 'react';

function readKey<T>(key: string, fallback: T): T {
    try {
        const stored = localStorage.getItem(key);
        return stored ? (JSON.parse(stored) as T) : fallback;
    } catch {
        return fallback;
    }
}

function writeKey<T>(key: string, value: T): void {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch {
        // Quota cheia ou storage indisponível: ignora silenciosamente.
    }
}

/**
 * Estado React com persistência automática em localStorage.
 *
 * Quando `key` muda, hidrata do novo slot em vez de sobrescrever — caso
 * contrário o estado anterior contaminaria a próxima entrada.
 */
export function useLocalState<T>(key: string, initial: T): [T, React.Dispatch<React.SetStateAction<T>>] {
    const [value, setValue] = useState<T>(() => readKey(key, initial));
    const lastKey = useRef(key);

    useEffect(() => {
        if (lastKey.current !== key) {
            // Key mudou — re-hidratar do novo slot e abortar o write deste turno.
            lastKey.current = key;
            setValue(readKey(key, initial));
            return;
        }
        writeKey(key, value);
    }, [key, value, initial]);

    return [value, setValue];
}

/**
 * Persistência de Set<number> (checklists) — armazena como array.
 *
 * Estabiliza a identidade do Set retornado entre renders quando o array
 * subjacente não mudou (evita re-render em quem usa em deps).
 */
export function useLocalNumberSet(key: string): [Set<number>, React.Dispatch<React.SetStateAction<Set<number>>>] {
    const [arr, setArr] = useState<number[]>(() => readKey<number[]>(key, []));
    const lastKey = useRef(key);

    useEffect(() => {
        if (lastKey.current !== key) {
            lastKey.current = key;
            setArr(readKey<number[]>(key, []));
            return;
        }
        writeKey(key, arr);
    }, [key, arr]);

    // useMemo garante que o mesmo array referencial → mesmo Set.
    const set = useMemo(() => new Set(arr), [arr]);

    const setSet: React.Dispatch<React.SetStateAction<Set<number>>> = (action) => {
        setArr((prev) => {
            const prevSet = new Set(prev);
            const next = typeof action === 'function'
                ? (action as (s: Set<number>) => Set<number>)(prevSet)
                : action;
            return Array.from(next);
        });
    };

    return [set, setSet];
}
