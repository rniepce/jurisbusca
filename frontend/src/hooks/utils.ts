// Internal helpers used by hooks.

export function safeResponseText(raw: unknown): string {
    if (typeof raw === 'string') return raw;
    if (raw && typeof raw === 'object') {
        const obj = raw as Record<string, unknown>;
        const candidate = obj.text ?? obj.content ?? obj.output;
        if (typeof candidate === 'string') return candidate;
        try {
            return JSON.stringify(raw);
        } catch {
            return String(raw);
        }
    }
    return String(raw);
}

export function errorText(e: unknown): string {
    if (e instanceof Error) return e.message;
    if (typeof e === 'string') return e;
    return 'Erro desconhecido';
}
