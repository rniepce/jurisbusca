/**
 * BYOK API-key storage.
 *
 * Keys are kept in sessionStorage (cleared when the tab closes), NEVER in
 * localStorage. Persisting provider secrets across sessions widens the window
 * for exfiltration via XSS; sessionStorage shrinks it to the active tab.
 * Any pre-existing localStorage value is migrated once and then purged so old
 * persistent copies do not linger.
 *
 * The truly hardened model is to keep these keys server-side only; this is the
 * minimum client-side mitigation while the BYOK product flow exists.
 */

export type ApiKeyName = 'azure_openai_key' | 'anthropic_api_key' | 'google_api_key';

export function getApiKey(name: ApiKeyName): string {
    try {
        const legacy = localStorage.getItem(name);
        if (legacy !== null) {
            if (legacy) sessionStorage.setItem(name, legacy);
            localStorage.removeItem(name);
        }
        return sessionStorage.getItem(name) || '';
    } catch {
        return '';
    }
}

export function setApiKey(name: ApiKeyName, value: string): void {
    try {
        if (value) sessionStorage.setItem(name, value);
        else sessionStorage.removeItem(name);
        // Make sure no persistent copy survives.
        localStorage.removeItem(name);
    } catch {
        /* storage unavailable (private mode/quota) — keys simply won't persist */
    }
}
