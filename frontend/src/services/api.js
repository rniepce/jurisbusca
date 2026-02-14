/**
 * API service module — communicates with the FastAPI backend.
 */

const API_BASE = '/api';

/**
 * Safely parse JSON from a response.
 * Detects HTML responses (common when backend is down and SPA catch-all serves index.html)
 * and provides actionable error messages.
 */
async function safeJson(res, context) {
    const contentType = res.headers.get('content-type') || '';

    // If response was redirected, the POST may have become a GET → served HTML
    if (res.redirected) {
        throw new Error(
            `${context}: requisição foi redirecionada para ${res.url}. ` +
            `Isso indica um problema de roteamento no servidor.`
        );
    }

    if (!contentType.includes('application/json')) {
        const text = await res.text().catch(() => '');
        const preview = text.slice(0, 120);
        throw new Error(
            `${context}: servidor retornou ${res.status} (${contentType || 'sem content-type'}). ` +
            `Verifique se o backend está rodando. Preview: ${preview}`
        );
    }
    return res.json();
}

/**
 * Upload a file and extract its text content.
 * @param {File} file
 * @param {string} ocrEngine
 * @returns {Promise<{filename: string, text: string, char_count: number}>}
 */
export async function uploadFile(file, ocrEngine = 'gemini_flash') {
    const form = new FormData();
    form.append('file', file);
    form.append('ocr_engine', ocrEngine);

    const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: form,
        redirect: 'error',    // Do NOT follow redirects — fail immediately
    }).catch((err) => {
        // redirect: 'error' causes a TypeError on redirect
        throw new Error(
            `Upload: requisição foi redirecionada ou bloqueada. ` +
            `Verifique se o backend Python está rodando no Railway. (${err.message})`
        );
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Upload').catch(() => ({}));
        throw new Error(err.detail || err.message || `Upload falhou (${res.status})`);
    }

    return safeJson(res, 'Upload');
}

/**
 * Send a chat message to the backend.
 * @param {object} params
 * @param {string} params.message
 * @param {string} params.model
 * @param {string|null} params.agentPrompt
 * @param {string|null} params.conversationId
 * @param {string|null} params.uploadedText
 * @returns {Promise<{conversation_id: string, response: string, model: string}>}
 */
export async function sendMessage({ message, model, agentPrompt, conversationId, uploadedText }) {
    const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message,
            model,
            conversation_id: conversationId,
            agent_prompt: agentPrompt || null,
            ocr_engine: 'gemini_flash',
            uploaded_text: uploadedText || null,
        }),
        redirect: 'error',    // Do NOT follow redirects
    }).catch((err) => {
        throw new Error(
            `Chat: requisição redirecionada ou bloqueada. ` +
            `Verifique se o backend Python está rodando. (${err.message})`
        );
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Chat').catch(() => ({}));
        throw new Error(err.detail || err.message || `Erro no chat (${res.status})`);
    }

    return safeJson(res, 'Chat');
}

/**
 * Upload multiple files for batch X-Ray clustering analysis.
 * @param {File[]} files
 * @returns {Promise<{report: object, file_count: number}>}
 */
export async function uploadBatchXray(files) {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));

    const res = await fetch(`${API_BASE}/xray`, {
        method: 'POST',
        body: form,
        redirect: 'error',
    }).catch((err) => {
        throw new Error(
            `Raio-X: requisição redirecionada ou bloqueada. (${err.message})`
        );
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Raio-X').catch(() => ({}));
        throw new Error(err.detail || err.message || `Raio-X falhou (${res.status})`);
    }

    return safeJson(res, 'Raio-X');
}
