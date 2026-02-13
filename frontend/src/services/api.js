/**
 * API service module — communicates with the FastAPI backend.
 */

const API_BASE = '/api';

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
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Upload falhou (${res.status})`);
    }

    return res.json();
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
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Erro no chat (${res.status})`);
    }

    return res.json();
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
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Raio-X falhou (${res.status})`);
    }

    return res.json();
}
