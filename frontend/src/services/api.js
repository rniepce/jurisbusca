/**
 * API service module — communicates with the FastAPI backend.
 */

const API_BASE = '/api';

/**
 * Get headers for API requests.
 * Includes Azure OpenAI key if stored.
 */
function getAuthHeaders(existingHeaders = {}) {
    const azureKey = localStorage.getItem('azure_openai_key');
    const headers = { ...existingHeaders };
    if (azureKey) headers['X-Azure-Key'] = azureKey;
    return headers;
}

/**
 * Validate an Azure OpenAI API key.
 * @param {string} key
 * @returns {Promise<{valid: boolean, message: string}>}
 */
export async function validateAzureKey(key) {
    const res = await fetch(`${API_BASE}/validate-key`, {
        method: 'POST',
        headers: { 'X-Azure-Key': key },
    });
    return safeJson(res, 'Validar Chave');
}

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
export async function uploadFile(file, ocrEngine = 'paddle', compress = true) {
    const form = new FormData();
    form.append('file', file);
    form.append('ocr_engine', ocrEngine);
    form.append('compress', compress.toString());

    // 1. Start background upload task
    const startRes = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: form,
        redirect: 'error',
    }).catch((err) => {
        throw new Error(
            `Upload: requisição redirecionada ou bloqueada. (${err.message})`
        );
    });

    if (!startRes.ok) {
        const err = await safeJson(startRes, 'Upload').catch(() => ({}));
        throw new Error(err.detail || err.message || `Upload falhou (${startRes.status})`);
    }

    const { task_id } = await safeJson(startRes, 'Upload');
    if (!task_id) throw new Error('Upload: servidor não retornou task_id.');

    // 2. Poll for results every 2 seconds (max ~5 minutes)
    const POLL_INTERVAL = 2000;
    const MAX_POLLS = 150;

    for (let i = 0; i < MAX_POLLS; i++) {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));

        const pollRes = await fetch(`${API_BASE}/upload/${task_id}`, {
            headers: getAuthHeaders(),
        }).catch(() => null);

        if (!pollRes || !pollRes.ok) continue;

        const data = await safeJson(pollRes, 'Upload Poll').catch(() => null);
        if (!data) continue;

        if (data.status === 'done') {
            return data.result;
        }
        if (data.status === 'error') {
            throw new Error(data.error || 'Upload falhou no servidor.');
        }
    }

    throw new Error('Upload: tempo máximo de espera excedido.');
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
export async function sendMessage({ message, model, llm, agentPrompt, conversationId, uploadedText, styleDossier, useRag = false }) {
    const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
            message,
            model,
            llm: llm || null,
            conversation_id: conversationId,
            agent_prompt: agentPrompt || null,
            ocr_engine: 'paddle',
            uploaded_text: uploadedText || null,
            style_dossier: styleDossier || null,
            use_rag: useRag,
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

    // 1. Start background task
    const startRes = await fetch(`${API_BASE}/xray`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: form,
        redirect: 'error',
    }).catch((err) => {
        throw new Error(
            `Raio-X: requisição redirecionada ou bloqueada. (${err.message})`
        );
    });

    if (!startRes.ok) {
        const err = await safeJson(startRes, 'Raio-X').catch(() => ({}));
        throw new Error(err.detail || err.message || `Raio-X falhou (${startRes.status})`);
    }

    const { task_id } = await safeJson(startRes, 'Raio-X');
    if (!task_id) throw new Error('Raio-X: servidor não retornou task_id.');

    // 2. Poll for results every 3 seconds (max ~10 minutes)
    const POLL_INTERVAL = 3000;
    const MAX_POLLS = 200;

    for (let i = 0; i < MAX_POLLS; i++) {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));

        const pollRes = await fetch(`${API_BASE}/xray/${task_id}`, {
            headers: getAuthHeaders(),
        }).catch(() => null);

        if (!pollRes || !pollRes.ok) continue; // Retry on network hiccup

        const data = await safeJson(pollRes, 'Raio-X Poll').catch(() => null);
        if (!data) continue;

        if (data.status === 'done') {
            return data.result;
        }
        if (data.status === 'error') {
            throw new Error(data.error || 'Raio-X falhou no servidor.');
        }
        // else: pending/running — keep polling
    }

    throw new Error('Raio-X: tempo máximo de espera excedido.');
}

/**
 * Analyze a cluster of processes individually, in parallel.
 * @param {Array<{filename: string, text: string}>} processes
 * @param {string} [agentPrompt] - Optional agent system prompt
 * @param {string} [model] - Engine model ID
 * @param {string} [llm] - LLM ID
 * @returns {Promise<{results: Array, total: number, ok_count: number}>}
 */
export async function analyzeCluster(processes, agentPrompt = '', model = 'claude', llm = null) {
    const res = await fetch(`${API_BASE}/cluster-analyze`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ processes, agent_prompt: agentPrompt, model, llm }),
        redirect: 'error',
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Cluster Analyze').catch(() => ({}));
        throw new Error(err.detail || `Análise em lote falhou (${res.status})`);
    }

    return safeJson(res, 'Cluster Analyze');
}


/**
 * Generate a Style Dossier from template decision files.
 * @param {File[]} files - Template files (PDF/DOCX/TXT)
 * @returns {Promise<{dossier: string, glossary: string, cloning_prompt: string, full_response: string, file_count: number}>}
 */
export async function generateStyleReport(files) {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));

    const res = await fetch(`${API_BASE}/style-report`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: form,
        redirect: 'error',
    }).catch((err) => {
        throw new Error(
            `Relatório de Estilo: requisição redirecionada ou bloqueada. (${err.message})`
        );
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Relatório de Estilo').catch(() => ({}));
        throw new Error(err.detail || err.message || `Relatório de Estilo falhou (${res.status})`);
    }

    return safeJson(res, 'Relatório de Estilo');
}

/**
 * Upload and index template files for persistent RAG.
 * Also auto-generates the style dossier.
 * @param {File[]} files - Template files (PDF/DOCX/TXT)
 * @returns {Promise<{indexed_chunks: number, file_count: number, has_dossier: boolean, cloning_prompt: string}>}
 */
export async function uploadTemplates(files) {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));

    const res = await fetch(`${API_BASE}/templates`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: form,
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Upload Templates').catch(() => ({}));
        throw new Error(err.detail || `Erro ao indexar modelos (${res.status})`);
    }

    return safeJson(res, 'Upload Templates');
}

/**
 * Check how many templates are indexed in the persistent RAG.
 * @returns {Promise<{indexed_chunks: number, has_dossier: boolean}>}
 */
export async function getTemplateStatus() {
    const res = await fetch(`${API_BASE}/templates/status`, {
        headers: getAuthHeaders()
    });
    if (!res.ok) return { indexed_chunks: 0, has_dossier: false };
    return safeJson(res, 'Template Status');
}

/**
 * Clear all indexed templates from persistent RAG.
 * @returns {Promise<{status: string, message: string}>}
 */
export async function clearTemplates() {
    const res = await fetch(`${API_BASE}/templates`, {
        method: 'DELETE',
        headers: getAuthHeaders()
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Clear Templates').catch(() => ({}));
        throw new Error(err.detail || `Erro ao limpar modelos (${res.status})`);
    }
    return safeJson(res, 'Clear Templates');
}
