/**
 * API service module — communicates with the FastAPI backend.
 */
import { supabase } from './supabase';

const API_BASE = '/api';

/**
 * Get headers for API requests.
 * Includes Azure OpenAI key if stored, and Supabase JWT.
 */
async function getAuthHeaders(existingHeaders = {}) {
    const azureKey = localStorage.getItem('azure_openai_key');
    const headers = { ...existingHeaders };
    if (azureKey) headers['X-Azure-Key'] = azureKey;

    // Anexar JWT do Supabase
    const { data: { session } } = await supabase.auth.getSession();
    if (session?.access_token) {
        headers['Authorization'] = `Bearer ${session.access_token}`;
    }

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
export async function uploadFile(file, ocrEngine = 'mistral_doc_ai', compress = true, vectorize = true, onProgress = null) {
    const form = new FormData();
    form.append('file', file);
    form.append('ocr_engine', ocrEngine);
    form.append('compress', compress.toString());
    form.append('vectorize', vectorize.toString());

    // 1. Start background upload task
    const startRes = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        headers: await getAuthHeaders(),
        body: form,
        redirect: 'follow',
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

    // Report initial progress
    if (onProgress) onProgress({ progress: '📤 Enviando arquivo...', percent: 5 });

    // 2. Poll for results every 2 seconds (max ~5 minutes)
    const POLL_INTERVAL = 2000;
    const MAX_POLLS = 150;

    for (let i = 0; i < MAX_POLLS; i++) {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));

        const pollRes = await fetch(`${API_BASE}/upload/${task_id}`, {
            headers: await getAuthHeaders(),
        }).catch(() => null);

        if (!pollRes || !pollRes.ok) continue;

        const data = await safeJson(pollRes, 'Upload Poll').catch(() => null);
        if (!data) continue;

        // Report progress to UI
        if (onProgress && data.progress) {
            onProgress({ progress: data.progress, percent: data.percent || 0 });
        }

        if (data.status === 'done') {
            if (onProgress) onProgress({ progress: '✅ Processamento concluído!', percent: 100 });
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
export async function sendMessage({ message, model, llm, agentPrompt, conversationId, uploadedText, styleDossier, useRag = false, jurisprudenceContext = null }) {
    const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
            message,
            model,
            llm: llm || null,
            conversation_id: conversationId,
            agent_prompt: agentPrompt || null,
            ocr_engine: 'marker',
            uploaded_text: uploadedText || null,
            style_dossier: styleDossier || null,
            use_rag: useRag,
            jurisprudence_context: jurisprudenceContext || null,
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
 * @param {function} [onProgress] - Optional callback for progress updates (receives progress string)
 * @returns {Promise<{report: object, file_count: number}>}
 */
export async function uploadBatchXray(files, onProgress = null) {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));

    // 1. Start background task
    const startRes = await fetch(`${API_BASE}/xray`, {
        method: 'POST',
        headers: await getAuthHeaders(),
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

    // 2. Poll for results every 2 seconds (max ~10 minutes)
    const POLL_INTERVAL = 2000;
    const MAX_POLLS = 300;

    for (let i = 0; i < MAX_POLLS; i++) {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));

        const pollRes = await fetch(`${API_BASE}/xray/${task_id}`, {
            headers: await getAuthHeaders(),
        }).catch(() => null);

        if (!pollRes || !pollRes.ok) continue; // Retry on network hiccup

        const data = await safeJson(pollRes, 'Raio-X Poll').catch(() => null);
        if (!data) continue;

        // Propagate progress to caller
        if (data.progress && onProgress) {
            onProgress(data.progress);
        }

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
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
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
        headers: await getAuthHeaders(),
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
        headers: await getAuthHeaders(),
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
        headers: await getAuthHeaders()
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
        headers: await getAuthHeaders()
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Clear Templates').catch(() => ({}));
        throw new Error(err.detail || `Erro ao limpar modelos (${res.status})`);
    }
    return safeJson(res, 'Clear Templates');
}

/**
 * List all template files for the current user.
 * @returns {Promise<{templates: Array, total: number}>}
 */
export async function listTemplates() {
    const res = await fetch(`${API_BASE}/templates/list`, {
        headers: await getAuthHeaders(),
    });
    if (!res.ok) return { templates: [], total: 0 };
    return safeJson(res, 'List Templates');
}

/**
 * Delete a specific template file by filename.
 * @param {string} filename - Source filename to delete
 * @returns {Promise<{status: string, removed_chunks: number, remaining_templates: number}>}
 */
export async function deleteTemplate(filename) {
    const res = await fetch(`${API_BASE}/templates/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
        headers: await getAuthHeaders(),
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Delete Template').catch(() => ({}));
        throw new Error(err.detail || `Erro ao remover modelo (${res.status})`);
    }
    return safeJson(res, 'Delete Template');
}

/**
 * AI search over user's templates (RAG + LLM summary).
 * @param {string} query - Natural language search query
 * @returns {Promise<{summary: string, results: Array, total: number, query: string}>}
 */
export async function askTemplates(query) {
    const res = await fetch(`${API_BASE}/templates/ask`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ query }),
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Ask Templates').catch(() => ({}));
        throw new Error(err.detail || `Erro na pesquisa nos modelos (${res.status})`);
    }
    return safeJson(res, 'Ask Templates');
}

/**
 * Extract main legal themes from user's templates.
 * @returns {Promise<{themes: Array<{id: number, title: string, description: string}>, total: number}>}
 */
export async function extractThemes() {
    const res = await fetch(`${API_BASE}/templates/extract-themes`, {
        method: 'POST',
        headers: await getAuthHeaders(),
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Extract Themes').catch(() => ({}));
        throw new Error(err.detail || `Erro ao extrair temas (${res.status})`);
    }
    return safeJson(res, 'Extract Themes');
}

/**
 * Verify a theme: compare user's templates vs jurisprudence.
 * @param {string} theme - Theme title to verify
 * @returns {Promise<{status: string, theme: string, majority_understanding: string, model_approach: string, comparison: string, alert: object|null, acordaos: Array}>}
 */
export async function verifyTheme(theme) {
    const res = await fetch(`${API_BASE}/templates/verify-theme`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ theme }),
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Verify Theme').catch(() => ({}));
        throw new Error(err.detail || `Erro ao verificar tema (${res.status})`);
    }
    return safeJson(res, 'Verify Theme');
}


// ── Jurisprudência Search ───────────────────────────────────────────────────

/**
 * Search TJMG case law database.
 * @param {string} query - Search terms
 * @param {object} [filters] - Optional filters
 * @param {number} [filters.anoInicio] - Start year
 * @param {number} [filters.anoFim] - End year
 * @param {string} [filters.tipo] - Case type filter
 * @param {number} [filters.page] - Page number (1-indexed)
 * @param {number} [filters.pageSize] - Results per page
 * @returns {Promise<{results: Array, total: number, page: number, pages: number}>}
 */
export async function searchJurisprudencia(query, filters = {}) {
    const params = new URLSearchParams({ q: query });
    if (filters.anoInicio) params.set('ano_inicio', filters.anoInicio);
    if (filters.anoFim) params.set('ano_fim', filters.anoFim);
    if (filters.tipo) params.set('tipo', filters.tipo);
    if (filters.page) params.set('page', filters.page);
    if (filters.pageSize) params.set('page_size', filters.pageSize);

    const res = await fetch(`${API_BASE}/jurisprudencia/search?${params}`, {
        headers: await getAuthHeaders(),
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Jurisprudência').catch(() => ({}));
        throw new Error(err.detail || `Erro na busca de jurisprudência (${res.status})`);
    }

    return safeJson(res, 'Jurisprudência');
}

/**
 * Get full text of a specific case law document.
 * @param {number} docId
 * @returns {Promise<{id: number, numero_processo: string, texto_completo: string, ...}>}
 */
export async function getJurisprudenciaDoc(docId) {
    const res = await fetch(`${API_BASE}/jurisprudencia/doc/${docId}`, {
        headers: await getAuthHeaders(),
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Jurisprudência Doc').catch(() => ({}));
        throw new Error(err.detail || `Erro ao buscar acórdão (${res.status})`);
    }

    return safeJson(res, 'Jurisprudência Doc');
}

/**
 * Get statistics about the case law database.
 * @returns {Promise<{total: number, por_ano: object, por_tipo: object, ano_min: number, ano_max: number}>}
 */
export async function getJurisprudenciaStats() {
    const res = await fetch(`${API_BASE}/jurisprudencia/stats`, {
        headers: await getAuthHeaders(),
    });

    if (!res.ok) return { total: 0, por_ano: {}, por_tipo: {}, ano_min: 2020, ano_max: 2026 };
    return safeJson(res, 'Jurisprudência Stats');
}

/**
 * Ask jurisprudência with LLM summary (RAG).
 * @param {string} query - Natural language question
 * @param {object} [filters] - Optional filters
 * @returns {Promise<{summary: string, results: Array, total: number, query: string, mode: string}>}
 */
export async function askJurisprudencia(query, filters = {}) {
    const body = { query };
    if (filters.anoInicio) body.ano_inicio = filters.anoInicio;
    if (filters.anoFim) body.ano_fim = filters.anoFim;
    if (filters.tipo) body.tipo = filters.tipo;

    const res = await fetch(`${API_BASE}/jurisprudencia/ask`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(body),
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Jurisprudência Ask').catch(() => ({}));
        throw new Error(err.detail || `Erro na pesquisa inteligente (${res.status})`);
    }

    return safeJson(res, 'Jurisprudência Ask');
}

/**
 * Manually trigger jurisprudence research for a given process.
 * @param {string} uploadedText - Full text of the uploaded process
 * @param {string} analysisText - Latest assistant analysis text (for theme extraction)
 * @returns {Promise<{task_id: string, status: string}>}
 */
export async function triggerJurisprudenciaResearch(uploadedText, analysisText = '') {
    const res = await fetch(`${API_BASE}/jurisprudencia/research`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
            uploaded_text: uploadedText,
            analysis_text: analysisText,
        }),
        redirect: 'error',
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Jurisprudência Research').catch(() => ({}));
        throw new Error(err.detail || `Erro ao iniciar pesquisa jurisprudencial (${res.status})`);
    }

    return safeJson(res, 'Jurisprudência Research');
}

/**
 * Poll for jurisprudence research results (V0.5 background task).
 * @param {string} taskId - Task ID returned from research trigger
 * @returns {Promise<{status: string, result?: object, error?: string, progress?: string}>}
 */
export async function pollJurisprudenciaResearch(taskId) {
    const res = await fetch(`${API_BASE}/jurisprudencia/research/${taskId}`, {
        headers: await getAuthHeaders(),
    }).catch(() => null);

    if (!res || !res.ok) return null;
    return safeJson(res, 'Jurisprudência Research Poll').catch(() => null);
}


// ── Custom Agents CRUD ──────────────────────────────────────────────────────

/**
 * Create a custom agent.
 * @param {{ name: string, prompt: string, color: string }} agent
 * @returns {Promise<{ id: string, name: string, prompt: string, color: string }>}
 */
export async function createCustomAgent(agent) {
    const res = await fetch(`${API_BASE}/custom-agents`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(agent),
        redirect: 'error',
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Create Agent').catch(() => ({}));
        throw new Error(err.detail || `Erro ao criar agente (${res.status})`);
    }
    return safeJson(res, 'Create Agent');
}

/**
 * List custom agents for the current user.
 * @returns {Promise<{ agents: Array }>}
 */
export async function getCustomAgents() {
    const res = await fetch(`${API_BASE}/custom-agents`, {
        headers: await getAuthHeaders(),
    });
    if (!res.ok) return { agents: [] };
    return safeJson(res, 'List Agents');
}

/**
 * Delete a custom agent.
 * @param {string} agentId
 * @returns {Promise<{ status: string }>}
 */
export async function deleteCustomAgent(agentId) {
    const res = await fetch(`${API_BASE}/custom-agents/${encodeURIComponent(agentId)}`, {
        method: 'DELETE',
        headers: await getAuthHeaders(),
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Delete Agent').catch(() => ({}));
        throw new Error(err.detail || `Erro ao apagar agente (${res.status})`);
    }
    return safeJson(res, 'Delete Agent');
}

/**
 * Share a custom agent with another user by email.
 * @param {string} agentId
 * @param {string} email
 * @returns {Promise<{ status: string }>}
 */
export async function shareCustomAgent(agentId, email) {
    const res = await fetch(`${API_BASE}/custom-agents/${encodeURIComponent(agentId)}/share`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ email }),
        redirect: 'error',
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Share Agent').catch(() => ({}));
        throw new Error(err.detail || `Erro ao compartilhar agente (${res.status})`);
    }
    return safeJson(res, 'Share Agent');
}
