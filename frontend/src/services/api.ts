/**
 * API service module — communicates with the FastAPI backend.
 */
import { supabase } from './supabase';

const API_BASE = '/api';

// ── TypeScript Interfaces ───────────────────────────────────────────────────

export interface JurisprudenciaFilters {
    anoInicio?: number;
    anoFim?: number;
    tipo?: string;
    page?: number;
    pageSize?: number;
}

export interface SendMessageParams {
    message: string;
    model: string;
    llm?: string | null;
    agentPrompt?: string | null;
    conversationId?: string | null;
    uploadedText?: string | null;
    styleDossier?: string | null;
    useRag?: boolean;
    jurisprudenceContext?: string | null;
    jurisEnabled?: boolean;
    reasoningEnabled?: boolean;
    agentKnowledge?: string | null;
}

export interface ReplicateBatchPilotParams {
    pilotInstructions: string;
    pilotMinuta: string;
    pilotSummary?: string | null;
    processes: Array<{ filename: string; text: string }>;
    model?: string;
    llm?: string | null;
    agentPrompt?: string | null;
}

export interface UploadProgressInfo {
    progress: string;
    percent: number;
}

export interface UploadResult {
    filename: string;
    text: string;
    char_count: number;
    rag_available: boolean;
}

/**
 * Get headers for API requests.
 * Includes Azure OpenAI key if stored, and Supabase JWT.
 */
async function getAuthHeaders(existingHeaders: Record<string, string> = {}): Promise<Record<string, string>> {
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
export async function validateAzureKey(key: string) {
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
async function safeJson(res: Response, context: string) {
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
export async function uploadFile(file: File, ocrEngine = 'mistral_doc_ai', compress = true, vectorize = true, onProgress: ((info: UploadProgressInfo) => void) | null = null): Promise<UploadResult> {
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
export async function sendMessage({ message, model, llm, agentPrompt, conversationId, uploadedText, styleDossier, useRag = false, jurisprudenceContext = null, jurisEnabled = true, reasoningEnabled = false, agentKnowledge = null }: SendMessageParams) {
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
            juris_enabled: jurisEnabled,
            reasoning_enabled: reasoningEnabled,
            agent_knowledge: agentKnowledge || null,
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
export async function uploadBatchXray(files: File[], onProgress: ((progress: string) => void) | null = null) {
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
 * Analyze a cluster of processes one-by-one, streaming results via SSE.
 * Optional onResult callback fires as each process finishes; the returned
 * promise resolves with the same shape as the old JSON response so callers
 * don't need to change.
 */
export async function analyzeCluster(
    processes: Array<{ filename: string; text: string }>,
    agentPrompt = '',
    model = 'claude',
    llm: string | null = null,
    clusterSituacao: string | null = null,
    clusterTags: string[] | null = null,
    onResult?: (result: { filename: string; status: string; response: string; model: string }) => void,
) {
    const res = await fetch(`${API_BASE}/cluster-analyze`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
            processes,
            agent_prompt: agentPrompt,
            model,
            llm,
            cluster_situacao: clusterSituacao,
            cluster_tags: clusterTags,
        }),
        redirect: 'error',
    });

    if (!res.ok || !res.body) {
        const err = await safeJson(res, 'Cluster Analyze').catch(() => ({}));
        throw new Error(err.detail || `Análise em lote falhou (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    const results: Array<{ filename: string; status: string; response: string; model: string }> = [];
    let donePayload: { total: number; ok_count: number; skill_applied: string | null } | null = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sepIdx: number;
        while ((sepIdx = buffer.indexOf('\n\n')) >= 0) {
            const raw = buffer.slice(0, sepIdx);
            buffer = buffer.slice(sepIdx + 2);
            const dataLine = raw.split('\n').find((l) => l.startsWith('data: '));
            if (!dataLine) continue;
            try {
                const parsed = JSON.parse(dataLine.slice(6));
                if (parsed.event === 'result') {
                    const r = { filename: parsed.filename, status: parsed.status, response: parsed.response, model: parsed.model };
                    results.push(r);
                    onResult?.(r);
                } else if (parsed.event === 'done') {
                    donePayload = { total: parsed.total, ok_count: parsed.ok_count, skill_applied: parsed.skill_applied };
                } else if (parsed.event === 'error') {
                    throw new Error(parsed.message || 'Erro na análise em lote');
                }
            } catch (e) {
                if (e instanceof SyntaxError) continue; // malformed SSE line
                throw e;
            }
        }
    }

    results.sort((a, b) => a.filename.localeCompare(b.filename));
    return {
        results,
        total: donePayload?.total ?? results.length,
        ok_count: donePayload?.ok_count ?? results.filter((r) => r.status === 'ok').length,
        skill_applied: donePayload?.skill_applied ?? null,
    };
}

/**
 * Replicate pilot case instructions + draft to remaining processes.
 * @param {object} params
 * @param {string} params.pilotInstructions - Consolidated judge instructions from pilot conversation
 * @param {string} params.pilotMinuta - Draft decision generated during pilot (template)
 * @param {string} [params.pilotSummary] - Summary of pilot case facts
 * @param {Array<{filename: string, text: string}>} params.processes - Remaining processes
 * @param {string} [params.model] - Engine model ID
 * @param {string} [params.llm] - LLM deployment name
 * @param {string} [params.agentPrompt] - Agent system prompt (optional)
 * @returns {Promise<{results: Array, total: number, ok_count: number, total_alerts: number}>}
 */
export async function replicateBatchPilot({ pilotInstructions, pilotMinuta, pilotSummary, processes, model, llm, agentPrompt }: ReplicateBatchPilotParams) {
    const res = await fetch(`${API_BASE}/batch-pilot/replicate`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
            pilot_instructions: pilotInstructions,
            pilot_minuta: pilotMinuta,
            pilot_summary: pilotSummary || null,
            processes,
            model: model || 'gpt53',
            llm: llm || null,
            agent_prompt: agentPrompt || null,
        }),
        redirect: 'error',
    });

    if (!res.ok) {
        const err = await safeJson(res, 'Batch Pilot Replicate').catch(() => ({}));
        throw new Error(err.detail || `Replicação piloto falhou (${res.status})`);
    }

    return safeJson(res, 'Batch Pilot Replicate');
}


// ── Deep Research ────────────────────────────────────────────────────────────

export type DeepResearchEvent =
    | { event: 'phase'; phase: string; message: string }
    | { event: 'plan'; questions: string[] }
    | { event: 'question_start'; index: number; total: number; question: string }
    | { event: 'question_done'; index: number; answer: string }
    | { event: 'question_error'; index: number; message: string }
    | { event: 'done'; dossier: string; qa_pairs: { question: string; answer: string }[]; chunks_indexed: number }
    | { event: 'error'; message: string };

/**
 * Run deep research on an uploaded process text. Streams progress events from
 * the backend via SSE; the consumer receives a callback per event.
 */
export async function runDeepResearch(
    text: string,
    onEvent: (event: DeepResearchEvent) => void,
    model: string | null = null,
): Promise<void> {
    const res = await fetch(`${API_BASE}/deep-research`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ text, model }),
        redirect: 'error',
    });

    if (!res.ok || !res.body) {
        const err = await safeJson(res, 'Deep Research').catch(() => ({}));
        throw new Error(err.detail || `Deep research falhou (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Parse SSE stream: events are separated by "\n\n", each line begins with "data: "
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sepIndex;
        while ((sepIndex = buffer.indexOf('\n\n')) >= 0) {
            const rawEvent = buffer.slice(0, sepIndex);
            buffer = buffer.slice(sepIndex + 2);
            const dataLine = rawEvent.split('\n').find((l) => l.startsWith('data: '));
            if (!dataLine) continue;
            try {
                const parsed = JSON.parse(dataLine.slice(6)) as DeepResearchEvent;
                onEvent(parsed);
            } catch {
                // ignore malformed events
            }
        }
    }
}


/**
 * Generate a Style Dossier from template decision files.
 * @param {File[]} files - Template files (PDF/DOCX/TXT)
 * @returns {Promise<{dossier: string, glossary: string, cloning_prompt: string, full_response: string, file_count: number}>}
 */
export async function generateStyleReport(files: File[]) {
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
export async function uploadTemplates(files: File[]) {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));

    const res = await fetch(`${API_BASE}/templates`, {
        method: 'POST',
        headers: await getAuthHeaders(),
        body: form,
        redirect: 'error',
    }).catch((err) => {
        throw new Error(
            `Indexação: requisição bloqueada ou redirecionada. (${err.message})`
        );
    });

    if (!res.ok) {
        // Preserve the actual error details instead of discarding them
        let detail = `Erro ao indexar modelos (${res.status})`;
        try {
            const err = await safeJson(res, 'Upload Templates');
            detail = err.detail || detail;
        } catch (parseErr: unknown) {
            // safeJson throws when response is not JSON — use its message (has content-type + preview)
            detail = (parseErr instanceof Error ? parseErr.message : String(parseErr)) || detail;
        }
        throw new Error(detail);
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
export async function deleteTemplate(filename: string) {
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
export async function askTemplates(query: string) {
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
export async function verifyTheme(theme: string) {
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
export async function searchJurisprudencia(query: string, filters: JurisprudenciaFilters = {}) {
    const params = new URLSearchParams({ q: query });
    if (filters.anoInicio) params.set('ano_inicio', String(filters.anoInicio));
    if (filters.anoFim) params.set('ano_fim', String(filters.anoFim));
    if (filters.tipo) params.set('tipo', filters.tipo);
    if (filters.page) params.set('page', String(filters.page));
    if (filters.pageSize) params.set('page_size', String(filters.pageSize));

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
export async function getJurisprudenciaDoc(docId: number) {
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
export async function askJurisprudencia(query: string, filters: JurisprudenciaFilters = {}) {
    const body: Record<string, string | number> = { query };
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
export async function triggerJurisprudenciaResearch(uploadedText: string, analysisText = '') {
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
export async function pollJurisprudenciaResearch(taskId: string) {
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
export async function createCustomAgent(agent: { name: string; prompt: string; color: string; description?: string; knowledge?: string }) {
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

export async function uploadAgentFiles(agentId: string, files: File[]): Promise<{ status: string; knowledge_chars: number }> {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));
    const res = await fetch(`${API_BASE}/custom-agents/${encodeURIComponent(agentId)}/files`, {
        method: 'POST',
        headers: await getAuthHeaders(),
        body: form,
        redirect: 'error',
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Upload Agent Files').catch(() => ({}));
        throw new Error(err.detail || `Erro ao enviar arquivos (${res.status})`);
    }
    return safeJson(res, 'Upload Agent Files');
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
export async function deleteCustomAgent(agentId: string) {
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
export async function shareCustomAgent(agentId: string, email: string) {
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


// ── SLM Status ──────────────────────────────────────────────────────────────

/**
 * Check the SLM (Small Language Model) server status.
 * @returns {Promise<{available: boolean, mode: string, ...}>}
 */
export async function getSlmStatus() {
    try {
        const res = await fetch(`${API_BASE}/slm/status`, {
            headers: await getAuthHeaders(),
        }).catch(() => null);
        if (!res || !res.ok) return { available: false, mode: 'none' };
        return safeJson(res, 'SLM Status').catch(() => ({ available: false, mode: 'none' }));
    } catch {
        return { available: false, mode: 'none' };
    }
}


// ── User Memory (Preferences) ───────────────────────────────────────────────

/**
 * Get user memory/preferences.
 * @returns {Promise<{content: string, enabled: boolean}>}
 */
export async function getMemory() {
    const res = await fetch(`${API_BASE}/memory`, {
        headers: await getAuthHeaders(),
    }).catch(() => null);
    if (!res || !res.ok) return { content: '', enabled: true };
    return safeJson(res, 'Get Memory').catch(() => ({ content: '', enabled: true }));
}

/**
 * Save user memory/preferences.
 * @param {string} content - Memory text (max 2000 chars)
 * @param {boolean} enabled - Whether memory injection is active
 * @returns {Promise<{content: string, enabled: boolean}>}
 */
export async function saveMemory(
    content: string,
    enabled = true,
    response_style = 'default',
    response_style_text = ''
) {
    const res = await fetch(`${API_BASE}/memory`, {
        method: 'PUT',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ content, enabled, response_style, response_style_text }),
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Save Memory').catch(() => ({}));
        throw new Error(err.detail || `Erro ao salvar memória (${res.status})`);
    }
    return safeJson(res, 'Save Memory');
}

// ── Auto-Memories ────────────────────────────────────────────────────────────

export async function getAutoMemories(): Promise<{ memories: AutoMemory[] }> {
    const res = await fetch(`${API_BASE}/auto-memories`, {
        headers: await getAuthHeaders(),
    }).catch(() => null);
    if (!res || !res.ok) return { memories: [] };
    return safeJson(res, 'Get Auto Memories').catch(() => ({ memories: [] }));
}

export async function updateAutoMemory(id: number, data: { content?: string; enabled?: boolean }) {
    const res = await fetch(`${API_BASE}/auto-memories/${id}`, {
        method: 'PUT',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`Erro ao atualizar memória (${res.status})`);
    return safeJson(res, 'Update Auto Memory');
}

export async function deleteAutoMemory(id: number) {
    const res = await fetch(`${API_BASE}/auto-memories/${id}`, {
        method: 'DELETE',
        headers: await getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`Erro ao deletar memória (${res.status})`);
    return safeJson(res, 'Delete Auto Memory');
}

export interface AutoMemory {
    id: number;
    content: string;
    source_conversation_id: string | null;
    enabled: boolean;
    created_at: string;
}


// ── Sustentação Oral / Audiência ────────────────────────────────────────────

export type TipoAto = 'sustentacao' | 'audiencia';
export type Modo = 'preparacao' | 'realizacao';

// Schema união — campos presentes variam por (tipo_ato × modo).
// Cada sub-view consome só o subconjunto que precisa.
export interface SustentacaoData {
    // Comuns / Sustentação
    numero_processo?: string | null;
    tipo_recursal?: string | null;
    camara?: string | null;
    relator?: string | null;
    data_sessao?: string | null;
    recorrente?: string | null;
    recorrido?: string | null;
    advogado_sustentante?: string | null;
    parte_sustentante?: string | null;
    sintese_recurso?: string | null;
    teses?: string[];
    preliminares?: string[];
    sintese_decisao_1grau?: string | null;
    pontos_criticos?: string[];
    pre_juizo?: string | null;

    // Audiência
    vara?: string | null;
    juiz?: string | null;
    tipo_acao?: string | null;
    autor?: string | null;
    reu?: string | null;
    data_audiencia?: string | null;
    pontos_controvertidos?: string[];
    onus_prova?: Array<{ de_quem: string; fato: string }>;
    provas_deferidas?: string[];
    provas_indeferidas?: string[];
    testemunhas_autor?: Array<{ nome: string; intimada: boolean | null; ja_depos: boolean | null }>;
    testemunhas_reu?: Array<{ nome: string; intimada: boolean | null; ja_depos: boolean | null }>;
    depoimento_pessoal_autor?: boolean | null;
    depoimento_pessoal_reu?: boolean | null;
    documentos_relevantes?: string[];
    quesitos_sugeridos?: Array<{ para: string; perguntas: string[] }>;
    depoentes?: Array<{ id: string; nome: string; tipo: string }>;
    perguntas_planejadas?: Array<{ depoente_id: string; perguntas: string[] }>;
}

export interface ExtractResult {
    process_id: string;
    data: SustentacaoData;
    tipo_ato: TipoAto;
    modo: Modo;
}

export async function extractSustentacao(text: string, tipoAto: TipoAto = 'sustentacao', modo: Modo = 'preparacao'): Promise<ExtractResult> {
    const res = await fetch(`${API_BASE}/sustentacao/extract`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ text, tipo_ato: tipoAto, modo }),
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Sustentação Extract').catch(() => ({}));
        throw new Error(err.detail || `Erro ao extrair processo (${res.status})`);
    }
    return safeJson(res, 'Sustentação Extract');
}

export interface AnaliseVotoResult {
    resultado: 'favoravel' | 'desfavoravel' | 'parcial';
    resumo: string;
    por_tese: Array<{
        tese: string;
        posicao: 'favoravel' | 'desfavoravel' | 'nao_apreciada';
        justificativa: string;
    }>;
}

export interface AnaliseSentencaResult {
    resultado: 'procedente' | 'improcedente' | 'parcial';
    resumo: string;
    por_ponto: Array<{
        ponto: string;
        decisao: string;
        fundamento: string;
        alerta: string | null;
    }>;
}

export async function analisarVoto(processId: string, documentoText: string): Promise<AnaliseVotoResult> {
    const res = await fetch(`${API_BASE}/sustentacao/analisar-voto`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ process_id: processId, documento_text: documentoText }),
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Análise voto').catch(() => ({}));
        throw new Error(err.detail || `Erro ao analisar voto (${res.status})`);
    }
    return safeJson(res, 'Análise voto');
}

export async function analisarSentenca(processId: string, documentoText: string): Promise<AnaliseSentencaResult> {
    const res = await fetch(`${API_BASE}/sustentacao/analisar-sentenca`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ process_id: processId, documento_text: documentoText }),
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Análise sentença').catch(() => ({}));
        throw new Error(err.detail || `Erro ao analisar sentença (${res.status})`);
    }
    return safeJson(res, 'Análise sentença');
}

export async function chatSustentacao(processId: string, messages: Array<{ role: string; content: string }>): Promise<{ reply: string }> {
    const res = await fetch(`${API_BASE}/sustentacao/chat`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ process_id: processId, messages }),
    });
    if (!res.ok) {
        const err = await safeJson(res, 'Sustentação Chat').catch(() => ({}));
        throw new Error(err.detail || `Erro no chat (${res.status})`);
    }
    return safeJson(res, 'Sustentação Chat');
}

// ── Agent Flows ────────────────────────────────────────────────────────────

export interface FlowSummary {
    id: string;
    name: string;
    created_at: string;
    updated_at: string;
}

export interface FlowConfig {
    nodes: Array<{ id: string; type: string; position: { x: number; y: number }; data: Record<string, string> }>;
    edges: Array<{ id: string; source: string; target: string; label?: string; sourceHandle?: string }>;
}

export interface Flow {
    id: string;
    name: string;
    config: FlowConfig;
    created_at: string;
    updated_at: string;
}

export async function listFlows(): Promise<FlowSummary[]> {
    const res = await fetch(`${API_BASE}/flows`, {
        headers: await getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`Erro ao listar fluxos (${res.status})`);
    const data = await safeJson(res, 'List flows');
    return data.flows ?? [];
}

export async function createFlow(name: string, config: FlowConfig): Promise<Flow> {
    const res = await fetch(`${API_BASE}/flows`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ name, config }),
    });
    if (!res.ok) throw new Error(`Erro ao criar fluxo (${res.status})`);
    return safeJson(res, 'Create flow');
}

export async function getFlow(flowId: string): Promise<Flow> {
    const res = await fetch(`${API_BASE}/flows/${flowId}`, {
        headers: await getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`Fluxo não encontrado (${res.status})`);
    return safeJson(res, 'Get flow');
}

export async function updateFlow(flowId: string, name: string, config: FlowConfig): Promise<void> {
    const res = await fetch(`${API_BASE}/flows/${flowId}`, {
        method: 'PUT',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ name, config }),
    });
    if (!res.ok) throw new Error(`Erro ao salvar fluxo (${res.status})`);
}

export async function deleteFlow(flowId: string): Promise<void> {
    const res = await fetch(`${API_BASE}/flows/${flowId}`, {
        method: 'DELETE',
        headers: await getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`Erro ao apagar fluxo (${res.status})`);
}

export async function runFlow(
    flowId: string,
    inputText: string,
    onEvent: (event: Record<string, unknown>) => void,
): Promise<void> {
    const res = await fetch(`${API_BASE}/flows/${flowId}/run`, {
        method: 'POST',
        headers: await getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ input_text: inputText }),
    });
    if (!res.ok) throw new Error(`Erro ao executar fluxo (${res.status})`);
    const reader = res.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    onEvent(JSON.parse(line.slice(6)));
                } catch {
                    // ignore malformed
                }
            }
        }
    }
}
