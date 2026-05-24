// Shared types for chat, agents, messages, and batch flows.

export type MessageRole =
    | 'user'
    | 'assistant'
    | 'ocr'
    | 'ocr-processing'
    | 'style-processing'
    | 'xray-processing'
    | 'slm-pipeline'
    | 'agent-activation'
    | 'batch-pilot-confirm';

export interface V2Section {
    title: string;
    content: string;
    type?: string;
}

export interface ModelContext {
    [key: string]: unknown;
}

export interface Message {
    role: MessageRole | string;
    content?: string;
    model?: string;
    v2Sections?: V2Section[] | null;
    modelContext?: ModelContext | null;
    isCanvasUpdate?: boolean;
    // OCR-specific fields
    filename?: string;
    text?: string;
    engine?: string;
    charCount?: number;
    // Agent activation fields
    agentName?: string;
    agentDesc?: string;
    agentColor?: string;
    agentIcon?: string;
    autoAction?: AgentAutoAction | null;
    // Batch pilot confirm fields
    remainingCount?: number;
    clusterName?: string;
}

export interface AgentAutoAction {
    label: string;
    requiresMinuta?: boolean;
    requiresUploadedText?: boolean;
}

export interface Agent {
    id: string;
    name: string;
    icon?: string;
    desc?: string;
    color?: string;
    promptModule?: () => Promise<{ default: string }>;
    prompt?: string;
    autoAction?: AgentAutoAction;
}

export interface CustomAgent extends Agent {
    prompt: string;
    description?: string;
    knowledge?: string;
    created_at?: string;
    type?: 'regular' | 'orquestrador';
    flow_id?: string;
}

export interface SelectedModel {
    id: string;
    name: string;
    color: string;
    llm: string;
}

export interface BatchProcess {
    filename: string;
    text: string;
}

export type BatchPilotPhase = 'pilot' | 'confirming' | 'replicating' | 'done';

export interface BatchPilotSession {
    clusterName: string;
    processes: BatchProcess[];
    pilotFilename: string;
    pilotMinuta: string | null;
    phase: BatchPilotPhase;
}

export interface BatchResult {
    filename: string;
    status: 'ok' | 'error';
    response: unknown;
    model: string;
    alerts_count?: number;
}

export interface XrayCluster {
    nome: string;
    arquivos: string[];
    [key: string]: unknown;
}

export interface XrayReport {
    clusters: XrayCluster[];
    [key: string]: unknown;
}

export interface RagStatus {
    indexed_chunks: number;
    has_dossier: boolean;
}

export interface OcrProgress {
    progress: string;
    percent: number;
}

export interface ChatHistoryEntry {
    id: string;
    title: string;
    messages: Message[];
    agent: Agent | null;
    timestamp: Date;
}

export interface JurisResearchItem {
    theme: string;
    results: Array<{
        ementa?: string;
        numero_processo?: string;
        tipo_recurso?: string;
        data_publicacao?: string;
        [key: string]: unknown;
    }>;
}

export interface JurisResearch {
    research: JurisResearchItem[];
    [key: string]: unknown;
}
