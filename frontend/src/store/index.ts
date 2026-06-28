import { create } from 'zustand';
import type {
    Message,
    Agent,
    BatchPilotSession,
    SelectedModel,
    CustomAgent,
} from '../types/chat';

// ── UI Store ───────────────────────────────────────────────────────────
interface UIStoreState {
    sidebarOpen: boolean;
    canvasMode: boolean;
    batchPanelOpen: boolean;
    selectedModel: SelectedModel;
    harnessEconomico: boolean;
    setSidebarOpen: (val: boolean) => void;
    toggleSidebar: () => void;
    setCanvasMode: (val: boolean) => void;
    setBatchPanelOpen: (val: boolean) => void;
    setSelectedModel: (model: SelectedModel) => void;
    setHarnessEconomico: (val: boolean) => void;
}

export const useUIStore = create<UIStoreState>((set) => ({
    sidebarOpen: typeof window !== 'undefined' ? window.innerWidth > 768 : true,
    canvasMode: false,
    batchPanelOpen: false,
    selectedModel: { id: 'deepseek', name: 'DeepSeek V4 Pro', color: '#0891B2', llm: 'DeepSeek-V4-Pro' },
    harnessEconomico: false,

    setSidebarOpen: (val) => set({ sidebarOpen: val }),
    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
    setCanvasMode: (val) => set({ canvasMode: val }),
    setBatchPanelOpen: (val) => set({ batchPanelOpen: val }),
    setSelectedModel: (model) => set({ selectedModel: model }),
    setHarnessEconomico: (val) => set({ harnessEconomico: val }),
}));

// ── Agent Store ────────────────────────────────────────────────────────
interface AgentStoreState {
    customAgents: CustomAgent[];
    setCustomAgents: (agents: CustomAgent[]) => void;
    addCustomAgent: (agent: CustomAgent) => void;
    removeCustomAgent: (id: string) => void;
}

export const useAgentStore = create<AgentStoreState>((set) => ({
    customAgents: [],
    setCustomAgents: (customAgents) => set({ customAgents }),
    addCustomAgent: (agent) => set((s) => ({ customAgents: [agent, ...s.customAgents] })),
    removeCustomAgent: (id) => set((s) => ({ customAgents: s.customAgents.filter((a) => a.id !== id) })),
}));

// ── Chat Store ─────────────────────────────────────────────────────────
// Single source of truth for chat-related state shared across many hooks.

type Updater<T> = T | ((prev: T) => T);

interface ChatStoreState {
    messages: Message[];
    conversationId: string | null;
    uploadedText: string | null;
    activeAgent: Agent | null;
    styleDossier: string | null;
    jurisContext: string;
    isLoading: boolean;
    batchPilotSession: BatchPilotSession | null;

    setMessages: (m: Updater<Message[]>) => void;
    addMessage: (m: Message) => void;
    addMessages: (m: Message[]) => void;
    setIsLoading: (val: boolean) => void;
    setConversationId: (id: string | null) => void;
    setUploadedText: (text: Updater<string | null>) => void;
    setActiveAgent: (agent: Agent | null) => void;
    setStyleDossier: (s: string | null) => void;
    setJurisContext: (ctx: Updater<string>) => void;
    setBatchPilotSession: (s: Updater<BatchPilotSession | null>) => void;
    resetChat: () => void;
}

function resolve<T>(updater: Updater<T>, prev: T): T {
    return typeof updater === 'function' ? (updater as (p: T) => T)(prev) : updater;
}

export const useChatStore = create<ChatStoreState>((set) => ({
    messages: [],
    conversationId: null,
    uploadedText: null,
    activeAgent: null,
    styleDossier: null,
    jurisContext: '',
    isLoading: false,
    batchPilotSession: null,

    setMessages: (m) => set((s) => ({ messages: resolve(m, s.messages) })),
    addMessage: (m) => set((s) => ({ messages: [...s.messages, m] })),
    addMessages: (m) => set((s) => ({ messages: [...s.messages, ...m] })),
    setIsLoading: (val) => set({ isLoading: val }),
    setConversationId: (id) => set({ conversationId: id }),
    setUploadedText: (text) => set((s) => ({ uploadedText: resolve(text, s.uploadedText) })),
    setActiveAgent: (agent) => set({ activeAgent: agent }),
    setStyleDossier: (sd) => set({ styleDossier: sd }),
    setJurisContext: (ctx) => set((s) => ({ jurisContext: resolve(ctx, s.jurisContext) })),
    setBatchPilotSession: (sess) =>
        set((s) => ({ batchPilotSession: resolve(sess, s.batchPilotSession) })),
    resetChat: () =>
        set({
            messages: [],
            conversationId: null,
            uploadedText: null,
            activeAgent: null,
            jurisContext: '',
            batchPilotSession: null,
            isLoading: false,
        }),
}));
