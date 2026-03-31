import { create } from 'zustand';

interface SelectedModel {
    id: string;
    name: string;
    color: string;
    llm: string;
}

interface UIStoreState {
    sidebarOpen: boolean;
    canvasMode: boolean;
    batchPanelOpen: boolean;
    selectedModel: SelectedModel;
    setSidebarOpen: (val: boolean) => void;
    toggleSidebar: () => void;
    setCanvasMode: (val: boolean) => void;
    setBatchPanelOpen: (val: boolean) => void;
    setSelectedModel: (model: SelectedModel) => void;
}

/**
 * UI state store — manages sidebar, canvas mode, batch panel, and selected model.
 * Chat/message state remains in App.jsx due to heavy interdependencies with API calls.
 */
export const useUIStore = create<UIStoreState>((set) => ({
    sidebarOpen: typeof window !== 'undefined' ? window.innerWidth > 768 : true,
    canvasMode: false,
    batchPanelOpen: false,
    selectedModel: { id: 'gpt53', name: 'GPT-5.3', color: '#4285F4', llm: 'gpt-5.3-chat' },

    setSidebarOpen: (val) => set({ sidebarOpen: val }),
    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
    setCanvasMode: (val) => set({ canvasMode: val }),
    setBatchPanelOpen: (val) => set({ batchPanelOpen: val }),
    setSelectedModel: (model) => set({ selectedModel: model }),
}));

/**
 * Agent store — manages agent list and active agent selection.
 * Active agent selection is partially kept in App.jsx for backward-compat with callbacks.
 */
export const useAgentStore = create((set) => ({
    agents: [],
    activeAgent: null,

    setAgents: (agents) => set({ agents }),
    setActiveAgent: (agent) => set({ activeAgent: agent }),
}));

/**
 * Chat store — messages, conversations, loading and streaming state.
 * NOTE: This store is defined but the actual state management for messages
 * remains in App.jsx due to the deep integration with async API calls,
 * batch pilots, canvas, jurisprudence, and OCR flows.
 * Use this store for future migration as those flows are extracted.
 */
export const useChatStore = create((set) => ({
    messages: [],
    conversations: [],
    activeConversationId: null,
    isLoading: false,
    streamingMessage: '',

    setMessages: (messages) => set({ messages }),
    addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
    setConversations: (conversations) => set({ conversations }),
    setActiveConversationId: (id) => set({ activeConversationId: id }),
    setIsLoading: (val) => set({ isLoading: val }),
    setStreamingMessage: (msg) => set({ streamingMessage: msg }),
    clearMessages: () => set({ messages: [], activeConversationId: null }),
}));
