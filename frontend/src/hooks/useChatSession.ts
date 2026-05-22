import { useCallback, useState } from 'react';
import { useChatStore } from '../store';
import type { Agent, ChatHistoryEntry, Message } from '../types/chat';

interface UseChatSessionArgs {
    onResetXray: () => void;
    onResetJurisprudencia: () => void;
    onResetCanvas: () => void;
    onResetSustentacao: () => void;
}

export function useChatSession({
    onResetXray,
    onResetJurisprudencia,
    onResetCanvas,
    onResetSustentacao,
}: UseChatSessionArgs) {
    const {
        messages,
        conversationId,
        activeAgent,
        addMessage,
        setMessages,
        setActiveAgent,
        resetChat,
    } = useChatStore();
    const [chatHistory, setChatHistory] = useState<ChatHistoryEntry[]>([]);

    const handleAgentSelect = useCallback(
        (agent: Agent) => {
            if (activeAgent?.id === agent.id) {
                setActiveAgent(null);
                setMessages((prev) => prev.filter((m) => m.role !== 'agent-activation'));
                return;
            }
            setActiveAgent(agent);
            const isCustom = !agent.promptModule && !!agent.prompt;
            const activationMsg: Message = {
                role: 'agent-activation',
                agentName: agent.name,
                agentDesc: agent.desc || (isCustom ? 'Agente personalizado' : ''),
                agentColor: agent.color || '#8B5CF6',
                agentIcon: agent.icon || 'FaRobot',
                autoAction: agent.autoAction || null,
            };
            setMessages((prev) => [
                ...prev.filter((m) => m.role !== 'agent-activation'),
                activationMsg,
            ]);
        },
        [activeAgent, setActiveAgent, setMessages]
    );

    const handleNewChat = useCallback(() => {
        const realMessages = messages.filter((m) => m.role === 'user' || m.role === 'assistant');
        if (realMessages.length > 0) {
            const firstUserMsg = realMessages.find((m) => m.role === 'user');
            const title = firstUserMsg?.content
                ? firstUserMsg.content.slice(0, 60) + (firstUserMsg.content.length > 60 ? '…' : '')
                : 'Conversa sem título';
            setChatHistory((prev) => [
                {
                    id: conversationId || Date.now().toString(),
                    title,
                    messages: [...messages],
                    agent: activeAgent,
                    timestamp: new Date(),
                },
                ...prev,
            ]);
        }
        resetChat();
        onResetXray();
        onResetJurisprudencia();
        onResetCanvas();
        onResetSustentacao();
    }, [
        messages,
        activeAgent,
        conversationId,
        resetChat,
        onResetXray,
        onResetJurisprudencia,
        onResetCanvas,
        onResetSustentacao,
    ]);

    const handleLoadChat = useCallback(
        (chatId: string) => {
            const chat = chatHistory.find((c) => c.id === chatId);
            if (!chat) return;
            setMessages(chat.messages);
            setActiveAgent(chat.agent);
            // setConversationId is encapsulated; do it via direct store access
            useChatStore.setState({ conversationId: chat.id, uploadedText: null });
            onResetXray();
        },
        [chatHistory, setMessages, setActiveAgent, onResetXray]
    );

    // Silence unused warning for addMessage when strict mode is enabled
    void addMessage;

    return {
        chatHistory,
        handleAgentSelect,
        handleNewChat,
        handleLoadChat,
    };
}
