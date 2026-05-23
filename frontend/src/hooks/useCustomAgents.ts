import { useCallback, useEffect, useState } from 'react';
import {
    getCustomAgents,
    createCustomAgent,
    deleteCustomAgent,
    shareCustomAgent,
    uploadAgentFiles,
} from '../services/api';
import { useAgentStore, useChatStore } from '../store';
import type { CustomAgent } from '../types/chat';

interface CreateAgentInput {
    name: string;
    description?: string;
    prompt: string;
    color?: string;
    files?: File[];
}

export function useCustomAgents() {
    const customAgents = useAgentStore((s) => s.customAgents);
    const { setCustomAgents, addCustomAgent, removeCustomAgent } = useAgentStore();
    const { activeAgent, setActiveAgent } = useChatStore();

    const [showAgentBuilder, setShowAgentBuilder] = useState(false);
    const [showCreateDialog, setShowCreateDialog] = useState(false);
    const [pendingPrompt, setPendingPrompt] = useState('');
    const [shareAgent, setShareAgent] = useState<CustomAgent | null>(null);

    useEffect(() => {
        getCustomAgents()
            .then((data) => setCustomAgents(data.agents || []))
            .catch(() => {});
    }, [setCustomAgents]);

    const handleOpenAgentBuilder = useCallback(() => setShowAgentBuilder(true), []);

    const handlePromptReady = useCallback((prompt: string) => {
        setPendingPrompt(prompt);
        setShowCreateDialog(true);
    }, []);

    const handleCreateAgent = useCallback(
        async ({ name, description, prompt, color, files }: CreateAgentInput) => {
            try {
                const created = await createCustomAgent({ name, description: description || '', prompt, color: color || '#8B5CF6' });
                if (files && files.length > 0 && created.id) {
                    try {
                        await uploadAgentFiles(created.id, files);
                    } catch (uploadErr) {
                        console.warn('File upload for agent failed (agent still created):', uploadErr);
                    }
                }
                addCustomAgent(created);
                setShowCreateDialog(false);
                setShowAgentBuilder(false);
                setPendingPrompt('');
            } catch (err) {
                console.error('Failed to create agent:', err);
            }
        },
        [addCustomAgent]
    );

    const handleDeleteAgent = useCallback(
        async (agentId: string) => {
            try {
                await deleteCustomAgent(agentId);
                removeCustomAgent(agentId);
                if (activeAgent?.id === agentId) setActiveAgent(null);
            } catch (err) {
                console.error('Failed to delete agent:', err);
            }
        },
        [activeAgent, removeCustomAgent, setActiveAgent]
    );

    const handleShareAgentOpen = useCallback((agent: CustomAgent) => {
        setShareAgent(agent);
    }, []);

    const handleShareAgentConfirm = useCallback(
        async (email: string) => {
            if (!shareAgent) return;
            await shareCustomAgent(shareAgent.id, email);
            setShareAgent(null);
        },
        [shareAgent]
    );

    return {
        customAgents,
        showAgentBuilder,
        showCreateDialog,
        pendingPrompt,
        shareAgent,
        setShowAgentBuilder,
        setShowCreateDialog,
        setPendingPrompt,
        setShareAgent,
        handleOpenAgentBuilder,
        handlePromptReady,
        handleCreateAgent,
        handleDeleteAgent,
        handleShareAgentOpen,
        handleShareAgentConfirm,
    };
}
