import { useCallback } from 'react';
import { sendMessage, uploadTemplates, generateStyleReport } from '../services/api';
import { useChatStore, useUIStore } from '../store';
import { safeResponseText, errorText } from './utils';
import type { Message, SelectedModel, RagStatus } from '../types/chat';

interface SendOptions {
    hideUserBubble?: boolean;
    overridePrompt?: string | null;
}

interface CanvasContext {
    canvasOpen: boolean;
    canvasContent: string;
    canvasSelection: string | null;
    setCanvasContent: (content: string) => void;
    setCanvasSelection: (sel: string | null) => void;
}

interface UseMessageSenderArgs {
    canvas: CanvasContext;
    ragStatus: RagStatus | null;
    setRagStatus: (s: RagStatus) => void;
}

export function useMessageSender({ canvas, ragStatus, setRagStatus }: UseMessageSenderArgs) {
    const {
        addMessage,
        addMessages,
        setIsLoading,
        setConversationId,
        setBatchPilotSession,
        setStyleDossier,
    } = useChatStore();
    const { selectedModel: globalSelectedModel } = useUIStore();

    const handleSend = useCallback(
        async (
            message: string,
            selectedModel: SelectedModel,
            _files: File[] | undefined,
            _ocrEngine: string | undefined,
            templateFiles: File[] | undefined,
            useRag = false,
            { hideUserBubble = false, overridePrompt = null }: SendOptions = {}
        ) => {
            const state = useChatStore.getState();
            const {
                activeAgent,
                conversationId,
                uploadedText,
                styleDossier,
                jurisContext,
                batchPilotSession,
            } = state;

            const userTyped = message.trim();
            const effectiveMessage =
                userTyped ||
                (uploadedText && activeAgent
                    ? 'Analise o documento anexado conforme as instruções do agente.'
                    : uploadedText
                        ? 'Analise o documento anexado.'
                        : '');
            if (!effectiveMessage) return;

            // Canvas context injection
            let finalMessage = effectiveMessage;
            if (canvas.canvasOpen && canvas.canvasContent && canvas.canvasContent.length > 50) {
                if (canvas.canvasSelection && canvas.canvasSelection.length > 5) {
                    finalMessage = [
                        '[TRECHO SELECIONADO PARA EDIÇÃO]:',
                        canvas.canvasSelection,
                        '',
                        '[DOCUMENTO COMPLETO NO CANVAS (contexto)]:',
                        canvas.canvasContent,
                        '',
                        `[INSTRUÇÃO DO USUÁRIO]: ${effectiveMessage}`,
                        '',
                        'Retorne O DOCUMENTO COMPLETO ATUALIZADO, com o trecho selecionado ajustado conforme a instrução. Preserve todo o restante do documento intacto.',
                    ].join('\n');
                } else {
                    finalMessage = [
                        '[DOCUMENTO ATUAL NO CANVAS]:',
                        canvas.canvasContent,
                        '',
                        `[INSTRUÇÃO DO USUÁRIO]: ${effectiveMessage}`,
                        '',
                        'Retorne O DOCUMENTO COMPLETO ATUALIZADO conforme a instrução. Preserve o que não precisa mudar.',
                    ].join('\n');
                }
                canvas.setCanvasSelection(null);
            }

            if (userTyped && !hideUserBubble) {
                addMessage({ role: 'user', content: userTyped });
            }
            setIsLoading(true);

            try {
                if (templateFiles && templateFiles.length > 0 && (!ragStatus || ragStatus.indexed_chunks === 0)) {
                    try {
                        const indexResult = await uploadTemplates(templateFiles);
                        setRagStatus({
                            indexed_chunks: indexResult.indexed_chunks,
                            has_dossier: indexResult.has_dossier,
                        });
                    } catch (err) {
                        console.warn('Auto-indexing templates failed:', errorText(err));
                    }
                }

                let currentStyleDossier = styleDossier;
                if (templateFiles && templateFiles.length > 0 && !currentStyleDossier) {
                    try {
                        const styleResult = await generateStyleReport(templateFiles);
                        if (styleResult.cloning_prompt) {
                            currentStyleDossier = styleResult.cloning_prompt;
                            setStyleDossier(styleResult.cloning_prompt);
                        }
                    } catch (styleErr) {
                        console.warn('Auto style report failed:', errorText(styleErr));
                    }
                }

                let agentPrompt: string | null = overridePrompt ?? null;
                if (!agentPrompt && activeAgent?.prompt && !activeAgent.promptModule) {
                    agentPrompt = activeAgent.prompt;
                } else if (!agentPrompt && activeAgent?.promptModule) {
                    try {
                        const mod = await activeAgent.promptModule();
                        agentPrompt = mod.default || null;
                    } catch {
                        console.warn('Could not load agent prompt');
                    }
                }

                const result = await sendMessage({
                    message: finalMessage,
                    model: selectedModel.id,
                    llm: selectedModel.llm || null,
                    agentPrompt,
                    conversationId,
                    uploadedText,
                    styleDossier: currentStyleDossier,
                    useRag,
                    jurisprudenceContext: jurisContext || null,
                });

                setConversationId(result.conversation_id);

                const safeResponse = safeResponseText(result.response);

                const isLongResponse = safeResponse.length > 500;
                const isErrorOrSystem =
                    safeResponse.includes('⚠️ **Erro') ||
                    safeResponse.includes('MESA DE DELIBERAÇÃO') ||
                    safeResponse.includes('AGUARDANDO DIRETRIZES');

                if (canvas.canvasOpen && isLongResponse && !isErrorOrSystem) {
                    const isFirstCanvas = !canvas.canvasContent;
                    canvas.setCanvasContent(safeResponse);
                    addMessage({
                        role: 'assistant',
                        content: isFirstCanvas
                            ? '📄 **Minuta gerada no Canvas.** Veja o documento no painel ao lado. Peça alterações aqui no chat e o Canvas será atualizado automaticamente.'
                            : '✏️ **Canvas atualizado** com as alterações solicitadas.',
                        model: result.model,
                        isCanvasUpdate: true,
                    });
                } else {
                    const assistantMsg: Message = {
                        role: 'assistant',
                        content: safeResponse,
                        model: result.model,
                        v2Sections: result.v2_sections || null,
                        modelContext: result.model_context || null,
                    };
                    addMessage(assistantMsg);

                    if (batchPilotSession && batchPilotSession.phase === 'pilot' && safeResponse.length > 800) {
                        const remainingCount = batchPilotSession.processes.length;
                        if (remainingCount > 0) {
                            setBatchPilotSession((prev) =>
                                prev ? { ...prev, phase: 'confirming', pilotMinuta: safeResponse } : prev
                            );
                            addMessage({
                                role: 'batch-pilot-confirm',
                                content: `✅ **Minuta do caso piloto gerada!**\n\nDeseja aplicar estas instruções e esta minuta como gabarito para os demais **${remainingCount} processo(s)** do grupo "${batchPilotSession.clusterName}"?\n\nCada processo será analisado individualmente, e diferenças serão destacadas com alertas.`,
                                remainingCount,
                                clusterName: batchPilotSession.clusterName,
                            });
                        }
                    }
                }
            } catch (err) {
                addMessage({
                    role: 'assistant',
                    content: `⚠️ **Erro:** ${errorText(err)}`,
                    model: 'erro',
                });
            } finally {
                setIsLoading(false);
            }
        },
        [canvas, ragStatus, setRagStatus, addMessage, setIsLoading, setConversationId, setBatchPilotSession, setStyleDossier]
    );

    const handleRetry = useCallback(() => {
        const { messages } = useChatStore.getState();
        const lastUserMsg = [...messages].reverse().find((m) => m.role === 'user');
        if (!lastUserMsg?.content) return;
        handleSend(lastUserMsg.content, globalSelectedModel, [], 'none', [], false);
    }, [globalSelectedModel, handleSend]);

    const handleAutoReview = useCallback(async () => {
        const { messages, uploadedText } = useChatStore.getState();
        const assistantMsgs = messages.filter(
            (m) => m.role === 'assistant' && m.content && m.content.length > 100
        );
        const lastMinuta = assistantMsgs[assistantMsgs.length - 1];

        if (!lastMinuta?.content) {
            addMessage({
                role: 'assistant',
                content:
                    '🛑 **Nenhuma minuta encontrada no chat.** Primeiro, use o agente Gabinete para gerar uma minuta. Depois, ative o Revisor (QA) para auditá-la.',
                model: 'sistema',
            });
            return;
        }

        if (!uploadedText) {
            addMessage({
                role: 'assistant',
                content:
                    '🛑 **Nenhum processo carregado.** Faça o upload do processo (PDF/DOCX) para que o Revisor possa cruzar os fatos com a minuta.',
                model: 'sistema',
            });
            return;
        }

        let qaPrompt: string | null = null;
        try {
            const mod = await import('../prompts/auditorQA');
            qaPrompt = mod.default || null;
        } catch {
            console.warn('Could not load auditor QA prompt');
        }

        const auditMessage = [
            'Execute a auditoria de conformidade cruzando os textos abaixo.',
            '',
            '[MINUTA PROPOSTA]:',
            lastMinuta.content,
            '',
            'Execute a auditoria de conformidade cruzando a minuta acima com os dados do processo. Gere o Dashboard de Conformidade completo.',
        ].join('\n');

        setIsLoading(true);
        try {
            const result = await sendMessage({
                message: auditMessage,
                model: globalSelectedModel.id,
                llm: globalSelectedModel.llm || null,
                agentPrompt: qaPrompt,
                conversationId: null,
                uploadedText,
                styleDossier: null,
                useRag: false,
                jurisprudenceContext: null,
            });
            addMessage({
                role: 'assistant',
                content: safeResponseText(result.response),
                model: result.model,
            });
        } catch (err) {
            addMessage({
                role: 'assistant',
                content: `⚠️ **Erro na auditoria QA:** ${errorText(err)}`,
                model: 'erro',
            });
        } finally {
            setIsLoading(false);
        }
        // Mark addMessages unused to satisfy linter w/o changing signature.
        void addMessages;
    }, [globalSelectedModel, addMessage, addMessages, setIsLoading]);

    return { handleSend, handleRetry, handleAutoReview };
}
