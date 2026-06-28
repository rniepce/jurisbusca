import { useCallback } from 'react';
import { sendMessage, uploadTemplates, generateStyleReport, runFlow, resumeFlow } from '../services/api';
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

    /**
     * Executa um fluxo (run ou resume) capturando eventos SSE e renderizando
     * no chat — incluindo pausa em HIL (emite bubble especial com botões).
     */
    const executeFlowAndRender = useCallback(async (
        flowId: string,
        inputOrResume: string | { state: Record<string, unknown>; startFrom: string; userInput?: string },
    ) => {
        const nodeOutputs: Array<{ label: string; output: string; downloadUrl?: string }> = [];
        let finalOutput = '';
        let hilEvent: {
            label?: string;
            question?: string;
            content?: string;
            next_node_id?: string;
            state?: Record<string, unknown>;
        } | null = null;

        const onEvent = (event: Record<string, unknown>) => {
            const ev = event as { event: string; label?: string; output?: string; error?: string;
                download_url?: string; question?: string; content?: string;
                next_node_id?: string; state?: Record<string, unknown> };
            if (ev.event === 'node_done' && ev.label && ev.output) {
                nodeOutputs.push({ label: ev.label, output: ev.output, downloadUrl: ev.download_url });
            }
            if (ev.event === 'flow_done' && ev.output) {
                finalOutput = ev.output;
            }
            if (ev.event === 'human_required') {
                hilEvent = {
                    label: ev.label,
                    question: ev.question,
                    content: ev.content,
                    next_node_id: ev.next_node_id,
                    state: ev.state,
                };
            }
            if (ev.event === 'node_error' || ev.event === 'error') {
                throw new Error(ev.error || 'Erro na execução do fluxo');
            }
        };

        try {
            if (typeof inputOrResume === 'string') {
                await runFlow(flowId, inputOrResume, onEvent);
            } else {
                await resumeFlow(flowId, inputOrResume.state, inputOrResume.startFrom, onEvent, inputOrResume.userInput);
            }

            // primeiro: render as saídas dos nós já executados
            if (nodeOutputs.length > 0 || finalOutput) {
                const steps = nodeOutputs
                    .map(s => {
                        const link = s.downloadUrl ? `\n\n[⬇ Baixar arquivo](${s.downloadUrl})` : '';
                        return `### ⚙️ ${s.label}\n\n${s.output}${link}`;
                    })
                    .join('\n\n---\n\n');
                const body = finalOutput
                    ? `${steps}\n\n---\n\n### ✅ Resultado Final\n\n${finalOutput}`
                    : steps;
                addMessage({
                    role: 'assistant',
                    content: body || '(fluxo concluído sem saída)',
                    model: 'orquestrador',
                });
            }

            // se houve pausa HIL, render bubble especial com botões
            if (hilEvent) {
                addMessage({
                    role: 'flow-hil-pending',
                    content: hilEvent.content || '',
                    flowId,
                    hilLabel: hilEvent.label,
                    hilQuestion: hilEvent.question,
                    hilNextNodeId: hilEvent.next_node_id,
                    hilState: hilEvent.state,
                });
            }
        } catch (err) {
            addMessage({
                role: 'assistant',
                content: `⚠️ **Erro no fluxo:** ${errorText(err)}`,
                model: 'erro',
            });
        }
    }, [addMessage]);

    const handleSend = useCallback(
        async (
            message: string,
            selectedModel: SelectedModel,
            _files: File[] | undefined,
            _ocrEngine: string | undefined,
            templateFiles: File[] | undefined,
            useRag = false,
            { hideUserBubble = false, overridePrompt = null }: SendOptions = {},
            jurisEnabled = true,
            reasoningEnabled = false
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

            // ─── Orquestrador: roda fluxo em vez de chat normal ──────────
            const orchAgent = activeAgent as { type?: string; flow_id?: string } | null;
            if (orchAgent?.type === 'orquestrador' && orchAgent.flow_id) {
                const flowInput = [
                    userTyped && `[INSTRUÇÃO DO USUÁRIO]:\n${userTyped}`,
                    uploadedText && `[DOCUMENTO ANEXADO]:\n${uploadedText}`,
                ].filter(Boolean).join('\n\n') || 'Execute o fluxo.';

                await executeFlowAndRender(orchAgent.flow_id, flowInput);
                setIsLoading(false);
                return;
            }

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

                const agentKnowledge = (activeAgent as any)?.knowledge || null;
                const harnessOn = useUIStore.getState().harnessEconomico;

                const result = await sendMessage({
                    message: finalMessage,
                    // Harness ligado → roda o pipeline de análise completo (V3) com modelos baratos.
                    model: harnessOn ? 'v3' : selectedModel.id,
                    llm: selectedModel.llm || null,
                    agentPrompt,
                    conversationId,
                    uploadedText,
                    styleDossier: currentStyleDossier,
                    useRag,
                    jurisprudenceContext: jurisContext || null,
                    jurisEnabled,
                    reasoningEnabled,
                    agentKnowledge,
                    profile: harnessOn ? 'economico' : 'premium',
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
                profile: useUIStore.getState().harnessEconomico ? 'economico' : 'premium',
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

    const handleFlowResume = useCallback(async (
        flowId: string,
        startFrom: string,
        state: Record<string, unknown>,
        decision: 'approve' | 'reject',
        userInput?: string,
    ) => {
        if (decision === 'reject') {
            addMessage({
                role: 'assistant',
                content: '🛑 **Fluxo cancelado** — execução interrompida pelo usuário.',
                model: 'orquestrador',
            });
            return;
        }
        setIsLoading(true);
        try {
            await executeFlowAndRender(flowId, { state, startFrom, userInput });
        } finally {
            setIsLoading(false);
        }
    }, [addMessage, setIsLoading, executeFlowAndRender]);

    return { handleSend, handleRetry, handleAutoReview, handleFlowResume };
}
