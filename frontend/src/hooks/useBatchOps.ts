import { useCallback, useState } from 'react';
import { uploadBatchXray, analyzeCluster, replicateBatchPilot } from '../services/api';
import { useChatStore, useUIStore } from '../store';
import { safeResponseText, errorText } from './utils';
import type { XrayReport, XrayCluster, BatchResult, Message } from '../types/chat';

export function useBatchOps() {
    const {
        addMessage,
        addMessages,
        setMessages,
        setUploadedText,
        setBatchPilotSession,
        setIsLoading,
    } = useChatStore();
    const { selectedModel: globalSelectedModel } = useUIStore();

    const [xrayReport, setXrayReport] = useState<XrayReport | null>(null);
    const [xrayLoading, setXrayLoading] = useState(false);
    const [xrayProgress, setXrayProgress] = useState('');
    const [xrayTextCache, setXrayTextCache] = useState<Record<string, string>>({});
    const [batchResults, setBatchResults] = useState<BatchResult[]>([]);
    const [batchSelectedIndex, setBatchSelectedIndex] = useState<number | null>(null);

    const handleXray = useCallback(
        async (files: File[]) => {
            if (files.length < 2) return;
            setXrayLoading(true);
            setXrayReport(null);
            setXrayProgress('Enviando processos...');

            try {
                const result = await uploadBatchXray(files, (progress) => setXrayProgress(progress));
                setXrayReport(result.report);
                setXrayTextCache(result.text_cache || {});
            } catch (err) {
                addMessage({
                    role: 'assistant',
                    content: `⚠️ **Erro no Raio-X:** ${errorText(err)}`,
                    model: 'erro',
                });
            } finally {
                setXrayLoading(false);
                setXrayProgress('');
            }
        },
        [addMessage]
    );

    const handleClusterAction = useCallback(
        async (cluster: XrayCluster) => {
            const filenames = cluster.arquivos || [];
            if (filenames.length === 0) return;

            const processes = filenames
                .filter((fname) => xrayTextCache[fname])
                .map((fname) => ({ filename: fname, text: xrayTextCache[fname] }));

            if (processes.length === 0) {
                addMessage({
                    role: 'assistant',
                    content: '⚠️ Textos dos processos não encontrados no cache. Tente refazer o Raio-X.',
                    model: 'erro',
                });
                return;
            }

            setXrayReport(null);
            setIsLoading(true);
            const { activeAgent } = useChatStore.getState();
            addMessage({
                role: 'assistant',
                content: `⚡ **Processando ${processes.length} processo(s) do grupo "${cluster.nome}" em paralelo...**\n\nAguarde, cada processo está sendo analisado individualmente.`,
                model: 'sistema',
            });

            try {
                const result = await analyzeCluster(
                    processes,
                    activeAgent?.prompt || '',
                    globalSelectedModel.id,
                    globalSelectedModel.llm
                );

                const resultMessages: Message[] = result.results.map((r: BatchResult) => {
                    const safeText = safeResponseText(r.response);
                    return {
                        role: 'assistant',
                        content:
                            r.status === 'ok'
                                ? `## 📄 ${r.filename}\n\n${safeText}`
                                : `## ⚠️ ${r.filename}\n\n${safeText}`,
                        model: r.model,
                    };
                });

                const summary: Message = {
                    role: 'assistant',
                    content: `✅ **Lote concluído:** ${result.ok_count}/${result.total} minutas geradas com sucesso.`,
                    model: 'sistema',
                };

                addMessages([summary, ...resultMessages]);
                setBatchResults(result.results);
                setBatchSelectedIndex(null);
            } catch (err) {
                addMessage({
                    role: 'assistant',
                    content: `⚠️ **Erro na análise em lote:** ${errorText(err)}`,
                    model: 'erro',
                });
            } finally {
                setIsLoading(false);
            }
        },
        [xrayTextCache, globalSelectedModel, addMessage, addMessages, setIsLoading]
    );

    const handleClusterPilotAction = useCallback(
        (cluster: XrayCluster) => {
            const filenames = cluster.arquivos || [];
            if (filenames.length === 0) return;

            const allProcesses = filenames
                .filter((fname) => xrayTextCache[fname])
                .map((fname) => ({ filename: fname, text: xrayTextCache[fname] }));

            if (allProcesses.length === 0) {
                addMessage({
                    role: 'assistant',
                    content: '⚠️ Textos dos processos não encontrados no cache. Tente refazer o Raio-X.',
                    model: 'erro',
                });
                return;
            }

            const [pilot, ...remaining] = allProcesses;
            setXrayReport(null);
            setUploadedText(pilot.text);

            setBatchPilotSession({
                clusterName: cluster.nome,
                processes: remaining,
                pilotFilename: pilot.filename,
                pilotMinuta: null,
                phase: 'pilot',
            });

            addMessage({
                role: 'assistant',
                content: `📋 **Caso Piloto ativado: ${pilot.filename}**\n\nO texto do processo foi carregado. Converse normalmente com o agente — faça perguntas, dê instruções e refine a minuta.\n\nQuando a minuta estiver pronta, o sistema perguntará se deseja replicar para os demais **${remaining.length} processo(s)** do grupo *"${cluster.nome}"*.`,
                model: 'sistema',
            });
        },
        [xrayTextCache, addMessage, setUploadedText, setBatchPilotSession]
    );

    const handlePilotReplicate = useCallback(async () => {
        const state = useChatStore.getState();
        const session = state.batchPilotSession;
        if (!session || !session.pilotMinuta) return;

        setBatchPilotSession((prev) => (prev ? { ...prev, phase: 'replicating' } : prev));
        setIsLoading(true);

        const pilotInstructions = state.messages
            .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content && m.content.length > 100))
            .map((m) => `[${m.role === 'user' ? 'MAGISTRADO' : 'AGENTE'}]: ${m.content}`)
            .join('\n\n');

        addMessage({
            role: 'assistant',
            content: `⚡ **Replicando para ${session.processes.length} processo(s)...**\n\nCada processo será analisado individualmente com as instruções e minuta-gabarito do caso piloto.`,
            model: 'sistema',
        });

        try {
            const result = await replicateBatchPilot({
                pilotInstructions,
                pilotMinuta: session.pilotMinuta,
                pilotSummary: null,
                processes: session.processes,
                model: globalSelectedModel.id,
                llm: globalSelectedModel.llm || null,
                agentPrompt: state.activeAgent?.prompt || null,
            });

            const resultMessages: Message[] = result.results.map((r: BatchResult) => {
                const safeText = safeResponseText(r.response);
                const alertsBadge =
                    (r.alerts_count ?? 0) > 0
                        ? ` (⚠️ ${r.alerts_count} alerta${(r.alerts_count ?? 0) > 1 ? 's' : ''})`
                        : '';
                return {
                    role: 'assistant',
                    content:
                        r.status === 'ok'
                            ? `## 📄 ${r.filename}${alertsBadge}\n\n${safeText}`
                            : `## ⚠️ ${r.filename}\n\n${safeText}`,
                    model: r.model,
                };
            });

            const alertsInfo =
                result.total_alerts > 0
                    ? ` — ⚠️ ${result.total_alerts} alerta(s) de diferença encontrado(s)`
                    : ' — Todos os processos são similares ao piloto';

            addMessages([
                {
                    role: 'assistant',
                    content: `✅ **Lote piloto concluído:** ${result.ok_count}/${result.total} minutas geradas${alertsInfo}.`,
                    model: 'sistema',
                },
                ...resultMessages,
            ]);

            setBatchResults(result.results);
            setBatchSelectedIndex(null);
            setBatchPilotSession((prev) => (prev ? { ...prev, phase: 'done' } : prev));
        } catch (err) {
            addMessage({
                role: 'assistant',
                content: `⚠️ **Erro na replicação piloto:** ${errorText(err)}`,
                model: 'erro',
            });
        } finally {
            setIsLoading(false);
        }
    }, [globalSelectedModel, addMessage, addMessages, setBatchPilotSession, setIsLoading]);

    const handlePilotDismiss = useCallback(() => {
        setBatchPilotSession((prev) => (prev ? { ...prev, phase: 'pilot' } : prev));
        setMessages((prev) => prev.filter((m) => m.role !== 'batch-pilot-confirm'));
    }, [setBatchPilotSession, setMessages]);

    const handleBatchSelect = useCallback(
        (index: number) => {
            setBatchSelectedIndex(index);
            const res = batchResults[index];
            if (!res) return;
            const safeText = safeResponseText(res.response);
            const content =
                res.status === 'ok'
                    ? `## 📄 ${res.filename}\n\n${safeText}`
                    : `## ⚠️ ${res.filename}\n\n${safeText}`;
            setMessages([{ role: 'assistant', content, model: res.model }]);
        },
        [batchResults, setMessages]
    );

    const handleBatchClose = useCallback(() => {
        setBatchResults([]);
        setBatchSelectedIndex(null);
    }, []);

    return {
        xrayReport,
        xrayLoading,
        xrayProgress,
        batchResults,
        batchSelectedIndex,
        setXrayReport,
        handleXray,
        handleClusterAction,
        handleClusterPilotAction,
        handlePilotReplicate,
        handlePilotDismiss,
        handleBatchSelect,
        handleBatchClose,
    };
}
