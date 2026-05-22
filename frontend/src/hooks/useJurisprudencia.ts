import { useCallback, useState } from 'react';
import { triggerJurisprudenciaResearch, pollJurisprudenciaResearch } from '../services/api';
import { useChatStore } from '../store';
import { errorText } from './utils';
import type { JurisResearch, JurisResearchItem } from '../types/chat';

const POLL_INTERVAL = 2500;
const MAX_POLLS = 60;

export function useJurisprudencia() {
    const { addMessage, setJurisContext } = useChatStore();
    const [jurisResearch, setJurisResearch] = useState<JurisResearch | null>(null);
    const [jurisResearchLoading, setJurisResearchLoading] = useState(false);
    const [jurisResearchProgress, setJurisResearchProgress] = useState('');

    const pollResearch = useCallback(async (taskId: string) => {
        for (let i = 0; i < MAX_POLLS; i++) {
            await new Promise((r) => setTimeout(r, POLL_INTERVAL));
            const data = await pollJurisprudenciaResearch(taskId);
            if (!data) continue;
            if (data.progress) setJurisResearchProgress(data.progress);
            if (data.status === 'done') {
                setJurisResearch(data.result);
                setJurisResearchLoading(false);
                return;
            }
            if (data.status === 'error') {
                console.error('Jurisprudence research failed:', data.error);
                setJurisResearchLoading(false);
                return;
            }
        }
        setJurisResearchLoading(false);
    }, []);

    const handleJurisSearch = useCallback(async () => {
        const { uploadedText, messages } = useChatStore.getState();
        if (!uploadedText) return;
        setJurisResearchLoading(true);
        setJurisResearch(null);
        setJurisResearchProgress('Extraindo temas jurídicos do processo...');

        try {
            const lastAssistant = [...messages]
                .reverse()
                .find((m) => m.role === 'assistant' && m.content && m.content.length > 100);
            const analysisText = lastAssistant?.content || '';

            const { task_id } = await triggerJurisprudenciaResearch(uploadedText, analysisText);
            pollResearch(task_id);
        } catch (err) {
            console.error('Juris research trigger failed:', err);
            setJurisResearchLoading(false);
            addMessage({
                role: 'assistant',
                content: `⚠️ **Erro na pesquisa jurisprudencial:** ${errorText(err)}`,
                model: 'erro',
            });
        }
    }, [pollResearch, addMessage]);

    const handleJurisImport = useCallback(
        (data: JurisResearch) => {
            const parts = data.research.map((r: JurisResearchItem) => {
                const resultsText = (r.results || [])
                    .map((res) => {
                        const ementa = (res.ementa || '').trim();
                        const processo = res.numero_processo || '?';
                        const tipo = res.tipo_recurso || 'Acórdão';
                        const data_pub = res.data_publicacao || '?';
                        return `Nesse sentido, assim entende o Eg. TJMG:\n"EMENTA: ${ementa}" (TJMG, ${tipo}, nº ${processo}, ${data_pub})`;
                    })
                    .join('\n\n');
                return `**Tema:** ${r.theme}\n\n${resultsText}`;
            });
            const newContext = parts.join('\n\n---\n\n');
            setJurisContext((prev) => (prev ? `${prev}\n\n---\n\n${newContext}` : newContext));

            addMessage({
                role: 'assistant',
                content:
                    '✅ **Jurisprudência incluída na fundamentação.** A próxima minuta gerada incluirá a ementa com a citação "Assim entende o TJMG:".',
                model: 'pesquisador',
            });
        },
        [setJurisContext, addMessage]
    );

    const resetJurisprudencia = useCallback(() => {
        setJurisResearch(null);
        setJurisResearchLoading(false);
    }, []);

    return {
        jurisResearch,
        jurisResearchLoading,
        jurisResearchProgress,
        handleJurisSearch,
        handleJurisImport,
        resetJurisprudencia,
    };
}
