import { useCallback, useState } from 'react';
import { generateStyleReport } from '../services/api';
import { useChatStore } from '../store';
import { errorText } from './utils';

export function useStyleReport() {
    const { addMessage, setStyleDossier } = useChatStore();
    const [styleAnalyzing, setStyleAnalyzing] = useState(false);

    const handleStyleReport = useCallback(
        async (templateFiles: File[]) => {
            if (!templateFiles || templateFiles.length === 0) return;
            setStyleAnalyzing(true);
            try {
                const result = await generateStyleReport(templateFiles);
                const dossierContent =
                    result.full_response || result.dossier || 'Dossiê gerado sem conteúdo.';
                addMessage({
                    role: 'assistant',
                    content: `🎨 **Dossiê de Identidade Decisional** (${result.file_count} modelo${
                        result.file_count > 1 ? 's' : ''
                    } analisado${result.file_count > 1 ? 's' : ''})\n\n${dossierContent}`,
                    model: 'gemini-flash',
                });
                if (result.cloning_prompt) setStyleDossier(result.cloning_prompt);
            } catch (err) {
                addMessage({
                    role: 'assistant',
                    content: `⚠️ **Erro no Relatório de Estilo:** ${errorText(err)}`,
                    model: 'erro',
                });
            } finally {
                setStyleAnalyzing(false);
            }
        },
        [addMessage, setStyleDossier]
    );

    return { styleAnalyzing, handleStyleReport };
}
