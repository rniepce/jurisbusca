import { useCallback, useState } from 'react';
import { uploadFile } from '../services/api';
import { useChatStore } from '../store';
import { errorText } from './utils';
import type { OcrProgress, Message } from '../types/chat';
import type { UploadResult } from '../services/api';

export function useOcrUpload() {
    const { addMessages, setUploadedText, addMessage } = useChatStore();
    const [ocrProcessing, setOcrProcessing] = useState(false);
    const [ocrEngineName, setOcrEngineName] = useState('none');
    const [ocrProgress, setOcrProgress] = useState<OcrProgress>({ progress: '', percent: 0 });

    const handleFilesUploaded = useCallback(
        async (files: File[], ocrEngine: string, compress = true): Promise<UploadResult[]> => {
            if (files.length === 0) return [];
            setOcrProcessing(true);
            setOcrEngineName(ocrEngine);
            setOcrProgress({ progress: '📤 Enviando arquivo...', percent: 5 });

            try {
                const results: UploadResult[] = [];
                for (const f of files) {
                    const result = await uploadFile(f, ocrEngine, compress, true, (p) => setOcrProgress(p));
                    results.push(result);
                }

                const newText = results.map((r) => r.text).join('\n\n---\n\n');
                setUploadedText((prev) => (prev ? prev + '\n\n---\n\n' + newText : newText));

                const ocrMessages: Message[] = results.map((r) => ({
                    role: 'ocr',
                    filename: r.filename,
                    text: r.text,
                    engine: ocrEngine,
                    charCount: r.char_count,
                }));
                addMessages(ocrMessages);

                return results;
            } catch (err) {
                addMessage({
                    role: 'assistant',
                    content: `⚠️ **Erro no OCR:** ${errorText(err)}`,
                    model: 'erro',
                });
                return [];
            } finally {
                setOcrProcessing(false);
            }
        },
        [addMessage, addMessages, setUploadedText]
    );

    return {
        ocrProcessing,
        ocrEngineName,
        ocrProgress,
        handleFilesUploaded,
    };
}
