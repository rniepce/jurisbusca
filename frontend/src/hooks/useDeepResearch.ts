import { useCallback, useState } from 'react';
import { runDeepResearch, type DeepResearchEvent } from '../services/api';
import { useChatStore } from '../store';
import { errorText } from './utils';

export interface DeepResearchQA {
    question: string;
    answer?: string;
    status: 'pending' | 'running' | 'done' | 'error';
    error?: string;
}

export interface DeepResearchState {
    active: boolean;
    phase: string;            // current phase message
    phaseId: string;          // 'chunking' | 'embedding' | 'planning' | 'researching' | 'synthesizing' | 'done'
    questions: DeepResearchQA[];
    dossier: string | null;
    error: string | null;
    chunksIndexed: number;
}

const INITIAL: DeepResearchState = {
    active: false,
    phase: '',
    phaseId: '',
    questions: [],
    dossier: null,
    error: null,
    chunksIndexed: 0,
};

/**
 * Runs a deep-research pipeline on the currently uploaded process text,
 * consuming SSE events from the backend and exposing progress so the UI
 * can render a live dossier-building experience.
 */
export function useDeepResearch() {
    const uploadedText = useChatStore((s) => s.uploadedText);
    const [state, setState] = useState<DeepResearchState>(INITIAL);

    const reset = useCallback(() => setState(INITIAL), []);

    const start = useCallback(
        async (model: string | null = null) => {
            if (!uploadedText || uploadedText.length < 500) {
                setState({
                    ...INITIAL,
                    error: 'Anexe um processo (mínimo ~500 caracteres) antes de rodar deep research.',
                });
                return;
            }

            setState({ ...INITIAL, active: true, phase: 'Iniciando...', phaseId: 'starting' });

            try {
                await runDeepResearch(
                    uploadedText,
                    (event: DeepResearchEvent) => {
                        setState((prev) => {
                            const next = { ...prev };
                            switch (event.event) {
                                case 'phase':
                                    next.phase = event.message;
                                    next.phaseId = event.phase;
                                    break;
                                case 'plan':
                                    next.questions = event.questions.map((q) => ({
                                        question: q,
                                        status: 'pending',
                                    }));
                                    next.phaseId = 'researching';
                                    next.phase = `Investigando ${event.questions.length} pontos do processo...`;
                                    break;
                                case 'question_start':
                                    next.questions = prev.questions.map((q, i) =>
                                        i === event.index ? { ...q, status: 'running' } : q
                                    );
                                    next.phase = `(${event.index + 1}/${event.total}) ${event.question}`;
                                    break;
                                case 'question_done':
                                    next.questions = prev.questions.map((q, i) =>
                                        i === event.index ? { ...q, answer: event.answer, status: 'done' } : q
                                    );
                                    break;
                                case 'question_error':
                                    next.questions = prev.questions.map((q, i) =>
                                        i === event.index ? { ...q, status: 'error', error: event.message } : q
                                    );
                                    break;
                                case 'done':
                                    next.dossier = event.dossier;
                                    next.chunksIndexed = event.chunks_indexed;
                                    next.phaseId = 'done';
                                    next.phase = 'Dossiê pronto.';
                                    next.active = false;
                                    break;
                                case 'error':
                                    next.error = event.message;
                                    next.active = false;
                                    break;
                            }
                            return next;
                        });
                    },
                    model,
                );
            } catch (err) {
                setState((prev) => ({
                    ...prev,
                    error: errorText(err),
                    active: false,
                }));
            }
        },
        [uploadedText]
    );

    return { state, start, reset, hasUploadedText: !!uploadedText };
}
