import { useCallback, useState } from 'react';
import { runDeepResearch, type DeepResearchEvent, type DeepResearchSection } from '../services/api';
import { useChatStore } from '../store';
import { errorText } from './utils';

export interface DeepResearchQA {
    question: string;
    answer?: string;
    status: 'pending' | 'running' | 'done' | 'error';
    error?: string;
}

export interface DeepResearchSectionState {
    id: string;
    title: string;
    content?: string;
    status: 'pending' | 'running' | 'done' | 'error';
    error?: string;
}

export interface DeepResearchState {
    active: boolean;
    phase: string;            // current phase message
    phaseId: string;          // 'chunking' | 'embedding' | 'planning' | 'researching' | 'synthesizing' | 'summarizing' | 'done'
    questions: DeepResearchQA[];
    sections: DeepResearchSectionState[];
    executiveSummary: string | null;
    dossier: string | null;
    error: string | null;
    chunksIndexed: number;
}

const INITIAL: DeepResearchState = {
    active: false,
    phase: '',
    phaseId: '',
    questions: [],
    sections: [],
    executiveSummary: null,
    dossier: null,
    error: null,
    chunksIndexed: 0,
};

/**
 * Runs a deep-research pipeline on the currently uploaded process text,
 * consuming SSE events from the backend and exposing progress so the UI
 * can render a live, section-by-section report-building experience.
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
                                case 'section_start':
                                    {
                                        // Ensure slot exists (first time we see this index)
                                        const existing = prev.sections.find((s) => s.id === event.section_id);
                                        if (!existing) {
                                            next.sections = [
                                                ...prev.sections,
                                                { id: event.section_id, title: event.title, status: 'running' },
                                            ];
                                        } else {
                                            next.sections = prev.sections.map((s) =>
                                                s.id === event.section_id ? { ...s, status: 'running' } : s
                                            );
                                        }
                                        next.phase = `Redigindo seção ${event.index + 1}/${event.total}: ${event.title}`;
                                    }
                                    break;
                                case 'section_done':
                                    next.sections = prev.sections.map((s) =>
                                        s.id === event.section_id
                                            ? { ...s, content: event.content, status: 'done' }
                                            : s
                                    );
                                    break;
                                case 'section_error':
                                    next.sections = prev.sections.map((s) =>
                                        s.id === event.section_id
                                            ? { ...s, status: 'error', error: event.message }
                                            : s
                                    );
                                    break;
                                case 'done':
                                    next.dossier = event.dossier;
                                    next.executiveSummary = event.executive_summary;
                                    next.chunksIndexed = event.chunks_indexed;
                                    // Replace section state with final, complete sections
                                    next.sections = event.sections.map((s: DeepResearchSection) => ({
                                        id: s.id,
                                        title: s.title,
                                        content: s.content,
                                        status: 'done' as const,
                                    }));
                                    next.phaseId = 'done';
                                    next.phase = 'Relatório completo pronto.';
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
