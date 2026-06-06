import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import {
    FaFlask, FaXmark, FaClockRotateLeft, FaPlay, FaArrowsRotate,
    FaCircleCheck, FaCircleXmark, FaSpinner, FaTrophy, FaPaperclip, FaFile,
} from 'react-icons/fa6';
import {
    runArenaCompare, voteArena, getCustomAgents, uploadFile,
    type ArenaVote, type ArenaVoteResult, type ArenaSlot,
} from '../services/api';
import { formatMarkdown } from '../utils/markdown';
import ArenaHistoryModal from './ArenaHistoryModal';
import './ArenaPanel.css';

// Modelos disponíveis na Arena — DEVE casar com ARENA_MODELS no backend.
const ARENA_MODEL_OPTIONS = [
    // Azure OpenAI (GPT)
    { name: 'GPT-5.5', color: '#4285F4', deployment: 'gpt-5.5' },
    { name: 'GPT-5.4 Pro', color: '#4285F4', deployment: 'gpt-5.4-pro' },
    { name: 'GPT-5.4 Mini', color: '#60A5FA', deployment: 'gpt-5.4-mini' },
    { name: 'GPT-5.3', color: '#4285F4', deployment: 'gpt-5.3-chat' },
    { name: 'GPT-5.2', color: '#4285F4', deployment: 'gpt-5.2' },
    { name: 'GPT-5.2 Chat', color: '#60A5FA', deployment: 'gpt-5.2-chat' },
    { name: 'GPT-4.1 Mini', color: '#93C5FD', deployment: 'gpt-4.1-mini' },
    // Azure AI Foundry (parceiros)
    { name: 'DeepSeek V4 Pro', color: '#0891B2', deployment: 'DeepSeek-V4-Pro' },
    { name: 'DeepSeek V4 Flash', color: '#22D3EE', deployment: 'DeepSeek-V4-Flash' },
    { name: 'DeepSeek V3.2 Speciale', color: '#0E7490', deployment: 'DeepSeek-V3.2-Speciale' },
    { name: 'Grok 4.3', color: '#1F2937', deployment: 'grok-4.3' },
    { name: 'Kimi K2.5', color: '#7C3AED', deployment: 'Kimi-K2.5' },
    { name: 'Kimi K2.6', color: '#8B5CF6', deployment: 'Kimi-K2.6' },
    // Nativos (chaves próprias)
    { name: 'Gemini 3.1 Pro', color: '#34A853', deployment: 'gemini-3.1-pro' },
    { name: 'Claude Sonnet 4.6', color: '#D97706', deployment: 'claude-sonnet-4-6' },
];

// Engines de OCR aceitos pelo backend (aplicam-se a PDF; DOCX/TXT extraem direto).
const OCR_OPTIONS = [
    { value: 'mistral_doc_ai', label: 'Mistral Document AI (recomendado)' },
    { value: 'marker', label: 'Marker (local)' },
    { value: 'tesseract', label: 'Tesseract (local)' },
    { value: 'none', label: 'Sem OCR — texto nativo do PDF' },
];

type SlotStatus = 'idle' | 'streaming' | 'done' | 'error';

interface AgentOpt { id: string; name: string; prompt: string; }

const VOTE_BUTTONS: Array<{ vote: ArenaVote; label: string }> = [
    { vote: 'A', label: '◀ Resposta A é melhor' },
    { vote: 'tie', label: 'Empate' },
    { vote: 'both_bad', label: 'Ambas ruins' },
    { vote: 'B', label: 'Resposta B é melhor ▶' },
];

function fmtLatency(ms: number): string {
    if (!ms) return '—';
    return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)}s`;
}
function fmtCost(usd: number): string {
    return usd ? `$${usd.toFixed(4)}` : '—';
}

export default function ArenaPanel({ onClose }: { onClose: () => void }) {
    const [modelA, setModelA] = useState('gpt-5.3-chat');
    const [modelB, setModelB] = useState('DeepSeek-V4-Pro');
    const [prompt, setPrompt] = useState('');
    const [docText, setDocText] = useState('');        // texto extraído do arquivo
    const [docFileName, setDocFileName] = useState('');
    const [uploading, setUploading] = useState(false);
    const [uploadMsg, setUploadMsg] = useState('');
    const [ocrEngine, setOcrEngine] = useState('mistral_doc_ai');
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [agents, setAgents] = useState<AgentOpt[]>([]);
    const [agentId, setAgentId] = useState('');

    // Chaves de API do usuário (Claude/Gemini) — persistidas no navegador
    const [anthropicKey, setAnthropicKey] = useState(() => localStorage.getItem('anthropic_api_key') || '');
    const [googleKey, setGoogleKey] = useState(() => localStorage.getItem('google_api_key') || '');
    const [showKeys, setShowKeys] = useState(false);
    const saveAnthropicKey = (v: string) => { setAnthropicKey(v); localStorage.setItem('anthropic_api_key', v); };
    const saveGoogleKey = (v: string) => { setGoogleKey(v); localStorage.setItem('google_api_key', v); };

    const [running, setRunning] = useState(false);
    const [error, setError] = useState('');
    const [comparisonId, setComparisonId] = useState('');
    const [responses, setResponses] = useState<{ A: string; B: string }>({ A: '', B: '' });
    const [status, setStatus] = useState<{ A: SlotStatus; B: SlotStatus }>({ A: 'idle', B: 'idle' });
    const [latency, setLatency] = useState<{ A: number; B: number }>({ A: 0, B: 0 });

    const [justification, setJustification] = useState('');
    const [reveal, setReveal] = useState<ArenaVoteResult | null>(null);
    const [voting, setVoting] = useState(false);

    const [showHistory, setShowHistory] = useState(false);

    // Buffers de streaming acumulados fora do React para não perder tokens entre renders
    const bufRef = useRef<{ A: string; B: string }>({ A: '', B: '' });

    useEffect(() => {
        getCustomAgents()
            .then((data: any) => {
                const regular = (data.agents || [])
                    .filter((a: any) => a.type === 'regular' && a.prompt)
                    .map((a: any) => ({ id: a.id, name: a.name, prompt: a.prompt }));
                setAgents(regular);
            })
            .catch(() => setAgents([]));
    }, []);

    const sameModel = modelA === modelB;
    const needsAnthropic = [modelA, modelB].some((m) => m.toLowerCase().startsWith('claude'));
    const needsGoogle = [modelA, modelB].some((m) => m.toLowerCase().startsWith('gemini'));
    const missingAnthropic = needsAnthropic && !anthropicKey.trim();
    const missingGoogle = needsGoogle && !googleKey.trim();
    const canRun = !running && !uploading && !sameModel && (prompt.trim() !== '' || docText.trim() !== '');
    const bothDone = status.A !== 'idle' && status.A !== 'streaming'
        && status.B !== 'idle' && status.B !== 'streaming';
    const selectedAgent = useMemo(() => agents.find((a) => a.id === agentId), [agents, agentId]);

    const handleFile = async (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (fileInputRef.current) fileInputRef.current.value = '';  // permite reanexar o mesmo arquivo
        if (!file) return;
        setUploading(true);
        setError('');
        setUploadMsg('📤 Enviando arquivo…');
        try {
            // vectorize=false: a Arena só precisa do texto extraído, não do índice RAG
            const res = await uploadFile(file, ocrEngine, true, false, (info) => setUploadMsg(info.progress));
            setDocText(res.text || '');
            setDocFileName(res.filename || file.name);
            setUploadMsg('');
        } catch (err: any) {
            setError(err?.message || 'Falha ao processar o arquivo.');
            setUploadMsg('');
        } finally {
            setUploading(false);
        }
    };

    const clearDoc = () => { setDocText(''); setDocFileName(''); };

    const resetResults = () => {
        bufRef.current = { A: '', B: '' };
        setResponses({ A: '', B: '' });
        setStatus({ A: 'idle', B: 'idle' });
        setLatency({ A: 0, B: 0 });
        setComparisonId('');
        setReveal(null);
        setJustification('');
        setError('');
    };

    const handleRun = async () => {
        if (!canRun) return;
        resetResults();
        setRunning(true);
        setStatus({ A: 'streaming', B: 'streaming' });
        try {
            await runArenaCompare(
                {
                    modelA,
                    modelB,
                    prompt,
                    uploadedText: docText.trim() || null,
                    agentPrompt: selectedAgent?.prompt || null,
                    agentId: selectedAgent?.id || null,
                    agentName: selectedAgent?.name || null,
                },
                {
                    onStart: (id) => setComparisonId(id),
                    onToken: (slot: ArenaSlot, text: string) => {
                        bufRef.current[slot] += text;
                        setResponses({ ...bufRef.current });
                    },
                    onSlotDone: (slot, ms) => {
                        setLatency((l) => ({ ...l, [slot]: ms }));
                        setStatus((s) => ({ ...s, [slot]: 'done' }));
                    },
                    onSlotError: (slot, err) => {
                        bufRef.current[slot] += `\n\n⚠️ Erro: ${err}`;
                        setResponses({ ...bufRef.current });
                        setStatus((s) => ({ ...s, [slot]: 'error' }));
                    },
                },
            );
        } catch (err: any) {
            setError(err?.message || 'Falha ao rodar a Arena.');
            setStatus({ A: 'error', B: 'error' });
        } finally {
            setRunning(false);
        }
    };

    const handleVote = async (vote: ArenaVote) => {
        if (!comparisonId || voting) return;
        setVoting(true);
        try {
            const result = await voteArena(comparisonId, vote, justification);
            setReveal(result);
        } catch (err: any) {
            setError(err?.message || 'Falha ao registrar voto.');
        } finally {
            setVoting(false);
        }
    };

    const renderColumn = (slot: ArenaSlot) => {
        const st = status[slot];
        const revealedName = reveal ? (slot === 'A' ? reveal.model_a : reveal.model_b) : null;
        const metrics = reveal ? reveal.metrics[slot] : null;
        const winner = reveal && (reveal.vote === slot);
        return (
            <div className={`arena-col ${winner ? 'winner' : ''}`}>
                <div className="arena-col-head">
                    <span className="arena-col-title">
                        {winner && <FaTrophy size={12} className="arena-trophy" />}
                        {revealedName ? revealedName : `Resposta ${slot}`}
                    </span>
                    <span className={`arena-col-status ${st}`}>
                        {st === 'streaming' && <><FaSpinner size={11} className="spin" /> gerando…</>}
                        {st === 'done' && <><FaCircleCheck size={11} /> {fmtLatency(latency[slot])}</>}
                        {st === 'error' && <><FaCircleXmark size={11} /> erro</>}
                    </span>
                </div>
                <div
                    className="arena-col-body markdown"
                    dangerouslySetInnerHTML={{ __html: formatMarkdown(responses[slot] || '') }}
                />
                {metrics && (
                    <div className="arena-col-metrics">
                        <span title="Latência">⏱ {fmtLatency(metrics.latency_ms)}</span>
                        <span title="Tokens entrada/saída">🔤 {metrics.input_tokens}/{metrics.output_tokens}</span>
                        <span title="Custo estimado (USD)">💰 {fmtCost(metrics.cost_usd)}</span>
                    </div>
                )}
            </div>
        );
    };

    const hasResults = status.A !== 'idle' || status.B !== 'idle';

    return (
        <div className="arena-panel">
            <div className="arena-header">
                <span className="arena-title">
                    <FaFlask size={15} /> Arena — comparação A/B cega
                </span>
                <div className="arena-header-actions">
                    <button className="arena-ghost-btn" onClick={() => setShowHistory(true)} title="Histórico">
                        <FaClockRotateLeft size={13} /> Histórico
                    </button>
                    <button className="arena-ghost-btn" onClick={onClose} title="Fechar">
                        <FaXmark size={15} />
                    </button>
                </div>
            </div>

            <div className="arena-config">
                <div className="arena-model-row">
                    <label className="arena-field">
                        <span>Modelo A</span>
                        <select value={modelA} onChange={(e) => setModelA(e.target.value)} disabled={running}>
                            {ARENA_MODEL_OPTIONS.map((m) => (
                                <option key={m.deployment} value={m.deployment}>{m.name}</option>
                            ))}
                        </select>
                    </label>
                    <span className="arena-vs">×</span>
                    <label className="arena-field">
                        <span>Modelo B</span>
                        <select value={modelB} onChange={(e) => setModelB(e.target.value)} disabled={running}>
                            {ARENA_MODEL_OPTIONS.map((m) => (
                                <option key={m.deployment} value={m.deployment}>{m.name}</option>
                            ))}
                        </select>
                    </label>
                    <label className="arena-field arena-field-grow">
                        <span>Agente (opcional)</span>
                        <select value={agentId} onChange={(e) => setAgentId(e.target.value)} disabled={running}>
                            <option value="">— nenhum —</option>
                            {agents.map((a) => (
                                <option key={a.id} value={a.id}>{a.name}</option>
                            ))}
                        </select>
                    </label>
                </div>

                {sameModel && <p className="arena-warn">Escolha dois modelos diferentes.</p>}

                {(needsAnthropic || needsGoogle) && (
                    <div className="arena-keys">
                        <button className="arena-doc-toggle" onClick={() => setShowKeys((v) => !v)} type="button">
                            {showKeys ? '▾' : '▸'} 🔑 Minhas chaves de API (Claude / Gemini)
                            {(missingAnthropic || missingGoogle) && <em className="arena-key-missing"> · faltando</em>}
                        </button>
                        {showKeys && (
                            <div className="arena-keys-fields">
                                {needsAnthropic && (
                                    <label className="arena-field arena-field-grow">
                                        <span>Anthropic API Key {missingAnthropic && <em className="arena-key-missing">(obrigatória)</em>}</span>
                                        <input
                                            type="password"
                                            autoComplete="off"
                                            placeholder="sk-ant-…"
                                            value={anthropicKey}
                                            onChange={(e) => saveAnthropicKey(e.target.value)}
                                            disabled={running}
                                        />
                                    </label>
                                )}
                                {needsGoogle && (
                                    <label className="arena-field arena-field-grow">
                                        <span>Google API Key {missingGoogle && <em className="arena-key-missing">(obrigatória)</em>}</span>
                                        <input
                                            type="password"
                                            autoComplete="off"
                                            placeholder="AIza…"
                                            value={googleKey}
                                            onChange={(e) => saveGoogleKey(e.target.value)}
                                            disabled={running}
                                        />
                                    </label>
                                )}
                                <p className="arena-keys-hint">
                                    As chaves ficam apenas no seu navegador e são enviadas só para chamar o modelo escolhido.
                                </p>
                            </div>
                        )}
                    </div>
                )}

                <textarea
                    className="arena-prompt"
                    placeholder="Digite o prompt que será enviado igualmente aos dois modelos…"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    disabled={running}
                    rows={3}
                />

                <div className="arena-doc-upload">
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf,.docx,.doc,.txt"
                        hidden
                        onChange={handleFile}
                        disabled={running || uploading}
                    />
                    {!docFileName && !uploading && (
                        <>
                            <button
                                className="arena-attach-btn"
                                type="button"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={running}
                            >
                                <FaPaperclip size={12} /> Anexar documento (PDF, DOCX, TXT)
                            </button>
                            <label className="arena-ocr-pick" title="Mecanismo de OCR aplicado a PDFs">
                                <span>OCR:</span>
                                <select value={ocrEngine} onChange={(e) => setOcrEngine(e.target.value)} disabled={running}>
                                    {OCR_OPTIONS.map((o) => (
                                        <option key={o.value} value={o.value}>{o.label}</option>
                                    ))}
                                </select>
                            </label>
                        </>
                    )}
                    {uploading && (
                        <span className="arena-upload-status">
                            <FaSpinner size={12} className="spin" /> {uploadMsg || 'Processando…'}
                        </span>
                    )}
                    {docFileName && !uploading && (
                        <div className="arena-doc-chip">
                            <FaFile size={12} />
                            <span className="arena-doc-chip-name">{docFileName}</span>
                            <em>{docText.length.toLocaleString('pt-BR')} caracteres</em>
                            <button type="button" onClick={clearDoc} title="Remover documento" disabled={running}>
                                <FaXmark size={12} />
                            </button>
                        </div>
                    )}
                </div>

                <div className="arena-run-row">
                    <button className="arena-run-btn" onClick={handleRun} disabled={!canRun}>
                        {running ? <><FaSpinner size={13} className="spin" /> Comparando…</>
                            : hasResults ? <><FaArrowsRotate size={13} /> Comparar de novo</>
                            : <><FaPlay size={13} /> Comparar</>}
                    </button>
                </div>

                {error && <p className="arena-error">{error}</p>}
            </div>

            {hasResults && (
                <div className="arena-results">
                    <div className="arena-cols">
                        {renderColumn('A')}
                        {renderColumn('B')}
                    </div>

                    {bothDone && !reveal && (
                        <div className="arena-vote">
                            <p className="arena-vote-q">
                                Qual resposta foi melhor? <em>(os modelos só serão revelados após o voto)</em>
                            </p>
                            <textarea
                                className="arena-justify"
                                placeholder="Justifique sua escolha (fica registrada no log como prova)…"
                                value={justification}
                                onChange={(e) => setJustification(e.target.value)}
                                rows={2}
                            />
                            <div className="arena-vote-btns">
                                {VOTE_BUTTONS.map((b) => (
                                    <button
                                        key={b.vote}
                                        className={`arena-vote-btn vote-${b.vote}`}
                                        onClick={() => handleVote(b.vote)}
                                        disabled={voting}
                                    >
                                        {b.label}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {reveal && (
                        <div className="arena-revealed">
                            <FaCircleCheck size={13} /> Voto registrado:{' '}
                            <strong>
                                {reveal.vote === 'A' ? `${reveal.model_a} (A)`
                                    : reveal.vote === 'B' ? `${reveal.model_b} (B)`
                                    : reveal.vote === 'tie' ? 'Empate'
                                    : 'Ambas ruins'}
                            </strong>
                            {reveal.justification && <span className="arena-revealed-just"> — “{reveal.justification}”</span>}
                        </div>
                    )}
                </div>
            )}

            <ArenaHistoryModal open={showHistory} onClose={() => setShowHistory(false)} />
        </div>
    );
}
