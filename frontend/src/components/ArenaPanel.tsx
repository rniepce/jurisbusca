import { useEffect, useMemo, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react';
import {
    FaFlask, FaXmark, FaClockRotateLeft, FaArrowsRotate,
    FaCircleCheck, FaCircleXmark, FaSpinner, FaTrophy, FaPaperclip, FaFile,
} from 'react-icons/fa6';
import { IoSend } from 'react-icons/io5';
import {
    runArenaCompare, continueArena, voteArena, getCustomAgents, uploadFile,
    type ArenaVote, type ArenaVoteResult, type ArenaSlot, type ArenaTurnHistory,
} from '../services/api';
import { formatMarkdown } from '../utils/markdown';
import ArenaHistoryModal from './ArenaHistoryModal';
import './ArenaPanel.css';

// Modelos disponíveis na Arena — DEVE casar com ARENA_MODELS no backend.
const ARENA_MODEL_OPTIONS = [
    // Azure OpenAI (GPT)
    { name: 'GPT-5.5', deployment: 'gpt-5.5' },
    { name: 'GPT-5.4 Pro', deployment: 'gpt-5.4-pro' },
    { name: 'GPT-5.4 Mini', deployment: 'gpt-5.4-mini' },
    { name: 'GPT-5.3', deployment: 'gpt-5.3-chat' },
    { name: 'GPT-5.2', deployment: 'gpt-5.2' },
    { name: 'GPT-5.2 Chat', deployment: 'gpt-5.2-chat' },
    { name: 'GPT-4.1 Mini', deployment: 'gpt-4.1-mini' },
    // Azure AI Foundry (parceiros)
    { name: 'DeepSeek V4 Pro', deployment: 'DeepSeek-V4-Pro' },
    { name: 'DeepSeek V4 Flash', deployment: 'DeepSeek-V4-Flash' },
    { name: 'DeepSeek V3.2 Speciale', deployment: 'DeepSeek-V3.2-Speciale' },
    { name: 'Grok 4.3', deployment: 'grok-4.3' },
    { name: 'Kimi K2.5', deployment: 'Kimi-K2.5' },
    { name: 'Kimi K2.6', deployment: 'Kimi-K2.6' },
    // Nativos (chaves próprias)
    { name: 'Gemini 3.1 Pro', deployment: 'gemini-3.1-pro' },
    { name: 'Claude Sonnet 4.6', deployment: 'claude-sonnet-4-6' },
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

interface Turn {
    user: string;
    a: string; b: string;
    aStatus: SlotStatus; bStatus: SlotStatus;
    aLatency: number; bLatency: number;
}

const VOTE_BUTTONS: Array<{ vote: ArenaVote; label: string }> = [
    { vote: 'A', label: '◀ Resposta A é melhor' },
    { vote: 'tie', label: 'Empate' },
    { vote: 'both_bad', label: 'Ambas ruins' },
    { vote: 'B', label: 'Resposta B é melhor ▶' },
];

const labelOf = (dep: string) => ARENA_MODEL_OPTIONS.find((o) => o.deployment === dep)?.name || dep;

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
    const [agents, setAgents] = useState<AgentOpt[]>([]);
    const [agentId, setAgentId] = useState('');

    // Chaves de API do usuário (Claude/Gemini) — persistidas no navegador
    const [anthropicKey, setAnthropicKey] = useState(() => localStorage.getItem('anthropic_api_key') || '');
    const [googleKey, setGoogleKey] = useState(() => localStorage.getItem('google_api_key') || '');
    const [showKeys, setShowKeys] = useState(false);
    const saveAnthropicKey = (v: string) => { setAnthropicKey(v); localStorage.setItem('anthropic_api_key', v); };
    const saveGoogleKey = (v: string) => { setGoogleKey(v); localStorage.setItem('google_api_key', v); };

    // Documento (upload + OCR)
    const [docText, setDocText] = useState('');
    const [docFileName, setDocFileName] = useState('');
    const [uploading, setUploading] = useState(false);
    const [uploadMsg, setUploadMsg] = useState('');
    const [ocrEngine, setOcrEngine] = useState('mistral_doc_ai');
    const fileInputRef = useRef<HTMLInputElement | null>(null);

    // Conversa multi-turno
    const [comparisonId, setComparisonId] = useState('');
    const [turns, setTurns] = useState<Turn[]>([]);
    const [input, setInput] = useState('');
    const [streaming, setStreaming] = useState(false);
    const [error, setError] = useState('');

    // Voto / reveal
    const [justification, setJustification] = useState('');
    const [reveal, setReveal] = useState<ArenaVoteResult | null>(null);
    const [voting, setVoting] = useState(false);

    const [showHistory, setShowHistory] = useState(false);

    const bufRef = useRef<{ a: string; b: string }>({ a: '', b: '' });
    const convoEndRef = useRef<HTMLDivElement | null>(null);

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

    useEffect(() => { convoEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [turns]);

    const started = turns.length > 0 || comparisonId !== '';
    const sameModel = modelA === modelB;
    const needsAnthropic = [modelA, modelB].some((m) => m.toLowerCase().startsWith('claude'));
    const needsGoogle = [modelA, modelB].some((m) => m.toLowerCase().startsWith('gemini'));
    const missingAnthropic = needsAnthropic && !anthropicKey.trim();
    const missingGoogle = needsGoogle && !googleKey.trim();
    const selectedAgent = useMemo(() => agents.find((a) => a.id === agentId), [agents, agentId]);

    const lastTurn = turns[turns.length - 1];
    const lastTurnDone = !!lastTurn && lastTurn.aStatus !== 'streaming' && lastTurn.bStatus !== 'streaming';
    const canSend = !streaming && !uploading && input.trim() !== '' && (started || !sameModel);

    // ── Upload ──
    const handleFile = async (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (fileInputRef.current) fileInputRef.current.value = '';
        if (!file) return;
        setUploading(true);
        setError('');
        setUploadMsg('📤 Enviando arquivo…');
        try {
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

    // ── Enviar mensagem (1º turno = compare; demais = continue) ──
    const handleSend = async () => {
        const msg = input.trim();
        if (!msg || streaming || uploading) return;
        if (!started && sameModel) { setError('Escolha dois modelos diferentes.'); return; }
        setError('');

        const priorTurns = turns;
        const idx = priorTurns.length;
        const newTurn: Turn = {
            user: msg, a: '', b: '',
            aStatus: 'streaming', bStatus: 'streaming', aLatency: 0, bLatency: 0,
        };
        setTurns([...priorTurns, newTurn]);
        setInput('');
        setStreaming(true);
        bufRef.current = { a: '', b: '' };

        const patch = (i: number, fn: (t: Turn) => Turn) =>
            setTurns((ts) => ts.map((t, j) => (j === i ? fn(t) : t)));

        const handlers = {
            onStart: (id: string) => setComparisonId((cur) => cur || id),
            onToken: (slot: ArenaSlot, text: string) => {
                if (slot === 'A') { bufRef.current.a += text; patch(idx, (t) => ({ ...t, a: bufRef.current.a })); }
                else { bufRef.current.b += text; patch(idx, (t) => ({ ...t, b: bufRef.current.b })); }
            },
            onSlotDone: (slot: ArenaSlot, ms: number) => {
                if (slot === 'A') patch(idx, (t) => ({ ...t, aStatus: 'done', aLatency: ms }));
                else patch(idx, (t) => ({ ...t, bStatus: 'done', bLatency: ms }));
            },
            onSlotError: (slot: ArenaSlot, err: string) => {
                if (slot === 'A') { bufRef.current.a += `\n\n⚠️ Erro: ${err}`; patch(idx, (t) => ({ ...t, a: bufRef.current.a, aStatus: 'error' })); }
                else { bufRef.current.b += `\n\n⚠️ Erro: ${err}`; patch(idx, (t) => ({ ...t, b: bufRef.current.b, bStatus: 'error' })); }
            },
        };

        try {
            if (!comparisonId) {
                await runArenaCompare({
                    modelA, modelB, prompt: msg,
                    uploadedText: docText.trim() || null,
                    agentPrompt: selectedAgent?.prompt || null,
                    agentId: selectedAgent?.id || null,
                    agentName: selectedAgent?.name || null,
                }, handlers);
            } else {
                const history: ArenaTurnHistory[] = priorTurns.map((t) => ({ user: t.user, a: t.a, b: t.b }));
                await continueArena(comparisonId, msg, history, handlers);
            }
        } catch (err: any) {
            setError(err?.message || 'Falha ao enviar.');
            patch(idx, (t) => ({
                ...t,
                aStatus: t.aStatus === 'streaming' ? 'error' : t.aStatus,
                bStatus: t.bStatus === 'streaming' ? 'error' : t.bStatus,
            }));
        } finally {
            setStreaming(false);
        }
    };

    const handleVote = async (vote: ArenaVote) => {
        if (!comparisonId || voting) return;
        setVoting(true);
        try {
            setReveal(await voteArena(comparisonId, vote, justification));
        } catch (err: any) {
            setError(err?.message || 'Falha ao registrar voto.');
        } finally {
            setVoting(false);
        }
    };

    const newComparison = () => {
        setTurns([]); setComparisonId(''); setReveal(null);
        setJustification(''); setError(''); setInput('');
        bufRef.current = { a: '', b: '' };
    };

    const onInputKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void handleSend(); }
    };

    // ── Render de uma célula (resposta de um slot num turno) ──
    const slotCell = (turn: Turn, slot: ArenaSlot) => {
        const text = slot === 'A' ? turn.a : turn.b;
        const st = slot === 'A' ? turn.aStatus : turn.bStatus;
        const lat = slot === 'A' ? turn.aLatency : turn.bLatency;
        return (
            <div className={`arena-col ${reveal && reveal.vote === slot ? 'winner' : ''}`}>
                <div
                    className="arena-col-body markdown"
                    dangerouslySetInnerHTML={{ __html: formatMarkdown(text || '') }}
                />
                <div className="arena-col-foot">
                    {st === 'streaming' && <span className="arena-col-status streaming"><FaSpinner size={10} className="spin" /> gerando…</span>}
                    {st === 'done' && <span className="arena-col-status done"><FaCircleCheck size={10} /> {fmtLatency(lat)}</span>}
                    {st === 'error' && <span className="arena-col-status error"><FaCircleXmark size={10} /> erro</span>}
                </div>
            </div>
        );
    };

    const headCell = (slot: ArenaSlot) => {
        const name = reveal ? (slot === 'A' ? reveal.model_a : reveal.model_b) : `Resposta ${slot}`;
        const m = reveal ? reveal.metrics[slot] : null;
        const winner = reveal && reveal.vote === slot;
        return (
            <div className={`arena-head-cell ${winner ? 'winner' : ''}`}>
                <span className="arena-head-title">
                    {winner && <FaTrophy size={11} className="arena-trophy" />}
                    {name}
                </span>
                {m && (
                    <span className="arena-head-metrics">
                        ⏱ {fmtLatency(m.latency_ms)} · 🔤 {m.input_tokens}/{m.output_tokens} · 💰 {fmtCost(m.cost_usd)}
                    </span>
                )}
            </div>
        );
    };

    return (
        <div className="arena-panel">
            <div className="arena-header">
                <span className="arena-title"><FaFlask size={15} /> Arena — chat A/B cego</span>
                <div className="arena-header-actions">
                    <button className="arena-ghost-btn" onClick={() => setShowHistory(true)} title="Histórico">
                        <FaClockRotateLeft size={13} /> Histórico
                    </button>
                    <button className="arena-ghost-btn" onClick={onClose} title="Fechar"><FaXmark size={15} /></button>
                </div>
            </div>

            {/* Setup (antes do 1º turno) ou barra travada (durante a conversa) */}
            {!started ? (
                <div className="arena-config">
                    <div className="arena-model-row">
                        <label className="arena-field">
                            <span>Modelo A</span>
                            <select value={modelA} onChange={(e) => setModelA(e.target.value)}>
                                {ARENA_MODEL_OPTIONS.map((m) => <option key={m.deployment} value={m.deployment}>{m.name}</option>)}
                            </select>
                        </label>
                        <span className="arena-vs">×</span>
                        <label className="arena-field">
                            <span>Modelo B</span>
                            <select value={modelB} onChange={(e) => setModelB(e.target.value)}>
                                {ARENA_MODEL_OPTIONS.map((m) => <option key={m.deployment} value={m.deployment}>{m.name}</option>)}
                            </select>
                        </label>
                        <label className="arena-field arena-field-grow">
                            <span>Agente (opcional)</span>
                            <select value={agentId} onChange={(e) => setAgentId(e.target.value)}>
                                <option value="">— nenhum —</option>
                                {agents.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
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
                                            <input type="password" autoComplete="off" placeholder="sk-ant-…" value={anthropicKey} onChange={(e) => saveAnthropicKey(e.target.value)} />
                                        </label>
                                    )}
                                    {needsGoogle && (
                                        <label className="arena-field arena-field-grow">
                                            <span>Google API Key {missingGoogle && <em className="arena-key-missing">(obrigatória)</em>}</span>
                                            <input type="password" autoComplete="off" placeholder="AIza…" value={googleKey} onChange={(e) => saveGoogleKey(e.target.value)} />
                                        </label>
                                    )}
                                    <p className="arena-keys-hint">As chaves ficam apenas no seu navegador e são enviadas só para chamar o modelo escolhido.</p>
                                </div>
                            )}
                        </div>
                    )}

                    <div className="arena-doc-upload">
                        <input ref={fileInputRef} type="file" accept=".pdf,.docx,.doc,.txt" hidden onChange={handleFile} disabled={uploading} />
                        {!docFileName && !uploading && (
                            <>
                                <button className="arena-attach-btn" type="button" onClick={() => fileInputRef.current?.click()}>
                                    <FaPaperclip size={12} /> Anexar documento (PDF, DOCX, TXT)
                                </button>
                                <label className="arena-ocr-pick" title="Mecanismo de OCR aplicado a PDFs">
                                    <span>OCR:</span>
                                    <select value={ocrEngine} onChange={(e) => setOcrEngine(e.target.value)}>
                                        {OCR_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                                    </select>
                                </label>
                            </>
                        )}
                        {uploading && <span className="arena-upload-status"><FaSpinner size={12} className="spin" /> {uploadMsg || 'Processando…'}</span>}
                        {docFileName && !uploading && (
                            <div className="arena-doc-chip">
                                <FaFile size={12} />
                                <span className="arena-doc-chip-name">{docFileName}</span>
                                <em>{docText.length.toLocaleString('pt-BR')} caracteres</em>
                                <button type="button" onClick={clearDoc} title="Remover documento"><FaXmark size={12} /></button>
                            </div>
                        )}
                    </div>
                </div>
            ) : (
                <div className="arena-locked-bar">
                    <span className="arena-locked-models">{labelOf(modelA)} <em>×</em> {labelOf(modelB)}</span>
                    {selectedAgent && <span className="arena-locked-tag">🤖 {selectedAgent.name}</span>}
                    {docFileName && <span className="arena-locked-tag">📄 {docFileName}</span>}
                    <button className="arena-ghost-btn" onClick={newComparison} title="Começar nova comparação">
                        <FaArrowsRotate size={12} /> Nova comparação
                    </button>
                </div>
            )}

            {/* Conversa */}
            {started && (
                <div className="arena-convo">
                    <div className="arena-convo-head">
                        {headCell('A')}
                        {headCell('B')}
                    </div>
                    {turns.map((t, i) => (
                        <div key={i} className="arena-turn">
                            <div className="arena-turn-user"><strong>Você:</strong> {t.user}</div>
                            <div className="arena-turn-cols">
                                {slotCell(t, 'A')}
                                {slotCell(t, 'B')}
                            </div>
                        </div>
                    ))}

                    {lastTurnDone && !reveal && (
                        <div className="arena-vote">
                            <p className="arena-vote-q">Qual coluna está melhor? <em>(os modelos só serão revelados após o voto)</em></p>
                            <textarea
                                className="arena-justify"
                                placeholder="Justifique sua escolha (fica registrada no log como prova)…"
                                value={justification}
                                onChange={(e) => setJustification(e.target.value)}
                                rows={2}
                            />
                            <div className="arena-vote-btns">
                                {VOTE_BUTTONS.map((b) => (
                                    <button key={b.vote} className={`arena-vote-btn vote-${b.vote}`} onClick={() => handleVote(b.vote)} disabled={voting}>
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
                                    : reveal.vote === 'tie' ? 'Empate' : 'Ambas ruins'}
                            </strong>
                            {reveal.justification && <span className="arena-revealed-just"> — “{reveal.justification}”</span>}
                            <span className="arena-revealed-hint"> · pode continuar conversando abaixo</span>
                        </div>
                    )}
                    <div ref={convoEndRef} />
                </div>
            )}

            {error && <p className="arena-error">{error}</p>}

            {/* Barra de input (sempre) */}
            <div className="arena-input-bar">
                <textarea
                    className="arena-input"
                    placeholder={started ? 'Continue a conversa com ambos os modelos…' : 'Digite o prompt inicial para os dois modelos…'}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={onInputKey}
                    disabled={streaming}
                    rows={2}
                />
                <button className="arena-send-btn" onClick={handleSend} disabled={!canSend} title="Enviar (Enter)">
                    {streaming ? <FaSpinner size={15} className="spin" /> : <IoSend size={15} />}
                </button>
            </div>

            <ArenaHistoryModal open={showHistory} onClose={() => setShowHistory(false)} />
        </div>
    );
}
