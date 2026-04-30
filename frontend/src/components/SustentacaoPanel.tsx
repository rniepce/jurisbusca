import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
    FaXmark, FaGavel, FaFileArrowUp, FaPaperPlane,
    FaCircleNotch, FaScaleBalanced, FaListUl, FaTriangleExclamation, FaFileLines
} from 'react-icons/fa6';
import { uploadFile, extractSustentacao, chatSustentacao, type SustentacaoData } from '../services/api';
import './SustentacaoPanel.css';

interface ChatMsg {
    role: 'user' | 'assistant';
    content: string;
}

interface Props {
    onClose: () => void;
}

const Field: React.FC<{ label: string; value: string | null }> = ({ label, value }) => (
    <div className="sust-field">
        <span className="sust-field-label">{label}</span>
        <span className="sust-field-value">{value || '—'}</span>
    </div>
);

const SustentacaoPanel: React.FC<Props> = ({ onClose }) => {
    const [phase, setPhase] = useState<'upload' | 'processing' | 'ready'>('upload');
    const [progress, setProgress] = useState('');
    const [error, setError] = useState('');
    const [processId, setProcessId] = useState<string | null>(null);
    const [data, setData] = useState<SustentacaoData | null>(null);
    const [notes, setNotes] = useState('');

    const [chatInput, setChatInput] = useState('');
    const [chatMessages, setChatMessages] = useState<ChatMsg[]>([]);
    const [chatLoading, setChatLoading] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const chatBottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatMessages, chatLoading]);

    const handleFile = useCallback(async (file: File) => {
        setError('');
        setPhase('processing');
        setProgress('Enviando PDF...');

        try {
            // 1. OCR via fluxo padrão de upload
            const upload = await uploadFile(file, 'mistral_doc_ai', true, false, (info) => {
                setProgress(info.progress || `Processando OCR... ${info.percent}%`);
            });
            const text = (upload as any)?.text || (upload as any)?.extracted_text || '';
            if (!text.trim()) {
                throw new Error('Não foi possível extrair texto do PDF.');
            }

            // 2. Extração estruturada
            setProgress('Extraindo dados do processo...');
            const result = await extractSustentacao(text);
            setProcessId(result.process_id);
            setData(result.data);
            setPhase('ready');
        } catch (e: any) {
            setError(e?.message || 'Erro desconhecido');
            setPhase('upload');
        }
    }, []);

    const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0];
        if (f) handleFile(f);
    }, [handleFile]);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        const f = e.dataTransfer.files?.[0];
        if (f && f.type === 'application/pdf') handleFile(f);
    }, [handleFile]);

    const handleSendChat = useCallback(async () => {
        const msg = chatInput.trim();
        if (!msg || !processId || chatLoading) return;
        setChatInput('');
        const newMessages: ChatMsg[] = [...chatMessages, { role: 'user', content: msg }];
        setChatMessages(newMessages);
        setChatLoading(true);
        try {
            const res = await chatSustentacao(processId, newMessages);
            setChatMessages([...newMessages, { role: 'assistant', content: res.reply }]);
        } catch (e: any) {
            setChatMessages([...newMessages, { role: 'assistant', content: `⚠️ ${e?.message || 'Erro'}` }]);
        } finally {
            setChatLoading(false);
        }
    }, [chatInput, processId, chatMessages, chatLoading]);

    const handleReset = useCallback(() => {
        setPhase('upload');
        setData(null);
        setProcessId(null);
        setChatMessages([]);
        setNotes('');
        setError('');
        setProgress('');
    }, []);

    return (
        <div className="sust-panel">
            <div className="sust-header">
                <div className="sust-header-content">
                    <div className="sust-title-row">
                        <FaGavel className="sust-header-icon" />
                        <div>
                            <h1 className="sust-title">Sustentação Oral</h1>
                            <p className="sust-subtitle">Apoio para audiências — extração estruturada + chat</p>
                        </div>
                    </div>
                    <div className="sust-header-actions">
                        {phase === 'ready' && (
                            <button className="sust-btn-secondary" onClick={handleReset}>
                                Novo processo
                            </button>
                        )}
                        <button className="sust-btn-icon" onClick={onClose} aria-label="Fechar">
                            <FaXmark />
                        </button>
                    </div>
                </div>
            </div>

            <div className="sust-body">
                {phase === 'upload' && (
                    <div
                        className="sust-upload-area"
                        onDrop={handleDrop}
                        onDragOver={(e) => e.preventDefault()}
                    >
                        <FaFileArrowUp className="sust-upload-icon" />
                        <h2>Faça upload do processo</h2>
                        <p>Arraste o PDF aqui ou clique para selecionar</p>
                        <button className="sust-btn-primary" onClick={() => fileInputRef.current?.click()}>
                            Selecionar PDF
                        </button>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="application/pdf"
                            onChange={handleFileInput}
                            style={{ display: 'none' }}
                        />
                        {error && <div className="sust-error">{error}</div>}
                    </div>
                )}

                {phase === 'processing' && (
                    <div className="sust-processing">
                        <FaCircleNotch className="sust-spinner" />
                        <h2>{progress || 'Processando...'}</h2>
                        <p>Isso pode levar alguns segundos para PDFs grandes ou digitalizados.</p>
                    </div>
                )}

                {phase === 'ready' && data && (
                    <div className="sust-dashboard">
                        <div className="sust-main">
                            {/* Cabeçalho do processo */}
                            <section className="sust-card">
                                <h3 className="sust-card-title">
                                    <FaFileLines /> Processo
                                </h3>
                                <div className="sust-fields-grid">
                                    <Field label="Número" value={data.numero_processo} />
                                    <Field label="Tipo recursal" value={data.tipo_recursal} />
                                    <Field label="Câmara" value={data.camara} />
                                    <Field label="Relator" value={data.relator} />
                                    <Field label="Data da sessão" value={data.data_sessao} />
                                    <Field label="Recorrente" value={data.recorrente} />
                                    <Field label="Recorrido" value={data.recorrido} />
                                    <Field label="Adv. sustentante" value={data.advogado_sustentante} />
                                    <Field label="Parte sustentante" value={data.parte_sustentante} />
                                </div>
                            </section>

                            {/* Teses */}
                            <section className="sust-card">
                                <h3 className="sust-card-title">
                                    <FaListUl /> Teses
                                </h3>
                                {data.teses && data.teses.length > 0 ? (
                                    <ul className="sust-list">
                                        {data.teses.map((t, i) => <li key={i}>{t}</li>)}
                                    </ul>
                                ) : (
                                    <p className="sust-empty">Nenhuma tese identificada.</p>
                                )}
                            </section>

                            {/* Preliminares */}
                            <section className="sust-card">
                                <h3 className="sust-card-title">
                                    <FaTriangleExclamation /> Preliminares
                                </h3>
                                {data.preliminares && data.preliminares.length > 0 ? (
                                    <ul className="sust-list">
                                        {data.preliminares.map((p, i) => <li key={i}>{p}</li>)}
                                    </ul>
                                ) : (
                                    <p className="sust-empty">Nenhuma preliminar identificada.</p>
                                )}
                            </section>

                            {/* Decisão de 1º grau */}
                            <section className="sust-card">
                                <h3 className="sust-card-title">
                                    <FaScaleBalanced /> Síntese da decisão de 1º grau
                                </h3>
                                <p className="sust-prose">
                                    {data.sintese_decisao_1grau || <span className="sust-empty">Não identificada.</span>}
                                </p>
                            </section>

                            {/* Notas */}
                            <section className="sust-card">
                                <h3 className="sust-card-title">📝 Anotações</h3>
                                <textarea
                                    className="sust-notes"
                                    value={notes}
                                    onChange={(e) => setNotes(e.target.value)}
                                    placeholder="Suas anotações para a sustentação oral..."
                                    rows={5}
                                />
                            </section>
                        </div>

                        {/* Chat lateral */}
                        <aside className="sust-chat">
                            <div className="sust-chat-header">
                                <FaGavel /> Pergunte sobre o processo
                            </div>
                            <div className="sust-chat-messages">
                                {chatMessages.length === 0 && !chatLoading && (
                                    <div className="sust-chat-empty">
                                        <p>Faça perguntas sobre o processo durante a sustentação.</p>
                                        <ul>
                                            <li>"Qual a tese principal do recorrente?"</li>
                                            <li>"Há precedente citado?"</li>
                                            <li>"Resumo dos pedidos."</li>
                                        </ul>
                                    </div>
                                )}
                                {chatMessages.map((m, i) => (
                                    <div key={i} className={`sust-chat-msg ${m.role}`}>
                                        {m.content}
                                    </div>
                                ))}
                                {chatLoading && (
                                    <div className="sust-chat-msg assistant loading">
                                        <FaCircleNotch className="sust-spinner-sm" /> Pensando...
                                    </div>
                                )}
                                <div ref={chatBottomRef} />
                            </div>
                            <div className="sust-chat-input">
                                <textarea
                                    value={chatInput}
                                    onChange={(e) => setChatInput(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                            e.preventDefault();
                                            handleSendChat();
                                        }
                                    }}
                                    placeholder="Pergunte algo sobre o processo..."
                                    rows={2}
                                    disabled={chatLoading}
                                />
                                <button
                                    className="sust-btn-send"
                                    onClick={handleSendChat}
                                    disabled={!chatInput.trim() || chatLoading}
                                >
                                    <FaPaperPlane />
                                </button>
                            </div>
                        </aside>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SustentacaoPanel;
