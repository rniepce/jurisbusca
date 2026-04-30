import React, { useRef, useState } from 'react';
import { FaFileArrowUp, FaCircleNotch, FaCircleCheck, FaCircleXmark, FaCircleHalfStroke } from 'react-icons/fa6';
import { uploadFile, analisarVoto, analisarSentenca, type AnaliseVotoResult, type AnaliseSentencaResult } from '../../../services/api';

type Mode = 'voto' | 'sentenca';

interface Props {
    processId: string;
    mode: Mode;
}

const labels: Record<Mode, { title: string; placeholder: string; cta: string }> = {
    voto: {
        title: 'Análise do voto do relator',
        cta: 'Carregar voto',
        placeholder: 'PDF do voto do relator',
    },
    sentenca: {
        title: 'Análise da minuta de sentença',
        cta: 'Carregar sentença',
        placeholder: 'PDF da minuta de sentença',
    },
};

const verdictBadge = (resultado: string) => {
    const r = resultado.toLowerCase();
    if (r === 'favoravel' || r === 'procedente') return { icon: <FaCircleCheck />, cls: 'favoravel', text: r === 'procedente' ? 'PROCEDENTE' : 'FAVORÁVEL' };
    if (r === 'desfavoravel' || r === 'improcedente') return { icon: <FaCircleXmark />, cls: 'desfavoravel', text: r === 'improcedente' ? 'IMPROCEDENTE' : 'DESFAVORÁVEL' };
    return { icon: <FaCircleHalfStroke />, cls: 'parcial', text: 'PARCIAL' };
};

const DocumentAnalysis: React.FC<Props> = ({ processId, mode }) => {
    const [phase, setPhase] = useState<'idle' | 'processing' | 'ready' | 'error'>('idle');
    const [progress, setProgress] = useState('');
    const [error, setError] = useState('');
    const [result, setResult] = useState<AnaliseVotoResult | AnaliseSentencaResult | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const cfg = labels[mode];

    const handleFile = async (file: File) => {
        setError('');
        setPhase('processing');
        setProgress('Enviando documento...');
        try {
            const upload = await uploadFile(file, 'mistral_doc_ai', true, false, (info) => {
                setProgress(info.progress || `OCR... ${info.percent}%`);
            });
            if (!upload.text?.trim()) throw new Error('Não foi possível extrair texto.');
            const text = upload.text;

            setProgress('Analisando...');
            const analysis = mode === 'voto'
                ? await analisarVoto(processId, text)
                : await analisarSentenca(processId, text);
            setResult(analysis);
            setPhase('ready');
        } catch (e: any) {
            setError(e?.message || 'Erro');
            setPhase('error');
        }
    };

    const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0];
        if (f) handleFile(f);
    };

    const reset = () => {
        setPhase('idle');
        setResult(null);
        setError('');
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    if (phase === 'idle' || phase === 'error') {
        return (
            <section className="sust-card">
                <h3 className="sust-card-title"><FaFileArrowUp /> {cfg.title}</h3>
                <button className="sust-btn-primary" onClick={() => fileInputRef.current?.click()}>
                    {cfg.cta}
                </button>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="application/pdf"
                    onChange={onFileInput}
                    style={{ display: 'none' }}
                />
                {error && <div className="sust-error">{error}</div>}
            </section>
        );
    }

    if (phase === 'processing') {
        return (
            <section className="sust-card">
                <h3 className="sust-card-title"><FaFileArrowUp /> {cfg.title}</h3>
                <div className="sust-inline-processing">
                    <FaCircleNotch className="sust-spinner-sm" /> {progress}
                </div>
            </section>
        );
    }

    if (phase === 'ready' && result) {
        const badge = verdictBadge(result.resultado);
        const isVoto = 'por_tese' in result;

        return (
            <section className="sust-card">
                <h3 className="sust-card-title"><FaFileArrowUp /> {cfg.title}</h3>

                <div
                    className={`sust-verdict ${badge.cls}`}
                    role="status"
                    aria-label={`Resultado: ${badge.text}. ${result.resumo}`}
                >
                    <span className="sust-verdict-icon" aria-hidden="true">{badge.icon}</span>
                    <div className="sust-verdict-body">
                        <strong>{badge.text}</strong>
                        <span>{result.resumo}</span>
                    </div>
                </div>

                {isVoto ? (
                    <div className="sust-analysis-list">
                        {(result as AnaliseVotoResult).por_tese.map((t, i) => {
                            const b = verdictBadge(t.posicao);
                            return (
                                <div key={i} className={`sust-analysis-item ${b.cls}`}>
                                    <div className="sust-analysis-item-head">
                                        {b.icon} <strong>{t.tese}</strong>
                                    </div>
                                    <p>{t.justificativa}</p>
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <div className="sust-analysis-list">
                        {(result as AnaliseSentencaResult).por_ponto.map((p, i) => (
                            <div key={i} className={`sust-analysis-item ${p.alerta ? 'parcial' : 'favoravel'}`}>
                                <div className="sust-analysis-item-head">
                                    <strong>{p.ponto}</strong>
                                </div>
                                <p><em>Decisão:</em> {p.decisao}</p>
                                <p><em>Fundamento:</em> {p.fundamento}</p>
                                {p.alerta && <p className="sust-analysis-alerta">⚠️ {p.alerta}</p>}
                            </div>
                        ))}
                    </div>
                )}

                <button className="sust-btn-secondary-light" onClick={reset} style={{ marginTop: 12 }}>
                    Analisar outro documento
                </button>
            </section>
        );
    }

    return null;
};

export default DocumentAnalysis;
