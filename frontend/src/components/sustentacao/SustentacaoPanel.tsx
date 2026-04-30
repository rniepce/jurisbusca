import React, { useState, useRef, useCallback } from 'react';
import { FaXmark, FaGavel, FaFileArrowUp, FaCircleNotch } from 'react-icons/fa6';
import { uploadFile, extractSustentacao, type SustentacaoData, type TipoAto, type Modo } from '../../services/api';
import ModeSelector from './ModeSelector';
import SustentacaoPrep from './SustentacaoPrep';
import SustentacaoLive from './SustentacaoLive';
import AudienciaPrep from './AudienciaPrep';
import AudienciaLive from './AudienciaLive';
import './SustentacaoPanel.css';

interface Props {
    onClose: () => void;
    onOpenJurisprudencia?: () => void;
}

type Phase = 'select' | 'processing' | 'ready';

const SustentacaoPanel: React.FC<Props> = ({ onClose, onOpenJurisprudencia }) => {
    const [tipoAto, setTipoAto] = useState<TipoAto>('sustentacao');
    const [modo, setModo] = useState<Modo>('preparacao');
    const [phase, setPhase] = useState<Phase>('select');
    const [progress, setProgress] = useState('');
    const [error, setError] = useState('');
    const [processId, setProcessId] = useState<string | null>(null);
    const [data, setData] = useState<SustentacaoData | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const titulo = tipoAto === 'sustentacao' ? 'Sustentação Oral' : 'Audiência';
    const subtitulo = `${tipoAto === 'sustentacao' ? '2ª instância' : '1ª instância'} · ${modo === 'preparacao' ? 'Preparação' : 'Realização'}`;

    const handleFile = useCallback(async (file: File) => {
        setError('');
        setPhase('processing');
        setProgress('Enviando PDF...');
        try {
            const upload = await uploadFile(file, 'mistral_doc_ai', true, false, (info) => {
                setProgress(info.progress || `Processando OCR... ${info.percent}%`);
            });
            const text = (upload as any)?.text || '';
            if (!text.trim()) throw new Error('Não foi possível extrair texto do PDF.');

            setProgress('Extraindo dados do processo...');
            const result = await extractSustentacao(text, tipoAto, modo);
            setProcessId(result.process_id);
            setData(result.data);
            setPhase('ready');
        } catch (e: any) {
            setError(e?.message || 'Erro desconhecido');
            setPhase('select');
        }
    }, [tipoAto, modo]);

    const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0];
        if (f) handleFile(f);
    }, [handleFile]);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        const f = e.dataTransfer.files?.[0];
        if (f && f.type === 'application/pdf') handleFile(f);
    }, [handleFile]);

    const handleReset = useCallback(() => {
        setPhase('select');
        setData(null);
        setProcessId(null);
        setError('');
        setProgress('');
    }, []);

    const renderView = () => {
        if (!data || !processId) return null;
        if (tipoAto === 'sustentacao' && modo === 'preparacao') return <SustentacaoPrep data={data} processId={processId} onOpenJurisprudencia={onOpenJurisprudencia} />;
        if (tipoAto === 'sustentacao' && modo === 'realizacao') return <SustentacaoLive data={data} processId={processId} />;
        if (tipoAto === 'audiencia' && modo === 'preparacao') return <AudienciaPrep data={data} processId={processId} onOpenJurisprudencia={onOpenJurisprudencia} />;
        if (tipoAto === 'audiencia' && modo === 'realizacao') return <AudienciaLive data={data} processId={processId} />;
        return null;
    };

    return (
        <div className="sust-panel">
            <div className="sust-header">
                <div className="sust-header-content">
                    <div className="sust-title-row">
                        <FaGavel className="sust-header-icon" />
                        <div>
                            <h1 className="sust-title">{titulo}</h1>
                            <p className="sust-subtitle">{subtitulo}</p>
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
                {phase === 'select' && (
                    <div className="sust-select-area">
                        <ModeSelector
                            tipoAto={tipoAto}
                            modo={modo}
                            onTipoChange={setTipoAto}
                            onModoChange={setModo}
                        />
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
                    </div>
                )}

                {phase === 'processing' && (
                    <div className="sust-processing">
                        <FaCircleNotch className="sust-spinner" />
                        <h2>{progress || 'Processando...'}</h2>
                        <p>Isso pode levar alguns segundos para PDFs grandes ou digitalizados.</p>
                    </div>
                )}

                {phase === 'ready' && renderView()}
            </div>
        </div>
    );
};

export default SustentacaoPanel;
