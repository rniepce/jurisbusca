import { useRef } from 'react';
import { FaLayerGroup, FaUpload, FaXmark } from 'react-icons/fa6';
import './LotePanel.css';

interface Props {
    /** Triggers the X-Ray flow with multiple files. */
    onUpload: (files: File[]) => void;
    onClose: () => void;
    isLoading?: boolean;
}

const ACCEPTED = '.pdf,.docx,.txt';

/**
 * Empty-state panel for batch analysis. Receives 2+ files and forwards them
 * to the X-Ray handler — the resulting dashboard takes over the view via the
 * existing `xrayReport` flow in App.tsx.
 */
export default function LotePanel({ onUpload, onClose, isLoading = false }: Props) {
    const inputRef = useRef<HTMLInputElement | null>(null);

    const pick = () => inputRef.current?.click();

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        e.target.value = '';
        if (files.length === 0) return;
        if (files.length < 2) {
            alert('Selecione pelo menos 2 processos para fazer análise em lote.');
            return;
        }
        onUpload(files);
    };

    return (
        <div className="lote-panel">
            <header className="lote-panel-header">
                <div className="lote-panel-title-group">
                    <FaLayerGroup size={20} />
                    <h2 className="lote-panel-title">Análise em lote</h2>
                </div>
                <button className="lote-panel-close" onClick={onClose} aria-label="Fechar">
                    <FaXmark />
                </button>
            </header>

            <div className="lote-panel-body">
                <p className="lote-panel-lead">
                    Anexe <strong>2 ou mais processos</strong> em PDF/DOCX para classificar e
                    agrupar automaticamente por matéria. Você poderá então analisar cada
                    grupo em paralelo ou usar um processo-piloto como gabarito.
                </p>

                <div className="lote-panel-card" role="region" aria-labelledby="lote-upload-label">
                    <FaUpload size={36} className="lote-panel-card-icon" />
                    <div id="lote-upload-label" className="lote-panel-card-title">
                        Anexar processos
                    </div>
                    <p className="lote-panel-card-hint">
                        Aceita PDF, DOCX e TXT — mínimo de 2 arquivos.
                    </p>
                    <button
                        type="button"
                        className="lote-panel-btn"
                        onClick={pick}
                        disabled={isLoading}
                    >
                        {isLoading ? 'Processando...' : 'Escolher arquivos'}
                    </button>
                    <input
                        ref={inputRef}
                        type="file"
                        accept={ACCEPTED}
                        multiple
                        onChange={handleChange}
                        style={{ display: 'none' }}
                    />
                </div>

                <div className="lote-panel-steps" aria-label="Como funciona">
                    <div className="lote-panel-step">
                        <span className="lote-panel-step-num">1</span>
                        <span>Anexa os processos</span>
                    </div>
                    <div className="lote-panel-step">
                        <span className="lote-panel-step-num">2</span>
                        <span>Sistema classifica e agrupa</span>
                    </div>
                    <div className="lote-panel-step">
                        <span className="lote-panel-step-num">3</span>
                        <span>Analisa grupos em paralelo ou via piloto</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
