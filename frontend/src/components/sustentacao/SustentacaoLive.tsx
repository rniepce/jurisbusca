import React, { useState } from 'react';
import { FaListUl, FaPenToSquare, FaTriangleExclamation } from 'react-icons/fa6';
import type { SustentacaoData } from '../../services/api';
import ProcessHeader from './shared/ProcessHeader';
import ChatPanel from './shared/ChatPanel';

interface Props {
    data: SustentacaoData;
    processId: string;
}

const SustentacaoLive: React.FC<Props> = ({ data, processId }) => {
    const [tesesMarcadas, setTesesMarcadas] = useState<Set<number>>(new Set());
    const [argumentosNovos, setArgumentosNovos] = useState('');
    const [notas, setNotas] = useState('');

    const toggleTese = (i: number) => {
        setTesesMarcadas(prev => {
            const next = new Set(prev);
            if (next.has(i)) next.delete(i); else next.add(i);
            return next;
        });
    };

    return (
        <div className="sust-live">
            <ProcessHeader data={data} tipoAto="sustentacao" modo="realizacao" compact />
            <div className="sust-dashboard">
                <div className="sust-main">
                    <section className="sust-card">
                        <h3 className="sust-card-title"><FaListUl /> Teses do recurso (checklist)</h3>
                        {data.teses && data.teses.length > 0 ? (
                            <ul className="sust-checklist">
                                {data.teses.map((t, i) => (
                                    <li key={i} className={tesesMarcadas.has(i) ? 'checked' : ''}>
                                        <label>
                                            <input
                                                type="checkbox"
                                                checked={tesesMarcadas.has(i)}
                                                onChange={() => toggleTese(i)}
                                            />
                                            <span>{t}</span>
                                        </label>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="sust-empty">Nenhuma tese identificada.</p>
                        )}
                    </section>

                    {data.preliminares && data.preliminares.length > 0 && (
                        <section className="sust-card">
                            <h3 className="sust-card-title"><FaTriangleExclamation /> Preliminares</h3>
                            <ul className="sust-list">
                                {data.preliminares.map((p, i) => <li key={i}>{p}</li>)}
                            </ul>
                        </section>
                    )}

                    <section className="sust-card">
                        <h3 className="sust-card-title"><FaPenToSquare /> Argumentos novos do sustentante</h3>
                        <textarea
                            className="sust-notes"
                            value={argumentosNovos}
                            onChange={(e) => setArgumentosNovos(e.target.value)}
                            placeholder="Anote argumentos que NÃO estavam nas razões recursais..."
                            rows={4}
                        />
                    </section>

                    <section className="sust-card">
                        <h3 className="sust-card-title">📝 Anotações</h3>
                        <textarea
                            className="sust-notes"
                            value={notas}
                            onChange={(e) => setNotas(e.target.value)}
                            placeholder="Anotações livres durante a sustentação..."
                            rows={6}
                        />
                    </section>
                </div>

                <ChatPanel
                    processId={processId}
                    title="Consulta rápida"
                    placeholders={[
                        'Qual o pedido na inicial?',
                        'O recorrente foi vencido em qual ponto?',
                    ]}
                />
            </div>
        </div>
    );
};

export default SustentacaoLive;
