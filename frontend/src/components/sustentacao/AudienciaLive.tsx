import React, { useMemo } from 'react';
import { FaUsers, FaQuestion, FaPenToSquare, FaTriangleExclamation, FaStar } from 'react-icons/fa6';
import type { SustentacaoData } from '../../services/api';
import ProcessHeader from './shared/ProcessHeader';
import ChatPanel from './shared/ChatPanel';
import DocumentAnalysis from './shared/DocumentAnalysis';
import { useLocalState } from './shared/useLocalState';

interface Props {
    data: SustentacaoData;
    processId: string;
}

const AudienciaLive: React.FC<Props> = ({ data, processId }) => {
    const depoentes = data.depoentes || [];
    const [depoenteId, setDepoenteId] = useLocalState<string>(`aud:${processId}:depoente`, depoentes[0]?.id || '');
    // Persistido como Record<string, number[]> (Sets não são serializáveis)
    const [perguntasMarcadasArr, setPerguntasMarcadasArr] = useLocalState<Record<string, number[]>>(`aud:${processId}:perguntas`, {});
    const [caderno, setCaderno] = useLocalState(`aud:${processId}:caderno`, '');

    const perguntasAtuais = useMemo(() => {
        const item = (data.perguntas_planejadas || []).find(p => p.depoente_id === depoenteId);
        return item?.perguntas || [];
    }, [data.perguntas_planejadas, depoenteId]);

    const togglePergunta = (i: number) => {
        setPerguntasMarcadasArr(prev => {
            const cur = new Set(prev[depoenteId] || []);
            if (cur.has(i)) cur.delete(i); else cur.add(i);
            return { ...prev, [depoenteId]: Array.from(cur) };
        });
    };

    const marcadas = useMemo(() => new Set(perguntasMarcadasArr[depoenteId] || []), [perguntasMarcadasArr, depoenteId]);

    const inserirMarca = (texto: string) => {
        const ts = new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
        setCaderno(prev => prev + (prev ? '\n' : '') + `[${ts}] ${texto} `);
    };

    return (
        <div className="sust-live">
            <ProcessHeader data={data} tipoAto="audiencia" modo="realizacao" compact />
            <div className="sust-dashboard">
                <div className="sust-main">
                    <section className="sust-card">
                        <h3 className="sust-card-title"><FaUsers /> Quem está depondo</h3>
                        {depoentes.length > 0 ? (
                            <div className="sust-mode-buttons">
                                {depoentes.map(d => (
                                    <button
                                        key={d.id}
                                        className={`sust-mode-btn ${depoenteId === d.id ? 'active' : ''}`}
                                        onClick={() => setDepoenteId(d.id)}
                                    >
                                        {d.nome} <small>({d.tipo})</small>
                                    </button>
                                ))}
                            </div>
                        ) : (
                            <p className="sust-empty">Nenhum depoente identificado.</p>
                        )}
                    </section>

                    <section className="sust-card">
                        <h3 className="sust-card-title"><FaQuestion /> Perguntas planejadas</h3>
                        {perguntasAtuais.length > 0 ? (
                            <ul className="sust-checklist">
                                {perguntasAtuais.map((p, i) => (
                                    <li key={i} className={marcadas.has(i) ? 'checked' : ''}>
                                        <label>
                                            <input
                                                type="checkbox"
                                                checked={marcadas.has(i)}
                                                onChange={() => togglePergunta(i)}
                                            />
                                            <span>{p}</span>
                                        </label>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="sust-empty">Sem perguntas pré-cadastradas para este depoente.</p>
                        )}
                    </section>

                    <section className="sust-card">
                        <h3 className="sust-card-title"><FaPenToSquare /> Caderno de audiência</h3>
                        <div className="sust-quick-actions">
                            <button className="sust-btn-secondary-light" onClick={() => inserirMarca('⚠️ CONTRADIÇÃO:')}>
                                <FaTriangleExclamation /> Contradição
                            </button>
                            <button className="sust-btn-secondary-light" onClick={() => inserirMarca('⭐ RELEVANTE:')}>
                                <FaStar /> Relevante
                            </button>
                        </div>
                        <textarea
                            className="sust-notes"
                            value={caderno}
                            onChange={(e) => setCaderno(e.target.value)}
                            placeholder="Anotações da audiência ao vivo..."
                            rows={12}
                        />
                    </section>

                    <DocumentAnalysis processId={processId} mode="sentenca" />
                </div>

                <ChatPanel
                    processId={processId}
                    title="Consulta rápida"
                    placeholders={[
                        'O que diz a fl. X da inicial?',
                        'Qual a versão dos fatos do réu?',
                    ]}
                />
            </div>
        </div>
    );
};

export default AudienciaLive;
