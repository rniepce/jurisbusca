import React from 'react';
import { FaListUl, FaScaleBalanced, FaUsers, FaFileLines, FaQuestion } from 'react-icons/fa6';
import type { SustentacaoData } from '../../services/api';
import ProcessHeader from './shared/ProcessHeader';
import ChatPanel from './shared/ChatPanel';

interface Props {
    data: SustentacaoData;
    processId: string;
}

const yesNo = (v: boolean | null | undefined): string => v == null ? '?' : v ? 'Sim' : 'Não';

const AudienciaPrep: React.FC<Props> = ({ data, processId }) => {
    const totalTestemunhas = (data.testemunhas_autor?.length || 0) + (data.testemunhas_reu?.length || 0);

    return (
        <div className="sust-dashboard">
            <div className="sust-main">
                <ProcessHeader data={data} tipoAto="audiencia" modo="preparacao" />

                <section className="sust-card">
                    <h3 className="sust-card-title"><FaListUl /> Pontos controvertidos</h3>
                    {data.pontos_controvertidos && data.pontos_controvertidos.length > 0 ? (
                        <ul className="sust-list">
                            {data.pontos_controvertidos.map((p, i) => <li key={i}>{p}</li>)}
                        </ul>
                    ) : (
                        <p className="sust-empty">Não identificados.</p>
                    )}
                </section>

                <section className="sust-card">
                    <h3 className="sust-card-title"><FaScaleBalanced /> Ônus da prova</h3>
                    {data.onus_prova && data.onus_prova.length > 0 ? (
                        <ul className="sust-list">
                            {data.onus_prova.map((o, i) => (
                                <li key={i}><strong>{o.de_quem}:</strong> {o.fato}</li>
                            ))}
                        </ul>
                    ) : (
                        <p className="sust-empty">—</p>
                    )}
                </section>

                <section className="sust-card">
                    <h3 className="sust-card-title">🧾 Provas</h3>
                    <div className="sust-fields-grid">
                        <div>
                            <span className="sust-field-label">Deferidas</span>
                            {data.provas_deferidas && data.provas_deferidas.length > 0 ? (
                                <ul className="sust-list">{data.provas_deferidas.map((p, i) => <li key={i}>{p}</li>)}</ul>
                            ) : <p className="sust-empty">—</p>}
                        </div>
                        <div>
                            <span className="sust-field-label">Indeferidas</span>
                            {data.provas_indeferidas && data.provas_indeferidas.length > 0 ? (
                                <ul className="sust-list">{data.provas_indeferidas.map((p, i) => <li key={i}>{p}</li>)}</ul>
                            ) : <p className="sust-empty">—</p>}
                        </div>
                    </div>
                </section>

                <section className="sust-card">
                    <h3 className="sust-card-title">
                        <FaUsers /> Pessoas a ouvir <small>({totalTestemunhas} testemunha{totalTestemunhas !== 1 ? 's' : ''})</small>
                    </h3>
                    <div className="sust-fields-grid">
                        <div>
                            <span className="sust-field-label">Depoimento pessoal autor</span>
                            <span className="sust-field-value">{yesNo(data.depoimento_pessoal_autor)}</span>
                        </div>
                        <div>
                            <span className="sust-field-label">Depoimento pessoal réu</span>
                            <span className="sust-field-value">{yesNo(data.depoimento_pessoal_reu)}</span>
                        </div>
                    </div>
                    {data.testemunhas_autor && data.testemunhas_autor.length > 0 && (
                        <div className="sust-subsection">
                            <strong>Testemunhas do autor</strong>
                            <table className="sust-table">
                                <thead><tr><th>Nome</th><th>Intimada</th><th>Já depôs</th></tr></thead>
                                <tbody>
                                    {data.testemunhas_autor.map((t, i) => (
                                        <tr key={i}><td>{t.nome}</td><td>{yesNo(t.intimada)}</td><td>{yesNo(t.ja_depos)}</td></tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                    {data.testemunhas_reu && data.testemunhas_reu.length > 0 && (
                        <div className="sust-subsection">
                            <strong>Testemunhas do réu</strong>
                            <table className="sust-table">
                                <thead><tr><th>Nome</th><th>Intimada</th><th>Já depôs</th></tr></thead>
                                <tbody>
                                    {data.testemunhas_reu.map((t, i) => (
                                        <tr key={i}><td>{t.nome}</td><td>{yesNo(t.intimada)}</td><td>{yesNo(t.ja_depos)}</td></tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </section>

                {data.documentos_relevantes && data.documentos_relevantes.length > 0 && (
                    <section className="sust-card">
                        <h3 className="sust-card-title"><FaFileLines /> Documentos relevantes</h3>
                        <ul className="sust-list">{data.documentos_relevantes.map((d, i) => <li key={i}>{d}</li>)}</ul>
                    </section>
                )}

                {data.quesitos_sugeridos && data.quesitos_sugeridos.length > 0 && (
                    <section className="sust-card">
                        <h3 className="sust-card-title"><FaQuestion /> Quesitos sugeridos</h3>
                        {data.quesitos_sugeridos.map((q, i) => (
                            <div key={i} className="sust-subsection">
                                <strong>Para {q.para}:</strong>
                                <ul className="sust-list">
                                    {q.perguntas.map((p, j) => <li key={j}>{p}</li>)}
                                </ul>
                            </div>
                        ))}
                    </section>
                )}
            </div>

            <ChatPanel
                processId={processId}
                placeholders={[
                    'Há contradição entre a inicial e a contestação?',
                    'Quais documentos foram juntados pela parte autora?',
                ]}
            />
        </div>
    );
};

export default AudienciaPrep;
