import React from 'react';
import { FaListUl, FaTriangleExclamation, FaScaleBalanced, FaEye, FaLightbulb } from 'react-icons/fa6';
import type { SustentacaoData } from '../../services/api';
import ProcessHeader from './shared/ProcessHeader';
import ChatPanel from './shared/ChatPanel';
import ResearchActions from './shared/ResearchActions';

interface Props {
    data: SustentacaoData;
    processId: string;
    onOpenJurisprudencia?: () => void;
}

const SustentacaoPrep: React.FC<Props> = ({ data, processId, onOpenJurisprudencia }) => {
    return (
        <div className="sust-dashboard">
            <div className="sust-main">
                <ProcessHeader data={data} tipoAto="sustentacao" modo="preparacao" />

                {data.sintese_recurso && (
                    <section className="sust-card">
                        <h3 className="sust-card-title"><FaScaleBalanced /> Síntese do recurso</h3>
                        <p className="sust-prose">{data.sintese_recurso}</p>
                    </section>
                )}

                <section className="sust-card">
                    <h3 className="sust-card-title"><FaListUl /> Teses</h3>
                    {data.teses && data.teses.length > 0 ? (
                        <ul className="sust-list">
                            {data.teses.map((t, i) => <li key={i}>{t}</li>)}
                        </ul>
                    ) : (
                        <p className="sust-empty">Nenhuma tese identificada.</p>
                    )}
                </section>

                <section className="sust-card">
                    <h3 className="sust-card-title"><FaTriangleExclamation /> Preliminares</h3>
                    {data.preliminares && data.preliminares.length > 0 ? (
                        <ul className="sust-list">
                            {data.preliminares.map((p, i) => <li key={i}>{p}</li>)}
                        </ul>
                    ) : (
                        <p className="sust-empty">Nenhuma preliminar identificada.</p>
                    )}
                </section>

                <section className="sust-card">
                    <h3 className="sust-card-title"><FaScaleBalanced /> Síntese da decisão de 1º grau</h3>
                    <p className="sust-prose">
                        {data.sintese_decisao_1grau || <span className="sust-empty">Não identificada.</span>}
                    </p>
                </section>

                <section className="sust-card">
                    <h3 className="sust-card-title"><FaEye /> Pontos críticos para acompanhar</h3>
                    {data.pontos_criticos && data.pontos_criticos.length > 0 ? (
                        <ul className="sust-list">
                            {data.pontos_criticos.map((p, i) => <li key={i}>{p}</li>)}
                        </ul>
                    ) : (
                        <p className="sust-empty">—</p>
                    )}
                </section>

                <section className="sust-card">
                    <h3 className="sust-card-title"><FaLightbulb /> Pré-juízo</h3>
                    <p className="sust-prose">
                        {data.pre_juizo || <span className="sust-empty">—</span>}
                    </p>
                </section>

                <ResearchActions onOpenJurisprudencia={onOpenJurisprudencia} />
            </div>

            <ChatPanel
                processId={processId}
                placeholders={[
                    'Qual é o pedido principal do recurso?',
                    'Houve preclusão de alguma tese?',
                    'Resumo da fundamentação da sentença.',
                ]}
            />
        </div>
    );
};

export default SustentacaoPrep;
