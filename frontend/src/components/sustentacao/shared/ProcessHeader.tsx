import React from 'react';
import type { SustentacaoData, TipoAto, Modo } from '../../../services/api';

interface Props {
    data: SustentacaoData;
    tipoAto: TipoAto;
    modo: Modo;
    compact?: boolean;
}

const Field: React.FC<{ label: string; value?: string | null }> = ({ label, value }) => (
    <div className="sust-field">
        <span className="sust-field-label">{label}</span>
        <span className="sust-field-value">{value || '—'}</span>
    </div>
);

const ProcessHeader: React.FC<Props> = ({ data, tipoAto, modo, compact = false }) => {
    if (compact) {
        const left = tipoAto === 'sustentacao'
            ? `${data.numero_processo || '?'} · ${data.tipo_recursal || ''} · ${data.camara || ''}`
            : `${data.numero_processo || '?'} · ${data.vara || ''}`;
        const right = tipoAto === 'sustentacao'
            ? `${data.recorrente || '?'} × ${data.recorrido || '?'}`
            : `${data.autor || '?'} × ${data.reu || '?'}`;
        return (
            <div className="sust-header-compact">
                <div className="sust-header-compact-left">{left}</div>
                <div className="sust-header-compact-right">{right}</div>
            </div>
        );
    }

    return (
        <section className="sust-card">
            <h3 className="sust-card-title">📄 Processo</h3>
            <div className="sust-fields-grid">
                {tipoAto === 'sustentacao' ? (
                    <>
                        <Field label="Número" value={data.numero_processo} />
                        <Field label="Tipo recursal" value={data.tipo_recursal} />
                        <Field label="Câmara" value={data.camara} />
                        <Field label="Relator" value={data.relator} />
                        {modo === 'preparacao' && <Field label="Data da sessão" value={data.data_sessao} />}
                        <Field label="Recorrente" value={data.recorrente} />
                        <Field label="Recorrido" value={data.recorrido} />
                        <Field label="Adv. sustentante" value={data.advogado_sustentante} />
                        <Field label="Parte sustentante" value={data.parte_sustentante} />
                    </>
                ) : (
                    <>
                        <Field label="Número" value={data.numero_processo} />
                        <Field label="Vara" value={data.vara} />
                        <Field label="Juiz(a)" value={data.juiz} />
                        <Field label="Tipo de ação" value={data.tipo_acao} />
                        {modo === 'preparacao' && <Field label="Data da audiência" value={data.data_audiencia} />}
                        <Field label="Autor" value={data.autor} />
                        <Field label="Réu" value={data.reu} />
                    </>
                )}
            </div>
        </section>
    );
};

export default ProcessHeader;
