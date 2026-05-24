import { Handle, Position } from '@xyflow/react';
import {
    FaShuffle, FaUserClock, FaFileWord,
    FaScaleBalanced, FaCubes, FaWandMagicSparkles,
    FaCode, FaPuzzlePiece,
} from 'react-icons/fa6';

type NodeStatus = 'idle' | 'running' | 'done' | 'error';

interface BaseData {
    label?: string;
    status?: NodeStatus;
}

function nodeClass(type: string, status?: NodeStatus, selected?: boolean) {
    return `flow-node ${type} ${selected ? 'selected' : ''} status-${status || 'idle'}`;
}

// ────────────────────────────────────────────────────────────────────
// SWITCH: classificador multi-saída (LLM decide qual rota seguir)
// ────────────────────────────────────────────────────────────────────
interface SwitchNodeData extends BaseData {
    categories?: string;    // separadas por |, ex: "Civil|Penal|Tributário"
    chosen?: string;
}

export function SwitchNode({ data, selected }: { data: SwitchNodeData; selected?: boolean }) {
    const cats = (data.categories || '').split('|').filter(Boolean);
    return (
        <div className={nodeClass('switch', data.status, selected)}>
            <Handle type="target" position={Position.Left} style={{ background: '#a855f7' }} />
            <div className="flow-node-header">
                <span className="flow-node-icon"><FaShuffle size={18} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">{data.label || 'Switch'}</span>
                    <span className="flow-node-meta">classificador LLM</span>
                </div>
            </div>
            {data.chosen && (
                <div className="flow-node-footer">
                    <span className="flow-node-branch true">→ {data.chosen}</span>
                </div>
            )}
            {cats.length > 0 && cats.map((c, i) => (
                <Handle
                    key={c}
                    id={c}
                    type="source"
                    position={Position.Right}
                    style={{
                        top: `${20 + ((60 / Math.max(cats.length - 1, 1)) * i)}%`,
                        background: '#a855f7',
                    }}
                />
            ))}
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────
// HUMAN-IN-THE-LOOP: pausa o fluxo e espera aprovação no chat
// ────────────────────────────────────────────────────────────────────
interface HILNodeData extends BaseData {
    question?: string;
}

export function HILNode({ data, selected }: { data: HILNodeData; selected?: boolean }) {
    return (
        <div className={nodeClass('hil', data.status, selected)}>
            <Handle type="target" position={Position.Left} style={{ background: '#eab308' }} />
            <div className="flow-node-header">
                <span className="flow-node-icon"><FaUserClock size={18} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">{data.label || 'Aprovação Humana'}</span>
                    <span className="flow-node-meta">pausa p/ revisão</span>
                </div>
            </div>
            <Handle type="source" position={Position.Right} style={{ background: '#eab308' }} />
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────
// GERAR DOCX: converte saída markdown em arquivo .docx
// ────────────────────────────────────────────────────────────────────
interface DocxNodeData extends BaseData {
    filename?: string;
}

export function DocxNode({ data, selected }: { data: DocxNodeData; selected?: boolean }) {
    return (
        <div className={nodeClass('docx', data.status, selected)}>
            <Handle type="target" position={Position.Left} style={{ background: '#1d4ed8' }} />
            <div className="flow-node-header">
                <span className="flow-node-icon"><FaFileWord size={18} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">{data.label || 'Gerar DOCX'}</span>
                    <span className="flow-node-meta">markdown → .docx</span>
                </div>
            </div>
            <Handle type="source" position={Position.Right} style={{ background: '#1d4ed8' }} />
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────
// PESQUISAR JURISPRUDÊNCIA: RAG sobre acordãos TJMG
// ────────────────────────────────────────────────────────────────────
interface JurisNodeData extends BaseData {
    query?: string;
    top_k?: string;
}

export function JurisNode({ data, selected }: { data: JurisNodeData; selected?: boolean }) {
    return (
        <div className={nodeClass('juris', data.status, selected)}>
            <Handle type="target" position={Position.Left} style={{ background: '#0891b2' }} />
            <div className="flow-node-header">
                <span className="flow-node-icon"><FaScaleBalanced size={18} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">{data.label || 'Pesquisar Jurisprudência'}</span>
                    <span className="flow-node-meta">RAG acordãos TJMG</span>
                </div>
            </div>
            <Handle type="source" position={Position.Right} style={{ background: '#0891b2' }} />
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────
// BUSCAR MODELO: RAG sobre templates de minutas
// ────────────────────────────────────────────────────────────────────
interface ModeloNodeData extends BaseData {
    query?: string;
    top_k?: string;
}

export function ModeloNode({ data, selected }: { data: ModeloNodeData; selected?: boolean }) {
    return (
        <div className={nodeClass('modelo', data.status, selected)}>
            <Handle type="target" position={Position.Left} style={{ background: '#16a34a' }} />
            <div className="flow-node-header">
                <span className="flow-node-icon"><FaCubes size={18} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">{data.label || 'Buscar Modelo'}</span>
                    <span className="flow-node-meta">RAG templates</span>
                </div>
            </div>
            <Handle type="source" position={Position.Right} style={{ background: '#16a34a' }} />
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────
// ESTILO DO JUIZ: aplica o style dossier do usuário no texto
// ────────────────────────────────────────────────────────────────────
interface EstiloNodeData extends BaseData {}

export function EstiloNode({ data, selected }: { data: EstiloNodeData; selected?: boolean }) {
    return (
        <div className={nodeClass('estilo', data.status, selected)}>
            <Handle type="target" position={Position.Left} style={{ background: '#e11d48' }} />
            <div className="flow-node-header">
                <span className="flow-node-icon"><FaWandMagicSparkles size={18} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">{data.label || 'Estilo do Juiz'}</span>
                    <span className="flow-node-meta">aplica style dossier</span>
                </div>
            </div>
            <Handle type="source" position={Position.Right} style={{ background: '#e11d48' }} />
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────
// EXTRACTOR: extrai JSON estruturado — cada campo vira variável
// ────────────────────────────────────────────────────────────────────
interface ExtractorNodeData extends BaseData {
    fields?: string; // "name:type:desc|name:type:desc"
}

export function ExtractorNode({ data, selected }: { data: ExtractorNodeData; selected?: boolean }) {
    const fieldCount = (data.fields || '').split('|').filter(Boolean).length;
    return (
        <div className={nodeClass('extractor', data.status, selected)}>
            <Handle type="target" position={Position.Left} style={{ background: '#0d9488' }} />
            <div className="flow-node-header">
                <span className="flow-node-icon"><FaCode size={18} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">{data.label || 'Extrator JSON'}</span>
                    <span className="flow-node-meta">
                        {fieldCount > 0 ? `${fieldCount} campo${fieldCount > 1 ? 's' : ''}` : 'sem campos'}
                    </span>
                </div>
            </div>
            <Handle type="source" position={Position.Right} style={{ background: '#0d9488' }} />
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────
// SUBFLOW: chama outro fluxo salvo como módulo
// ────────────────────────────────────────────────────────────────────
interface SubflowNodeData extends BaseData {
    flow_id?: string;
    flow_name?: string;
}

export function SubflowNode({ data, selected }: { data: SubflowNodeData; selected?: boolean }) {
    return (
        <div className={nodeClass('subflow', data.status, selected)}>
            <Handle type="target" position={Position.Left} style={{ background: '#7c3aed' }} />
            <div className="flow-node-header">
                <span className="flow-node-icon"><FaPuzzlePiece size={18} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">{data.label || 'Sub-fluxo'}</span>
                    <span className="flow-node-meta">
                        {data.flow_name || (data.flow_id ? 'fluxo configurado' : 'sem fluxo')}
                    </span>
                </div>
            </div>
            <Handle type="source" position={Position.Right} style={{ background: '#7c3aed' }} />
        </div>
    );
}
