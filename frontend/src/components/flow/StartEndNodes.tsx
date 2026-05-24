import { Handle, Position } from '@xyflow/react';
import { FaPlay, FaFlag } from 'react-icons/fa6';

export function StartNode({ selected }: { selected?: boolean }) {
    return (
        <div className={`flow-node start ${selected ? 'selected' : ''}`} style={{ minWidth: 170 }}>
            <div className="flow-node-header">
                <span className="flow-node-icon"><FaPlay size={16} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">Início</span>
                    <span className="flow-node-meta">entrada do usuário</span>
                </div>
            </div>
            <Handle type="source" position={Position.Right} style={{ background: '#22c55e' }} />
        </div>
    );
}

export function EndNode({ selected }: { selected?: boolean }) {
    return (
        <div className={`flow-node end ${selected ? 'selected' : ''}`} style={{ minWidth: 170 }}>
            <Handle type="target" position={Position.Left} style={{ background: '#64748b' }} />
            <div className="flow-node-header">
                <span className="flow-node-icon"><FaFlag size={16} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">Fim</span>
                    <span className="flow-node-meta">resultado final</span>
                </div>
            </div>
        </div>
    );
}
