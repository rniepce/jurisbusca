import { Handle, Position } from '@xyflow/react';
import { FaCodeBranch } from 'react-icons/fa6';

interface RouterNodeData {
    label?: string;
    condition?: string;
    status?: 'idle' | 'running' | 'done' | 'error';
    branch?: 'true' | 'false';
}

export default function RouterNode({ data, selected }: { data: RouterNodeData; selected?: boolean }) {
    const status = data.status || 'idle';
    const klass = `flow-node router ${selected ? 'selected' : ''} status-${status}`;

    return (
        <div className={klass}>
            <Handle type="target" position={Position.Left} style={{ background: '#f59e0b' }} />

            <div className="flow-node-header">
                <span className="flow-node-icon"><FaCodeBranch size={18} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">{data.label || 'Roteador'}</span>
                    {data.condition && <span className="flow-node-meta">{data.condition}</span>}
                </div>
            </div>

            {data.branch && (
                <div className="flow-node-footer">
                    <span className={`flow-node-branch ${data.branch}`}>
                        → {data.branch === 'true' ? 'verdadeiro' : 'falso'}
                    </span>
                </div>
            )}

            <Handle
                id="true"
                type="source"
                position={Position.Right}
                style={{ top: '32%', background: '#22c55e' }}
            />
            <Handle
                id="false"
                type="source"
                position={Position.Right}
                style={{ top: '68%', background: '#fb923c' }}
            />
        </div>
    );
}
