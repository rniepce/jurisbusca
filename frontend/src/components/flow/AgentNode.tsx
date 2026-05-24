import { Handle, Position } from '@xyflow/react';
import { FaRobot, FaPaperclip } from 'react-icons/fa6';

interface AgentNodeData {
    label?: string;
    model?: string;
    knowledge?: string;
    knowledge_files?: string;
    status?: 'idle' | 'running' | 'done' | 'error';
}

export default function AgentNode({ data, selected }: { data: AgentNodeData; selected?: boolean }) {
    const status = data.status || 'idle';
    const klass = `flow-node agent ${selected ? 'selected' : ''} status-${status}`;
    const fileCount = data.knowledge_files
        ? data.knowledge_files.split('|').filter(Boolean).length
        : 0;

    return (
        <div className={klass}>
            <Handle type="target" position={Position.Left} style={{ background: 'var(--primary-color)' }} />

            <div className="flow-node-header">
                <span className="flow-node-icon"><FaRobot size={20} /></span>
                <div className="flow-node-title">
                    <span className="flow-node-name">{data.label || 'Agente'}</span>
                    {data.model && <span className="flow-node-meta">{data.model}</span>}
                </div>
            </div>

            {fileCount > 0 && (
                <div className="flow-node-footer">
                    <span className="flow-node-chip knowledge">
                        <FaPaperclip size={9} /> {fileCount} arquivo{fileCount > 1 ? 's' : ''}
                    </span>
                </div>
            )}

            <Handle type="source" position={Position.Right} style={{ background: 'var(--primary-color)' }} />
        </div>
    );
}
