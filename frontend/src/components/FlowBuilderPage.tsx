import { useCallback, useEffect, useRef, useState } from 'react';
import {
    ReactFlow,
    addEdge,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    type Connection,
    type Node,
    type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
    FaPlus, FaFloppyDisk, FaPlay, FaTrash, FaChevronLeft,
    FaCodeBranch, FaRobot, FaFlag, FaListUl, FaEye, FaXmark,
} from 'react-icons/fa6';
import AgentNode from './flow/AgentNode';
import RouterNode from './flow/RouterNode';
import { StartNode, EndNode } from './flow/StartEndNodes';
import NodeConfigPanel from './flow/NodeConfigPanel';
import {
    listFlows, createFlow, getFlow, updateFlow, deleteFlow, previewFlow,
    type FlowSummary, type FlowConfig,
} from '../services/api';
import './FlowBuilderPage.css';

const NODE_TYPES = {
    start: StartNode,
    end: EndNode,
    agent: AgentNode,
    router: RouterNode,
};

const AGENT_COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4'];

let nodeCounter = 1;

function makeNode(type: string, position = { x: 200, y: 200 }): Node {
    const id = `${type}_${Date.now()}_${nodeCounter++}`;
    const defaults: Record<string, Record<string, string>> = {
        agent: { label: 'Agente', model: 'gpt-5.3-chat', prompt: '', output_var: `saida_${nodeCounter}`, knowledge: '', knowledge_files: '' },
        router: { label: 'Roteador', condition: '' },
        start: {},
        end: {},
    };
    return { id, type, position, data: defaults[type] ?? {} };
}

interface RunEvent {
    event: string;
    node_id?: string;
    label?: string;
    output?: string;
    branch?: string;
    error?: string;
    state?: Record<string, string>;
}

export default function FlowBuilderPage({ onClose }: { onClose?: () => void }) {
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);
    const [flowId, setFlowId] = useState<string | null>(null);
    const [flowName, setFlowName] = useState('Meu Fluxo');
    const [flowDescription, setFlowDescription] = useState('');
    const [flowColor, setFlowColor] = useState(AGENT_COLORS[0]);
    const [saving, setSaving] = useState(false);
    const [running, setRunning] = useState(false);
    const [runLog, setRunLog] = useState<RunEvent[]>([]);
    const [showRunModal, setShowRunModal] = useState(false);
    const [runIsPreview, setRunIsPreview] = useState(false);
    const [inputText, setInputText] = useState('');
    const [showFlowList, setShowFlowList] = useState(false);
    const [showSaveDialog, setShowSaveDialog] = useState(false);
    const [flows, setFlows] = useState<FlowSummary[]>([]);
    const [loadingFlows, setLoadingFlows] = useState(false);
    const [finalOutput, setFinalOutput] = useState('');
    const logEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [runLog]);

    const onConnect = useCallback(
        (connection: Connection) => {
            const edge: Edge = {
                ...connection,
                id: `e_${Date.now()}`,
                label: connection.sourceHandle === 'false' ? 'falso' : connection.sourceHandle === 'true' ? 'verdadeiro' : '',
                style: { stroke: '#94a3b8', strokeWidth: 2 },
            } as Edge;
            setEdges(eds => addEdge(edge, eds));
        },
        [setEdges],
    );

    const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
        if (node.type === 'start' || node.type === 'end') {
            setSelectedNode(null);
            return;
        }
        setSelectedNode(node);
    }, []);

    const onPaneClick = useCallback(() => setSelectedNode(null), []);

    const handleNodeDataChange = useCallback((nodeId: string, data: Record<string, string>) => {
        setNodes(ns => ns.map(n => n.id === nodeId ? { ...n, data } : n));
        setSelectedNode(prev => prev?.id === nodeId ? { ...prev, data } : prev);
    }, [setNodes]);

    const addNode = (type: string) => {
        const offset = nodes.length * 25;
        const n = makeNode(type, { x: 300 + offset, y: 180 + offset });
        setNodes(ns => [...ns, n]);
    };

    const deleteSelectedNode = () => {
        if (!selectedNode) return;
        setNodes(ns => ns.filter(n => n.id !== selectedNode.id));
        setEdges(es => es.filter(e => e.source !== selectedNode.id && e.target !== selectedNode.id));
        setSelectedNode(null);
    };

    const buildConfig = (): FlowConfig => ({
        nodes: nodes.map(n => ({
            id: n.id,
            type: n.type ?? 'agent',
            position: n.position,
            data: n.data as Record<string, string>,
        })),
        edges: edges.map(e => ({
            id: e.id,
            source: e.source,
            target: e.target,
            label: typeof e.label === 'string' ? e.label : '',
            sourceHandle: e.sourceHandle ?? '',
        })),
    });

    const handleSaveConfirm = async () => {
        setSaving(true);
        try {
            const config = buildConfig();
            if (flowId) {
                await updateFlow(flowId, flowName, config, flowDescription, flowColor);
            } else {
                const created = await createFlow(flowName, config, flowDescription, flowColor);
                setFlowId(created.id);
            }
            setShowSaveDialog(false);
        } catch (err) {
            alert(`Erro ao salvar: ${err}`);
        } finally {
            setSaving(false);
        }
    };

    const handleSaveClick = () => {
        // Se já está salvo, salva direto. Caso contrário, abre modal pedindo nome/desc/cor.
        if (flowId) {
            void handleSaveConfirm();
        } else {
            setShowSaveDialog(true);
        }
    };

    const startRun = async (preview: boolean) => {
        setRunning(true);
        setRunIsPreview(preview);
        setRunLog([]);
        setFinalOutput('');
        setShowRunModal(true);
        setNodes(ns => ns.map(n => ({ ...n, data: { ...n.data, status: 'idle' } })));
    };

    const handlePreview = () => {
        void startRun(true);
    };

    const handleRunSavedFlow = () => {
        if (!flowId) {
            setShowSaveDialog(true);
            return;
        }
        void startRun(false);
    };

    const executeRun = async () => {
        if (!inputText.trim()) {
            alert('Forneça um texto de entrada para executar o fluxo.');
            return;
        }
        setNodes(ns => ns.map(n => ({ ...n, data: { ...n.data, status: 'idle' } })));
        setRunLog([]);
        setFinalOutput('');
        setRunning(true);

        const onEvent = (event: Record<string, unknown>) => {
            const ev = event as RunEvent;
            setRunLog(prev => [...prev, ev]);
            if (ev.event === 'node_start' && ev.node_id) {
                setNodes(ns => ns.map(n => n.id === ev.node_id ? { ...n, data: { ...n.data, status: 'running' } } : n));
            }
            if (ev.event === 'node_done' && ev.node_id) {
                setNodes(ns => ns.map(n => n.id === ev.node_id ? { ...n, data: { ...n.data, status: 'done', branch: ev.branch ?? '' } } : n));
            }
            if (ev.event === 'node_error' && ev.node_id) {
                setNodes(ns => ns.map(n => n.id === ev.node_id ? { ...n, data: { ...n.data, status: 'error' } } : n));
            }
            if (ev.event === 'flow_done') {
                setFinalOutput(ev.output ?? '');
            }
        };

        try {
            if (runIsPreview || !flowId) {
                await previewFlow(buildConfig(), inputText, onEvent);
            } else {
                const { runFlow } = await import('../services/api');
                await runFlow(flowId, inputText, onEvent);
            }
        } catch (err) {
            setRunLog(prev => [...prev, { event: 'error', error: String(err) }]);
        } finally {
            setRunning(false);
        }
    };

    const handleLoadFlowList = async () => {
        setLoadingFlows(true);
        try {
            setFlows(await listFlows());
            setShowFlowList(true);
        } catch (err) {
            alert(`Erro ao carregar fluxos: ${err}`);
        } finally {
            setLoadingFlows(false);
        }
    };

    const handleOpenFlow = async (id: string) => {
        try {
            const flow = await getFlow(id);
            setFlowId(flow.id);
            setFlowName(flow.name);
            setFlowDescription(flow.description ?? '');
            setFlowColor(flow.color ?? AGENT_COLORS[0]);
            const loadedNodes = flow.config.nodes.map(n => ({
                id: n.id,
                type: n.type,
                position: n.position,
                data: { ...n.data, status: 'idle' },
            }));
            setNodes(loadedNodes);
            setEdges(flow.config.edges.map(e => ({
                id: e.id,
                source: e.source,
                target: e.target,
                label: e.label || '',
                sourceHandle: e.sourceHandle || null,
                style: { stroke: '#94a3b8', strokeWidth: 2 },
            })));
            setShowFlowList(false);
            setSelectedNode(null);
        } catch (err) {
            alert(`Erro ao abrir fluxo: ${err}`);
        }
    };

    const handleDeleteFlow = async (id: string) => {
        if (!confirm('Apagar este fluxo? Ele também sumirá da lista de agentes.')) return;
        try {
            await deleteFlow(id);
            setFlows(f => f.filter(x => x.id !== id));
            if (flowId === id) {
                setFlowId(null);
                handleNewFlow();
            }
        } catch (err) {
            alert(`Erro ao apagar: ${err}`);
        }
    };

    const handleNewFlow = () => {
        setFlowId(null);
        setFlowName('Meu Fluxo');
        setFlowDescription('');
        setFlowColor(AGENT_COLORS[0]);
        setNodes([
            makeNode('start', { x: 80, y: 200 }),
            makeNode('end', { x: 720, y: 200 }),
        ]);
        setEdges([]);
        setSelectedNode(null);
        setShowFlowList(false);
    };

    useEffect(() => {
        handleNewFlow();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const closeIcon = (<FaXmark size={16} />);

    return (
        <div className="flow-builder">
            {/* Toolbar */}
            <div className="flow-toolbar">
                {onClose && (
                    <button onClick={onClose} className="flow-btn flow-btn-ghost" title="Voltar">
                        <FaChevronLeft size={11} /> Voltar
                    </button>
                )}

                <div className="flow-name-display" title={flowName}>
                    <span className="flow-name-dot" style={{ background: flowColor }} />
                    <span className="flow-name-text">{flowName}</span>
                    {flowId && <span className="flow-name-badge">Orquestrador</span>}
                </div>

                <div className="flow-toolbar-divider" />

                <button onClick={() => addNode('agent')} className="flow-btn flow-btn-add agent">
                    <FaRobot size={12} /> Agente
                </button>
                <button onClick={() => addNode('router')} className="flow-btn flow-btn-add router">
                    <FaCodeBranch size={12} /> Roteador
                </button>
                <button onClick={() => addNode('start')} className="flow-btn flow-btn-add start">
                    <FaPlay size={10} /> Início
                </button>
                <button onClick={() => addNode('end')} className="flow-btn flow-btn-add end">
                    <FaFlag size={11} /> Fim
                </button>

                {selectedNode && (
                    <>
                        <div className="flow-toolbar-divider" />
                        <button onClick={deleteSelectedNode} className="flow-btn flow-btn-danger">
                            <FaTrash size={11} /> Apagar nó
                        </button>
                    </>
                )}

                <div className="flow-toolbar-spacer" />

                <button onClick={handleLoadFlowList} disabled={loadingFlows} className="flow-btn flow-btn-ghost">
                    <FaListUl size={11} /> Meus Fluxos
                </button>
                <button onClick={handleNewFlow} className="flow-btn flow-btn-ghost">
                    <FaPlus size={11} /> Novo
                </button>
                <button onClick={handlePreview} disabled={running} className="flow-btn flow-btn-ghost">
                    <FaEye size={11} /> Preview
                </button>
                <button onClick={handleSaveClick} disabled={saving} className="flow-btn flow-btn-primary">
                    <FaFloppyDisk size={11} /> {saving ? 'Salvando...' : flowId ? 'Salvar' : 'Salvar como Agente'}
                </button>
                {flowId && (
                    <button onClick={handleRunSavedFlow} disabled={running} className="flow-btn flow-btn-success">
                        <FaPlay size={11} /> Executar
                    </button>
                )}
            </div>

            {/* Canvas */}
            <div className="flow-canvas-wrap">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onNodeClick={onNodeClick}
                    onPaneClick={onPaneClick}
                    nodeTypes={NODE_TYPES}
                    fitView
                    colorMode="light"
                    deleteKeyCode={null}
                    proOptions={{ hideAttribution: false }}
                >
                    <Background color="#475569" gap={28} size={1.5} />
                    <Controls />
                    <MiniMap
                        nodeColor={(n) => {
                            const t = n.type;
                            if (t === 'agent') return '#3b82f6';
                            if (t === 'router') return '#f59e0b';
                            if (t === 'start') return '#22c55e';
                            return '#64748b';
                        }}
                        maskColor="rgba(15, 23, 42, 0.6)"
                    />
                </ReactFlow>

                {selectedNode && (
                    <NodeConfigPanel
                        node={selectedNode as { id: string; type: string; data: Record<string, string> }}
                        onChange={handleNodeDataChange}
                        onClose={() => setSelectedNode(null)}
                    />
                )}
            </div>

            {/* Save dialog */}
            {showSaveDialog && (
                <div className="flow-modal-backdrop" onClick={() => !saving && setShowSaveDialog(false)}>
                    <div className="flow-modal" onClick={e => e.stopPropagation()}>
                        <div className="flow-modal-header">
                            <span className="flow-modal-title">{flowId ? 'Salvar Agente Orquestrador' : 'Criar Agente Orquestrador'}</span>
                            <button onClick={() => setShowSaveDialog(false)} className="flow-config-close">{closeIcon}</button>
                        </div>

                        <p className="flow-config-help" style={{ marginTop: -6 }}>
                            Após salvar, este fluxo aparecerá na barra lateral como um agente especial.
                            Você poderá selecioná-lo em qualquer conversa para que ele execute todos os passos automaticamente.
                        </p>

                        <div className="flow-config-field">
                            <label className="flow-config-label">Nome do Agente</label>
                            <input
                                className="flow-input"
                                value={flowName}
                                onChange={e => setFlowName(e.target.value)}
                                placeholder="Ex: Triagem + Minuta + Revisão"
                                autoFocus
                            />
                        </div>

                        <div className="flow-config-field">
                            <label className="flow-config-label">Descrição</label>
                            <input
                                className="flow-input"
                                value={flowDescription}
                                onChange={e => setFlowDescription(e.target.value)}
                                placeholder="O que esse fluxo faz?"
                            />
                        </div>

                        <div className="flow-config-field">
                            <label className="flow-config-label">Cor</label>
                            <div className="flow-color-row">
                                {AGENT_COLORS.map(c => (
                                    <button
                                        key={c}
                                        className={`flow-color-swatch ${flowColor === c ? 'active' : ''}`}
                                        style={{ background: c }}
                                        onClick={() => setFlowColor(c)}
                                        aria-label={`Cor ${c}`}
                                    />
                                ))}
                            </div>
                        </div>

                        <button
                            onClick={handleSaveConfirm}
                            disabled={saving || !flowName.trim()}
                            className="flow-btn flow-btn-primary"
                            style={{ justifyContent: 'center', padding: '10px' }}
                        >
                            <FaFloppyDisk size={12} /> {saving ? 'Salvando...' : 'Salvar e criar agente'}
                        </button>
                    </div>
                </div>
            )}

            {/* Flow list modal */}
            {showFlowList && (
                <div className="flow-modal-backdrop" onClick={() => setShowFlowList(false)}>
                    <div className="flow-modal" onClick={e => e.stopPropagation()}>
                        <div className="flow-modal-header">
                            <span className="flow-modal-title">Meus Fluxos / Orquestradores</span>
                            <button onClick={() => setShowFlowList(false)} className="flow-config-close">{closeIcon}</button>
                        </div>
                        <div className="flow-list-scroll">
                            {flows.length === 0 && (
                                <p className="flow-list-empty">Nenhum fluxo salvo ainda.</p>
                            )}
                            {flows.map(f => (
                                <div key={f.id} className="flow-list-item">
                                    <span className="flow-list-item-dot" style={{ background: f.color || '#3b82f6' }} />
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                        <div className="flow-list-item-name">{f.name}</div>
                                        {f.description && <div className="flow-list-item-desc">{f.description}</div>}
                                    </div>
                                    <span className="flow-list-item-date">
                                        {new Date(f.updated_at).toLocaleDateString('pt-BR')}
                                    </span>
                                    <button onClick={() => handleOpenFlow(f.id)} className="flow-btn flow-btn-primary">
                                        Abrir
                                    </button>
                                    <button onClick={() => handleDeleteFlow(f.id)} className="flow-btn flow-btn-danger">
                                        <FaTrash size={10} />
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Run / Preview modal */}
            {showRunModal && (
                <div className="flow-modal-backdrop">
                    <div className="flow-modal flow-run-modal">
                        <div className="flow-modal-header">
                            <span className="flow-modal-title">
                                {runIsPreview ? '👁️ Preview do Fluxo' : '▶ Executar Fluxo'}
                            </span>
                            {!running && (
                                <button onClick={() => setShowRunModal(false)} className="flow-config-close">{closeIcon}</button>
                            )}
                        </div>

                        <div className="flow-config-field">
                            <label className="flow-config-label">Texto de Entrada</label>
                            <textarea
                                value={inputText}
                                onChange={e => setInputText(e.target.value)}
                                disabled={running}
                                className="flow-run-textarea"
                                placeholder="Cole o texto do processo ou a entrada para o fluxo..."
                            />
                        </div>

                        {!running && (
                            <button onClick={executeRun} className="flow-btn flow-btn-success" style={{ justifyContent: 'center', padding: '10px' }}>
                                <FaPlay size={12} /> {runIsPreview ? 'Iniciar Preview' : 'Iniciar Execução'}
                            </button>
                        )}

                        {runLog.length > 0 && (
                            <div className="flow-run-log">
                                {runLog.map((ev, i) => (
                                    <div key={i} className={`flow-run-log-line ${ev.event}`}>
                                        {ev.event === 'node_start' && `▶ ${ev.label} — iniciando...`}
                                        {ev.event === 'node_done' && `✓ ${ev.label} — concluído${ev.branch ? ` (${ev.branch})` : ''}`}
                                        {ev.event === 'node_error' && `✗ ${ev.label} — erro: ${ev.error}`}
                                        {ev.event === 'flow_done' && '🏁 Fluxo concluído'}
                                        {ev.event === 'error' && `✗ Erro: ${ev.error}`}
                                    </div>
                                ))}
                                <div ref={logEndRef} />
                            </div>
                        )}

                        {finalOutput && (
                            <div className="flow-config-field">
                                <label className="flow-config-label">Resultado Final</label>
                                <div className="flow-run-output">{finalOutput}</div>
                            </div>
                        )}

                        {running && (
                            <div className="flow-run-status">⚙️ Executando... aguarde.</div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
