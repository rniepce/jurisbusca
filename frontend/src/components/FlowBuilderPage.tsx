import { useCallback, useEffect, useRef, useState } from 'react';
import {
    ReactFlow,
    addEdge,
    Background,
    Controls,
    useNodesState,
    useEdgesState,
    type Connection,
    type Node,
    type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
    FaPlus, FaFloppyDisk, FaPlay, FaTrash, FaChevronLeft,
    FaListUl, FaEye, FaXmark, FaShapes, FaWandMagicSparkles, FaShareNodes,
    FaClockRotateLeft,
} from 'react-icons/fa6';
import AgentNode from './flow/AgentNode';
import RouterNode from './flow/RouterNode';
import { StartNode, EndNode } from './flow/StartEndNodes';
import NodeConfigPanel from './flow/NodeConfigPanel';
import NodeCatalog from './flow/NodeCatalog';
import FlowTemplates from './flow/FlowTemplates';
import FlowVersionsModal from './flow/FlowVersionsModal';
import {
    SwitchNode, HILNode, DocxNode, JurisNode, ModeloNode, EstiloNode,
    ExtractorNode, SubflowNode,
} from './flow/SpecialNodes';
import {
    listFlows, createFlow, getFlow, updateFlow, deleteFlow, previewFlow,
    extractFlowFiles, shareFlow,
    type FlowSummary, type FlowConfig,
} from '../services/api';
import './FlowBuilderPage.css';

const NODE_TYPES = {
    start: StartNode,
    end: EndNode,
    agent: AgentNode,
    router: RouterNode,
    switch: SwitchNode,
    hil: HILNode,
    docx: DocxNode,
    juris: JurisNode,
    modelo: ModeloNode,
    estilo: EstiloNode,
    extractor: ExtractorNode,
    subflow: SubflowNode,
};

const AGENT_COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4'];

let nodeCounter = 1;

function makeNode(type: string, position = { x: 200, y: 200 }, dataOverrides: Record<string, string> = {}): Node {
    const id = `${type}_${Date.now()}_${nodeCounter++}`;
    const defaults: Record<string, Record<string, string>> = {
        agent: { label: 'Agente', model: 'gpt-5.3-chat', prompt: '', output_var: `saida_${nodeCounter}`, knowledge: '', knowledge_files: '' },
        router: { label: 'Roteador', condition: '' },
        switch: { label: 'Classificador', categories: 'Civil|Penal|Tributário', model: 'gpt-5.4-mini' },
        hil: { label: 'Aprovação Humana', question: 'Por favor, revise antes de continuar.' },
        docx: { label: 'Gerar DOCX', filename: 'documento.docx' },
        juris: { label: 'Pesquisar Jurisprudência', query: '', top_k: '5' },
        modelo: { label: 'Buscar Modelo', query: '', top_k: '3' },
        estilo: { label: 'Aplicar Estilo' },
        extractor: {
            label: 'Extrator JSON',
            fields: 'numero_processo:string:Número CNJ do processo|valor_causa:number:Valor em reais|partes:array:Lista das partes envolvidas',
            model: 'gpt-5.4-mini',
        },
        subflow: { label: 'Sub-fluxo', flow_id: '', flow_name: '' },
        start: {},
        end: {},
    };
    return {
        id, type, position,
        data: { ...(defaults[type] ?? {}), output_var: `saida_${nodeCounter}`, ...dataOverrides },
    };
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

export default function FlowBuilderPage({ onClose, customAgents = [] }: { onClose?: () => void; customAgents?: { id: string; name: string; prompt: string; color?: string }[] }) {
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);
    const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
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
    const [showCatalog, setShowCatalog] = useState(false);
    const [showTemplates, setShowTemplates] = useState(false);
    const [showVersions, setShowVersions] = useState(false);
    const [flows, setFlows] = useState<FlowSummary[]>([]);
    const [loadingFlows, setLoadingFlows] = useState(false);
    const [finalOutput, setFinalOutput] = useState('');
    const [dragOver, setDragOver] = useState(false);
    const [pdfDropping, setPdfDropping] = useState(false);
    const [pdfFiles, setPdfFiles] = useState<string[]>([]);
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
        // start/end podem ser selecionados (para deletar via toolbar) mas
        // não abrem painel de configuração — não há nada a configurar.
        setSelectedNode(node);
        setSelectedEdgeId(null);
    }, []);

    const onEdgeClick = useCallback((_: React.MouseEvent, edge: Edge) => {
        setSelectedEdgeId(edge.id);
        setSelectedNode(null);
    }, []);

    const onPaneClick = useCallback(() => {
        setSelectedNode(null);
        setSelectedEdgeId(null);
    }, []);

    const deleteSelectedEdge = useCallback(() => {
        if (!selectedEdgeId) return;
        setEdges(es => es.filter(e => e.id !== selectedEdgeId));
        setSelectedEdgeId(null);
    }, [selectedEdgeId, setEdges]);

    const handleNodeDataChange = useCallback((nodeId: string, data: Record<string, string>) => {
        setNodes(ns => ns.map(n => n.id === nodeId ? { ...n, data } : n));
        setSelectedNode(prev => prev?.id === nodeId ? { ...prev, data } : prev);
    }, [setNodes]);

    const addNode = (type: string, dataOverrides: Record<string, string> = {}) => {
        const offset = nodes.length * 25;
        const n = makeNode(type, { x: 300 + offset, y: 180 + offset }, dataOverrides);
        setNodes(ns => [...ns, n]);
    };

    // Construo a lista de labels disponíveis (saídas dos nós) para autocompletar prompts
    const availableLabels = nodes
        .filter(n => n.type !== 'start' && n.type !== 'end')
        .map(n => (n.data as Record<string, string>).label || n.id)
        .filter(Boolean);

    // ── Drag-and-drop de PDF/DOCX no canvas → vira input do Início ──
    const handleCanvasDragOver = (e: React.DragEvent) => {
        if (e.dataTransfer.types.includes('Files')) {
            e.preventDefault();
            setDragOver(true);
        }
    };
    const handleCanvasDragLeave = (e: React.DragEvent) => {
        // só desativa se sair do wrapper (não de filho)
        if (e.currentTarget === e.target) setDragOver(false);
    };
    const handleCanvasDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const files = Array.from(e.dataTransfer.files || []);
        if (files.length === 0) return;
        const acceptable = files.filter(f => /\.(pdf|docx?|txt|md)$/i.test(f.name));
        if (acceptable.length === 0) {
            alert('Solte arquivos PDF, DOCX, TXT ou Markdown.');
            return;
        }
        setPdfDropping(true);
        try {
            const result = await extractFlowFiles(acceptable);
            setInputText(prev => prev ? `${prev}\n\n${result.text}` : result.text);
            setPdfFiles(prev => [...prev, ...result.files]);
        } catch (err) {
            alert(`Erro ao processar arquivo: ${err}`);
        } finally {
            setPdfDropping(false);
        }
    };

    const loadTemplateFlow = (template: { name: string; color: string; desc: string; config: FlowConfig }) => {
        setFlowId(null);
        setFlowName(template.name);
        setFlowDescription(template.desc);
        setFlowColor(template.color);
        const loadedNodes = template.config.nodes.map(n => ({
            id: n.id,
            type: n.type,
            position: n.position,
            data: { ...n.data, status: 'idle' },
        }));
        setNodes(loadedNodes);
        setEdges(template.config.edges.map(e => ({
            id: e.id,
            source: e.source,
            target: e.target,
            label: e.label || '',
            sourceHandle: e.sourceHandle || null,
            style: { stroke: '#94a3b8', strokeWidth: 2 },
        })));
        setSelectedNode(null);
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

    const handleShareFlow = async (id: string, name: string) => {
        const email = window.prompt(`Compartilhar "${name}" com qual e-mail?`, '');
        if (!email || !email.includes('@')) return;
        try {
            await shareFlow(id, email.trim());
            alert(`✓ Fluxo compartilhado com ${email.trim()}.\n\nA pessoa verá esse fluxo como um agente orquestrador na barra lateral assim que recarregar.`);
        } catch (err) {
            alert(`Erro ao compartilhar: ${err}`);
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

                <button onClick={() => setShowCatalog(true)} className="flow-btn flow-btn-add agent">
                    <FaShapes size={12} /> Adicionar Nó
                </button>
                <button onClick={() => setShowTemplates(true)} className="flow-btn flow-btn-add modelo">
                    <FaWandMagicSparkles size={12} /> Templates
                </button>

                {selectedNode && (
                    <>
                        <div className="flow-toolbar-divider" />
                        <button onClick={deleteSelectedNode} className="flow-btn flow-btn-danger">
                            <FaTrash size={11} /> Apagar nó
                        </button>
                    </>
                )}
                {selectedEdgeId && (
                    <>
                        <div className="flow-toolbar-divider" />
                        <button onClick={deleteSelectedEdge} className="flow-btn flow-btn-danger">
                            <FaTrash size={11} /> Apagar conexão
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
                {flowId && (
                    <button onClick={() => setShowVersions(true)} className="flow-btn flow-btn-ghost" title="Histórico de versões">
                        <FaClockRotateLeft size={11} /> Histórico
                    </button>
                )}
                <button onClick={handleSaveClick} disabled={saving} className="flow-btn flow-btn-primary">
                    <FaFloppyDisk size={11} /> {saving ? 'Salvando...' : flowId ? 'Salvar' : 'Salvar como Agente'}
                </button>
                {flowId && (
                    <button onClick={handleRunSavedFlow} disabled={running} className="flow-btn flow-btn-success">
                        <FaPlay size={11} /> Executar
                    </button>
                )}
            </div>

            {/* Canvas (suporta drag-and-drop de PDF/DOCX) */}
            <div
                className={`flow-canvas-wrap ${dragOver ? 'drag-over' : ''}`}
                onDragOver={handleCanvasDragOver}
                onDragLeave={handleCanvasDragLeave}
                onDrop={handleCanvasDrop}
            >
                <ReactFlow
                    nodes={nodes}
                    edges={edges.map(e => e.id === selectedEdgeId
                        ? { ...e, style: { ...(e.style || {}), stroke: '#ef4444', strokeWidth: 3 }, animated: true }
                        : { ...e, style: { stroke: '#94a3b8', strokeWidth: 2 }, animated: false })}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onNodeClick={onNodeClick}
                    onEdgeClick={onEdgeClick}
                    onPaneClick={onPaneClick}
                    nodeTypes={NODE_TYPES}
                    fitView
                    colorMode="light"
                    deleteKeyCode={['Backspace', 'Delete']}
                    proOptions={{ hideAttribution: false }}
                >
                    <Background color="#475569" gap={28} size={1.5} />
                    <Controls />
                </ReactFlow>

                {selectedNode && selectedNode.type !== 'start' && selectedNode.type !== 'end' && (
                    <NodeConfigPanel
                        node={selectedNode as { id: string; type: string; data: Record<string, string> }}
                        onChange={handleNodeDataChange}
                        onClose={() => setSelectedNode(null)}
                        availableLabels={availableLabels.filter(l => l !== (selectedNode.data as Record<string, string>).label)}
                        customAgents={customAgents}
                    />
                )}

                {/* Overlay de drag-and-drop */}
                {dragOver && (
                    <div className="flow-canvas-dropzone">
                        <div className="flow-canvas-dropzone-content">
                            📄 Solte aqui para usar como entrada do fluxo
                            <span>PDF, DOCX, TXT, MD</span>
                        </div>
                    </div>
                )}
                {pdfDropping && (
                    <div className="flow-canvas-dropzone uploading">
                        <div className="flow-canvas-dropzone-content">⏳ Extraindo texto...</div>
                    </div>
                )}

                {/* Chip dos arquivos prontos */}
                {pdfFiles.length > 0 && (
                    <div className="flow-canvas-pdf-chip">
                        📎 {pdfFiles.length} arquivo{pdfFiles.length > 1 ? 's' : ''} pronto{pdfFiles.length > 1 ? 's' : ''}: {pdfFiles.slice(-2).join(', ')}{pdfFiles.length > 2 ? '...' : ''}
                        <button onClick={() => { setPdfFiles([]); setInputText(''); }} title="Limpar"><FaXmark size={10} /></button>
                    </div>
                )}
            </div>

            {/* Catálogo de nós */}
            <NodeCatalog
                open={showCatalog}
                onClose={() => setShowCatalog(false)}
                onAdd={(item) => addNode(item.type, item.defaults)}
            />

            {/* Templates de fluxo */}
            <FlowTemplates
                open={showTemplates}
                onClose={() => setShowTemplates(false)}
                onPick={loadTemplateFlow}
            />

            {/* Histórico de versões */}
            <FlowVersionsModal
                open={showVersions}
                flowId={flowId}
                onClose={() => setShowVersions(false)}
                onRestored={() => { if (flowId) void handleOpenFlow(flowId); }}
            />

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
                                    <button onClick={() => handleShareFlow(f.id, f.name)} className="flow-btn flow-btn-ghost" title="Compartilhar">
                                        <FaShareNodes size={10} />
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
                                {runLog.map((ev, i) => {
                                    const evAny = ev as RunEvent & { node_ids?: string[]; labels?: string[] };
                                    return (
                                    <div key={i} className={`flow-run-log-line ${ev.event}`}>
                                        {ev.event === 'parallel_start' && `⫴ Executando em paralelo: ${(evAny.labels || []).join(', ')}`}
                                        {ev.event === 'parallel_done' && `⫴ Etapa paralela concluída`}
                                        {ev.event === 'node_start' && `▶ ${ev.label} — iniciando...`}
                                        {ev.event === 'node_done' && `✓ ${ev.label} — concluído${ev.branch ? ` (${ev.branch})` : ''}`}
                                        {ev.event === 'node_error' && `✗ ${ev.label} — erro: ${ev.error}`}
                                        {ev.event === 'human_required' && `⏸ ${ev.label} — aguardando aprovação humana (no chat)`}
                                        {ev.event === 'flow_done' && '🏁 Fluxo concluído'}
                                        {ev.event === 'error' && `✗ Erro: ${ev.error}`}
                                    </div>
                                );})}
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
