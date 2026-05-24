import { useState, useEffect, useCallback, lazy } from 'react';
import { Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import WelcomeContent from './components/WelcomeContent';
import ChatArea from './components/ChatArea';
import ChatInput from './components/ChatInput';
import BatchPanel from './components/BatchPanel';
import CreateAgentDialog from './components/CreateAgentDialog';
import SettingsPanel from './components/SettingsPanel';
import ShareAgentDialog from './components/ShareAgentDialog';
import LoginPage from './components/LoginPage';
import SignupPage from './components/SignupPage';
import ForgotPasswordPage from './components/ForgotPasswordPage';
import ResetPasswordPage from './components/ResetPasswordPage';
import LazyPanel from './components/LazyPanel';
import { AuthProvider, useAuth } from './components/AuthContext';
import { getTemplateStatus } from './services/api';
import { useUIStore, useChatStore } from './store';
import { useMessageSender } from './hooks/useMessageSender';
import { useOcrUpload } from './hooks/useOcrUpload';
import { useBatchOps } from './hooks/useBatchOps';
import { useDeepResearch } from './hooks/useDeepResearch';
import { useJurisprudencia } from './hooks/useJurisprudencia';
import { useCustomAgents } from './hooks/useCustomAgents';
import { useStyleReport } from './hooks/useStyleReport';
import { useCanvas } from './hooks/useCanvas';
import { useChatSession } from './hooks/useChatSession';
import type { Agent, RagStatus } from './types/chat';
import './App.css';

// Painéis pesados — carregados sob demanda
const CanvasEditor = lazy(() => import('./components/CanvasEditor'));
const XRayDashboard = lazy(() => import('./components/XRayDashboard'));
const JurisprudenciaPanel = lazy(() => import('./components/JurisprudenciaPanel'));
const SustentacaoPanel = lazy(() => import('./components/sustentacao/SustentacaoPanel'));
const ModelManagerPanel = lazy(() => import('./components/ModelManagerPanel'));
const AgentBuilderChat = lazy(() => import('./components/AgentBuilderChat'));
const LotePage = lazy(() => import('./components/LotePage'));
const DeepResearchPanel = lazy(() => import('./components/DeepResearchPanel'));
const FlowBuilderPage = lazy(() => import('./components/FlowBuilderPage'));

// ── Route constants ──
const ROUTES = {
    home: '/',
    login: '/login',
    signup: '/signup',
    forgotPassword: '/forgot-password',
    resetPassword: '/reset-password',
    jurisprudencia: '/jurisprudencia',
    sustentacao: '/sustentacao',
    modelos: '/modelos',
    engenheiroPrompts: '/engenheiro-prompts',
    lote: '/lote',
    flows: '/flows',
} as const;

const PANEL_PATHS = new Set<string>([
    ROUTES.jurisprudencia,
    ROUTES.sustentacao,
    ROUTES.modelos,
    ROUTES.engenheiroPrompts,
    ROUTES.lote,
    ROUTES.flows,
]);

function MainApp() {
    const navigate = useNavigate();
    const location = useLocation();
    const currentPath = location.pathname;

    const { sidebarOpen, setSidebarOpen, toggleSidebar: toggleSidebarStore, selectedModel: globalSelectedModel, setSelectedModel } = useUIStore();
    const messages = useChatStore((s) => s.messages);
    const isLoading = useChatStore((s) => s.isLoading);
    const activeAgent = useChatStore((s) => s.activeAgent);
    const uploadedText = useChatStore((s) => s.uploadedText);
    const conversationId = useChatStore((s) => s.conversationId);

    // Local UI state (modals + sustentacao overlay)
    const [showMemoryPanel, setShowMemoryPanel] = useState(false);
    const [showJurisOverlay, setShowJurisOverlay] = useState(false);
    const [ragStatus, setRagStatus] = useState<RagStatus | null>(null);

    // Hooks own their respective flows
    const canvas = useCanvas();
    const ocr = useOcrUpload();
    const sender = useMessageSender({ canvas, ragStatus, setRagStatus });
    const batch = useBatchOps();
    const deepResearch = useDeepResearch();
    const juris = useJurisprudencia();
    const style = useStyleReport();
    const agents = useCustomAgents();
    const session = useChatSession({
        onResetXray: () => batch.setXrayReport(null),
        onResetJurisprudencia: juris.resetJurisprudencia,
        onResetCanvas: () => {
            canvas.setCanvasOpen(false);
            canvas.setCanvasContent('');
        },
        onResetSustentacao: () => {
            if (currentPath === ROUTES.sustentacao) navigate(ROUTES.home);
            setShowJurisOverlay(false);
        },
    });

    // Bootstrap RAG status
    useEffect(() => {
        getTemplateStatus().then(setRagStatus).catch(() => {});
    }, []);

    // Close juris overlay automatically when leaving /sustentacao
    useEffect(() => {
        if (currentPath !== ROUTES.sustentacao) setShowJurisOverlay(false);
    }, [currentPath]);

    const toggleSidebar = useCallback(() => toggleSidebarStore(), [toggleSidebarStore]);
    const closeSidebarMobile = useCallback(() => {
        if (window.innerWidth <= 768) setSidebarOpen(false);
    }, [setSidebarOpen]);

    const goHome = useCallback(() => navigate(ROUTES.home), [navigate]);

    const sidebarHistory = session.chatHistory.length > 0
        ? [{
            label: 'Conversas anteriores',
            items: session.chatHistory.map((c) => ({ id: c.id, title: c.title })),
        }]
        : [];

    const hasMessages = messages.length > 0;
    const showBatchPanel = batch.batchResults.length > 0;
    const isPanelRoute = PANEL_PATHS.has(currentPath);
    const deepResearchOpen = deepResearch.state.active || !!deepResearch.state.dossier || !!deepResearch.state.error;
    const showFullPanel = isPanelRoute || !!batch.xrayReport || deepResearchOpen;

    // ── Content area: either deep research, xray report, lazy panel routes, or chat/welcome ──
    const renderMainContent = () => {
        if (deepResearchOpen) {
            return (
                <LazyPanel label="Deep Research" onReset={deepResearch.reset}>
                    <DeepResearchPanel
                        state={deepResearch.state}
                        onClose={deepResearch.reset}
                        onSendToCanvas={(dossier: string) => {
                            canvas.setCanvasContent(dossier);
                            canvas.setCanvasOpen(true);
                        }}
                    />
                </LazyPanel>
            );
        }
        if (batch.xrayReport && currentPath !== ROUTES.lote) {
            return (
                <LazyPanel label="Raio-X" onReset={() => batch.setXrayReport(null)}>
                    <XRayDashboard
                        report={batch.xrayReport}
                        onClose={() => batch.setXrayReport(null)}
                        onClusterAction={batch.handleClusterAction}
                        onClusterPilotAction={batch.handleClusterPilotAction}
                    />
                </LazyPanel>
            );
        }

        return (
            <Routes>
                <Route
                    path={ROUTES.engenheiroPrompts}
                    element={
                        <LazyPanel label="Engenheiro de Prompts" onReset={goHome}>
                            <AgentBuilderChat
                                onClose={goHome}
                                onPromptReady={agents.handlePromptReady}
                            />
                        </LazyPanel>
                    }
                />
                <Route
                    path={ROUTES.modelos}
                    element={
                        <LazyPanel label="Modelos" onReset={goHome}>
                            <ModelManagerPanel
                                onClose={goHome}
                                onRagStatusChange={setRagStatus}
                            />
                        </LazyPanel>
                    }
                />
                <Route
                    path={ROUTES.jurisprudencia}
                    element={
                        <LazyPanel label="Jurisprudência" onReset={goHome}>
                            <JurisprudenciaPanel onClose={goHome} />
                        </LazyPanel>
                    }
                />
                <Route
                    path={ROUTES.sustentacao}
                    element={
                        <LazyPanel label="Sustentação" onReset={goHome}>
                            <SustentacaoPanel
                                onClose={goHome}
                                onOpenJurisprudencia={() => setShowJurisOverlay(true)}
                            />
                        </LazyPanel>
                    }
                />
                <Route
                    path={ROUTES.lote}
                    element={
                        <LazyPanel label="Análise em Lote" onReset={goHome}>
                            <LotePage
                                onClose={goHome}
                                onPilotMode={batch.handleClusterPilotAction}
                            />
                        </LazyPanel>
                    }
                />
                <Route
                    path={ROUTES.flows}
                    element={
                        <LazyPanel label="Construtor de Fluxos" onReset={goHome}>
                            <FlowBuilderPage onClose={goHome} />
                        </LazyPanel>
                    }
                />
                <Route
                    path="*"
                    element={
                        hasMessages || style.styleAnalyzing || batch.xrayLoading || ocr.ocrProcessing ? (
                            <ChatArea
                                messages={messages}
                                isLoading={isLoading}
                                selectedModel={globalSelectedModel}
                                activeAgent={activeAgent}
                                ocrProcessing={ocr.ocrProcessing}
                                ocrEngineName={ocr.ocrEngineName}
                                ocrProgress={ocr.ocrProgress}
                                styleAnalyzing={style.styleAnalyzing}
                                xrayLoading={batch.xrayLoading}
                                xrayProgress={batch.xrayProgress}
                                onAutoAction={sender.handleAutoReview}
                                onQAReview={sender.handleAutoReview}
                                hasUploadedText={!!uploadedText}
                                jurisResearch={juris.jurisResearch}
                                jurisResearchLoading={juris.jurisResearchLoading}
                                jurisResearchProgress={juris.jurisResearchProgress}
                                onJurisImport={juris.handleJurisImport}
                                onJurisSearch={juris.handleJurisSearch}
                                onPilotReplicate={batch.handlePilotReplicate}
                                onPilotDismiss={batch.handlePilotDismiss}
                                onRetry={sender.handleRetry}
                            />
                        ) : (
                            <WelcomeContent
                                onOpenJurisprudencia={() => navigate(ROUTES.jurisprudencia)}
                                onOpenSustentacao={() => navigate(ROUTES.sustentacao)}
                                onOpenLote={() => navigate(ROUTES.lote)}
                                onOpenModelos={() => navigate(ROUTES.modelos)}
                                onCreateAgent={() => {
                                    agents.setPendingPrompt('');
                                    agents.setShowCreateDialog(true);
                                }}
                                onOpenPromptEngineer={() => navigate(ROUTES.engenheiroPrompts)}
                            />
                        )
                    }
                />
            </Routes>
        );
    };

    return (
        <div className="app-layout">
            <div
                className={`sidebar-backdrop ${sidebarOpen ? 'visible' : ''}`}
                onClick={() => setSidebarOpen(false)}
            />
            <Sidebar
                isOpen={sidebarOpen}
                onToggle={toggleSidebar}
                onCloseMobile={closeSidebarMobile}
                activeAgent={activeAgent}
                onAgentSelect={session.handleAgentSelect}
                onNewChat={() => {
                    session.handleNewChat();
                    if (currentPath !== ROUTES.home) navigate(ROUTES.home);
                }}
                history={sidebarHistory}
                onLoadChat={(id: string) => {
                    session.handleLoadChat(id);
                    if (currentPath !== ROUTES.home) navigate(ROUTES.home);
                }}
                customAgents={agents.customAgents}
                onCreateAgent={() => { agents.setPendingPrompt(''); agents.setShowCreateDialog(true); }}
                onDeleteAgent={agents.handleDeleteAgent}
                onShareAgent={agents.handleShareAgentOpen}
                onOpenMemory={() => setShowMemoryPanel(true)}
                onOpenSustentacao={() => navigate(ROUTES.sustentacao)}
            />

            <div className={`main-panel ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
                <Header
                    onMenuClick={toggleSidebar}
                    isOpen={sidebarOpen}
                    onOpenModelManager={() => navigate(ROUTES.modelos)}
                />

                <div className={`main-content ${showBatchPanel ? 'with-batch' : ''} ${canvas.canvasOpen ? 'with-canvas' : ''} ${showFullPanel ? 'with-full-panel' : ''}`}>
                    <div className={`chat-content-wrapper ${canvas.canvasOpen ? 'canvas-chat-pane' : ''}`}>
                        {renderMainContent()}
                    </div>
                    {canvas.canvasOpen && (
                        <LazyPanel label="Canvas" fallback={<div className="lazy-fallback">Carregando editor...</div>}>
                            <CanvasEditor
                                content={canvas.canvasContent}
                                onClose={canvas.handleCanvasClose}
                                onContentChange={canvas.setCanvasContent}
                                onSelectionChange={canvas.setCanvasSelection}
                                onFocusChat={(selectedText: string) => {
                                    canvas.setCanvasSelection(selectedText);
                                    const ta = canvas.chatTextareaRef.current;
                                    if (ta) {
                                        ta.focus();
                                        ta.placeholder = `Refinar: "${selectedText.slice(0, 40)}..." — digite sua instrução`;
                                    }
                                }}
                                isUpdating={isLoading}
                                conversationId={conversationId}
                            />
                        </LazyPanel>
                    )}
                    {showBatchPanel && (
                        <BatchPanel
                            results={batch.batchResults}
                            selectedIndex={batch.batchSelectedIndex}
                            onSelect={batch.handleBatchSelect}
                            onClose={batch.handleBatchClose}
                        />
                    )}
                </div>

                {!isPanelRoute && !batch.xrayReport && !deepResearchOpen && (
                    <ChatInput
                        onSend={sender.handleSend}
                        onXray={batch.handleXray}
                        onFilesUploaded={ocr.handleFilesUploaded}
                        onStyleReport={style.handleStyleReport}
                        onModelChange={setSelectedModel}
                        onOpenModelManager={() => navigate(ROUTES.modelos)}
                        isLoading={isLoading || batch.xrayLoading || style.styleAnalyzing}
                        ocrProcessing={ocr.ocrProcessing}
                        hasContext={!!(uploadedText || activeAgent)}
                        ragStatus={ragStatus}
                        onRagStatusChange={setRagStatus}
                        activeAgent={activeAgent}
                        canvasOpen={canvas.canvasOpen}
                        canvasSelection={canvas.canvasSelection}
                        onCanvasToggle={canvas.handleCanvasToggle}
                        onDeepResearch={() => deepResearch.start()}
                        chatTextareaRef={canvas.chatTextareaRef}
                    />
                )}
            </div>

            <CreateAgentDialog
                isOpen={agents.showCreateDialog}
                onClose={() => {
                    agents.setShowCreateDialog(false);
                    agents.setPendingPrompt('');
                }}
                onConfirm={async (input: { name: string; prompt: string; color?: string }) => {
                    await agents.handleCreateAgent(input);
                    if (currentPath === ROUTES.engenheiroPrompts) navigate(ROUTES.home);
                }}
                initialPrompt={agents.pendingPrompt}
            />

            <ShareAgentDialog
                isOpen={!!agents.shareAgent}
                onClose={() => agents.setShareAgent(null)}
                onShare={agents.handleShareAgentConfirm}
                agentName={agents.shareAgent?.name || ''}
            />

            <SettingsPanel isOpen={showMemoryPanel} onClose={() => setShowMemoryPanel(false)} />

            {/* Jurisprudência overlay when sustentação is the active route */}
            {currentPath === ROUTES.sustentacao && showJurisOverlay && (
                <div className="sust-jurisprudencia-overlay">
                    <LazyPanel
                        label="Jurisprudência"
                        onReset={() => setShowJurisOverlay(false)}
                    >
                        <JurisprudenciaPanel onClose={() => setShowJurisOverlay(false)} />
                    </LazyPanel>
                </div>
            )}
        </div>
    );
}

function App() {
    return (
        <AuthProvider>
            <AppRoutes />
        </AuthProvider>
    );
}

function AppRoutes() {
    const { user, loading, passwordRecovery } = useAuth();
    const navigate = useNavigate();

    if (loading) {
        return (
            <div className="flex h-screen w-full items-center justify-center bg-background text-primary">
                <div className="animate-spin text-3xl">⚙️</div>
            </div>
        );
    }

    // Password recovery overrides normal routing — user must finish reset before anything else.
    if (passwordRecovery) {
        return (
            <Routes>
                <Route path="*" element={<ResetPasswordPage onDone={() => navigate(ROUTES.login)} />} />
            </Routes>
        );
    }

    if (!user) {
        return (
            <Routes>
                <Route
                    path={ROUTES.login}
                    element={
                        <LoginPage
                            onNavigateSignup={() => navigate(ROUTES.signup)}
                            onNavigateForgot={() => navigate(ROUTES.forgotPassword)}
                        />
                    }
                />
                <Route path={ROUTES.signup} element={<SignupPage onNavigateLogin={() => navigate(ROUTES.login)} />} />
                <Route
                    path={ROUTES.forgotPassword}
                    element={<ForgotPasswordPage onNavigateLogin={() => navigate(ROUTES.login)} />}
                />
                {/* Direct visits to /reset-password without a recovery session land back at login. */}
                <Route path={ROUTES.resetPassword} element={<Navigate to={ROUTES.login} replace />} />
                <Route path="*" element={<Navigate to={ROUTES.login} replace />} />
            </Routes>
        );
    }

    return (
        <Routes>
            <Route path={ROUTES.login} element={<Navigate to={ROUTES.home} replace />} />
            <Route path={ROUTES.signup} element={<Navigate to={ROUTES.home} replace />} />
            <Route path={ROUTES.forgotPassword} element={<Navigate to={ROUTES.home} replace />} />
            <Route path="/*" element={<MainApp />} />
        </Routes>
    );
}

export default App;
