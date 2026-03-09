import React, { useState, useCallback, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import WelcomeContent from './components/WelcomeContent';
import ChatArea from './components/ChatArea';
import ChatInput from './components/ChatInput';
import XRayDashboard from './components/XRayDashboard';
import BatchPanel from './components/BatchPanel';
import JurisprudenciaPanel from './components/JurisprudenciaPanel';
import ModelManagerPanel from './components/ModelManagerPanel';
import AgentBuilderChat from './components/AgentBuilderChat';
import CreateAgentDialog from './components/CreateAgentDialog';
import ShareAgentDialog from './components/ShareAgentDialog';
import LoginPage from './components/LoginPage';
import SignupPage from './components/SignupPage';
import { AuthProvider, useAuth } from './components/AuthContext';
import { sendMessage, uploadFile, uploadBatchXray, generateStyleReport, getTemplateStatus, analyzeCluster, uploadTemplates, triggerJurisprudenciaResearch, pollJurisprudenciaResearch, getCustomAgents, createCustomAgent, deleteCustomAgent, shareCustomAgent } from './services/api';
import agentDefinitions from './config/agents';
import './App.css';

function MainApp() {
  const isMobile = typeof window !== 'undefined' && window.innerWidth <= 768;
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeAgent, setActiveAgent] = useState(agentDefinitions[0]);
  const [conversationId, setConversationId] = useState(null);
  const [uploadedText, setUploadedText] = useState(null);
  const [xrayReport, setXrayReport] = useState(null);
  const [xrayLoading, setXrayLoading] = useState(false);
  const [xrayProgress, setXrayProgress] = useState('');
  const [xrayTextCache, setXrayTextCache] = useState({});
  const [ocrProcessing, setOcrProcessing] = useState(false);
  const [ocrEngineName, setOcrEngineName] = useState('none');
  const [ocrProgress, setOcrProgress] = useState({ progress: '', percent: 0 });
  const [styleAnalyzing, setStyleAnalyzing] = useState(false);
  const [styleDossier, setStyleDossier] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [ragStatus, setRagStatus] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [batchSelectedIndex, setBatchSelectedIndex] = useState(null);
  const [globalSelectedModel, setGlobalSelectedModel] = useState({ id: agentDefinitions[0]?.engineId || 'v1', name: agentDefinitions[0]?.name || 'Gabinete 2.0', color: agentDefinitions[0]?.color || '#10B981', llm: 'gpt-5.2-chat' });
  const [showJurisprudencia, setShowJurisprudencia] = useState(false);
  const [showModelManager, setShowModelManager] = useState(false);
  // Jurisprudence toggle state — always enabled
  // V0.5 jurisprudence research state
  const [jurisResearch, setJurisResearch] = useState(null);
  const [jurisResearchLoading, setJurisResearchLoading] = useState(false);
  const [jurisResearchProgress, setJurisResearchProgress] = useState('');
  const [jurisContext, setJurisContext] = useState(''); // accumulated imported jurisprudence for LLM
  // Custom agents state
  const [customAgents, setCustomAgents] = useState([]);
  const [showAgentBuilder, setShowAgentBuilder] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [pendingPrompt, setPendingPrompt] = useState('');
  const [shareAgent, setShareAgent] = useState(null); // agent being shared



  // Fetch template/RAG status and custom agents on mount
  useEffect(() => {
    getTemplateStatus().then(setRagStatus).catch(() => { });
    getCustomAgents().then((data) => setCustomAgents(data.agents || [])).catch(() => { });
  }, []);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);
  const closeSidebarMobile = () => { if (window.innerWidth <= 768) setSidebarOpen(false); };

  // ── Send message handler ────────────────────────────────────────────
  const handleSend = useCallback(async (message, selectedModel, files, ocrEngine, templateFiles, useRag = false, { hideUserBubble = false, overridePrompt = null } = {}) => {
    // Use a default prompt if user sends empty message but has context
    const userTyped = message.trim();
    const effectiveMessage = userTyped ||
      (uploadedText && activeAgent ? 'Analise o documento anexado conforme as instruções do agente.' :
        uploadedText ? 'Analise o documento anexado.' : '');
    if (!effectiveMessage) return;

    // 1. Add user message immediately (hide auto-generated messages)
    if (userTyped && !hideUserBubble) {
      const userMsg = { role: 'user', content: userTyped };
      setMessages((prev) => [...prev, userMsg]);
    }
    setIsLoading(true);

    try {
      // ── Auto-index templates in ChromaDB if not yet indexed ──
      if (templateFiles && templateFiles.length > 0 && (!ragStatus || ragStatus.indexed_chunks === 0)) {
        try {
          const indexResult = await uploadTemplates(templateFiles);
          setRagStatus({ indexed_chunks: indexResult.indexed_chunks, has_dossier: indexResult.has_dossier });
        } catch (err) {
          console.warn('Auto-indexing templates failed:', err.message);
        }
      }

      // ── Auto-generate style dossier in background (no chat display) ──
      // The dossier is used internally for styleDossier state (injected into LLM calls).
      // Users can view it explicitly via "Relatório de Estilo" button.
      let currentStyleDossier = styleDossier;
      if (templateFiles && templateFiles.length > 0 && !currentStyleDossier) {
        try {
          const styleResult = await generateStyleReport(templateFiles);
          if (styleResult.cloning_prompt) {
            currentStyleDossier = styleResult.cloning_prompt;
            setStyleDossier(styleResult.cloning_prompt);
          }
        } catch (styleErr) {
          console.warn('Auto style report failed:', styleErr.message);
        }
      }

      // Files are already processed by handleFilesUploaded (auto-OCR),
      // so we just use the uploadedText that was populated earlier.

      // 2. Load agent prompt — use overridePrompt if provided (e.g. auto-review QA)
      let agentPrompt = overridePrompt || null;
      const engineId = selectedModel.id;

      if (!agentPrompt && activeAgent?.prompt && !activeAgent?.promptModule) {
        // Custom agent — use stored prompt directly
        agentPrompt = activeAgent.prompt;
      } else if (activeAgent?.promptModule) {
        // Agent explicitly selected from sidebar — load its prompt
        try {
          const mod = await activeAgent.promptModule();
          agentPrompt = mod.default || null;
        } catch {
          console.warn('Could not load agent prompt');
        }
      } else if (['v0', 'v1', 'v2'].includes(engineId)) {
        // No agent selected but engine known — auto-load unified prompt
        try {
          const mod = await import('./prompts/gabineteCivelV2.js');
          agentPrompt = mod.default || null;
        } catch {
          console.warn('Could not auto-load Gabinete prompt');
        }
      }

      // 3. Call the backend
      const result = await sendMessage({
        message: effectiveMessage,
        model: selectedModel.id,
        llm: selectedModel.llm || null,
        agentPrompt,
        conversationId,
        uploadedText,
        styleDossier: currentStyleDossier,
        useRag: useRag,
        jurisprudenceContext: jurisContext || null,
      });

      setConversationId(result.conversation_id);

      // 4. Add assistant response — ensure content is always a string
      let rawResponse = result.response;
      if (typeof rawResponse === 'object' && rawResponse !== null) {
        rawResponse = rawResponse.text || rawResponse.content || rawResponse.output || JSON.stringify(rawResponse);
      }

      const assistantMsg = {
        role: 'assistant',
        content: typeof rawResponse === 'string' ? rawResponse : String(rawResponse),
        model: result.model,
        v2Sections: result.v2_sections || null,
        modelContext: result.model_context || null,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg = {
        role: 'assistant',
        content: `⚠️ **Erro:** ${err.message}`,
        model: 'erro',
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [activeAgent, conversationId, uploadedText, styleDossier, ragStatus, jurisContext]);

  // ── V0.5: Manually trigger jurisprudence research ─────────────────
  const handleJurisSearch = useCallback(async () => {
    if (!uploadedText) return;
    setJurisResearchLoading(true);
    setJurisResearch(null);
    setJurisResearchProgress('Extraindo temas jurídicos do processo...');

    try {
      // Get the latest assistant message for better theme extraction
      const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant' && m.content?.length > 100);
      const analysisText = lastAssistant?.content || '';

      const { task_id } = await triggerJurisprudenciaResearch(uploadedText, analysisText);
      _pollJurisResearch(task_id);
    } catch (err) {
      console.error('Juris research trigger failed:', err);
      setJurisResearchLoading(false);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ **Erro na pesquisa jurisprudencial:** ${err.message}`,
        model: 'erro',
      }]);
    }
  }, [uploadedText, messages]);

  // ── V0.5: Poll jurisprudence research task ──────────────────────
  const _pollJurisResearch = async (taskId) => {
    const POLL_INTERVAL = 2500;
    const MAX_POLLS = 60; // ~2.5 min max

    for (let i = 0; i < MAX_POLLS; i++) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL));

      const data = await pollJurisprudenciaResearch(taskId);
      if (!data) continue;

      if (data.progress) setJurisResearchProgress(data.progress);

      if (data.status === 'done') {
        setJurisResearch(data.result);
        setJurisResearchLoading(false);
        return;
      }
      if (data.status === 'error') {
        console.error('Jurisprudence research failed:', data.error);
        setJurisResearchLoading(false);
        return;
      }
    }
    // Timeout
    setJurisResearchLoading(false);
  };

  // ── X-Ray handler (batch clustering) ────────────────────────────────
  const handleXray = useCallback(async (files) => {
    if (files.length < 2) return;
    setXrayLoading(true);
    setXrayReport(null);
    setXrayProgress('Enviando processos...');

    try {
      const result = await uploadBatchXray(files, (progress) => {
        setXrayProgress(progress);
      });
      setXrayReport(result.report);
      setXrayTextCache(result.text_cache || {});
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `⚠️ **Erro no Raio-X:** ${err.message}`, model: 'erro' },
      ]);
    } finally {
      setXrayLoading(false);
      setXrayProgress('');
    }
  }, []);

  // ── Cluster action: analyze all processes in a cluster individually ──
  const handleClusterAction = useCallback(async (cluster) => {
    const filenames = cluster.arquivos || [];
    if (filenames.length === 0) return;

    // Build process list from text cache
    const processes = filenames
      .filter((fname) => xrayTextCache[fname])
      .map((fname) => ({ filename: fname, text: xrayTextCache[fname] }));

    if (processes.length === 0) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: '⚠️ Textos dos processos não encontrados no cache. Tente refazer o Raio-X.', model: 'erro' },
      ]);
      return;
    }

    // Switch to chat view and show progress
    setXrayReport(null);
    setIsLoading(true);
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content: `⚡ **Processando ${processes.length} processo(s) do grupo "${cluster.nome}" em paralelo...**\n\nAguarde, cada processo está sendo analisado individualmente.`,
        model: 'sistema',
      },
    ]);

    try {
      // Usa o globalSelectedModel que veio do ChatInput (incluindo engine e llm)
      const result = await analyzeCluster(processes, activeAgent?.prompt || '', globalSelectedModel.id, globalSelectedModel.llm);

      // Add each individual result as a separate message
      const resultMessages = result.results.map((r) => {
        let raw = r.response;
        if (typeof raw === 'object' && raw !== null) {
          raw = raw.text || raw.content || raw.output || JSON.stringify(raw);
        }
        const safeText = typeof raw === 'string' ? raw : String(raw);

        return {
          role: 'assistant',
          content: r.status === 'ok'
            ? `## 📄 ${r.filename}\n\n${safeText}`
            : `## ⚠️ ${r.filename}\n\n${safeText}`,
          model: r.model,
        };
      });

      // Summary header
      const summary = {
        role: 'assistant',
        content: `✅ **Lote concluído:** ${result.ok_count}/${result.total} minutas geradas com sucesso.`,
        model: 'sistema',
      };

      setMessages((prev) => [...prev, summary, ...resultMessages]);

      // Populate batch panel with results
      setBatchResults(result.results);
      setBatchSelectedIndex(null);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `⚠️ **Erro na análise em lote:** ${err.message}`, model: 'erro' },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [xrayTextCache, activeAgent]);

  // ── When user selects a batch card, show its content in chat ────────
  const handleBatchSelect = useCallback((index) => {
    setBatchSelectedIndex(index);
    const res = batchResults[index];
    if (!res) return;

    let raw = res.response;
    if (typeof raw === 'object' && raw !== null) {
      raw = raw.text || raw.content || raw.output || JSON.stringify(raw);
    }
    const safeText = typeof raw === 'string' ? raw : String(raw);

    const content = res.status === 'ok'
      ? `## 📄 ${res.filename}\n\n${safeText}`
      : `## ⚠️ ${res.filename}\n\n${safeText}`;

    // Replace messages with just this result
    setMessages([{
      role: 'assistant',
      content,
      model: res.model,
    }]);
  }, [batchResults]);

  const handleBatchClose = useCallback(() => {
    setBatchResults([]);
    setBatchSelectedIndex(null);
  }, []);

  // ── Files uploaded → run OCR immediately ─────────────────────────────
  const handleFilesUploaded = useCallback(async (files, ocrEngine, compress = true) => {
    if (files.length === 0) return [];
    // Show processing animation for all uploads (OCR or plain reading)
    setOcrProcessing(true);
    setOcrEngineName(ocrEngine);
    setOcrProgress({ progress: '📤 Enviando arquivo...', percent: 5 });

    try {
      // Upload files sequentially (not parallel) to avoid overwhelming backend
      const results = [];
      for (const f of files) {
        const result = await uploadFile(f, ocrEngine, compress, true, (p) => setOcrProgress(p));
        results.push(result);
      }

      // Store the extracted text for future chat messages
      const newText = results.map((r) => r.text).join('\n\n---\n\n');
      setUploadedText((prev) => prev ? prev + '\n\n---\n\n' + newText : newText);

      // Inject OCR preview messages into the chat
      const ocrMessages = results.map((r) => ({
        role: 'ocr',
        filename: r.filename,
        text: r.text,
        engine: ocrEngine,
        charCount: r.char_count,
      }));
      setMessages((prev) => [...prev, ...ocrMessages]);

      return results; // Return results so ChatInput can check rag_available
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `⚠️ **Erro no OCR:** ${err.message}`, model: 'erro' },
      ]);
      return [];
    } finally {
      setOcrProcessing(false);
    }
  }, []);

  // ── Agent selection handler ─────────────────────────────────────────
  const handleAgentSelect = useCallback((agent) => {
    // Toggle: if the clicked agent is already active, deactivate it
    if (activeAgent?.id === agent.id) {
      setActiveAgent(null);
      // Opcional: remover a mensagem de ativação ao desativar
      setMessages((prev) => prev.filter((m) => m.role !== 'agent-activation'));
      return;
    }

    setActiveAgent(agent);
    setShowAgentBuilder(false); // Close builder if open
    // Sync the global model with the agent's engine
    setGlobalSelectedModel((prev) => ({
      ...prev,
      id: agent.engineId || 'v0',
      name: agent.name,
      color: agent.color || '#8B5CF6',
    }));
    // For custom agents, use their prompt as the promptModule
    const isCustom = !agent.engineId && agent.prompt;
    const activationMsg = {
      role: 'agent-activation',
      agentName: agent.name,
      agentDesc: agent.desc || (isCustom ? 'Agente personalizado' : ''),
      agentColor: agent.color || '#8B5CF6',
      agentIcon: agent.icon || 'FaRobot',
      autoAction: agent.autoAction || null,
    };

    // Filtra qualquer mensagem de ativação anterior e adiciona a nova
    setMessages((prev) => [
      ...prev.filter((m) => m.role !== 'agent-activation'),
      activationMsg,
    ]);
  }, [activeAgent]);

  // ── Custom Agent CRUD handlers ──────────────────────────────────────
  const handleOpenAgentBuilder = useCallback(() => {
    setShowAgentBuilder(true);
  }, []);

  const handlePromptReady = useCallback((prompt) => {
    setPendingPrompt(prompt);
    setShowCreateDialog(true);
  }, []);

  const handleCreateAgent = useCallback(async ({ name, prompt, color }) => {
    try {
      const created = await createCustomAgent({ name, prompt, color });
      setCustomAgents((prev) => [created, ...prev]);
      setShowCreateDialog(false);
      setShowAgentBuilder(false);
      setPendingPrompt('');
    } catch (err) {
      console.error('Failed to create agent:', err);
    }
  }, []);

  const handleDeleteAgent = useCallback(async (agentId) => {
    try {
      await deleteCustomAgent(agentId);
      setCustomAgents((prev) => prev.filter((a) => a.id !== agentId));
      // If the deleted agent was active, deactivate it
      if (activeAgent?.id === agentId) setActiveAgent(null);
    } catch (err) {
      console.error('Failed to delete agent:', err);
    }
  }, [activeAgent]);

  const handleShareAgentOpen = useCallback((agent) => {
    setShareAgent(agent);
  }, []);

  const handleShareAgentConfirm = useCallback(async (email) => {
    if (!shareAgent) return;
    await shareCustomAgent(shareAgent.id, email);
    setShareAgent(null);
  }, [shareAgent]);

  // ── Auto-review handler (Revisor QA) ────────────────────────────────
  // Makes an ISOLATED API call (no conversation history) to avoid token limit overflow.
  // Only sends: QA prompt + OCR process text + minuta extracted from chat.
  const handleAutoReview = useCallback(async (activationMsg) => {
    // 1. Find the last substantial assistant message (the minuta)
    const assistantMsgs = messages.filter(
      (m) => m.role === 'assistant' && m.content && m.content.length > 100
    );
    const lastMinuta = assistantMsgs.length > 0 ? assistantMsgs[assistantMsgs.length - 1] : null;

    if (!lastMinuta) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: '🛑 **Nenhuma minuta encontrada no chat.** Primeiro, use o agente Gabinete para gerar uma minuta. Depois, ative o Revisor (QA) para auditá-la.',
          model: 'sistema',
        },
      ]);
      return;
    }

    if (!uploadedText) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: '🛑 **Nenhum processo carregado.** Faça o upload do processo (PDF/DOCX) para que o Revisor possa cruzar os fatos com a minuta.',
          model: 'sistema',
        },
      ]);
      return;
    }

    // 2. Load the QA prompt
    let qaPrompt = null;
    try {
      const mod = await import('./prompts/auditorQA.js');
      qaPrompt = mod.default || null;
    } catch {
      console.warn('Could not load auditor QA prompt');
    }

    // 3. Build the audit message with ONLY process data + minuta (sandwich format)
    const auditMessage = [
      'Execute a auditoria de conformidade cruzando os textos abaixo.',
      '',
      '[MINUTA PROPOSTA]:',
      lastMinuta.content,
      '',
      'Execute a auditoria de conformidade cruzando a minuta acima com os dados do processo. Gere o Dashboard de Conformidade completo.',
    ].join('\n');

    // 4. Make an ISOLATED API call — no conversationId (fresh conversation, no history)
    // The uploadedText goes via uploaded_text param (backend injects it as system prompt context)
    // The minuta goes in the message body
    setIsLoading(true);
    try {
      const result = await sendMessage({
        message: auditMessage,
        model: globalSelectedModel.id,
        llm: globalSelectedModel.llm || null,
        agentPrompt: qaPrompt,
        conversationId: null, // ← ISOLATED: no history, fresh call
        uploadedText: uploadedText,
        styleDossier: null,
        useRag: false,
        jurisprudenceContext: null,
      });

      // 5. Add the QA result to chat
      let rawResponse = result.response;
      if (typeof rawResponse === 'object' && rawResponse !== null) {
        rawResponse = rawResponse.text || rawResponse.content || rawResponse.output || JSON.stringify(rawResponse);
      }
      const assistantMsg = {
        role: 'assistant',
        content: typeof rawResponse === 'string' ? rawResponse : String(rawResponse),
        model: result.model,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `⚠️ **Erro na auditoria QA:** ${err.message}`, model: 'erro' },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [messages, uploadedText, globalSelectedModel]);



  // ── New chat handler — saves current conversation to history ────────
  const handleNewChat = useCallback(() => {
    // Save current conversation to history (only if it has real messages)
    const realMessages = messages.filter((m) => m.role === 'user' || m.role === 'assistant');
    if (realMessages.length > 0) {
      const firstUserMsg = realMessages.find((m) => m.role === 'user');
      const title = firstUserMsg
        ? firstUserMsg.content.slice(0, 60) + (firstUserMsg.content.length > 60 ? '…' : '')
        : 'Conversa sem título';

      setChatHistory((prev) => [
        {
          id: conversationId || Date.now().toString(),
          title,
          messages: [...messages],
          agent: activeAgent,
          timestamp: new Date(),
        },
        ...prev,
      ]);
    }


    // Reset everything for new chat
    setMessages([]);
    setActiveAgent(null);
    setConversationId(null);
    setUploadedText(null);
    setXrayReport(null);
    setJurisResearch(null);
    setJurisResearchLoading(false);
    setJurisContext('');
  }, [messages, activeAgent, conversationId]);

  // ── Load chat from history ──────────────────────────────────────────
  const handleLoadChat = useCallback((chatId) => {
    const chat = chatHistory.find((c) => c.id === chatId);
    if (!chat) return;

    setMessages(chat.messages);
    setActiveAgent(chat.agent);
    setConversationId(chat.id);
    setUploadedText(null);
    setXrayReport(null);
  }, [chatHistory]);

  // ── Style report handler — calls backend API ────────────────────────
  const handleStyleReport = useCallback(async (templateFiles) => {
    if (!templateFiles || templateFiles.length === 0) return;

    setStyleAnalyzing(true);

    try {
      const result = await generateStyleReport(templateFiles);

      // Display the full dossier as an assistant message
      const dossierContent = result.full_response || result.dossier || 'Dossiê gerado sem conteúdo.';
      const assistantMsg = {
        role: 'assistant',
        content: `🎨 **Dossiê de Identidade Decisional** (${result.file_count} modelo${result.file_count > 1 ? 's' : ''} analisado${result.file_count > 1 ? 's' : ''})\n\n${dossierContent}`,
        model: 'gemini-flash',
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Store the cloning prompt for subsequent LLM analysis
      if (result.cloning_prompt) {
        setStyleDossier(result.cloning_prompt);
      }
    } catch (err) {
      const errorMsg = {
        role: 'assistant',
        content: `⚠️ **Erro no Relatório de Estilo:** ${err.message}`,
        model: 'erro',
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setStyleAnalyzing(false);
    }
  }, []);

  const hasMessages = messages.length > 0;

  // Build sidebar history format
  const sidebarHistory = chatHistory.length > 0
    ? [{
      label: 'Conversas anteriores',
      items: chatHistory.map((c) => ({ id: c.id, title: c.title })),
    }]
    : [];

  // Determine what to show in main content
  const renderContent = () => {
    if (showAgentBuilder) {
      return (
        <AgentBuilderChat
          onClose={() => setShowAgentBuilder(false)}
          onPromptReady={handlePromptReady}
        />
      );
    }
    if (showModelManager) {
      return (
        <ModelManagerPanel
          onClose={() => setShowModelManager(false)}
          onRagStatusChange={setRagStatus}
        />
      );
    }
    if (showJurisprudencia) {
      return (
        <JurisprudenciaPanel
          onClose={() => setShowJurisprudencia(false)}
        />
      );
    }
    if (xrayReport) {
      return (
        <XRayDashboard
          report={xrayReport}
          onClose={() => setXrayReport(null)}
          onClusterAction={handleClusterAction}
        />
      );
    }
    if (hasMessages || styleAnalyzing || xrayLoading || ocrProcessing) {
      return (
        <ChatArea
          messages={messages}
          isLoading={isLoading}
          selectedModel={globalSelectedModel}
          activeAgent={activeAgent}
          ocrProcessing={ocrProcessing}
          ocrEngineName={ocrEngineName}
          ocrProgress={ocrProgress}
          styleAnalyzing={styleAnalyzing}
          xrayLoading={xrayLoading}
          xrayProgress={xrayProgress}
          onAutoAction={handleAutoReview}
          onQAReview={handleAutoReview}
          hasUploadedText={!!uploadedText}
          jurisResearch={jurisResearch}
          jurisResearchLoading={jurisResearchLoading}
          jurisResearchProgress={jurisResearchProgress}
          onJurisImport={(data) => {
            // Format imported jurisprudence with explicit ementa + citation for LLM
            const parts = data.research.map((r) => {
              const resultsText = (r.results || []).map((res) => {
                const ementa = (res.ementa || '').trim();
                const processo = res.numero_processo || '?';
                const tipo = res.tipo_recurso || 'Acórdão';
                const data_pub = res.data_publicacao || '?';
                return (
                  `Nesse sentido, assim entende o Eg. TJMG:\n` +
                  `"EMENTA: ${ementa}" (TJMG, ${tipo}, nº ${processo}, ${data_pub})`
                );
              }).join('\n\n');
              return `**Tema:** ${r.theme}\n\n${resultsText}`;
            });
            const newContext = parts.join('\n\n---\n\n');
            setJurisContext((prev) => prev ? `${prev}\n\n---\n\n${newContext}` : newContext);

            // Also show visual confirmation in chat
            const importMsg = {
              role: 'assistant',
              content: `✅ **Jurisprudência incluída na fundamentação.** A próxima minuta gerada incluirá a ementa com a citação \"Assim entende o TJMG:\".`,
              model: 'pesquisador',
            };
            setMessages((prev) => [...prev, importMsg]);
          }}
          onJurisSearch={handleJurisSearch}
          selectedModel={globalSelectedModel}
        />
      );
    }
    return <WelcomeContent onOpenJurisprudencia={() => setShowJurisprudencia(true)} />;
  };

  const showBatchPanel = batchResults.length > 0;

  return (
    <div className="app-layout">
      {/* Mobile backdrop */}
      <div
        className={`sidebar-backdrop ${sidebarOpen ? 'visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={toggleSidebar}
        onCloseMobile={closeSidebarMobile}
        activeAgent={activeAgent}
        onAgentSelect={handleAgentSelect}
        onNewChat={handleNewChat}
        history={sidebarHistory}
        onLoadChat={handleLoadChat}
        customAgents={customAgents}
        onCreateAgent={handleOpenAgentBuilder}
        onDeleteAgent={handleDeleteAgent}
        onShareAgent={handleShareAgentOpen}
      />

      <div className={`main-panel ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
        <Header onMenuClick={toggleSidebar} isOpen={sidebarOpen} />

        <div className={`main-content ${showBatchPanel ? 'with-batch' : ''}`}>
          {renderContent()}
          {showBatchPanel && (
            <BatchPanel
              results={batchResults}
              selectedIndex={batchSelectedIndex}
              onSelect={handleBatchSelect}
              onClose={handleBatchClose}
            />
          )}
        </div>

        {!showAgentBuilder && (
          <ChatInput
            onSend={handleSend}
            onXray={handleXray}
            onFilesUploaded={handleFilesUploaded}
            onStyleReport={handleStyleReport}
            onModelChange={setGlobalSelectedModel}
            onOpenModelManager={() => setShowModelManager(true)}
            isLoading={isLoading || xrayLoading || styleAnalyzing}
            ocrProcessing={ocrProcessing}
            hasContext={!!(uploadedText || activeAgent)}
            ragStatus={ragStatus}
            onRagStatusChange={setRagStatus}
            activeAgent={activeAgent}
          />
        )}
      </div>

      {/* Create Agent Dialog */}
      <CreateAgentDialog
        isOpen={showCreateDialog}
        onClose={() => { setShowCreateDialog(false); setPendingPrompt(''); }}
        onConfirm={handleCreateAgent}
        initialPrompt={pendingPrompt}
      />

      {/* Share Agent Dialog */}
      <ShareAgentDialog
        isOpen={!!shareAgent}
        onClose={() => setShareAgent(null)}
        onShare={handleShareAgentConfirm}
        agentName={shareAgent?.name || ''}
      />
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <AuthGate />
    </AuthProvider>
  );
}

function AuthGate() {
  const { user, loading } = useAuth();
  const [showSignup, setShowSignup] = useState(false);

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background text-primary">
        <div className="animate-spin text-3xl">⚙️</div>
      </div>
    );
  }

  if (!user) {
    return showSignup ? (
      <SignupPage onNavigateLogin={() => setShowSignup(false)} />
    ) : (
      <LoginPage onNavigateSignup={() => setShowSignup(true)} />
    );
  }

  return <MainApp />;
}

export default App;
