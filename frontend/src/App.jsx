import React, { useState, useCallback, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import WelcomeContent from './components/WelcomeContent';
import ChatArea from './components/ChatArea';
import ChatInput from './components/ChatInput';
import XRayDashboard from './components/XRayDashboard';
import BatchPanel from './components/BatchPanel';
import { sendMessage, uploadFile, uploadBatchXray, generateStyleReport, getTemplateStatus, analyzeCluster, uploadTemplates } from './services/api';
import agentDefinitions from './config/agents';
import './App.css';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeAgent, setActiveAgent] = useState(agentDefinitions[0]);
  const [conversationId, setConversationId] = useState(null);
  const [uploadedText, setUploadedText] = useState(null);
  const [xrayReport, setXrayReport] = useState(null);
  const [xrayLoading, setXrayLoading] = useState(false);
  const [xrayTextCache, setXrayTextCache] = useState({});
  const [ocrProcessing, setOcrProcessing] = useState(false);
  const [styleAnalyzing, setStyleAnalyzing] = useState(false);
  const [styleDossier, setStyleDossier] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [ragStatus, setRagStatus] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [batchSelectedIndex, setBatchSelectedIndex] = useState(null);
  const [globalSelectedModel, setGlobalSelectedModel] = useState({ id: 'v0', name: 'Gabinete V0', color: '#10B981', llm: 'gpt-5.2-chat' });

  // Fetch template/RAG status on mount
  useEffect(() => {
    getTemplateStatus().then(setRagStatus).catch(() => { });
  }, []);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  // ── Send message handler ────────────────────────────────────────────
  const handleSend = useCallback(async (message, selectedModel, files, ocrEngine, templateFiles, useRag = false) => {
    // Use a default prompt if user sends empty message but has context
    const userTyped = message.trim();
    const effectiveMessage = userTyped ||
      (uploadedText && activeAgent ? 'Analise o documento anexado conforme as instruções do agente.' :
        uploadedText ? 'Analise o documento anexado.' : '');
    if (!effectiveMessage) return;

    // 1. Add user message immediately (hide auto-generated messages)
    if (userTyped) {
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

      // 2. Load agent prompt if active (version-aware: V0 uses 4.5, V1+ uses 4.6)
      let agentPrompt = null;
      if (activeAgent?.promptModule) {
        try {
          // Check if V0 is selected — use the V4.5 prompt instead
          if (selectedModel.id === 'v0') {
            const mod = await import('./prompts/gabineteCivelV0.js');
            agentPrompt = mod.default || null;
          } else {
            const mod = await activeAgent.promptModule();
            agentPrompt = mod.default || null;
          }
        } catch {
          console.warn('Could not load agent prompt');
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
  }, [activeAgent, conversationId, uploadedText, styleDossier, ragStatus]);

  // ── X-Ray handler (batch clustering) ────────────────────────────────
  const handleXray = useCallback(async (files) => {
    if (files.length < 2) return;
    setXrayLoading(true);
    setXrayReport(null);

    try {
      const result = await uploadBatchXray(files);
      setXrayReport(result.report);
      setXrayTextCache(result.text_cache || {});
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `⚠️ **Erro no Raio-X:** ${err.message}`, model: 'erro' },
      ]);
    } finally {
      setXrayLoading(false);
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
    // Show OCR animation only when actually running OCR (not for 'Sem OCR')
    if (ocrEngine !== 'none') setOcrProcessing(true);

    try {
      const uploadPromises = files.map((f) => uploadFile(f, ocrEngine, compress));
      const results = await Promise.all(uploadPromises);

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
    setActiveAgent(agent);
    const activationMsg = {
      role: 'agent-activation',
      agentName: agent.name,
      agentDesc: agent.desc,
      agentColor: agent.color,
      agentIcon: agent.icon,
      autoAction: agent.autoAction || null,
    };
    setMessages((prev) => [...prev, activationMsg]);
  }, []);

  // ── Auto-review handler (Revisor QA) ────────────────────────────────
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

    // 2. Build the "sandwich" audit message
    const auditMessage = [
      'Execute a auditoria de conformidade cruzando os textos abaixo.',
      '',
      '[DADOS DO PROCESSO]:',
      uploadedText,
      '',
      '[MINUTA PROPOSTA]:',
      lastMinuta.content,
      '',
      'Execute a auditoria de conformidade cruzando os textos acima. Gere o Dashboard de Conformidade completo.',
    ].join('\n');

    // 3. Send using the current active agent (QA) and selected model
    await handleSend(auditMessage, globalSelectedModel, [], 'paddle', [], false);
  }, [messages, uploadedText, handleSend, globalSelectedModel]);

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
    if (xrayReport) {
      return (
        <XRayDashboard
          report={xrayReport}
          onClose={() => setXrayReport(null)}
          onClusterAction={handleClusterAction}
        />
      );
    }
    if (hasMessages || styleAnalyzing) {
      return (
        <ChatArea
          messages={messages}
          isLoading={isLoading}
          activeAgent={activeAgent}
          ocrProcessing={ocrProcessing}
          styleAnalyzing={styleAnalyzing}
          onAutoAction={handleAutoReview}
        />
      );
    }
    return <WelcomeContent />;
  };

  const showBatchPanel = batchResults.length > 0;

  return (
    <div className="app-layout">
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={toggleSidebar}
        activeAgent={activeAgent}
        onAgentSelect={handleAgentSelect}
        onNewChat={handleNewChat}
        history={sidebarHistory}
        onLoadChat={handleLoadChat}
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

        <ChatInput
          onSend={handleSend}
          onXray={handleXray}
          onFilesUploaded={handleFilesUploaded}
          onStyleReport={handleStyleReport}
          onModelChange={setGlobalSelectedModel}
          isLoading={isLoading || xrayLoading || styleAnalyzing}
          ocrProcessing={ocrProcessing}
          hasContext={!!(uploadedText || activeAgent)}
          ragStatus={ragStatus}
          onRagStatusChange={setRagStatus}
        />
      </div>
    </div>
  );
}

export default App;
