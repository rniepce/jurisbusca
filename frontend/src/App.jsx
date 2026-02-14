import React, { useState, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import WelcomeContent from './components/WelcomeContent';
import ChatArea from './components/ChatArea';
import ChatInput from './components/ChatInput';
import XRayDashboard from './components/XRayDashboard';
import { sendMessage, uploadFile, uploadBatchXray, generateStyleReport } from './services/api';
import './App.css';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeAgent, setActiveAgent] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [uploadedText, setUploadedText] = useState(null);
  const [xrayReport, setXrayReport] = useState(null);
  const [xrayLoading, setXrayLoading] = useState(false);
  const [ocrProcessing, setOcrProcessing] = useState(false);
  const [styleAnalyzing, setStyleAnalyzing] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  // ── Send message handler ────────────────────────────────────────────
  const handleSend = useCallback(async (message, selectedModel, files, ocrEngine, templateFiles) => {
    if (!message.trim() && !uploadedText) return;

    // 1. Add user message immediately
    const userMsg = { role: 'user', content: message };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      // Files are already processed by handleFilesUploaded (auto-OCR),
      // so we just use the uploadedText that was populated earlier.

      // 2. Load agent prompt if active
      let agentPrompt = null;
      if (activeAgent?.promptModule) {
        try {
          const mod = await activeAgent.promptModule();
          agentPrompt = mod.default || null;
        } catch {
          console.warn('Could not load agent prompt');
        }
      }

      // 3. Call the backend
      const result = await sendMessage({
        message,
        model: selectedModel.id,
        agentPrompt,
        conversationId,
        uploadedText,
      });

      setConversationId(result.conversation_id);

      // 4. Add assistant response
      const assistantMsg = {
        role: 'assistant',
        content: result.response,
        model: result.model,
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
  }, [activeAgent, conversationId, uploadedText]);

  // ── X-Ray handler (batch clustering) ────────────────────────────────
  const handleXray = useCallback(async (files) => {
    if (files.length < 2) return;
    setXrayLoading(true);
    setXrayReport(null);

    try {
      const result = await uploadBatchXray(files);
      setXrayReport(result.report);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `⚠️ **Erro no Raio-X:** ${err.message}`, model: 'erro' },
      ]);
    } finally {
      setXrayLoading(false);
    }
  }, []);

  // ── Files uploaded → run OCR immediately ─────────────────────────────
  const handleFilesUploaded = useCallback(async (files, ocrEngine) => {
    if (files.length === 0) return;
    setOcrProcessing(true);

    try {
      const uploadPromises = files.map((f) => uploadFile(f, ocrEngine));
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
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `⚠️ **Erro no OCR:** ${err.message}`, model: 'erro' },
      ]);
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
    };
    setMessages((prev) => [...prev, activationMsg]);
  }, []);

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
        />
      );
    }
    if (hasMessages) {
      return (
        <ChatArea
          messages={messages}
          isLoading={isLoading}
          activeAgent={activeAgent}
          ocrProcessing={ocrProcessing}
          styleAnalyzing={styleAnalyzing}
        />
      );
    }
    return <WelcomeContent />;
  };

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
        <Header onMenuClick={toggleSidebar} />

        <div className="main-content">
          {renderContent()}
        </div>

        <ChatInput
          onSend={handleSend}
          onXray={handleXray}
          onFilesUploaded={handleFilesUploaded}
          onStyleReport={handleStyleReport}
          isLoading={isLoading || xrayLoading || styleAnalyzing}
          ocrProcessing={ocrProcessing}
        />
      </div>
    </div>
  );
}

export default App;
