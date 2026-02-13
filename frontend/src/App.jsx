import React, { useState, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import WelcomeContent from './components/WelcomeContent';
import ChatArea from './components/ChatArea';
import ChatInput from './components/ChatInput';
import XRayDashboard from './components/XRayDashboard';
import { sendMessage, uploadFile, uploadBatchXray } from './services/api';
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

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  // ── Send message handler ────────────────────────────────────────────
  const handleSend = useCallback(async (message, selectedModel, files, ocrEngine, templateFiles) => {
    if (!message.trim() && files.length === 0) return;

    // 1. Add user message immediately
    const userMsg = { role: 'user', content: message };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      // 2. Upload files first (if any)
      let fileText = uploadedText;
      if (files.length > 0) {
        const uploadPromises = files.map((f) => uploadFile(f, ocrEngine));
        const results = await Promise.all(uploadPromises);
        const newText = results.map((r) => r.text).join('\n\n---\n\n');
        fileText = fileText ? fileText + '\n\n---\n\n' + newText : newText;
        setUploadedText(fileText);
      }

      // 3. Load agent prompt if active
      let agentPrompt = null;
      if (activeAgent?.promptModule) {
        try {
          const mod = await activeAgent.promptModule();
          agentPrompt = mod.default || null;
        } catch {
          console.warn('Could not load agent prompt');
        }
      }

      // 4. Call the backend
      const result = await sendMessage({
        message,
        model: selectedModel.id,
        agentPrompt,
        conversationId,
        uploadedText: fileText,
      });

      setConversationId(result.conversation_id);

      // 5. Add assistant response
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

  // ── Agent selection handler ─────────────────────────────────────────
  const handleAgentSelect = useCallback((agent) => {
    setActiveAgent(agent);
  }, []);

  // ── New chat handler ────────────────────────────────────────────────
  const handleNewChat = useCallback(() => {
    setMessages([]);
    setActiveAgent(null);
    setConversationId(null);
    setUploadedText(null);
    setXrayReport(null);
  }, []);

  // ── Welcome action handler (quick prompts) ──────────────────────────
  const handleWelcomeAction = useCallback((action) => {
    const fakeModel = { id: 'gemini', name: 'Gemini 2.5 Pro', color: '#4285F4' };
    handleSend(action, fakeModel, [], 'gemini_flash', []);
  }, [handleSend]);

  const hasMessages = messages.length > 0;

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
        />
      );
    }
    return <WelcomeContent onAction={handleWelcomeAction} />;
  };

  return (
    <div className="app-layout">
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={toggleSidebar}
        activeAgent={activeAgent}
        onAgentSelect={handleAgentSelect}
        onNewChat={handleNewChat}
      />

      <div className={`main-panel ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
        <Header onMenuClick={toggleSidebar} />

        <div className="main-content">
          {renderContent()}
        </div>

        <ChatInput
          onSend={handleSend}
          onXray={handleXray}
          isLoading={isLoading || xrayLoading}
        />
      </div>
    </div>
  );
}

export default App;
