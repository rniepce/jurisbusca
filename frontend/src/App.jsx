import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import WelcomeContent from './components/WelcomeContent';
import ChatInput from './components/ChatInput';
import './App.css';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="app-layout">
      <Sidebar isOpen={sidebarOpen} onToggle={toggleSidebar} />

      <div className={`main-panel ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
        <Header onMenuClick={toggleSidebar} />

        <div className="main-content">
          <WelcomeContent />
        </div>

        <ChatInput />
      </div>
    </div>
  );
}

export default App;
