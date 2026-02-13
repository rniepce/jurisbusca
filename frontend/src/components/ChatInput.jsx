import React, { useState } from 'react';
import {
    FaPaperclip, FaBook, FaSlash,
    FaArrowRotateRight, FaChevronDown
} from 'react-icons/fa6';
import { IoSend } from 'react-icons/io5';
import './ChatInput.css';

const ChatInput = ({ onSend }) => {
    const [message, setMessage] = useState('');

    const handleSend = () => {
        if (message.trim()) {
            if (onSend) onSend(message);
            setMessage('');
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="chat-footer">
            {/* Toolbar */}
            <div className="chat-toolbar">
                <div className="toolbar-left">
                    <button className="toolbar-btn model-selector">
                        <span className="model-dot" />
                        <FaChevronDown size={10} />
                    </button>
                    <button className="toolbar-btn" aria-label="Anexar"><FaPaperclip /></button>
                    <button className="toolbar-btn" aria-label="Modelos"><FaBook /></button>
                    <button className="toolbar-btn" aria-label="Prompts"><FaSlash /></button>
                </div>
                <div className="toolbar-right">
                    <button className="toolbar-btn" aria-label="Recarregar"><FaArrowRotateRight /></button>
                </div>
            </div>

            {/* Input Area */}
            <div className="chat-input-box">
                <textarea
                    className="chat-textarea"
                    placeholder="Insira o seu prompt aqui. @ para modelos, / para prompts"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={3}
                />
                <button
                    className={`send-btn ${message.trim() ? 'active' : ''}`}
                    onClick={handleSend}
                    aria-label="Enviar"
                >
                    <IoSend size={14} />
                </button>
            </div>
        </div>
    );
};

export default ChatInput;
