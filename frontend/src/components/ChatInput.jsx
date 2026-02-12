import React, { useState } from 'react';
import { FaPaperclip, FaWandMagicSparkles, FaArrowUp, FaPlus } from 'react-icons/fa6';
import './ChatInput.css';

const ChatInput = ({ onSend }) => {
    const [message, setMessage] = useState('');

    const handleSend = () => {
        if (message.trim()) {
            onSend(message);
            setMessage('');
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleSend();
        }
    };

    return (
        <div className="footer-container">
            <div className="chat-input-wrapper">
                <button className="input-action-btn attach" aria-label="Anexar arquivo">
                    <FaPaperclip />
                </button>

                <input
                    type="text"
                    className="chat-input"
                    placeholder="Inicie o seu prompt aqui... / para prompts"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyPress}
                />

                <div className="right-actions">
                    <button className="input-action-btn prompt-lib" aria-label="Biblioteca de Prompts">
                        <FaWandMagicSparkles />
                    </button>
                    <button className="input-action-btn send-btn" aria-label="Enviar" onClick={handleSend}>
                        <FaArrowUp />
                    </button>
                </div>
            </div>

            <div className="footer-info">
                <button className="new-chat-btn">
                    <FaPlus size={12} /> <span>Nova chat</span>
                </button>
            </div>
        </div>
    );
};

export default ChatInput;
