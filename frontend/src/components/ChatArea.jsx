import React, { useEffect, useRef } from 'react';
import {
    FaRobot, FaUser,
    FaScaleBalanced, FaFileLines, FaMagnifyingGlass,
    FaBookOpen, FaPenNib
} from 'react-icons/fa6';
import OcrPreview from './OcrPreview';
import './ChatArea.css';

// Icon map for agent activation cards
const iconMap = {
    FaScaleBalanced: FaScaleBalanced,
    FaFileLines: FaFileLines,
    FaMagnifyingGlass: FaMagnifyingGlass,
    FaBookOpen: FaBookOpen,
    FaPenNib: FaPenNib,
};

const ChatArea = ({ messages, isLoading, activeAgent }) => {
    const endRef = useRef(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    return (
        <div className="chat-area">
            {/* Active agent banner */}
            {activeAgent && (
                <div className="agent-banner" style={{ borderLeftColor: activeAgent.color }}>
                    <span className="agent-banner-label">Agente ativo:</span>
                    <span className="agent-banner-name" style={{ color: activeAgent.color }}>
                        {activeAgent.name}
                    </span>
                </div>
            )}

            {/* Messages */}
            <div className="messages-list">
                {messages.map((msg, i) => {
                    // ── Agent Activation Card ──
                    if (msg.role === 'agent-activation') {
                        const IconComp = iconMap[msg.agentIcon] || FaScaleBalanced;
                        return (
                            <div key={i} className="agent-activation-card" style={{ '--agent-color': msg.agentColor }}>
                                <div className="activation-icon-ring">
                                    <IconComp size={22} />
                                </div>
                                <div className="activation-info">
                                    <span className="activation-title">
                                        🎯 Agente <strong>{msg.agentName}</strong> ativado
                                    </span>
                                    <span className="activation-desc">
                                        {msg.agentDesc}
                                    </span>
                                    <span className="activation-hint">
                                        Todas as mensagens usarão o prompt especializado deste agente.
                                    </span>
                                </div>
                            </div>
                        );
                    }

                    // ── OCR Preview Card ──
                    if (msg.role === 'ocr') {
                        return (
                            <OcrPreview
                                key={i}
                                filename={msg.filename}
                                text={msg.text}
                                engine={msg.engine}
                                charCount={msg.charCount}
                            />
                        );
                    }

                    // ── Normal messages ──
                    return (
                        <div key={i} className={`message-row ${msg.role}`}>
                            <div className="message-avatar">
                                {msg.role === 'user' ? (
                                    <FaUser size={14} />
                                ) : (
                                    <FaRobot size={14} />
                                )}
                            </div>
                            <div className={`message-bubble ${msg.role}`}>
                                {msg.role === 'assistant' ? (
                                    <div
                                        className="message-content markdown"
                                        dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }}
                                    />
                                ) : (
                                    <div className="message-content">{msg.content}</div>
                                )}
                                {msg.model && (
                                    <span className="message-model">{msg.model}</span>
                                )}
                            </div>
                        </div>
                    );
                })}

                {/* Typing indicator */}
                {isLoading && (
                    <div className="message-row assistant">
                        <div className="message-avatar">
                            <FaRobot size={14} />
                        </div>
                        <div className="message-bubble assistant">
                            <div className="typing-indicator">
                                <span className="dot" />
                                <span className="dot" />
                                <span className="dot" />
                            </div>
                        </div>
                    </div>
                )}

                <div ref={endRef} />
            </div>
        </div>
    );
};

/**
 * Lightweight markdown → HTML (bold, italic, headers, code blocks, lists, line breaks).
 * No external dependency needed.
 */
function formatMarkdown(text) {
    if (!text) return '';
    // Pre-processing: convert escaped newlines to real newlines
    let processed = text.replace(/\\n/g, '\n');
    let html = processed
        // Escape HTML
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        // Code blocks (triple backtick)
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Headers
        .replace(/^### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^## (.+)$/gm, '<h3>$1</h3>')
        .replace(/^# (.+)$/gm, '<h2>$1</h2>')
        // Bold + italic
        .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Horizontal rule
        .replace(/^---$/gm, '<hr/>')
        // Tables (markdown pipe tables)
        .replace(/^(\|.+\|)\n\|[-| :]+\|\n((?:\|.+\|\n?)+)/gm, (match, header, body) => {
            const headers = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
            const rows = body.trim().split('\n').map(row => {
                const cols = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
                return `<tr>${cols}</tr>`;
            }).join('');
            return `<table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
        })
        // Unordered list items
        .replace(/^\s*[-*] (.+)$/gm, '<li>$1</li>')
        // Ordered list items
        .replace(/^\s*\d+\.\s(.+)$/gm, '<li>$1</li>')
        // Line breaks (double newline → paragraph, single → <br>)
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br/>');

    // Post-processing: wrap consecutive <li> items in <ul>
    html = html.replace(/((?:<li>.*?<\/li>(?:<br\/>)?)+)/g, '<ul>$1</ul>');
    // Clean up any <br/> inside <ul> between list items
    html = html.replace(/<ul>([\s\S]*?)<\/ul>/g, (match, inner) => {
        return '<ul>' + inner.replace(/<br\/>/g, '') + '</ul>';
    });

    // Wrap in paragraph if not already structured
    if (!html.startsWith('<h') && !html.startsWith('<pre') && !html.startsWith('<ul') && !html.startsWith('<table')) {
        html = '<p>' + html + '</p>';
    }

    return html;
}

export default ChatArea;
