import React, { useEffect, useRef, useState } from 'react';
import {
    FaUser,
    FaScaleBalanced, FaFileLines, FaMagnifyingGlass,
    FaBookOpen, FaPenNib, FaClipboardCheck
} from 'react-icons/fa6';
import OcrPreview from './OcrPreview';
import logoSvg from '../assets/logo.svg';
import './ChatArea.css';

// ── Analysis phases for the style animation ──
const STYLE_PHASES = [
    { icon: '📄', title: 'Extraindo texto dos modelos...', sub: 'Lendo e processando os arquivos enviados' },
    { icon: '🔍', title: 'Analisando estrutura e vocabulário...', sub: 'Identificando padrões de redação e jargões' },
    { icon: '🧬', title: 'Mapeando DNA da escrita judicial...', sub: 'Analisando os 5 Pilares de Identidade Decisional' },
    { icon: '📝', title: 'Compilando glossário do magistrado...', sub: 'Catalogando expressões e conectivos característicos' },
    { icon: '✨', title: 'Finalizando dossiê de identidade...', sub: 'Gerando System Prompt de Clonagem Estilística' },
];

// ── Raio-X phases for the xray processing animation ──
const XRAY_PHASES = [
    { icon: '📤', title: 'Enviando processos para análise...', sub: 'Fazendo upload e leitura dos arquivos' },
    { icon: '🔍', title: 'Lendo e classificando documentos...', sub: 'Extraindo texto e identificando tipos processuais' },
    { icon: '🧩', title: 'Agrupando processos por matéria...', sub: 'Clusterizando por similaridade temática' },
    { icon: '📊', title: 'Gerando painel de Raio-X da Carteira...', sub: 'Compilando estatísticas e recomendações' },
];

/** Multi-step animation shown during Style Report generation */
function StyleAnalysisAnimation() {
    const [phase, setPhase] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setPhase((p) => (p + 1) % STYLE_PHASES.length);
        }, 4000);
        return () => clearInterval(interval);
    }, []);

    const current = STYLE_PHASES[phase];
    const progress = ((phase + 1) / STYLE_PHASES.length) * 100;

    return (
        <div className="style-processing-card">
            <div className="style-phase-icon" key={phase}>
                <span>{current.icon}</span>
            </div>
            <div className="style-processing-body">
                <div className="style-steps-row">
                    {STYLE_PHASES.map((_, i) => (
                        <span
                            key={i}
                            className={`style-step-dot ${i === phase ? 'active' : ''} ${i < phase ? 'done' : ''}`}
                        />
                    ))}
                    <span className="style-step-label">Etapa {phase + 1}/{STYLE_PHASES.length}</span>
                </div>
                <span className="style-processing-title" key={`t-${phase}`}>{current.title}</span>
                <span className="style-processing-sub" key={`s-${phase}`}>{current.sub}</span>
                <div className="style-progress-track">
                    <div className="style-progress-fill" style={{ width: `${progress}%` }} />
                </div>
            </div>
        </div>
    );
}

/** Multi-step animation shown during Raio-X batch processing */
function XRayProcessingAnimation() {
    const [phase, setPhase] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setPhase((p) => (p + 1) % XRAY_PHASES.length);
        }, 3500);
        return () => clearInterval(interval);
    }, []);

    const current = XRAY_PHASES[phase];
    const progress = ((phase + 1) / XRAY_PHASES.length) * 100;

    return (
        <div className="xray-processing-card">
            <div className="xray-phase-icon" key={phase}>
                <span>{current.icon}</span>
            </div>
            <div className="xray-processing-body">
                <div className="xray-steps-row">
                    {XRAY_PHASES.map((_, i) => (
                        <span
                            key={i}
                            className={`xray-step-dot ${i === phase ? 'active' : ''} ${i < phase ? 'done' : ''}`}
                        />
                    ))}
                    <span className="xray-step-label">Etapa {phase + 1}/{XRAY_PHASES.length}</span>
                </div>
                <span className="xray-processing-title" key={`t-${phase}`}>{current.title}</span>
                <span className="xray-processing-sub" key={`s-${phase}`}>{current.sub}</span>
                <div className="xray-progress-track">
                    <div className="xray-progress-fill" style={{ width: `${progress}%` }} />
                </div>
            </div>
        </div>
    );
}


// Icon map for agent activation cards
const iconMap = {
    FaScaleBalanced: FaScaleBalanced,
    FaFileLines: FaFileLines,
    FaMagnifyingGlass: FaMagnifyingGlass,
    FaBookOpen: FaBookOpen,
    FaPenNib: FaPenNib,
    FaClipboardCheck: FaClipboardCheck,
};

/** Collapsible card for V2 engine sections (triage/audit) */
function V2CollapsibleCard({ icon, title, content }) {
    const [open, setOpen] = useState(false);

    return (
        <div className={`v2-card ${open ? 'v2-card-open' : ''}`} onClick={() => setOpen(!open)}>
            <div className="v2-card-header">
                <span className="v2-card-icon">{icon}</span>
                <span className="v2-card-title">{title}</span>
                <span className={`v2-card-chevron ${open ? 'rotated' : ''}`}>▸</span>
            </div>
            {open && (
                <div
                    className="v2-card-body markdown"
                    dangerouslySetInnerHTML={{ __html: formatMarkdown(content) }}
                    onClick={(e) => e.stopPropagation()}
                />
            )}
        </div>
    );
}

const ChatArea = ({ messages, isLoading, activeAgent, ocrProcessing = false, ocrEngineName = 'none', styleAnalyzing = false, xrayLoading = false, onAutoAction }) => {
    const endRef = useRef(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading, styleAnalyzing, xrayLoading]);

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
                                    {msg.autoAction && (
                                        <button
                                            className="auto-action-btn"
                                            style={{
                                                '--agent-color': msg.agentColor,
                                                marginTop: '10px',
                                                padding: '8px 18px',
                                                background: `linear-gradient(135deg, ${msg.agentColor}, ${msg.agentColor}dd)`,
                                                color: '#fff',
                                                border: 'none',
                                                borderRadius: '8px',
                                                cursor: isLoading ? 'not-allowed' : 'pointer',
                                                fontSize: '13px',
                                                fontWeight: 600,
                                                letterSpacing: '0.3px',
                                                display: 'inline-flex',
                                                alignItems: 'center',
                                                gap: '6px',
                                                opacity: isLoading ? 0.6 : 1,
                                                transition: 'all 0.2s ease',
                                                boxShadow: `0 2px 8px ${msg.agentColor}44`,
                                            }}
                                            onClick={() => onAutoAction && onAutoAction(msg)}
                                            disabled={isLoading}
                                        >
                                            {msg.autoAction.label}
                                        </button>
                                    )}
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
                    // ── V2 Structured Response with collapsible cards ──
                    if (msg.role === 'assistant' && msg.v2Sections) {
                        return (
                            <div key={i} className={`message-row assistant`}>
                                <div className="message-avatar">
                                    <img src={logoSvg} alt="Assistente" style={{ width: 14, height: 14, borderRadius: '2px' }} />
                                </div>
                                <div className={`message-bubble assistant`}>
                                    {/* Main content: the minuta/draft */}
                                    <div
                                        className="message-content markdown"
                                        dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.v2Sections.draft || msg.content) }}
                                    />

                                    {/* Collapsible cards */}
                                    <div className="v2-cards-row">
                                        {msg.v2Sections.triage && (
                                            <V2CollapsibleCard
                                                icon="🔍"
                                                title="Relatório de Triagem"
                                                content={msg.v2Sections.triage}
                                            />
                                        )}
                                        {msg.v2Sections.audit && (
                                            <V2CollapsibleCard
                                                icon="🛡️"
                                                title="Auditoria (QA)"
                                                content={msg.v2Sections.audit}
                                            />
                                        )}
                                    </div>

                                    {msg.model && (
                                        <span className="message-model">{msg.model}</span>
                                    )}
                                </div>
                            </div>
                        );
                    }

                    return (
                        <div key={i} className={`message-row ${msg.role}`}>
                            <div className="message-avatar">
                                {msg.role === 'user' ? (
                                    <FaUser size={14} />
                                ) : (
                                    <img src={logoSvg} alt="Assistente" style={{ width: 14, height: 14, borderRadius: '2px' }} />
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

                {/* OCR Processing Animation */}
                {ocrProcessing && (
                    <div className="ocr-processing-card">
                        <div className="ocr-processing-icon">
                            <FaFileLines size={18} />
                        </div>
                        <div className="ocr-processing-info">
                            <span className="ocr-processing-title">
                                {ocrEngineName === 'none' ? 'Fazendo a leitura do processo...' : 'Processando OCR...'}
                            </span>
                            <span className="ocr-processing-sub">
                                {ocrEngineName === 'none' ? 'Extraindo texto nativo do documento' : 'Extraindo texto do documento via OCR'}
                            </span>
                        </div>
                        <div className="ocr-processing-dots">
                            <span className="dot" />
                            <span className="dot" />
                            <span className="dot" />
                        </div>
                    </div>
                )}

                {/* Raio-X Processing Animation — Multi-step */}
                {xrayLoading && (
                    <XRayProcessingAnimation />
                )}

                {/* Style Analysis Processing Animation — Multi-step */}
                {styleAnalyzing && (
                    <StyleAnalysisAnimation />
                )}

                {/* Typing indicator */}
                {isLoading && (
                    <div className="message-row assistant">
                        <div className="message-avatar">
                            <img src={logoSvg} alt="Assistente" style={{ width: 14, height: 14, borderRadius: '2px' }} />
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

function formatMarkdown(text) {
    // Robust type coercion: handle non-string inputs (objects, arrays, null, etc.)
    if (text === null || text === undefined) return '';

    // Deep Extraction algorithm for bizarre LLM/Langchain JSON formats arriving at the UI
    if (typeof text !== 'string') {
        if (Array.isArray(text)) {
            // Aggregate all array items into a single string recursively
            text = text.map(item => {
                if (typeof item === 'string') return item;
                if (item?.type === 'thinking') return ''; // ignore logic blocks from deepseek/claude if they leak
                return item?.text || item?.content || item?.message || JSON.stringify(item);
            }).filter(Boolean).join('\n');
        } else if (typeof text === 'object') {
            // It's a dict. Try known LLM response keys
            text = text.text || text.content || text.message || text.output || JSON.stringify(text);
        } else {
            // Fallback for numbers, booleans, etc
            text = String(text);
        }
    }

    if (!text || typeof text !== 'string') return '';

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
