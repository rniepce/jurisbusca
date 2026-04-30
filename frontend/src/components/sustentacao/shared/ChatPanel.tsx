import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FaPaperPlane, FaCircleNotch, FaGavel } from 'react-icons/fa6';
import { chatSustentacao } from '../../../services/api';

interface ChatMsg {
    role: 'user' | 'assistant';
    content: string;
}

interface Props {
    processId: string;
    placeholders?: string[];
    title?: string;
}

const ChatPanel: React.FC<Props> = ({ processId, placeholders, title = 'Pergunte sobre o processo' }) => {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState<ChatMsg[]>([]);
    const [loading, setLoading] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    const handleSend = useCallback(async () => {
        const msg = input.trim();
        if (!msg || loading) return;
        setInput('');
        const newMessages: ChatMsg[] = [...messages, { role: 'user', content: msg }];
        setMessages(newMessages);
        setLoading(true);
        try {
            const res = await chatSustentacao(processId, newMessages);
            setMessages([...newMessages, { role: 'assistant', content: res.reply }]);
        } catch (e: any) {
            setMessages([...newMessages, { role: 'assistant', content: `⚠️ ${e?.message || 'Erro'}` }]);
        } finally {
            setLoading(false);
        }
    }, [input, processId, messages, loading]);

    return (
        <aside className="sust-chat">
            <div className="sust-chat-header">
                <FaGavel /> {title}
            </div>
            <div className="sust-chat-messages">
                {messages.length === 0 && !loading && placeholders && placeholders.length > 0 && (
                    <div className="sust-chat-empty">
                        <p>Exemplos de perguntas:</p>
                        <ul>
                            {placeholders.map((p, i) => <li key={i}>{p}</li>)}
                        </ul>
                    </div>
                )}
                {messages.map((m, i) => (
                    <div key={i} className={`sust-chat-msg ${m.role}`}>{m.content}</div>
                ))}
                {loading && (
                    <div className="sust-chat-msg assistant loading">
                        <FaCircleNotch className="sust-spinner-sm" /> Pensando...
                    </div>
                )}
                <div ref={bottomRef} />
            </div>
            <div className="sust-chat-input">
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSend();
                        }
                    }}
                    placeholder="Pergunte algo sobre o processo..."
                    rows={2}
                    disabled={loading}
                />
                <button className="sust-btn-send" onClick={handleSend} disabled={!input.trim() || loading}>
                    <FaPaperPlane />
                </button>
            </div>
        </aside>
    );
};

export default ChatPanel;
