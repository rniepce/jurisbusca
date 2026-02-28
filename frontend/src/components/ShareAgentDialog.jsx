import React, { useState } from 'react';
import { FaXmark, FaShareNodes, FaCheck } from 'react-icons/fa6';
import './CreateAgentDialog.css'; // Reuses same overlay/dialog styles

const ShareAgentDialog = ({ isOpen, onClose, onShare, agentName = '' }) => {
    const [email, setEmail] = useState('');
    const [sending, setSending] = useState(false);
    const [success, setSuccess] = useState(false);

    if (!isOpen) return null;

    const handleShare = async () => {
        if (!email.trim()) return;
        setSending(true);
        try {
            await onShare(email.trim());
            setSuccess(true);
            setTimeout(() => {
                setSuccess(false);
                setEmail('');
                onClose();
            }, 1500);
        } catch (err) {
            console.error('Share failed:', err);
        } finally {
            setSending(false);
        }
    };

    return (
        <div className="share-agent-overlay" onClick={onClose}>
            <div className="share-agent-dialog" onClick={(e) => e.stopPropagation()}>
                <div className="share-agent-header">
                    <h2><FaShareNodes /> Compartilhar "{agentName}"</h2>
                    <button className="create-agent-close" onClick={onClose}>
                        <FaXmark />
                    </button>
                </div>

                <div className="share-agent-body">
                    {success ? (
                        <div className="share-success">
                            <FaCheck /> Agente compartilhado com sucesso!
                        </div>
                    ) : (
                        <div className="create-agent-field">
                            <label>Email do destinatário</label>
                            <input
                                type="email"
                                placeholder="usuario@exemplo.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                autoFocus
                                id="share-agent-email"
                            />
                        </div>
                    )}
                </div>

                <div className="share-agent-footer">
                    <button className="btn-cancel" onClick={onClose}>Cancelar</button>
                    <button
                        className="btn-confirm"
                        onClick={handleShare}
                        disabled={!email.trim() || sending || success}
                        id="btn-share-agent"
                    >
                        {sending ? 'Enviando...' : 'Compartilhar'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ShareAgentDialog;
