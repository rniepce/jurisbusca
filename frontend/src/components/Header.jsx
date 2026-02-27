import React, { useState, useEffect, useRef } from 'react';
import { FaKey } from 'react-icons/fa';
import { LuPanelLeftClose, LuPanelLeftOpen } from 'react-icons/lu';
import { validateAzureKey } from '../services/api';
import { useAuth } from './AuthContext';
import { FaUserCircle } from 'react-icons/fa';
import logoSvg from '../assets/logo.svg';
import './Header.css';

const Header = ({ onMenuClick, isOpen }) => {
    const [keyOpen, setKeyOpen] = useState(false);
    const [keyValue, setKeyValue] = useState('');
    const [keyStatus, setKeyStatus] = useState('unknown'); // 'unknown' | 'valid' | 'invalid' | 'checking'
    const [keyMessage, setKeyMessage] = useState('');
    const popoverRef = useRef(null);
    const { user, signOut } = useAuth();

    // Load stored key status on mount
    useEffect(() => {
        const stored = localStorage.getItem('azure_openai_key');
        if (stored) {
            setKeyValue(stored);
            setKeyStatus('valid');
        }
    }, []);

    // Close popover on outside click
    useEffect(() => {
        const handler = (e) => {
            if (popoverRef.current && !popoverRef.current.contains(e.target)) {
                setKeyOpen(false);
            }
        };
        if (keyOpen) document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [keyOpen]);

    const handleValidate = async () => {
        const trimmed = keyValue.trim();
        if (!trimmed) {
            setKeyStatus('invalid');
            setKeyMessage('Cole a chave primeiro.');
            return;
        }
        setKeyStatus('checking');
        setKeyMessage('Verificando...');
        try {
            const result = await validateAzureKey(trimmed);
            if (result.valid) {
                setKeyStatus('valid');
                setKeyMessage('✅ Chave válida!');
                localStorage.setItem('azure_openai_key', trimmed);
            } else {
                setKeyStatus('invalid');
                setKeyMessage(result.message || 'Chave inválida.');
                localStorage.removeItem('azure_openai_key');
            }
        } catch (err) {
            setKeyStatus('invalid');
            setKeyMessage('Erro de conexão. Backend rodando?');
        }
    };

    const handleClear = () => {
        setKeyValue('');
        setKeyStatus('unknown');
        setKeyMessage('');
        localStorage.removeItem('azure_openai_key');
    };

    const statusDot = keyStatus === 'valid' ? 'key-dot valid'
        : keyStatus === 'invalid' ? 'key-dot invalid'
            : keyStatus === 'checking' ? 'key-dot checking'
                : 'key-dot';

    return (
        <header className="top-header">
            <div className="header-left">
                <button className="menu-toggle" onClick={onMenuClick} aria-label="Toggle sidebar">
                    {isOpen ? <LuPanelLeftClose /> : <LuPanelLeftOpen />}
                </button>
                <div className="header-brand">
                    <img src={logoSvg} alt="Logo TJMG" className="brand-icon" />
                    <span className="brand-title">Assistente TJMG</span>
                </div>
            </div>

            <div className="header-right">
                {/* API Key Configuration */}
                <div className="key-config" ref={popoverRef}>
                    <button
                        className="key-button"
                        onClick={() => setKeyOpen(!keyOpen)}
                        title="Configurar chave Azure OpenAI"
                    >
                        <FaKey size={14} />
                        <span className={statusDot} />
                    </button>

                    {keyOpen && (
                        <div className="key-popover">
                            <div className="key-popover-title">🔑 Chave Azure OpenAI</div>
                            <input
                                type="password"
                                className="key-input"
                                placeholder="Cole sua API Key aqui..."
                                value={keyValue}
                                onChange={(e) => {
                                    setKeyValue(e.target.value);
                                    setKeyStatus('unknown');
                                    setKeyMessage('');
                                }}
                                autoFocus
                            />
                            <div className="key-actions">
                                <button className="key-validate-btn" onClick={handleValidate} disabled={keyStatus === 'checking'}>
                                    {keyStatus === 'checking' ? 'Verificando...' : 'Validar'}
                                </button>
                                <button className="key-clear-btn" onClick={handleClear}>Limpar</button>
                            </div>
                            {keyMessage && (
                                <div className={`key-message ${keyStatus}`}>{keyMessage}</div>
                            )}
                        </div>
                    )}
                </div>

                {/* User Profile */}
                {user && (
                    <div className="user-profile">
                        <div className="user-info">
                            <FaUserCircle className="user-avatar" size={20} />
                            <span className="user-name">{user.user_metadata?.full_name || user.email}</span>
                        </div>
                        <button className="logout-btn" onClick={signOut} title="Sair da conta">
                            Sair
                        </button>
                    </div>
                )}
            </div>
        </header>
    );
};

export default Header;
