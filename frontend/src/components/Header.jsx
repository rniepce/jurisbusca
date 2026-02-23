import React, { useState, useEffect, useRef } from 'react';
import { FaSignOutAlt, FaKey } from 'react-icons/fa';
import { LuPanelLeftClose, LuPanelLeftOpen } from 'react-icons/lu';
import { GoogleLogin } from '@react-oauth/google';
import { useAuth } from '../context/AuthContext';
import { validateAzureKey } from '../services/api';
import logoSvg from '../assets/logo.svg';
import './Header.css';

const Header = ({ onMenuClick, isOpen }) => {
    const { user, isAuthLoaded, login, logout } = useAuth();
    const [keyOpen, setKeyOpen] = useState(false);
    const [keyValue, setKeyValue] = useState('');
    const [keyStatus, setKeyStatus] = useState('unknown'); // 'unknown' | 'valid' | 'invalid' | 'checking'
    const [keyMessage, setKeyMessage] = useState('');
    const popoverRef = useRef(null);

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

                {/* Auth Section */}
                {isAuthLoaded && (
                    user ? (
                        <div className="user-profile">
                            <img src={user.picture} alt="Avatar" className="user-avatar" />
                            <span className="user-name">{user.name}</span>
                            <button className="logout-button" onClick={logout} title="Sair">
                                <FaSignOutAlt />
                            </button>
                        </div>
                    ) : (
                        <GoogleLogin
                            onSuccess={login}
                            onError={() => console.error('Login Failed')}
                            useOneTap
                            shape="pill"
                            size="medium"
                        />
                    )
                )}
            </div>
        </header>
    );
};

export default Header;
