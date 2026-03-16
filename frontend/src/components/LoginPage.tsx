import React, { useState } from 'react';
import { supabase } from '../services/supabase';
import { AiOutlineLoading3Quarters } from 'react-icons/ai';
import logoSvg from '../assets/logo.svg';
import './AuthPage.css';

export default function LoginPage({ onNavigateSignup }) {
    const [isLoading, setIsLoading] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [errorMsg, setErrorMsg] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMsg(null);
        setIsLoading(true);

        const { error } = await supabase.auth.signInWithPassword({
            email,
            password,
        });

        if (error) {
            setErrorMsg(error.message);
            setIsLoading(false);
        }
    };

    return (
        <div className="auth-container">
            <div className="auth-blobs pointer-events-none">
                <div className="blob blob-1"></div>
                <div className="blob blob-2"></div>
            </div>

            <div className="auth-card">
                <div className="auth-header">
                    <img src={logoSvg} alt="Assistente" className="auth-logo" />
                    <div>
                        <h2 className="auth-title">
                            Entrar no <span className="text-gradient">Assistente</span>
                        </h2>
                        <p className="auth-description">
                            Autenticação Inteligente
                        </p>
                    </div>
                </div>

                <div className="auth-content">
                    {errorMsg && (
                        <div className="auth-error">
                            ⚠️ {errorMsg === 'Invalid login credentials' ? 'Email ou senha incorretos' : errorMsg}
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="auth-form">
                        <div className="form-group">
                            <label htmlFor="email">Email</label>
                            <input
                                id="email"
                                name="email"
                                type="email"
                                placeholder="seu@email.com"
                                required
                                disabled={isLoading}
                                className="auth-input"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="password">Senha</label>
                            <input
                                id="password"
                                name="password"
                                type="password"
                                placeholder="••••••••"
                                required
                                disabled={isLoading}
                                className="auth-input"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                        </div>

                        <button
                            type="submit"
                            className="auth-btn w-full"
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <>
                                    <AiOutlineLoading3Quarters className="spinner" />
                                    Entrando...
                                </>
                            ) : (
                                "Entrar"
                            )}
                        </button>
                    </form>

                    <p className="auth-footer">
                        Não tem conta?{" "}
                        <button
                            type="button"
                            className="text-primary hover-underline font-medium"
                            onClick={onNavigateSignup}
                        >
                            Criar conta
                        </button>
                    </p>
                </div>
            </div>
        </div>
    );
}
