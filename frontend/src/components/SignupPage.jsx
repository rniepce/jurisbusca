import React, { useState } from 'react';
import { supabase } from '../services/supabase';
import { AiOutlineLoading3Quarters } from 'react-icons/ai';
import logoSvg from '../assets/logo.svg';
import './AuthPage.css';

export default function SignupPage({ onNavigateLogin }) {
    const [isLoading, setIsLoading] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: '',
        confirmPassword: ''
    });
    const [errorMsg, setErrorMsg] = useState(null);
    const [successMsg, setSuccessMsg] = useState(null);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMsg(null);
        setSuccessMsg(null);

        if (formData.password !== formData.confirmPassword) {
            setErrorMsg("As senhas não coincidem.");
            return;
        }
        if (formData.password.length < 6) {
            setErrorMsg("A senha deve ter pelo menos 6 caracteres.");
            return;
        }

        setIsLoading(true);

        const { data, error } = await supabase.auth.signUp({
            email: formData.email,
            password: formData.password,
            options: {
                data: {
                    full_name: formData.name,
                }
            }
        });

        setIsLoading(false);

        if (error) {
            setErrorMsg(error.message);
        } else {
            setSuccessMsg("Conta criada com sucesso! Você já pode entrar.");
            setTimeout(() => {
                onNavigateLogin();
            }, 2500);
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
                            Criar conta no <span className="text-gradient">Assistente</span>
                        </h2>
                        <p className="auth-description">
                            Cadastre-se para acessar as ferramentas
                        </p>
                    </div>
                </div>

                <div className="auth-content">
                    {errorMsg && <div className="auth-error">⚠️ {errorMsg}</div>}
                    {successMsg && <div className="auth-success">✅ {successMsg}</div>}

                    <form onSubmit={handleSubmit} className="auth-form">
                        <div className="form-group">
                            <label htmlFor="name">Nome completo</label>
                            <input
                                id="name"
                                name="name"
                                type="text"
                                placeholder="Dr. João Silva"
                                required
                                disabled={isLoading}
                                className="auth-input"
                                value={formData.name}
                                onChange={handleChange}
                            />
                        </div>

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
                                value={formData.email}
                                onChange={handleChange}
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="password">Senha <span className="text-xs text-muted font-normal ml-2">(Mín. 6 caracteres)</span></label>
                            <input
                                id="password"
                                name="password"
                                type="password"
                                placeholder="••••••••"
                                minLength={6}
                                required
                                disabled={isLoading}
                                className="auth-input"
                                value={formData.password}
                                onChange={handleChange}
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="confirmPassword">Confirmar senha</label>
                            <input
                                id="confirmPassword"
                                name="confirmPassword"
                                type="password"
                                placeholder="Repita sua senha"
                                minLength={6}
                                required
                                disabled={isLoading}
                                className="auth-input"
                                value={formData.confirmPassword}
                                onChange={handleChange}
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
                                    Criando conta...
                                </>
                            ) : (
                                "Criar conta"
                            )}
                        </button>
                    </form>

                    <p className="auth-footer">
                        Já tem conta?{" "}
                        <button
                            type="button"
                            className="text-primary hover-underline font-medium"
                            onClick={onNavigateLogin}
                        >
                            Entrar
                        </button>
                    </p>
                </div>
            </div>
        </div>
    );
}
