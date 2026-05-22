import React, { useState } from 'react';
import { supabase } from '../services/supabase';
import { AiOutlineLoading3Quarters } from 'react-icons/ai';
import { forgotPasswordSchema, firstError } from '../validation/schemas';
import logoSvg from '../assets/logo.svg';
import './AuthPage.css';

interface Props {
    onNavigateLogin: () => void;
}

export default function ForgotPasswordPage({ onNavigateLogin }: Props) {
    const [email, setEmail] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);
    const [successMsg, setSuccessMsg] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setErrorMsg(null);
        setSuccessMsg(null);

        const parsed = forgotPasswordSchema.safeParse({ email });
        if (!parsed.success) {
            setErrorMsg(firstError(parsed.error));
            return;
        }

        setIsLoading(true);
        // Supabase envia o usuário de volta para esta URL com um token na hash.
        // O SDK detecta automaticamente e dispara o evento PASSWORD_RECOVERY.
        const redirectTo = `${window.location.origin}/reset-password`;
        const { error } = await supabase.auth.resetPasswordForEmail(parsed.data.email, {
            redirectTo,
        });

        setIsLoading(false);

        if (error) {
            setErrorMsg(error.message);
        } else {
            // Mensagem genérica de propósito: não revela se o email existe (evita user enumeration).
            setSuccessMsg(
                'Se houver uma conta vinculada a este email, enviamos um link de recuperação. Verifique sua caixa de entrada (e o spam).'
            );
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
                            Recuperar <span className="text-gradient">Senha</span>
                        </h2>
                        <p className="auth-description">
                            Vamos enviar um link de redefinição para seu email.
                        </p>
                    </div>
                </div>

                <div className="auth-content">
                    {errorMsg && (
                        <div className="auth-error">⚠️ {errorMsg}</div>
                    )}
                    {successMsg && (
                        <div
                            className="auth-error"
                            style={{
                                background: 'color-mix(in srgb, var(--success-color) 10%, transparent)',
                                color: 'var(--success-color)',
                                borderColor: 'var(--success-color)',
                            }}
                        >
                            ✅ {successMsg}
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
                                disabled={isLoading || !!successMsg}
                                className="auth-input"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                autoFocus
                            />
                        </div>

                        <button
                            type="submit"
                            className="auth-btn w-full"
                            disabled={isLoading || !!successMsg}
                        >
                            {isLoading ? (
                                <>
                                    <AiOutlineLoading3Quarters className="spinner" />
                                    Enviando...
                                </>
                            ) : (
                                'Enviar link de recuperação'
                            )}
                        </button>
                    </form>

                    <p className="auth-footer">
                        Lembrou da senha?{' '}
                        <button
                            type="button"
                            className="text-primary hover-underline font-medium"
                            onClick={onNavigateLogin}
                        >
                            Voltar para o login
                        </button>
                    </p>
                </div>
            </div>
        </div>
    );
}
