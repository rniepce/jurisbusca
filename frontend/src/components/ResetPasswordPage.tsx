import React, { useState } from 'react';
import { supabase } from '../services/supabase';
import { AiOutlineLoading3Quarters } from 'react-icons/ai';
import { resetPasswordSchema, firstError } from '../validation/schemas';
import { useAuth } from './AuthContext';
import logoSvg from '../assets/logo.svg';
import './AuthPage.css';

interface Props {
    onDone: () => void;
}

/**
 * Renders the new-password form. The user lands here from the email link;
 * Supabase SDK has already created a recovery session by the time this page mounts.
 */
export default function ResetPasswordPage({ onDone }: Props) {
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);
    const [successMsg, setSuccessMsg] = useState<string | null>(null);
    const { signOut, clearPasswordRecovery } = useAuth();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setErrorMsg(null);
        setSuccessMsg(null);

        const parsed = resetPasswordSchema.safeParse({ password, confirmPassword });
        if (!parsed.success) {
            setErrorMsg(firstError(parsed.error));
            return;
        }

        setIsLoading(true);
        const { error } = await supabase.auth.updateUser({ password: parsed.data.password });
        setIsLoading(false);

        if (error) {
            setErrorMsg(error.message);
            return;
        }

        setSuccessMsg('Senha redefinida com sucesso! Você será redirecionado para o login...');
        // Sign out to force a fresh login with the new password.
        setTimeout(async () => {
            clearPasswordRecovery();
            await signOut();
            onDone();
        }, 2000);
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
                            Redefinir <span className="text-gradient">Senha</span>
                        </h2>
                        <p className="auth-description">
                            Crie uma senha nova para entrar.
                        </p>
                    </div>
                </div>

                <div className="auth-content">
                    {errorMsg && <div className="auth-error">⚠️ {errorMsg}</div>}
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
                            <label htmlFor="password">Nova senha</label>
                            <input
                                id="password"
                                name="password"
                                type="password"
                                placeholder="Mínimo 6 caracteres"
                                required
                                disabled={isLoading || !!successMsg}
                                className="auth-input"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                autoFocus
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="confirmPassword">Confirmar nova senha</label>
                            <input
                                id="confirmPassword"
                                name="confirmPassword"
                                type="password"
                                placeholder="Repita a senha"
                                required
                                disabled={isLoading || !!successMsg}
                                className="auth-input"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
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
                                    Salvando...
                                </>
                            ) : (
                                'Redefinir senha'
                            )}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}
