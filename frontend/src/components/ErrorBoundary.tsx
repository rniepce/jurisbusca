import { Component, type ReactNode, type ErrorInfo } from 'react';

interface Props {
    /** Optional human label used in the fallback message (e.g. "Jurisprudência"). */
    label?: string;
    /** Optional callback invoked when the user clicks "Voltar". */
    onReset?: () => void;
    /** Optional custom fallback renderer. */
    fallback?: (error: Error, reset: () => void) => ReactNode;
    children: ReactNode;
}

interface State {
    error: Error | null;
}

/**
 * Generic error boundary for lazy-loaded panels.
 * Prevents a single panel crash from taking down the whole app.
 */
export default class ErrorBoundary extends Component<Props, State> {
    state: State = { error: null };

    static getDerivedStateFromError(error: Error): State {
        return { error };
    }

    componentDidCatch(error: Error, info: ErrorInfo) {
        // Log to console; can be wired to a remote logger later.
        console.error(`[ErrorBoundary${this.props.label ? ` · ${this.props.label}` : ''}]`, error, info);
    }

    private reset = () => {
        this.setState({ error: null });
        this.props.onReset?.();
    };

    render() {
        const { error } = this.state;
        if (!error) return this.props.children;

        if (this.props.fallback) return this.props.fallback(error, this.reset);

        const label = this.props.label || 'painel';
        return (
            <div
                role="alert"
                style={{
                    padding: '2rem',
                    margin: '1rem',
                    border: '1px solid var(--border-color, #e5e7eb)',
                    borderRadius: '0.75rem',
                    background: 'var(--surface, #fff)',
                    color: 'var(--text-primary, #111827)',
                    fontFamily: 'inherit',
                    maxWidth: '640px',
                }}
            >
                <h2 style={{ margin: '0 0 0.5rem', fontSize: '1.15rem', fontWeight: 600 }}>
                    Não foi possível carregar {label}.
                </h2>
                <p style={{ margin: '0 0 1rem', color: 'var(--text-secondary, #6b7280)', fontSize: '0.9rem' }}>
                    Ocorreu um erro inesperado. Você pode tentar novamente ou voltar para o chat.
                </p>
                <details style={{ marginBottom: '1rem', fontSize: '0.8rem', color: 'var(--text-secondary, #6b7280)' }}>
                    <summary style={{ cursor: 'pointer' }}>Detalhes técnicos</summary>
                    <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', marginTop: '0.5rem' }}>
                        {error.message}
                    </pre>
                </details>
                <button
                    type="button"
                    onClick={this.reset}
                    style={{
                        padding: '0.5rem 1rem',
                        border: '1px solid var(--border-color, #e5e7eb)',
                        borderRadius: '0.5rem',
                        background: 'var(--primary, #2563eb)',
                        color: 'white',
                        cursor: 'pointer',
                        fontSize: '0.9rem',
                    }}
                >
                    Voltar
                </button>
            </div>
        );
    }
}
