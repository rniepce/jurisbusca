import { z } from 'zod';

/**
 * Centralized form schemas. Used by dialogs and auth pages for client-side
 * validation. Each schema's `parse()` returns the cleaned value; `safeParse()`
 * returns `{ success, error }` for inline error display.
 */

// ── Auth ───────────────────────────────────────────────────────────────
export const loginSchema = z.object({
    email: z.string().trim().min(1, 'Informe o email.').email('Email inválido.'),
    password: z.string().min(1, 'Informe a senha.'),
});

export const signupSchema = z
    .object({
        name: z.string().trim().min(2, 'Informe seu nome completo.'),
        email: z.string().trim().min(1, 'Informe o email.').email('Email inválido.'),
        password: z.string().min(6, 'A senha deve ter pelo menos 6 caracteres.'),
        confirmPassword: z.string(),
    })
    .refine((d) => d.password === d.confirmPassword, {
        message: 'As senhas não coincidem.',
        path: ['confirmPassword'],
    });

// ── Custom agent creation ──────────────────────────────────────────────
const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;
export const createAgentSchema = z.object({
    name: z.string().trim().min(2, 'O nome precisa ter ao menos 2 caracteres.').max(60, 'O nome é muito longo.'),
    prompt: z.string().trim().min(20, 'O prompt deve ter ao menos 20 caracteres.'),
    color: z.string().regex(HEX_COLOR, 'Cor inválida.'),
});

// ── Helper: pick the first error message from a ZodError ───────────────
export function firstError(err: z.ZodError): string {
    const issue = err.issues[0];
    return issue?.message || 'Dados inválidos.';
}

export type LoginInput = z.infer<typeof loginSchema>;
export type SignupInput = z.infer<typeof signupSchema>;
export type CreateAgentInput = z.infer<typeof createAgentSchema>;
