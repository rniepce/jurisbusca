import React, { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from '../services/supabase';
import type { User, Session } from '@supabase/supabase-js';

interface AuthContextType {
    user: User | null;
    session: Session | null;
    loading: boolean;
    /** True when the current session originated from a password-recovery email link.
     *  AppRoutes uses this to force-route to /reset-password regardless of `user`. */
    passwordRecovery: boolean;
    signOut: () => Promise<void>;
    clearPasswordRecovery: () => void;
}

const AuthContext = createContext<AuthContextType>({
    user: null,
    session: null,
    loading: true,
    passwordRecovery: false,
    signOut: async () => {},
    clearPasswordRecovery: () => {},
});

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [user, setUser] = useState<User | null>(null);
    const [session, setSession] = useState<Session | null>(null);
    const [loading, setLoading] = useState(true);
    const [passwordRecovery, setPasswordRecovery] = useState(false);

    useEffect(() => {
        // Initial session
        supabase.auth.getSession().then(({ data: { session } }) => {
            setSession(session);
            setUser(session?.user ?? null);
            setLoading(false);
        });

        // Auth state listener
        const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
            setSession(session);
            setUser(session?.user ?? null);
            setLoading(false);
            // Supabase fires PASSWORD_RECOVERY when user lands from the reset email link.
            // We keep the flag on until the user finishes resetting (or signs out).
            if (event === 'PASSWORD_RECOVERY') setPasswordRecovery(true);
        });

        return () => subscription.unsubscribe();
    }, []);

    const signOut = async () => {
        await supabase.auth.signOut();
        setPasswordRecovery(false);
    };

    const clearPasswordRecovery = () => setPasswordRecovery(false);

    return (
        <AuthContext.Provider
            value={{ user, session, loading, passwordRecovery, signOut, clearPasswordRecovery }}
        >
            {children}
        </AuthContext.Provider>
    );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useAuth = () => {
    return useContext(AuthContext);
};
