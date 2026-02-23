import React, { createContext, useContext, useState } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(() => {
        try {
            const storedUser = localStorage.getItem('jurisbusca_user');
            return storedUser ? JSON.parse(storedUser) : null;
        } catch (e) {
            console.error("Erro ao fazer parse do usuário salvo", e);
            return null;
        }
    });
    const [token, setToken] = useState(() => localStorage.getItem('jurisbusca_token'));
    const [isAuthLoaded] = useState(true);

    const login = (credentialResponse) => {
        // credentialResponse possui a JWT (credential)
        const jwt = credentialResponse.credential;
        setToken(jwt);
        localStorage.setItem('jurisbusca_token', jwt);

        // Decodificar o payload JWT para pegar email e nome (básico, frontend apenas para exibição)
        try {
            const payload = JSON.parse(atob(jwt.split('.')[1]));
            const userData = {
                name: payload.name,
                email: payload.email,
                picture: payload.picture,
            };
            setUser(userData);
            localStorage.setItem('jurisbusca_user', JSON.stringify(userData));
        } catch (e) {
            console.error("Erro ao decodificar JWT", e);
        }
    };

    const logout = () => {
        setUser(null);
        setToken(null);
        localStorage.removeItem('jurisbusca_token');
        localStorage.removeItem('jurisbusca_user');
    };

    return (
        <AuthContext.Provider value={{ user, token, isAuthLoaded, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
