import React from 'react';
import { FaBars, FaSignOutAlt } from 'react-icons/fa';
import { GoogleLogin } from '@react-oauth/google';
import { useAuth } from '../context/AuthContext';
import logoSvg from '../assets/logo.svg';
import './Header.css';

const Header = ({ onMenuClick }) => {
    const { user, isAuthLoaded, login, logout } = useAuth();

    return (
        <header className="top-header">
            <div className="header-left">
                <button className="menu-toggle" onClick={onMenuClick} aria-label="Toggle sidebar">
                    <FaBars />
                </button>
                <div className="header-brand">
                    <img src={logoSvg} alt="Logo TJMG" className="brand-icon" />
                    <span className="brand-title">Assistente TJMG</span>
                </div>
            </div>

            <div className="header-right">
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
