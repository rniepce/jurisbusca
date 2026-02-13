import React from 'react';
import { FaBars } from 'react-icons/fa6';
import './Header.css';

const Header = ({ onMenuClick }) => {
    return (
        <header className="top-header">
            <button className="menu-toggle" onClick={onMenuClick} aria-label="Toggle sidebar">
                <FaBars />
            </button>
            <div className="header-brand">
                <svg className="brand-icon" width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <rect width="24" height="24" rx="4" fill="#c62828" />
                    <path d="M6 8h12M6 12h8M6 16h10" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
                </svg>
                <span className="brand-title">Assistente TJMG</span>
            </div>
        </header>
    );
};

export default Header;
