import React from 'react';
import { FaBars } from 'react-icons/fa6';
import logoSvg from '../assets/logo.svg';
import './Header.css';

const Header = ({ onMenuClick }) => {
    return (
        <header className="top-header">
            <button className="menu-toggle" onClick={onMenuClick} aria-label="Toggle sidebar">
                <FaBars />
            </button>
            <div className="header-brand">
                <img src={logoSvg} alt="Logo TJMG" className="brand-icon" />
                <span className="brand-title">Assistente TJMG</span>
            </div>
        </header>
    );
};

export default Header;
