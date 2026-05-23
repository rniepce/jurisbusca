import { FaSun, FaMoon, FaDesktop } from 'react-icons/fa6';
import { LuPanelLeftClose, LuPanelLeftOpen } from 'react-icons/lu';
import { NavLink } from 'react-router-dom';
import { useTheme } from '../hooks/useTheme';
import './Header.css';

const HEADER_TABS = [
    { to: '/', label: 'Chat', end: true },
    { to: '/sustentacao', label: 'Sessão', end: false },
    { to: '/jurisprudencia', label: 'Jurisprudência', end: false },
    { to: '/modelos', label: 'Modelos', end: false },
] as const;

interface Props {
    onMenuClick: () => void;
    isOpen: boolean;
    onOpenModelManager?: () => void; // mantido por compat, mas não usado
}

const Header = ({ onMenuClick, isOpen }: Props) => {
    const { mode, toggle: toggleTheme } = useTheme();

    const themeIcon = mode === 'light' ? <FaSun size={16} /> : mode === 'dark' ? <FaMoon size={16} /> : <FaDesktop size={16} />;
    const themeLabel =
        mode === 'light' ? 'Tema claro (clique para escuro)' :
        mode === 'dark' ? 'Tema escuro (clique para automático)' :
        'Tema automático (clique para claro)';

    return (
        <header className="top-header">
            <div className="header-left">
                <button className="menu-toggle" onClick={onMenuClick} aria-label="Toggle sidebar">
                    {isOpen ? <LuPanelLeftClose /> : <LuPanelLeftOpen />}
                </button>
            </div>

            <nav className="header-tabs" aria-label="Navegação principal">
                {HEADER_TABS.map((tab) => (
                    <NavLink
                        key={tab.to}
                        to={tab.to}
                        end={tab.end}
                        className={({ isActive }) =>
                            `header-tab ${isActive ? 'active' : ''}`
                        }
                    >
                        {tab.label}
                    </NavLink>
                ))}
            </nav>

            <div className="header-right">
                <button
                    className="header-icon-btn"
                    aria-label={themeLabel}
                    title={themeLabel}
                    onClick={toggleTheme}
                >
                    {themeIcon}
                </button>
            </div>
        </header>
    );
};

export default Header;
