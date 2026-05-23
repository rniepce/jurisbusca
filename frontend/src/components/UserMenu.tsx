import { useEffect, useRef, useState } from 'react';
import { FaChevronUp, FaGear, FaArrowRightFromBracket } from 'react-icons/fa6';
import { useAuth } from './AuthContext';
import './UserMenu.css';

interface Props {
    onOpenMemory: () => void;
}

function getInitials(name?: string | null, email?: string | null): string {
    if (name) {
        const parts = name.trim().split(/\s+/);
        const first = parts[0]?.[0] || '';
        const last = parts.length > 1 ? parts[parts.length - 1][0] : '';
        return (first + last).toUpperCase();
    }
    if (email) return email[0].toUpperCase();
    return '?';
}

/**
 * Single bottom-of-sidebar button: shows user name + avatar; opens a popover
 * with Configurações (memory) and Sair (signOut).
 */
export default function UserMenu({ onOpenMemory }: Props) {
    const { user, signOut } = useAuth();
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLDivElement | null>(null);

    const fullName = (user?.user_metadata?.full_name as string | undefined) || null;
    const email = user?.email || null;
    const displayName = fullName || email || 'Conta';
    const initials = getInitials(fullName, email);

    // Close popover on outside click
    useEffect(() => {
        if (!open) return;
        const handler = (e: MouseEvent) => {
            if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [open]);

    // Close on ESC
    useEffect(() => {
        if (!open) return;
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setOpen(false);
        };
        document.addEventListener('keydown', onKey);
        return () => document.removeEventListener('keydown', onKey);
    }, [open]);

    return (
        <div className="user-menu-wrapper" ref={ref}>
            {open && (
                <div className="user-menu-popover" role="menu">
                    <div className="user-menu-header">
                        <div className="user-menu-header-avatar">{initials}</div>
                        <div className="user-menu-header-text">
                            {fullName && <span className="user-menu-header-name">{fullName}</span>}
                            {email && <span className="user-menu-header-email">{email}</span>}
                        </div>
                    </div>

                    <div className="user-menu-divider" />

                    <button
                        type="button"
                        className="user-menu-item"
                        onClick={() => {
                            setOpen(false);
                            onOpenMemory();
                        }}
                        role="menuitem"
                    >
                        <FaGear size={13} />
                        <span>Configurações</span>
                    </button>

                    <button
                        type="button"
                        className="user-menu-item danger"
                        onClick={() => {
                            setOpen(false);
                            signOut();
                        }}
                        role="menuitem"
                    >
                        <FaArrowRightFromBracket size={13} />
                        <span>Sair</span>
                    </button>
                </div>
            )}

            <button
                type="button"
                className={`user-menu-trigger ${open ? 'open' : ''}`}
                onClick={() => setOpen((v) => !v)}
                aria-haspopup="menu"
                aria-expanded={open}
                title={displayName}
            >
                <span className="user-menu-avatar">{initials}</span>
                <span className="user-menu-name">{displayName}</span>
                <FaChevronUp size={11} className="user-menu-chevron" />
            </button>
        </div>
    );
}
