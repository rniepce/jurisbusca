import { useCallback, useRef, useState } from 'react';
import { useUIStore } from '../store';

export function useCanvas() {
    const { setSidebarOpen } = useUIStore();
    const [canvasOpen, setCanvasOpen] = useState(false);
    const [canvasContent, setCanvasContent] = useState('');
    const [canvasSelection, setCanvasSelection] = useState<string | null>(null);
    const chatTextareaRef = useRef<HTMLTextAreaElement | null>(null);

    const handleCanvasToggle = useCallback(() => {
        setCanvasOpen((prev) => {
            const next = !prev;
            if (next) {
                setSidebarOpen(false);
            } else if (window.innerWidth > 768) {
                setSidebarOpen(true);
            }
            return next;
        });
    }, [setSidebarOpen]);

    const handleCanvasClose = useCallback(() => {
        setCanvasOpen(false);
        if (window.innerWidth > 768) setSidebarOpen(true);
    }, [setSidebarOpen]);

    return {
        canvasOpen,
        canvasContent,
        canvasSelection,
        chatTextareaRef,
        setCanvasOpen,
        setCanvasContent,
        setCanvasSelection,
        handleCanvasToggle,
        handleCanvasClose,
    };
}
