import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    FaFileLines, FaCopy, FaCheck, FaDownload,
    FaXmark, FaPrint, FaBold, FaItalic,
    FaUnderline, FaStrikethrough, FaListUl, FaListOl,
    FaRotateLeft, FaRotateRight, FaChevronDown,
    FaAlignLeft, FaAlignCenter, FaAlignJustify
} from 'react-icons/fa6';
import { Document, Paragraph, TextRun, Packer } from 'docx';
import { saveAs } from 'file-saver';
import { formatMarkdown } from '../utils/markdown';
import './CanvasEditor.css';

/**
 * Extract a short title from content.
 */
function extractTitle(content) {
    if (!content) return 'Documento';
    const headingMatch = content.match(/^#{1,3}\s+(.+)$/m);
    if (headingMatch) return headingMatch[1].slice(0, 60);
    const firstLine = content.split('\n').find(l => l.trim().length > 0);
    if (firstLine) return firstLine.replace(/[*#_]/g, '').trim().slice(0, 60);
    return 'Documento';
}

/** Heading style options */
const HEADING_OPTIONS = [
    { value: 'p', label: 'Texto normal', tag: 'P' },
    { value: 'h1', label: 'Título 1', tag: 'H1' },
    { value: 'h2', label: 'Título 2', tag: 'H2' },
    { value: 'h3', label: 'Título 3', tag: 'H3' },
    { value: 'h4', label: 'Título 4', tag: 'H4' },
];

const CanvasEditor = ({ content, onClose, onContentChange, onSelectionChange, onFocusChat, isUpdating = false, conversationId }: any) => {
    const [copied, setCopied] = useState(false);
    const [flashKey, setFlashKey] = useState(0);
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
    const [headingDropdownOpen, setHeadingDropdownOpen] = useState(false);
    const [currentHeading, setCurrentHeading] = useState('Texto normal');
    const [activeFormats, setActiveFormats] = useState<Record<string, boolean>>({});
    const [selectionFab, setSelectionFab] = useState<{ text: string; x: number; y: number } | null>(null);
    const [autoSaveIndicator, setAutoSaveIndicator] = useState(false); // "Salvo ✓" badge
    const [draftToast, setDraftToast] = useState(false); // "Rascunho restaurado ✓" toast
    const editorRef = useRef<HTMLDivElement | null>(null);
    const headingRef = useRef<HTMLDivElement | null>(null);
    const contentChangeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const autoSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const lastLlmContent = useRef(''); // tracks LLM content to avoid overwriting user edits

    // Set content into the editor when it changes from LLM
    useEffect(() => {
        if (content && editorRef.current && content !== lastLlmContent.current) {
            lastLlmContent.current = content;
            const html = formatMarkdown(content);
            // Only update if content is from LLM (not user typing)
            if (editorRef.current.dataset.source !== 'user') {
                editorRef.current.innerHTML = html;
            }
            // Defer state updates to avoid cascading-render lint warning
            const now = new Date();
            setTimeout(() => {
                setFlashKey(prev => prev + 1);
                setLastUpdated(now);
            }, 0);
        }
    }, [content]);

    // On mount: restore draft from localStorage if available for this conversationId
    useEffect(() => {
        if (!conversationId || !editorRef.current) return;
        const key = `canvas_draft_${conversationId}`;
        const saved = localStorage.getItem(key);
        if (saved && !content) {
            editorRef.current.innerHTML = formatMarkdown(saved);
            if (onContentChange) onContentChange(saved);
            // Defer to avoid setState-in-effect lint error
            setTimeout(() => {
                setDraftToast(true);
                setTimeout(() => setDraftToast(false), 3000);
            }, 0);
        }
    }, [conversationId]); // eslint-disable-line react-hooks/exhaustive-deps

    // Track text selection for floating action button
    useEffect(() => {
        const handleSelectionEnd = () => {
            if (!editorRef.current) return;
            const sel = window.getSelection();
            if (!sel || sel.isCollapsed || !editorRef.current.contains(sel.anchorNode)) {
                setSelectionFab(null);
                if (onSelectionChange) onSelectionChange(null);
                return;
            }
            const text = sel.toString().trim();
            if (text.length < 5) {
                setSelectionFab(null);
                if (onSelectionChange) onSelectionChange(null);
                return;
            }
            // Position FAB near the selection
            const range = sel.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            const editorRect = editorRef.current.getBoundingClientRect();
            setSelectionFab({
                text,
                x: rect.left - editorRect.left + rect.width / 2,
                y: rect.top - editorRect.top - 44,
            });
            if (onSelectionChange) onSelectionChange(text);
        };
        document.addEventListener('mouseup', handleSelectionEnd);
        return () => document.removeEventListener('mouseup', handleSelectionEnd);
    }, [onSelectionChange]);

    // Close heading dropdown on click outside
    useEffect(() => {
        const handler = (e) => {
            if (headingRef.current && !headingRef.current.contains(e.target)) {
                setHeadingDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    // Track active formatting state on selection change
    const updateActiveFormats = useCallback(() => {
        const formats = {
            bold: document.queryCommandState('bold'),
            italic: document.queryCommandState('italic'),
            underline: document.queryCommandState('underline'),
            strikeThrough: document.queryCommandState('strikeThrough'),
            insertUnorderedList: document.queryCommandState('insertUnorderedList'),
            insertOrderedList: document.queryCommandState('insertOrderedList'),
        };
        setActiveFormats(formats);

        // Detect current block heading
        const block = document.queryCommandValue('formatBlock');
        const match = HEADING_OPTIONS.find(h => h.tag.toLowerCase() === block.toLowerCase());
        setCurrentHeading(match ? match.label : 'Texto normal');
    }, []);

    useEffect(() => {
        document.addEventListener('selectionchange', updateActiveFormats);
        return () => document.removeEventListener('selectionchange', updateActiveFormats);
    }, [updateActiveFormats]);

    // ── Formatting commands ──
    const execFormat = (command: string, value: string | undefined = undefined) => {
        editorRef.current?.focus();
        document.execCommand(command, false, value);
        updateActiveFormats();
    };

    const handleHeadingSelect = (option) => {
        editorRef.current?.focus();
        if (option.value === 'p') {
            document.execCommand('formatBlock', false, 'p');
        } else {
            document.execCommand('formatBlock', false, option.value);
        }
        setCurrentHeading(option.label);
        setHeadingDropdownOpen(false);
        updateActiveFormats();
    };

    // ── Content edit handler — marks edits as user-originated + debounced sync ──
    const handleInput = () => {
        if (editorRef.current) {
            const editor = editorRef.current as HTMLDivElement & { _userTimeout?: ReturnType<typeof setTimeout> };
            editor.dataset.source = 'user';
            // Clear the 'user' flag after a short delay so LLM updates can flow again
            if (editor._userTimeout) clearTimeout(editor._userTimeout);
            editor._userTimeout = setTimeout(() => {
                if (editorRef.current) editorRef.current.dataset.source = '';
            }, 2000);

            // Debounced content sync → parent state
            if (contentChangeTimer.current) clearTimeout(contentChangeTimer.current);
            contentChangeTimer.current = setTimeout(() => {
                if (editorRef.current && onContentChange) {
                    onContentChange(editorRef.current.innerText);
                }
            }, 1000);

            // Auto-save to localStorage (debounced 2s)
            if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current);
            autoSaveTimer.current = setTimeout(() => {
                if (editorRef.current && conversationId) {
                    const textContent = editorRef.current.innerText;
                    localStorage.setItem(`canvas_draft_${conversationId}`, textContent);
                    setAutoSaveIndicator(true);
                    setTimeout(() => setAutoSaveIndicator(false), 2000);
                }
            }, 2000);
        }
    };

    // ── Copy plain text ──
    const handleCopy = async () => {
        const text = editorRef.current?.innerText || content || '';
        if (!text) return;
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    // ── Download TXT ──
    const handleDownload = () => {
        const text = editorRef.current?.innerText || content || '';
        if (!text) return;
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const t = extractTitle(content).replace(/[^a-zA-Z0-9À-ÿ\s_-]/g, '').trim().replace(/\s+/g, '_');
        a.download = `${t || 'minuta'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    // ── Download real DOCX using docx library ──
    const handleDownloadDocx = async () => {
        const html = editorRef.current?.innerHTML || '';
        if (!html) return;

        // Parse HTML into plain text paragraphs
        // Split on block-level boundaries: <br>, </p>, </div>, </h1>-</h6>, </li>
        const rawText = html
            .replace(/<br\s*\/?>/gi, '\n')
            .replace(/<\/p>/gi, '\n')
            .replace(/<\/div>/gi, '\n')
            .replace(/<\/h[1-6]>/gi, '\n')
            .replace(/<\/li>/gi, '\n')
            .replace(/<[^>]+>/g, ''); // strip remaining tags

        // Decode HTML entities
        const decoded = rawText
            .replace(/&amp;/g, '&')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&nbsp;/g, ' ')
            .replace(/&#39;/g, "'")
            .replace(/&quot;/g, '"');

        const lines = decoded.split('\n').map(l => l.trim()).filter(Boolean);

        const docParagraphs = lines.map(line => new Paragraph({
            children: [new TextRun({ text: line, size: 24, font: 'Times New Roman' })],
            spacing: { after: 120 },
        }));

        const doc = new Document({
            sections: [{
                properties: {},
                children: docParagraphs,
            }],
        });

        try {
            const blob = await Packer.toBlob(doc);
            const t = extractTitle(content).replace(/[^a-zA-Z0-9À-ÿ\s_-]/g, '').trim().replace(/\s+/g, '_');
            saveAs(blob, `${t || 'documento'}.docx`);
        } catch (err) {
            console.error('DOCX export failed:', err);
        }
    };

    // ── Handle FAB actions ──
    const handleRefineSelection = () => {
        if (selectionFab?.text && onFocusChat) {
            onFocusChat(selectionFab.text);
        }
        setSelectionFab(null);
    };

    const handleCopySelection = async () => {
        if (!selectionFab?.text) return;
        try {
            await navigator.clipboard.writeText(selectionFab.text);
        } catch {
            // fallback
        }
        setSelectionFab(null);
    };

    // ── Print ──
    const handlePrint = () => {
        const html = editorRef.current?.innerHTML || '';
        const printWindow = window.open('', '_blank');
        if (!printWindow) return;
        printWindow.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>${extractTitle(content)}</title>
                <style>
                    body { font-family: 'Times New Roman', serif; font-size: 13pt; line-height: 1.8; margin: 2cm; color: #000; }
                    h1 { font-size: 18pt; margin: 24px 0 10px; }
                    h2 { font-size: 16pt; margin: 20px 0 8px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
                    h3 { font-size: 14pt; margin: 16px 0 6px; }
                    h4 { font-size: 13pt; margin: 12px 0 4px; }
                    p { margin: 0 0 10px; text-align: justify; }
                    table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                    th, td { border: 1px solid #999; padding: 6px 10px; text-align: left; }
                    th { background: #f0f0f0; }
                    ul, ol { margin: 6px 0 10px 24px; }
                    hr { border: none; border-top: 1px solid #ccc; margin: 14px 0; }
                </style>
            </head>
            <body>${html}</body>
            </html>
        `);
        printWindow.document.close();
        printWindow.print();
    };

    const title = extractTitle(content);
    const timeStr = lastUpdated
        ? lastUpdated.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' })
        : null;

    return (
        <div className="canvas-editor" id="canvas-editor">
            {/* ── Draft restored toast ── */}
            {draftToast && (
                <div className="canvas-draft-toast" aria-live="polite">
                    Rascunho restaurado ✓
                </div>
            )}

            {/* ── Top bar: title + file actions ── */}
            <div className="canvas-toolbar">
                <div className="canvas-title-section">
                    <span className="canvas-title-icon"><FaFileLines /></span>
                    <span className="canvas-title">{title}</span>
                    {timeStr && (
                        <span className="canvas-updated-badge">
                            <span className="canvas-updated-dot" />
                            {timeStr}
                        </span>
                    )}
                    {autoSaveIndicator && (
                        <span className="canvas-autosave-badge" aria-live="polite">
                            Salvo ✓
                        </span>
                    )}
                </div>
                <div className="canvas-toolbar-actions">
                    <button className={`canvas-action-btn ${copied ? 'copied' : ''}`} onClick={handleCopy} title="Copiar conteúdo" aria-label="Copiar conteúdo">
                        {copied ? <FaCheck size={13} /> : <FaCopy size={13} />}
                    </button>
                    <button className="canvas-action-btn" onClick={handleDownload} title="Baixar como .txt" aria-label="Baixar como TXT">
                        <FaDownload size={13} />
                    </button>
                    <button className="canvas-action-btn" onClick={handleDownloadDocx} title="Baixar como .docx" aria-label="Baixar como DOCX">
                        <FaFileLines size={13} />
                    </button>
                    <button className="canvas-action-btn" onClick={handlePrint} title="Imprimir" aria-label="Imprimir documento">
                        <FaPrint size={13} />
                    </button>
                    <button className="canvas-action-btn close-btn" onClick={onClose} title="Fechar Canvas" aria-label="Fechar Canvas">
                        <FaXmark size={14} />
                    </button>
                </div>
            </div>

            {/* ── Formatting toolbar ── */}
            {content && (
                <div className="canvas-format-bar">
                    {/* Undo / Redo */}
                    <button className="fmt-btn" onClick={() => execFormat('undo')} title="Desfazer (Ctrl+Z)">
                        <FaRotateLeft size={12} />
                    </button>
                    <button className="fmt-btn" onClick={() => execFormat('redo')} title="Refazer (Ctrl+Y)">
                        <FaRotateRight size={12} />
                    </button>

                    <span className="fmt-separator" />

                    {/* Heading selector */}
                    <div className="heading-selector" ref={headingRef}>
                        <button
                            className={`heading-selector-btn ${headingDropdownOpen ? 'open' : ''}`}
                            onClick={() => setHeadingDropdownOpen(!headingDropdownOpen)}
                        >
                            <span>{currentHeading}</span>
                            <FaChevronDown size={8} />
                        </button>
                        {headingDropdownOpen && (
                            <div className="heading-dropdown">
                                {HEADING_OPTIONS.map((opt) => (
                                    <button
                                        key={opt.value}
                                        className={`heading-dropdown-item ${currentHeading === opt.label ? 'active' : ''}`}
                                        onClick={() => handleHeadingSelect(opt)}
                                    >
                                        <span className={`heading-preview heading-preview-${opt.value}`}>{opt.label}</span>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    <span className="fmt-separator" />

                    {/* Text formatting */}
                    <button className={`fmt-btn ${activeFormats.bold ? 'active' : ''}`} onClick={() => execFormat('bold')} title="Negrito (Ctrl+B)">
                        <FaBold size={12} />
                    </button>
                    <button className={`fmt-btn ${activeFormats.italic ? 'active' : ''}`} onClick={() => execFormat('italic')} title="Itálico (Ctrl+I)">
                        <FaItalic size={12} />
                    </button>
                    <button className={`fmt-btn ${activeFormats.underline ? 'active' : ''}`} onClick={() => execFormat('underline')} title="Sublinhado (Ctrl+U)">
                        <FaUnderline size={12} />
                    </button>
                    <button className={`fmt-btn ${activeFormats.strikeThrough ? 'active' : ''}`} onClick={() => execFormat('strikeThrough')} title="Tachado">
                        <FaStrikethrough size={12} />
                    </button>

                    <span className="fmt-separator" />

                    {/* Lists */}
                    <button className={`fmt-btn ${activeFormats.insertUnorderedList ? 'active' : ''}`} onClick={() => execFormat('insertUnorderedList')} title="Lista com marcadores">
                        <FaListUl size={12} />
                    </button>
                    <button className={`fmt-btn ${activeFormats.insertOrderedList ? 'active' : ''}`} onClick={() => execFormat('insertOrderedList')} title="Lista numerada">
                        <FaListOl size={12} />
                    </button>

                    <span className="fmt-separator" />

                    {/* Alignment */}
                    <button className="fmt-btn" onClick={() => execFormat('justifyLeft')} title="Alinhar à esquerda">
                        <FaAlignLeft size={12} />
                    </button>
                    <button className="fmt-btn" onClick={() => execFormat('justifyCenter')} title="Centralizar">
                        <FaAlignCenter size={12} />
                    </button>
                    <button className="fmt-btn" onClick={() => execFormat('justifyFull')} title="Justificar">
                        <FaAlignJustify size={12} />
                    </button>
                </div>
            )}

            {/* ── Editable content area ── */}
            {content ? (
                <div className="canvas-content-wrapper" style={{ position: 'relative' }}>
                    <div
                        ref={editorRef}
                        key={flashKey}
                        className={`canvas-document ${isUpdating ? 'updating' : ''}`}
                        contentEditable
                        suppressContentEditableWarning
                        onInput={handleInput}
                        spellCheck
                        dangerouslySetInnerHTML={{ __html: formatMarkdown(content) }}
                    />
                    {/* ── Floating Action Button for selection ── */}
                    {selectionFab && (
                        <div
                            className="canvas-selection-fab"
                            style={{
                                left: `${Math.max(16, Math.min(selectionFab.x - 60, 300))}px`,
                                top: `${Math.max(0, selectionFab.y)}px`,
                            }}
                        >
                            <button className="fab-action" onClick={handleRefineSelection} title="Pedir à IA para refinar este trecho">
                                ✏️ Refinar
                            </button>
                            <button className="fab-action" onClick={handleCopySelection} title="Copiar trecho">
                                📋 Copiar
                            </button>
                        </div>
                    )}
                </div>
            ) : (
                <div className="canvas-empty">
                    <span className="canvas-empty-icon">📄</span>
                    <span className="canvas-empty-title">Canvas Pronto</span>
                    <span className="canvas-empty-sub">
                        Envie uma mensagem pedindo a geração de uma minuta.
                        O documento aparecerá aqui para edição contínua.
                    </span>
                </div>
            )}
        </div>
    );
};

export default CanvasEditor;
