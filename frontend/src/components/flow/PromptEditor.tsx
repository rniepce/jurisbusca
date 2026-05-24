import { useRef, useState, useEffect } from 'react';

interface Props {
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
    availableLabels: string[];
    minHeight?: number;
}

/**
 * Textarea com autocomplete de variáveis: digite @ e escolha um nó anterior.
 * Insere {{Label do Nó}} no texto, que o engine substitui em runtime.
 */
export default function PromptEditor({ value, onChange, placeholder, availableLabels, minHeight = 100 }: Props) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const [open, setOpen] = useState(false);
    const [filter, setFilter] = useState('');
    const [hoverIdx, setHoverIdx] = useState(0);
    const [pos, setPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });

    const filtered = availableLabels.filter(l => l.toLowerCase().includes(filter.toLowerCase()));

    useEffect(() => {
        if (filtered.length === 0) setOpen(false);
    }, [filtered.length]);

    const insertVar = (label: string) => {
        const ta = textareaRef.current;
        if (!ta) return;
        const start = ta.selectionStart ?? value.length;
        // procura o @ que abriu o autocomplete
        const before = value.slice(0, start);
        const atIndex = before.lastIndexOf('@');
        if (atIndex === -1) return;
        const newValue = value.slice(0, atIndex) + `{{${label}}}` + value.slice(start);
        onChange(newValue);
        setOpen(false);
        // reposiciona cursor depois do {{}}
        setTimeout(() => {
            const newPos = atIndex + label.length + 4;
            ta.focus();
            ta.setSelectionRange(newPos, newPos);
        }, 0);
    };

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const v = e.target.value;
        onChange(v);
        const ta = e.target;
        const cursor = ta.selectionStart ?? v.length;
        const before = v.slice(0, cursor);
        const atIndex = before.lastIndexOf('@');
        if (atIndex >= 0 && (atIndex === 0 || /\s/.test(before[atIndex - 1]))) {
            const fragment = before.slice(atIndex + 1);
            if (!/\s/.test(fragment)) {
                setFilter(fragment);
                setHoverIdx(0);
                setOpen(true);
                // estima posição (relativa ao textarea)
                const lines = before.split('\n');
                const lineCount = lines.length;
                const colCount = lines[lines.length - 1].length;
                setPos({
                    top: 18 + lineCount * 18,
                    left: Math.min(colCount * 7 + 12, ta.offsetWidth - 220),
                });
                return;
            }
        }
        setOpen(false);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (!open || filtered.length === 0) return;
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setHoverIdx(i => (i + 1) % filtered.length);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setHoverIdx(i => (i - 1 + filtered.length) % filtered.length);
        } else if (e.key === 'Enter' || e.key === 'Tab') {
            e.preventDefault();
            insertVar(filtered[hoverIdx]);
        } else if (e.key === 'Escape') {
            setOpen(false);
        }
    };

    return (
        <div className="flow-prompt-wrap">
            <textarea
                ref={textareaRef}
                className="flow-textarea"
                value={value}
                onChange={handleChange}
                onKeyDown={handleKeyDown}
                onBlur={() => setTimeout(() => setOpen(false), 150)}
                placeholder={placeholder}
                style={{ minHeight }}
            />
            {open && filtered.length > 0 && (
                <div className="flow-autocomplete" style={{ top: pos.top, left: pos.left }}>
                    <div className="flow-autocomplete-hint">↑↓ navega · Enter insere · Esc fecha</div>
                    {filtered.map((label, i) => (
                        <button
                            key={label}
                            type="button"
                            className={`flow-autocomplete-item ${i === hoverIdx ? 'hover' : ''}`}
                            onMouseDown={(e) => { e.preventDefault(); insertVar(label); }}
                            onMouseEnter={() => setHoverIdx(i)}
                        >
                            <span className="flow-autocomplete-at">@</span> {label}
                        </button>
                    ))}
                </div>
            )}
            {availableLabels.length > 0 && (
                <div className="flow-prompt-hint">
                    💡 Digite <code>@</code> para usar a saída de outro nó.
                </div>
            )}
        </div>
    );
}
