import DOMPurify from 'dompurify';

/**
 * Sanitizes an HTML fragment that already contains a small set of trusted tags
 * (e.g. the FTS5 snippet, which wraps matched terms in <mark>). All other tags
 * and every attribute/event handler are stripped. Use for any server- or
 * document-derived HTML that is NOT produced by formatMarkdown.
 */
export function sanitizeSnippet(html: any): string {
    if (typeof html !== 'string') return '';
    return DOMPurify.sanitize(html, { ALLOWED_TAGS: ['mark'], ALLOWED_ATTR: [] });
}

/**
 * Minimal markdown-to-HTML renderer used across the app.
 * Handles: headers, bold/italic, code blocks, inline code, lists, tables, hr,
 * line breaks, and robust coercion of non-string LLM payloads (arrays/objects).
 *
 * Output is injected via dangerouslySetInnerHTML: it escapes & < > before
 * processing markdown AND runs the final HTML through DOMPurify, so LLM- or
 * document-derived content cannot smuggle <script>/onerror/javascript: through.
 */
export function formatMarkdown(text: any): string {
    if (text === null || text === undefined) return '';

    // Coerce non-string payloads from LLM/Langchain responses
    if (typeof text !== 'string') {
        if (Array.isArray(text)) {
            text = text.map((item: any) => {
                if (typeof item === 'string') return item;
                if (item?.type === 'thinking') return '';
                return item?.text || item?.content || item?.message || JSON.stringify(item);
            }).filter(Boolean).join('\n');
        } else if (typeof text === 'object') {
            text = text.text || text.content || text.message || text.output || JSON.stringify(text);
        } else {
            text = String(text);
        }
    }

    if (!text || typeof text !== 'string') return '';

    const processed = text.replace(/\\n/g, '\n');
    let html = processed
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/^### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^## (.+)$/gm, '<h3>$1</h3>')
        .replace(/^# (.+)$/gm, '<h2>$1</h2>')
        .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/^---$/gm, '<hr/>')
        .replace(/^(\|.+\|)\n\|[-| :]+\|\n((?:\|.+\|\n?)+)/gm, (_match, header: string, body: string) => {
            const headers = header.split('|').filter((c: string) => c.trim()).map((c: string) => `<th>${c.trim()}</th>`).join('');
            const rows = body.trim().split('\n').map((row: string) => {
                const cols = row.split('|').filter((c: string) => c.trim()).map((c: string) => `<td>${c.trim()}</td>`).join('');
                return `<tr>${cols}</tr>`;
            }).join('');
            return `<table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
        })
        .replace(/^\s*[-*] (.+)$/gm, '<li>$1</li>')
        .replace(/^\s*\d+\.\s(.+)$/gm, '<li>$1</li>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br/>');

    html = html.replace(/((?:<li>.*?<\/li>(?:<br\/>)?)+)/g, '<ul>$1</ul>');
    html = html.replace(/<ul>([\s\S]*?)<\/ul>/g, (_m, inner: string) => '<ul>' + inner.replace(/<br\/>/g, '') + '</ul>');

    if (!html.startsWith('<h') && !html.startsWith('<pre') && !html.startsWith('<ul') && !html.startsWith('<table')) {
        html = '<p>' + html + '</p>';
    }

    // Final safety net: strip any script/event-handler/javascript: that slipped
    // through the regex pipeline (e.g. from LLM output or RAG source documents).
    return DOMPurify.sanitize(html);
}
