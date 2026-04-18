/**
 * Ensures window.um8EscapeHtml exists before any inline or page script runs.
 * auth-stack.js overwrites with the same implementation when loaded later.
 */
(function () {
    if (typeof window.um8EscapeHtml === 'function') return;
    window.um8EscapeHtml = function (value) {
        if (value === null || value === undefined) return '';
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    };
})();
