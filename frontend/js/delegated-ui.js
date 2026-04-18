/**
 * Replaces inline onclick= on buttons/links/overlays.
 * - data-um8-fn="fnName" → window.fnName (after api-base + auth-stack). Page-local
 *   functions are invisible unless assigned (e.g. window.closeModal = closeModal).
 *   Handlers are invoked as fn.call(triggerElement, ev, …args) — use `this` (or ev.target)
 *   for the clicked control; ev.currentTarget is the document, not the trigger.
 * - data-um8-arg="single" → passed as second argument (string).
 * - data-um8-args='[1,"a"]' → JSON array; fn.apply(el, [ev, ...parsed]) (numbers stay numbers).
 * - data-um8-open-blank="1" + data-um8-href="https://..." → window.open (no fn needed).
 * - data-um8-stop-propagation="1" on a container → click.stopPropagation() (for nested modals).
 * For <a href="#"> + data-um8-fn, default click is prevented.
 *
 * Modal panels: unconditional stopPropagation on `.modal > .modal-content` breaks this delegate
 * (clicks never reach document). We bind a safe handler on all such panels — call
 * `um8BindModalContentBubbleGuards()` after injecting new modals.
 */
(function () {
    if (window.__UM8_DELEGATED_UI) return;
    window.__UM8_DELEGATED_UI = true;

    window.um8HistoryBack = function () {
        history.back();
    };
    window.um8LocationReload = function () {
        location.reload();
    };
    window.um8WindowPrint = function () {
        window.print();
    };

    document.addEventListener(
        'click',
        function (ev) {
            const t = ev.target;
            const stopEl = t && t.closest && t.closest('[data-um8-stop-propagation="1"]');
            if (stopEl) {
                ev.stopPropagation();
            }

            const el = t && t.closest && t.closest('[data-um8-fn],[data-um8-open-blank]');
            if (!el) return;

            const openBlank = el.getAttribute('data-um8-open-blank');
            if (openBlank === '1' || openBlank === 'true') {
                const href = el.getAttribute('data-um8-href') || '';
                if (href) {
                    ev.preventDefault();
                    window.open(href, '_blank', 'noopener,noreferrer');
                }
                return;
            }

            const name = (el.getAttribute('data-um8-fn') || '').trim();
            if (!name) return;

            if (el.getAttribute('data-um8-stop-before-fn') === '1') {
                ev.stopPropagation();
            }

            const tag = (el.tagName || '').toLowerCase();
            const href = el.getAttribute('href');
            if (tag === 'a' && (href === '#' || href === '#!' || (href || '').startsWith('#'))) {
                ev.preventDefault();
            }

            const fn = window[name];
            if (typeof fn !== 'function') return;

            const rawArgs = el.getAttribute('data-um8-args');
            try {
                if (rawArgs !== null && rawArgs !== '') {
                    const parsed = JSON.parse(rawArgs);
                    const list = Array.isArray(parsed) ? parsed : [parsed];
                    fn.apply(el, [ev].concat(list));
                    return;
                }
                const rawArg = el.getAttribute('data-um8-arg');
                if (rawArg !== null && rawArg !== '') {
                    fn.call(el, ev, rawArg);
                } else {
                    fn.call(el, ev);
                }
            } catch (err) {
                console.error('[data-um8-fn]', name, err);
            }
        },
        false
    );
})();

(function () {
    if (window.__UM8_MODAL_CONTENT_BUBBLE) return;
    window.__UM8_MODAL_CONTENT_BUBBLE = true;

    function bindModalContentBubbleGuards() {
        document.querySelectorAll('.modal > .modal-content:not([data-um8-modal-bubble-bound])').forEach(function (panel) {
            panel.setAttribute('data-um8-modal-bubble-bound', '1');
            panel.addEventListener('click', function (e) {
                if (e.target === e.currentTarget) e.stopPropagation();
            });
        });
    }

    window.um8BindModalContentBubbleGuards = bindModalContentBubbleGuards;
    bindModalContentBubbleGuards();
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bindModalContentBubbleGuards);
    }
})();
