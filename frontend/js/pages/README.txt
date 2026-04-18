Page-specific JS (progressive extraction from inline onclick/onchange).

Target: move handlers from HTML into js/pages/<name>.js and bind from a single DOMContentLoaded.
Start with high-traffic pages (queue, scheduled, dashboard) when touching those files.

Pattern (queue.html example — apply when refactoring):
  <button type="button" data-queue-action="refresh">…</button>
  // js/pages/queue-actions.js
  document.addEventListener('DOMContentLoaded', () => {
    document.body.addEventListener('click', (ev) => {
      const btn = ev.target.closest('[data-queue-action]');
      if (!btn) return;
      const act = btn.getAttribute('data-queue-action');
      if (act === 'refresh' && typeof refreshQueue === 'function') refreshQueue();
    });
  });
