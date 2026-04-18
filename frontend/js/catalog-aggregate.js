/**
 * catalog-aggregate.js
 * ─────────────────────
 * Shared "All Connected Accounts" catalog aggregate widget.
 *
 * Usage — add this anywhere in a page body:
 *   <div data-catalog-agg="true"></div>
 *   <script src="js/catalog-aggregate.js"></script>
 *
 * Or call directly from page JS:
 *   window.CatalogAgg.load('30d');
 *   window.CatalogAgg.render(containerId, data);
 */
(function () {
    'use strict';

    const PLAT_COLORS = { tiktok: '#69c9d0', youtube: '#ff0000', instagram: '#e1306c', facebook: '#1877f2' };
    const PLAT_ICONS  = { tiktok: 'fab fa-tiktok', youtube: 'fab fa-youtube', instagram: 'fab fa-instagram', facebook: 'fab fa-facebook-f' };
    const PLAT_ORDER  = ['tiktok', 'youtube', 'instagram', 'facebook'];

    /** Merge catalog by_platform with linked accounts so platforms with no catalog rows still show a chip (e.g. TikTok connected but not yet synced). */
    function _mergeByPlatformWithAccounts(d, accountsPayload) {
        const byp = d.by_platform || {};
        const accounts = (accountsPayload && accountsPayload.accounts) ? accountsPayload.accounts : [];
        const connected = new Set();
        accounts.forEach(function (a) {
            var p = String((a && a.platform) || '').toLowerCase();
            if (p) connected.add(p);
        });
        var keys = new Set(Object.keys(byp));
        connected.forEach(function (k) { keys.add(k); });
        var empty = { video_count: 0, views: 0, likes: 0, comments: 0, shares: 0, engagement_rate: 0 };
        var out = {};
        PLAT_ORDER.forEach(function (p) {
            if (!keys.has(p)) return;
            out[p] = byp[p] ? Object.assign({}, byp[p]) : Object.assign({}, empty);
        });
        keys.forEach(function (p) {
            if (out[p] != null) return;
            out[p] = byp[p] ? Object.assign({}, byp[p]) : Object.assign({}, empty);
        });
        d.by_platform = out;
        return d;
    }

    // Preset period options
    const PERIOD_PRESETS = [
        { label: '1 Hour',    value: '1h' },
        { label: '6 Hours',   value: '6h' },
        { label: '24 Hours',  value: '24h' },
        { label: '7 Days',    value: '7d' },
        { label: '30 Days',   value: '30d' },
        { label: '90 Days',   value: '90d' },
        { label: '1 Year',    value: '365d' },
        { label: 'All Time',  value: 'all' },
        { label: 'Custom…',   value: '__custom__' },
    ];

    function fmtNum(n) {
        n = Number(n || 0);
        if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
        if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
        if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
        return n.toLocaleString();
    }

    function agoLabel(isoTs) {
        if (!isoTs) return '';
        const ago = Math.round((Date.now() - new Date(isoTs)) / 60000);
        if (ago < 2) return 'just now';
        if (ago < 60) return ago + 'm ago';
        if (ago < 1440) return Math.floor(ago / 60) + 'h ago';
        return Math.floor(ago / 1440) + 'd ago';
    }

    /**
     * Build an HTML string for the full widget.
     * Called once per container; subsequent refreshes only update inner numbers.
     */
    function buildWidgetHtml(containerId, currentPeriod) {
        const opts = PERIOD_PRESETS.map(p =>
            `<option value="${p.value}"${p.value === currentPeriod ? ' selected' : ''}>${p.label}</option>`
        ).join('');

        return `
<div style="background:rgba(139,92,246,0.07);border:1px solid rgba(139,92,246,0.2);border-radius:12px;padding:1rem 1.25rem;">
  <!-- Header row -->
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.75rem;">
    <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#a78bfa;display:flex;align-items:center;gap:0.4rem;">
      <i class="fas fa-globe"></i> All Connected Accounts
      <span id="${containerId}-sync-lbl" style="font-size:0.67rem;font-weight:400;color:var(--text-muted);font-style:italic;margin-left:0.25rem;"></span>
    </div>
    <!-- Controls: period selector + custom input + refresh + sync -->
    <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">
      <select id="${containerId}-period-sel"
              style="padding:0.25rem 0.5rem;border-radius:7px;border:1px solid var(--border-color);background:var(--bg-card);color:var(--text-primary);font-size:0.75rem;font-family:inherit;cursor:pointer;"
              onchange="window.CatalogAgg._onSelectChange('${containerId}')">
        ${opts}
      </select>
      <input id="${containerId}-custom-input" type="text" placeholder="e.g. 48h, 14d, 90m"
             style="display:none;width:9rem;padding:0.25rem 0.5rem;border-radius:7px;border:1px solid var(--border-color);background:var(--bg-card);color:var(--text-primary);font-size:0.75rem;font-family:inherit;"
             onkeydown="if(event.key==='Enter')window.CatalogAgg._applyCustom('${containerId}')">
      <button id="${containerId}-apply-btn"
              style="display:none;padding:0.25rem 0.65rem;border-radius:7px;border:1px solid var(--border-color);background:var(--bg-secondary);color:var(--text-primary);font-size:0.72rem;font-weight:600;cursor:pointer;font-family:inherit;"
              onclick="window.CatalogAgg._applyCustom('${containerId}')">Apply</button>
      <button id="${containerId}-refresh-btn"
              style="padding:0.25rem 0.75rem;border-radius:7px;border:1px solid rgba(139,92,246,0.35);background:rgba(139,92,246,0.1);color:#a78bfa;font-size:0.72rem;font-weight:600;cursor:pointer;font-family:inherit;display:flex;align-items:center;gap:0.3rem;"
              onclick="window.CatalogAgg._refresh('${containerId}')">
        <i class="fas fa-sync-alt" id="${containerId}-refresh-icon"></i> Refresh
      </button>
      <button id="${containerId}-sync-btn"
              style="padding:0.25rem 0.75rem;border-radius:7px;border:1px solid rgba(249,115,22,0.35);background:rgba(249,115,22,0.08);color:#f97316;font-size:0.72rem;font-weight:600;cursor:pointer;font-family:inherit;display:flex;align-items:center;gap:0.3rem;"
              onclick="window.CatalogAgg._sync('${containerId}')">
        <i class="fas fa-cloud-download-alt" id="${containerId}-sync-icon"></i> Sync Catalog
      </button>
    </div>
  </div>

  <!-- KPI grid -->
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.6rem;margin-bottom:0.75rem;"
       id="${containerId}-kpi-grid">
    ${_kpiCell(containerId,'views','fa-eye','Views')}
    ${_kpiCell(containerId,'likes','fa-thumbs-up','Likes')}
    ${_kpiCell(containerId,'comments','fa-comment','Comments')}
    ${_kpiCell(containerId,'eng','fa-chart-line','Engagement','#f97316')}
  </div>
  <div style="font-size:0.68rem;color:var(--text-muted);margin-bottom:0.5rem;" id="${containerId}-sub-label"></div>

  <!-- Per-platform chips -->
  <div id="${containerId}-plat-chips" style="display:flex;gap:0.4rem;flex-wrap:wrap;"></div>
</div>`;
    }

    function _kpiCell(cid, key, icon, label, color) {
        const valColor = color || 'var(--text-primary)';
        return `<div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:0.6rem;text-align:center;">
    <div style="font-size:1.05rem;font-weight:700;color:${valColor};" id="${cid}-${key}">--</div>
    <div style="font-size:0.6rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-muted);margin-top:2px;">
      <i class="fas ${icon}" style="opacity:.55;"></i> ${label}
    </div>
  </div>`;
    }

    function _applyData(containerId, d) {
        const setT = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        setT(containerId + '-views',    fmtNum(d.views));
        setT(containerId + '-likes',    fmtNum(d.likes));
        setT(containerId + '-comments', fmtNum(d.comments));
        setT(containerId + '-eng',      (d.engagement_rate || 0).toFixed(1) + '%');

        // Sub-label: source breakdown
        const bs = d.by_source || {};
        const extV = bs.external?.video_count || 0;
        const m8V  = (bs.uploadm8?.video_count || 0) + (bs.linked?.video_count || 0);
        const total = d.total_videos || 0;
        const sub = document.getElementById(containerId + '-sub-label');
        if (sub) sub.textContent =
            `${fmtNum(total)} videos tracked · ${fmtNum(m8V)} UploadM8 · ${fmtNum(extV)} external`;

        // Sync label
        const syncRow = (d.sync_status || [])[0];
        const lbl = document.getElementById(containerId + '-sync-lbl');
        if (lbl && syncRow && syncRow.last_synced_at)
            lbl.textContent = '· synced ' + agoLabel(syncRow.last_synced_at);

        // Period label in selector (reflect period returned from API)
        const sel = document.getElementById(containerId + '-period-sel');
        if (sel && d.period && d.period !== sel.value) {
            // Try to match; if not a preset, keep selector as-is
            const match = Array.from(sel.options).find(o => o.value === d.period);
            // Only auto-select if it's a known preset, not __custom__
            if (match && match.value !== '__custom__') sel.value = d.period;
        }

        // Per-platform chips
        const chips = document.getElementById(containerId + '-plat-chips');
        if (chips) {
            const entries = Object.entries(d.by_platform || {});
            if (entries.length === 0) {
                chips.innerHTML = '<span style="font-size:.72rem;color:var(--text-muted);font-style:italic;">No catalog data yet — click Sync Catalog to import</span>';
            } else {
                chips.innerHTML = entries.map(([plat, pd]) => {
                    const col  = PLAT_COLORS[plat] || '#6b7280';
                    const icon = PLAT_ICONS[plat]  || 'fas fa-globe';
                    const eng  = (pd.engagement_rate || 0).toFixed(1);
                    return `<span style="display:inline-flex;align-items:center;gap:.35rem;padding:.2rem .65rem;border-radius:20px;background:var(--bg-card);border:1px solid var(--border-color);font-size:.72rem;font-weight:600;">
                        <i class="${icon}" style="color:${col};font-size:.78rem;"></i>
                        <span style="color:var(--text-muted);text-transform:capitalize;">${plat}</span>
                        <span style="color:var(--text-primary);">${fmtNum(pd.views || 0)}</span>
                        <span style="color:var(--text-muted);font-size:.6rem;">${fmtNum(pd.video_count || 0)} vids</span>
                        <span style="color:#f97316;font-size:.65rem;">${eng}%</span>
                    </span>`;
                }).join('');
            }
        }
    }

    // State per container
    const _state = {};

    function _currentPeriod(containerId) {
        return (_state[containerId] || {}).period || '30d';
    }

    async function _load(containerId, period) {
        _state[containerId] = _state[containerId] || {};
        _state[containerId].period = period || _currentPeriod(containerId);

        const periodParam = _state[containerId].period;
        const qs = periodParam && periodParam !== 'all' ? `?period=${encodeURIComponent(periodParam)}` : '?period=all';

        try {
            const [d, acct] = await Promise.all([
                window.apiCall('/api/catalog/aggregate' + qs).catch(function () { return null; }),
                window.apiCall('/api/platform-accounts').catch(function () { return null; }),
            ]);
            if (!d) return;
            _mergeByPlatformWithAccounts(d, acct);
            _applyData(containerId, d);
            _state[containerId].lastData = d;
        } catch (e) {
            console.warn('[CatalogAgg]', e);
        }
    }

    function _onSelectChange(containerId) {
        const sel = document.getElementById(containerId + '-period-sel');
        const customInput = document.getElementById(containerId + '-custom-input');
        const applyBtn    = document.getElementById(containerId + '-apply-btn');
        if (!sel) return;
        if (sel.value === '__custom__') {
            if (customInput) customInput.style.display = '';
            if (applyBtn)    applyBtn.style.display = '';
            if (customInput) customInput.focus();
        } else {
            if (customInput) customInput.style.display = 'none';
            if (applyBtn)    applyBtn.style.display = 'none';
            _load(containerId, sel.value);
        }
    }

    function _applyCustom(containerId) {
        const input = document.getElementById(containerId + '-custom-input');
        if (!input) return;
        const raw = (input.value || '').trim();
        if (!raw) return;
        _load(containerId, raw);
    }

    function _refresh(containerId) {
        const icon = document.getElementById(containerId + '-refresh-icon');
        if (icon) icon.classList.add('fa-spin');
        _load(containerId, _currentPeriod(containerId)).finally(() => {
            if (icon) icon.classList.remove('fa-spin');
        });
    }

    async function _sync(containerId) {
        const btn  = document.getElementById(containerId + '-sync-btn');
        const icon = document.getElementById(containerId + '-sync-icon');
        if (btn)  { btn.disabled = true; }
        if (icon) icon.className = 'fas fa-spinner fa-spin';
        try {
            await window.apiCall('/api/catalog/sync?async_mode=true', { method: 'POST', body: '{}' });
            window.showToast && window.showToast('Catalog sync queued — refreshing in 12s', 'info');
            setTimeout(() => _refresh(containerId), 12000);
        } catch (e) {
            window.showToast && window.showToast('Sync failed: ' + (e.message || ''), 'error');
        } finally {
            if (btn)  btn.disabled = false;
            if (icon) icon.className = 'fas fa-cloud-download-alt';
        }
    }

    /**
     * Mount the widget into a container element.
     * @param {string} containerId  id of the wrapper <div>
     * @param {string} [initialPeriod]  e.g. '30d', '7h', 'all'
     */
    function mount(containerId, initialPeriod) {
        const el = document.getElementById(containerId);
        if (!el) return;
        const period = initialPeriod || '30d';
        el.innerHTML = buildWidgetHtml(containerId, period);
        _load(containerId, period);
    }

    // Auto-mount any elements with data-catalog-agg="true"
    function _autoMount() {
        document.querySelectorAll('[data-catalog-agg]').forEach(function (el) {
            if (!el.id) el.id = 'catalog-agg-' + Math.random().toString(36).slice(2, 7);
            const period = el.getAttribute('data-catalog-agg-period') || '30d';
            mount(el.id, period);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _autoMount);
    } else {
        _autoMount();
    }

    // Public API
    window.CatalogAgg = {
        mount,
        load: _load,
        refresh: _refresh,
        _onSelectChange,
        _applyCustom,
        _refresh,
        _sync,
    };
})();
