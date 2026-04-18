/**
 * m8-stats.js — Canonical engagement stats module
 * ─────────────────────────────────────────────────
 * Single source of truth for Views / Likes / Comments / Shares /
 * Engagement Rate across all pages.
 *
 * Backend source: GET /api/analytics?range=X
 *   → services/canonical_engagement.compute_canonical_engagement_rollup()
 *
 * Usage:
 *   await window.M8Stats.load('30d');          // fetch + cache + fire event
 *   window.M8Stats.applyToElements(data, map); // write values to DOM ids
 *
 * Event: window fires 'uploadm8:stats-loaded' with { detail: { range, data } }
 *
 * Canonical field names (from /api/analytics response):
 *   data.views            — canonical views
 *   data.likes            — canonical likes
 *   data.comments         — canonical comments
 *   data.shares           — canonical shares
 *   data.total_uploads    — upload count in window
 *   data.success_rate_pct — upload success %
 *   data.engagement_rollup_rule  — provenance string
 *   data.breakdown        — detailed sourcing breakdown
 */
(function () {
    'use strict';

    var _cache   = {};   // range → response data
    var _inflight = {};  // range → Promise (dedup concurrent requests)

    // ── Formatting ───────────────────────────────────────────────────────────
    function fmtNum(n) {
        n = Number(n || 0);
        if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
        if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
        if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
        return n.toLocaleString();
    }

    function fmtPct(n) {
        return Number(n || 0).toFixed(1) + '%';
    }

    // ── Load canonical stats ─────────────────────────────────────────────────
    /**
     * Load engagement stats from the canonical backend source.
     * Deduplicates concurrent calls for the same range.
     * Fires 'uploadm8:stats-loaded' on the window when done.
     *
     * @param {string} range  e.g. '30d', '7d', '90d', 'all'
     * @param {object} [opts]
     * @param {boolean} [opts.force]  bypass cache
     * @returns {Promise<object|null>}
     */
    function load(range, opts) {
        range = String(range || '30d');
        opts  = opts || {};

        // Return cached immediately (but still fire event so late-binding widgets update)
        if (_cache[range] && !opts.force) {
            _dispatch(range, _cache[range]);
            return Promise.resolve(_cache[range]);
        }

        // Deduplicate in-flight
        if (_inflight[range]) return _inflight[range];

        var fn = window.apiCall;
        if (typeof fn !== 'function') {
            console.warn('[M8Stats] window.apiCall not available — ensure app.js is loaded first');
            return Promise.resolve(null);
        }

        var qs = range && range !== 'all' ? '?range=' + encodeURIComponent(range) : '?range=all';
        _inflight[range] = fn('/api/analytics' + qs)
            .then(function (data) {
                _cache[range] = data;
                delete _inflight[range];
                _dispatch(range, data);
                return data;
            })
            .catch(function (e) {
                delete _inflight[range];
                console.warn('[M8Stats] load failed for range=' + range, e);
                return null;
            });

        return _inflight[range];
    }

    function _dispatch(range, data) {
        window.dispatchEvent(new CustomEvent('uploadm8:stats-loaded', {
            detail: { range: range, data: data }
        }));
    }

    // ── DOM helpers ──────────────────────────────────────────────────────────
    function _set(id, text) {
        var el = id && document.getElementById(id);
        if (el) el.textContent = text;
    }

    /**
     * Apply canonical stats to DOM elements by id.
     *
     * @param {object} data   Response from /api/analytics
     * @param {object} map    Keys are stat names, values are element ids:
     *   {
     *     views:           'elementId',
     *     likes:           'elementId',
     *     comments:        'elementId',
     *     shares:          'elementId',
     *     engagement_rate: 'elementId',   // derived: (l+c+s)/v
     *     total_uploads:   'elementId',
     *     success_rate:    'elementId',
     *   }
     */
    function applyToElements(data, map) {
        if (!data || !map) return;

        var v   = Number(data.views    || 0);
        var l   = Number(data.likes    || 0);
        var c   = Number(data.comments || 0);
        var s   = Number(data.shares   || 0);
        var eng = v > 0 ? fmtPct((l + c + s) / v * 100) : '0.0%';

        if (map.views)           _set(map.views,           fmtNum(v));
        if (map.likes)           _set(map.likes,           fmtNum(l));
        if (map.comments)        _set(map.comments,        fmtNum(c));
        if (map.shares)          _set(map.shares,          fmtNum(s));
        if (map.engagement_rate) _set(map.engagement_rate, eng);

        if (map.total_uploads)   _set(map.total_uploads,   fmtNum(Number(data.total_uploads || 0)));
        if (map.success_rate) {
            var sr = data.success_rate_pct != null
                ? fmtPct(Number(data.success_rate_pct))
                : (Number(data.total_uploads || 0) > 0
                    ? fmtPct((Number(data.completed || 0) / Number(data.total_uploads)) * 100)
                    : '0.0%');
            _set(map.success_rate, sr);
        }
    }

    /**
     * Invalidate the in-memory cache so the next load() hits the API.
     * @param {string} [range]  specific range to invalidate; omit to clear all
     */
    function invalidate(range) {
        if (range) {
            delete _cache[range];
        } else {
            _cache   = {};
            _inflight = {};
        }
    }

    // ── Public API ───────────────────────────────────────────────────────────
    window.M8Stats = {
        load:           load,
        applyToElements: applyToElements,
        invalidate:     invalidate,
        fmtNum:         fmtNum,
        fmtPct:         fmtPct,
    };

})();
