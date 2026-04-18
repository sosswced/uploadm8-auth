/**
 * UploadM8 — personalized tips coach (GET /api/me/coach).
 * Renders into elements with [data-ai-coach]. Dispatches uploadm8:coach for wallet hooks.
 * Surfaces content_attribution_insights from the coach payload (performance rollups + apply-optimized).
 */
(function () {
  'use strict';

  var coachBootStarted = false;
  var coachFetchInFlight = false;
  var coachHasRendered = false;
  var coachPendingForce = false;
  var COACH_FALLBACK_MS = 900;

  function esc(s) {
    if (s == null) return '';
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }

  function sevClass(sev) {
    if (sev === 'warning') return 'border-color:rgba(251,191,36,.45);background:rgba(251,191,36,.08);';
    if (sev === 'info') return 'border-color:rgba(59,130,246,.4);background:rgba(59,130,246,.07);';
    return 'border-color:rgba(139,92,246,.35);background:rgba(139,92,246,.06);';
  }

  function _isLikelyLoggedIn() {
    try {
      if (window.getAccessToken && window.getAccessToken()) return true;
      if (window.currentUser && window.currentUser.email) return true;
      if (typeof window.isLoggedIn === 'function' && window.isLoggedIn()) return true;
    } catch (_) {}
    return false;
  }

  function bindApplyInsightsButton(container) {
    var btn = container.querySelector('#um8CoachApplyBtn');
    var st = container.querySelector('#um8CoachApplyStatus');
    if (!btn) return;
    btn.addEventListener('click', async function () {
      btn.disabled = true;
      if (st) st.textContent = 'Applying…';
      try {
        var resp;
        if (typeof window.apiCall === 'function') {
          resp = await window.apiCall('/api/me/content-insights/apply-optimized', {
            method: 'POST',
            body: JSON.stringify({ confirm: true }),
          });
        } else if (typeof window.apiFetch === 'function') {
          var r = await window.apiFetch('/api/me/content-insights/apply-optimized', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confirm: true }),
            authRedirectOn401: true,
          });
          resp = r.ok ? await r.json().catch(function () { return {}; }) : null;
          if (!r.ok) {
            var errText = await r.text().catch(function () { return ''; });
            throw new Error(errText || r.status);
          }
        } else {
          throw new Error('apiCall unavailable');
        }
        if (st) {
          st.textContent = resp && resp.ok ? 'Saved to Settings.' : 'Done — review Settings.';
        }
      } catch (e) {
        if (st) st.textContent = 'Could not apply — try Settings manually.';
        console.warn('[ai-coach] apply-optimized', e);
      } finally {
        btn.disabled = false;
      }
    });
  }

  function renderSlot(el, data) {
    if (!el) return;
    if (!data) {
      if (_isLikelyLoggedIn()) {
        el.innerHTML =
          '<span class="text-secondary">Tips could not be loaded (network or server).</span> '
          + '<button type="button" class="btn btn-secondary btn-sm" style="margin-left:.35rem;" '
          + 'onclick="location.reload()">Retry</button>';
        return;
      }
      el.innerHTML =
        '<span class="text-secondary">Sign in to load personalized tips.</span> '
        + '<a class="btn btn-primary btn-sm" style="margin-left:.35rem;" href="/login.html">Log in</a> '
        + '<span class="text-secondary" style="font-size:.78rem;"> · New here? <a href="/signup.html">Create an account</a></span>';
      return;
    }

    const ins = data.content_attribution_insights;
    const ranked = ins && ins.ok && Array.isArray(ins.ranked_strategies) ? ins.ranked_strategies : [];
    const hasInsights = !!(ins && ins.ok && (ins.narrative || ranked.length));

    let html = '';

    if (hasInsights) {
      html += '<div style="margin-bottom:.85rem;padding:.65rem .8rem;border-radius:10px;border:1px solid rgba(34,197,94,.4);background:rgba(34,197,94,.08);">';
      html += '<div style="font-weight:700;font-size:.88rem;color:var(--text-primary);margin-bottom:.25rem;">'
        + '<i class="fas fa-chart-line" style="margin-right:.4rem;color:#22c55e;"></i>'
        + 'What\'s working for you</div>';
      html += '<div class="text-secondary" style="font-size:.72rem;line-height:1.4;margin-bottom:.4rem;opacity:.9;">'
        + 'We compare your recent posts so suggestions match how your audience actually responds.</div>';
      if (ins.narrative) {
        html += '<div class="text-secondary" style="font-size:.82rem;line-height:1.5;margin-bottom:.5rem;">' + esc(ins.narrative) + '</div>';
      }
      if (ranked.length) {
        html += '<div style="font-size:.74rem;font-weight:600;margin-bottom:.3rem;color:var(--text-secondary);">Strongest choices by interaction</div>';
        html += '<ul style="margin:0;padding-left:1.1rem;font-size:.8rem;line-height:1.45;color:var(--text-secondary);">';
        ranked.slice(0, 4).forEach(function (row) {
          var pct = row.weighted_mean_engagement_pct != null ? Number(row.weighted_mean_engagement_pct).toFixed(2) : '—';
          html += '<li style="margin-bottom:.2rem;"><strong style="color:var(--text-primary);">' + esc(row.summary || row.strategy_key || '') + '</strong>'
            + ' · about ' + esc(pct) + '% engagement · from ' + esc(String(row.samples || 0)) + ' similar posts</li>';
        });
        html += '</ul>';
      }
      var rec = ins.recommended;
      var patch = rec && rec.preferences_patch && typeof rec.preferences_patch === 'object' ? rec.preferences_patch : null;
      var patchKeys = patch
        ? Object.keys(patch).filter(function (k) {
            var v = patch[k];
            return v != null && v !== '';
          })
        : [];
      if (patchKeys.length) {
        html += '<div style="margin-top:.55rem;display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;">'
          + '<button type="button" class="btn btn-primary btn-sm" id="um8CoachApplyBtn">Apply recommended prefs to Settings</button>'
          + '<span id="um8CoachApplyStatus" class="text-secondary" style="font-size:.72rem;"></span></div>';
        if (rec.confidence_note) {
          html += '<div class="text-secondary" style="font-size:.7rem;margin-top:.35rem;opacity:.85;">' + esc(rec.confidence_note) + '</div>';
        }
      }
      html += '</div>';
    }

    const offer = data.smart_offer;
    const sugs = Array.isArray(data.suggestions) ? data.suggestions : [];
    const base = data.baselines || {};
    const gav = base.global_avg_views != null ? Number(base.global_avg_views).toFixed(0) : '—';

    if (offer && offer.headline) {
      const href = esc(offer.cta_href || '/settings.html#billing-panel');
      html += '<div style="padding:.65rem .75rem;border-radius:10px;border:1px solid rgba(249,115,22,.4);background:rgba(249,115,22,.09);margin-bottom:.75rem;">'
        + '<div style="font-weight:700;color:var(--text-primary);">' + esc(offer.headline) + '</div>'
        + '<div class="text-secondary" style="font-size:.82rem;margin-top:.35rem;line-height:1.45;">' + esc(offer.body || '') + '</div>'
        + '<div style="margin-top:.5rem;"><a class="btn btn-primary btn-sm" href="' + href + '">' + esc(offer.cta_label || 'View') + '</a></div>'
        + '</div>';
    }

    html += '<div class="text-secondary" style="font-size:.78rem;margin-bottom:.5rem;">Typical views in the last 30 days (across everyone on UploadM8): <strong>' + esc(gav) + '</strong> · your tips blend this with your own results as you post more.</div>';

    if (!sugs.length && !hasInsights) {
      html += '<div class="text-secondary">No strong signals yet — keep uploading to unlock more specific guidance.</div>';
      el.innerHTML = html;
      bindApplyInsightsButton(el);
      return;
    }

    if (sugs.length) {
      html += '<div style="display:flex;flex-direction:column;gap:.55rem;">';
      sugs.slice(0, 5).forEach(function (s) {
        const st = sevClass(s.severity);
        const cta = s.cta_href
          ? '<a class="btn btn-secondary btn-sm" style="margin-top:.45rem;" href="' + esc(s.cta_href) + '">' + esc(s.cta_label || 'Open') + '</a>'
          : '';
        const conf = s.confidence != null ? '<span style="font-size:.68rem;opacity:.75;"> · about ' + esc(String(Math.round(Number(s.confidence) * 100))) + '% sure</span>' : '';
        html += '<div style="border:1px solid var(--border-color);border-radius:10px;padding:.55rem .65rem;' + st + '">'
          + '<div style="font-weight:600;font-size:.88rem;">' + esc(s.title || 'Tip') + conf + '</div>'
          + '<div class="text-secondary" style="font-size:.8rem;margin-top:.25rem;line-height:1.45;">' + esc(s.body || '') + '</div>'
          + cta
          + '</div>';
      });
      html += '</div>';
    } else if (hasInsights) {
      html += '<div class="text-secondary" style="margin-top:.45rem;font-size:.8rem;">More tips appear here as your account builds history.</div>';
    }

    el.innerHTML = html;
    bindApplyInsightsButton(el);
  }

  async function loadCoach() {
    if (typeof window.apiFetch !== 'function') return null;
    try {
      var resp = await window.apiFetch('/api/me/coach', {
        method: 'GET',
        authRedirectOn401: false,
      });
      if (!resp.ok) return null;
      return await resp.json();
    } catch (_) {
      return null;
    }
  }

  async function loadAndRenderCoach(opts) {
    var force = opts && opts.force;
    if (coachFetchInFlight) {
      if (force) coachPendingForce = true;
      return;
    }
    if (coachHasRendered && !force) return;
    var slots = document.querySelectorAll('[data-ai-coach]');
    if (!slots.length) return;

    coachFetchInFlight = true;
    try {
      if (!_isLikelyLoggedIn() && typeof window.checkAuth === 'function') {
        try {
          await window.checkAuth({ redirectOnFail: false, silent: true, forceRefresh: false });
        } catch (_) {}
      }
      var data = await loadCoach();
      try {
        window.dispatchEvent(new CustomEvent('uploadm8:coach', { detail: data }));
      } catch (_) {}
      slots.forEach(function (el) {
        renderSlot(el, data);
      });
      if (data != null || !_isLikelyLoggedIn()) {
        coachHasRendered = true;
      }
    } finally {
      coachFetchInFlight = false;
      if (coachPendingForce) {
        coachPendingForce = false;
        void loadAndRenderCoach({ force: true });
      }
    }
  }

  function bootCoach() {
    if (coachBootStarted) return;
    coachBootStarted = true;

    if (!document.querySelectorAll('[data-ai-coach]').length) return;

    window.addEventListener(
      'userLoaded',
      function () {
        // Defer coach fetch so it does not compete with upload/dashboard wallet + prefs on the wire.
        var run = function () { void loadAndRenderCoach({ force: true }); };
        if (typeof requestIdleCallback === 'function') {
          requestIdleCallback(run, { timeout: 4500 });
        } else {
          setTimeout(run, 200);
        }
      },
      false
    );

    setTimeout(function () {
      void loadAndRenderCoach({ force: false });
    }, COACH_FALLBACK_MS);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      void bootCoach();
    });
  } else {
    void bootCoach();
  }
})();
