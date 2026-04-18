/**
 * UploadM8 — tier metadata from GET /api/entitlements/tiers (+ /api/pricing for Stripe keys).
 * Include after js/api-base.js. Call UploadM8TierCatalog.load() before reading.
 */
(function (global) {
  'use strict';

  var TIER_UI = {
    free: {
      color: '#9ca3af',
      bg: 'rgba(156,163,175,.1)',
      border: 'rgba(156,163,175,.25)',
      icon: 'fa-circle',
    },
    creator_lite: {
      color: '#f97316',
      bg: 'rgba(249,115,22,.1)',
      border: 'rgba(249,115,22,.3)',
      icon: 'fa-bolt',
    },
    creator_pro: {
      color: '#8b5cf6',
      bg: 'rgba(139,92,246,.1)',
      border: 'rgba(139,92,246,.3)',
      icon: 'fa-rocket',
    },
    studio: {
      color: '#3b82f6',
      bg: 'rgba(59,130,246,.1)',
      border: 'rgba(59,130,246,.3)',
      icon: 'fa-film',
    },
    agency: {
      color: '#22c55e',
      bg: 'rgba(34,197,94,.1)',
      border: 'rgba(34,197,94,.3)',
      icon: 'fa-building',
    },
    friends_family: {
      color: '#fbbf24',
      bg: 'rgba(251,191,36,.1)',
      border: 'rgba(251,191,36,.3)',
      icon: 'fa-heart',
    },
    lifetime: {
      color: '#fbbf24',
      bg: 'rgba(251,191,36,.1)',
      border: 'rgba(251,191,36,.3)',
      icon: 'fa-infinity',
    },
    master_admin: {
      color: '#ef4444',
      bg: 'rgba(239,68,68,.1)',
      border: 'rgba(239,68,68,.3)',
      icon: 'fa-shield-alt',
    },
  };

  function apiBase() {
    if (typeof global.resolveUploadM8ApiOrigin === 'function') return global.resolveUploadM8ApiOrigin();
    if (typeof global.getUploadM8ApiBase === 'function') return global.getUploadM8ApiBase();
    var b = global.API_BASE;
    return b ? String(b).replace(/\/$/, '') : '';
  }

  function n(x, d) {
    var v = Number(x);
    return isFinite(v) ? v : d;
  }

  function mergeRow(apiRow, pricingRow) {
    if (!apiRow || !apiRow.slug) return null;
    var slug = apiRow.slug;
    var ui = TIER_UI[slug] || {};
    var stripe_lookup_key = pricingRow && pricingRow.stripe_lookup_key ? pricingRow.stripe_lookup_key : null;
    var name = apiRow.name;
    return {
      slug: slug,
      name: name,
      put: n(apiRow.put_monthly, 0),
      aic: n(apiRow.aic_monthly, 0),
      price: n(apiRow.price, 0),
      max_accounts_per_platform: n(apiRow.max_accounts_per_platform, 0),
      max_accounts: n(apiRow.max_accounts, 0),
      queue_depth: n(apiRow.queue_depth, 0),
      lookahead_hours: n(apiRow.lookahead_hours, 0),
      trial_days: n(apiRow.trial_days, 0),
      team_seats: n(apiRow.team_seats, 1),
      analytics: apiRow.analytics || 'basic',
      analytics_label: apiRow.analytics_label || apiRow.analytics || 'Basic analytics',
      ai_depth: apiRow.ai_depth || 'basic',
      webhooks: !!apiRow.webhooks,
      white_label: !!apiRow.white_label,
      hud: !!apiRow.hud,
      excel: !!apiRow.excel,
      flex: !!apiRow.flex,
      watermark: !!apiRow.watermark,
      max_thumbnails: n(apiRow.max_thumbnails, 1),
      max_caption_frames: n(apiRow.max_caption_frames, 3),
      priority_class: apiRow.priority_class || 'p4',
      queue_lane_label: apiRow.queue_lane_label || 'Standard',
      scheduling_window_label: apiRow.scheduling_window_label || '4 hours',
      internal: !!apiRow.internal,
      color: ui.color || '#9ca3af',
      bg: ui.bg,
      border: ui.border,
      icon: ui.icon || 'fa-circle',
      stripe_lookup_key: stripe_lookup_key,
    };
  }

  function aiCaptionLabel(depth) {
    var d = (depth || 'basic').toLowerCase();
    if (d === 'enhanced') return 'AI captions (enhanced)';
    if (d === 'advanced') return 'Advanced AI depth';
    if (d === 'max') return 'Max AI depth';
    return 'AI captions';
  }

  function buildBillingSuccessFeats(m) {
    if (!m) return [];
    var slug = m.slug;
    var pp = m.max_accounts_per_platform;
    if (slug === 'free') {
      return [
        { ok: true, icon: 'fa-cloud-upload-alt', label: pp + ' accounts per platform' },
        { ok: true, icon: 'fa-robot', label: 'AI captions' },
        { ok: true, icon: 'fa-calendar-alt', label: 'Scheduled posts' },
        { ok: false, icon: 'fa-film', label: 'HUD overlay' },
        { ok: true, icon: 'fa-chart-line', label: m.analytics_label || 'Basic analytics' },
        { ok: false, icon: 'fa-bolt', label: 'Priority queue' },
      ];
    }
    if (slug === 'creator_lite') {
      return [
        { ok: true, icon: 'fa-cloud-upload-alt', label: pp + ' accounts per platform' },
        { ok: true, icon: 'fa-robot', label: aiCaptionLabel(m.ai_depth) },
        { ok: true, icon: 'fa-calendar-alt', label: 'Scheduled posts' },
        { ok: true, icon: 'fa-image', label: 'Up to ' + m.max_thumbnails + ' thumbnails' },
        { ok: true, icon: 'fa-bell', label: 'Webhook notifications' },
        { ok: true, icon: 'fa-chart-line', label: m.analytics_label || 'Standard analytics' },
      ];
    }
    if (slug === 'creator_pro') {
      return [
        { ok: true, icon: 'fa-cloud-upload-alt', label: pp + ' accounts per platform' },
        { ok: true, icon: 'fa-robot', label: aiCaptionLabel(m.ai_depth) },
        { ok: true, icon: 'fa-calendar-alt', label: 'Scheduled posts' },
        { ok: true, icon: 'fa-film', label: 'HUD burn-in overlay' },
        { ok: true, icon: 'fa-bolt', label: 'Priority queue (' + m.priority_class + ')' },
        { ok: true, icon: 'fa-image', label: 'Up to ' + m.max_thumbnails + ' candidate thumbnails' },
        { ok: true, icon: 'fa-users', label: m.team_seats + ' team seats' },
      ];
    }
    if (slug === 'studio') {
      return [
        { ok: true, icon: 'fa-cloud-upload-alt', label: pp + ' accounts per platform' },
        {
          ok: true,
          icon: 'fa-robot',
          label: 'Max AI — ' + m.max_caption_frames + ' frame scan',
        },
        { ok: true, icon: 'fa-file-excel', label: 'Excel export reports' },
        { ok: true, icon: 'fa-bolt', label: 'Turbo priority queue (' + m.priority_class + ')' },
        { ok: true, icon: 'fa-film', label: 'HUD burn-in overlay' },
        { ok: true, icon: 'fa-users', label: m.team_seats + ' team seats' },
      ];
    }
    if (slug === 'agency') {
      return [
        { ok: true, icon: 'fa-cloud-upload-alt', label: pp + ' accounts per platform' },
        { ok: true, icon: 'fa-robot', label: 'Max AI + dense frame scan' },
        { ok: true, icon: 'fa-tag', label: 'White-label branding' },
        { ok: true, icon: 'fa-bolt', label: 'Top priority queue (' + m.priority_class + ')' },
        { ok: true, icon: 'fa-exchange-alt', label: 'Flex token transfers' },
        { ok: true, icon: 'fa-users', label: m.team_seats + ' team seats' },
      ];
    }
    var feats = [
      { ok: true, icon: 'fa-cloud-upload-alt', label: pp + ' accounts per platform' },
      { ok: true, icon: 'fa-robot', label: aiCaptionLabel(m.ai_depth) },
      { ok: true, icon: 'fa-calendar-alt', label: 'Scheduled posts' },
    ];
    if (m.hud) feats.push({ ok: true, icon: 'fa-film', label: 'HUD burn-in overlay' });
    feats.push({
      ok: true,
      icon: 'fa-bolt',
      label: 'Priority queue (' + m.priority_class + ')',
    });
    feats.push({
      ok: true,
      icon: 'fa-users',
      label: (m.team_seats >= 500 ? 'Unlimited' : m.team_seats) + ' team seats',
    });
    return feats;
  }

  function buildDashboardBillingFeats(m) {
    if (!m) return [];
    var slug = m.slug;
    var pp = m.max_accounts_per_platform;
    if (slug === 'free') {
      return [
        { icon: 'fa-cloud-upload-alt', label: 'Upload to ' + pp + ' accounts/platform', ok: true },
        { icon: 'fa-robot', label: 'AI captions', ok: true },
        { icon: 'fa-calendar-alt', label: 'Scheduled posts', ok: true },
        { icon: 'fa-bolt', label: 'Priority processing', ok: false },
        { icon: 'fa-film', label: 'HUD overlay', ok: false },
        { icon: 'fa-chart-line', label: m.analytics_label || 'Basic analytics', ok: true },
      ];
    }
    if (slug === 'creator_lite') {
      return [
        { icon: 'fa-cloud-upload-alt', label: pp + ' accounts per platform', ok: true },
        { icon: 'fa-robot', label: aiCaptionLabel(m.ai_depth), ok: true },
        { icon: 'fa-calendar-alt', label: 'Scheduled posts', ok: true },
        { icon: 'fa-image', label: 'Up to ' + m.max_thumbnails + ' thumbnails', ok: true },
        { icon: 'fa-paint-brush', label: 'Styled thumbnails', ok: true },
        { icon: 'fa-bell', label: 'Webhook notifications', ok: true },
        { icon: 'fa-chart-line', label: m.analytics_label || 'Standard analytics', ok: true },
      ];
    }
    if (slug === 'creator_pro') {
      return [
        { icon: 'fa-cloud-upload-alt', label: pp + ' accounts per platform', ok: true },
        { icon: 'fa-robot', label: aiCaptionLabel(m.ai_depth), ok: true },
        { icon: 'fa-film', label: 'HUD burn-in overlay', ok: true },
        { icon: 'fa-bolt', label: 'Priority queue (' + m.priority_class + ')', ok: true },
        { icon: 'fa-image', label: 'Up to ' + m.max_thumbnails + ' thumbnails', ok: true },
        { icon: 'fa-magic', label: 'AI thumbnail styling', ok: true },
        { icon: 'fa-users', label: m.team_seats + ' team seats', ok: true },
      ];
    }
    if (slug === 'studio') {
      return [
        { icon: 'fa-cloud-upload-alt', label: pp + ' accounts per platform', ok: true },
        {
          icon: 'fa-robot',
          label: 'Max AI depth (' + m.max_caption_frames + ' frames)',
          ok: true,
        },
        { icon: 'fa-file-excel', label: 'Excel export reports', ok: true },
        { icon: 'fa-bolt', label: 'Turbo priority queue (' + m.priority_class + ')', ok: true },
        { icon: 'fa-film', label: 'HUD burn-in overlay', ok: true },
        { icon: 'fa-users', label: m.team_seats + ' team seats', ok: true },
      ];
    }
    if (slug === 'agency') {
      return [
        { icon: 'fa-cloud-upload-alt', label: pp + ' accounts per platform', ok: true },
        { icon: 'fa-robot', label: 'Max AI + dense frame scan', ok: true },
        { icon: 'fa-tag', label: 'White-label branding', ok: true },
        { icon: 'fa-bolt', label: 'Top priority queue (' + m.priority_class + ')', ok: true },
        { icon: 'fa-exchange-alt', label: 'Flex token transfers', ok: true },
        { icon: 'fa-users', label: m.team_seats + ' team seats', ok: true },
      ];
    }
    return buildBillingSuccessFeats(m).map(function (f) {
      return { icon: f.icon, label: f.label, ok: f.ok };
    });
  }

  function teamSeatsPhrase(m) {
    if (!m) return '1';
    var ts = n(m.team_seats, 1);
    if (ts >= 500) return '∞';
    return String(ts);
  }

  function buildSuccessModalFeatureList(tier, ent, getMeta) {
    var m = getMeta(tier) || getMeta('free');
    var unlimited = m.put >= 999999;
    var pc = m.priority_class || 'p4';
    var priorityTiers = ['p2', 'p1', 'p0'];
    var hasPriority = priorityTiers.indexOf(pc) !== -1;
    var entCanWatermark = ent ? !!ent.can_watermark : m.watermark;
    var noWatermark = !entCanWatermark;

    return [
      [
        ' ' + (unlimited ? 'Unlimited' : m.put.toLocaleString()) + ' PUT / month',
        'Processing tokens for uploading & distributing videos',
        true,
        null,
      ],
      [
        ' ' + (unlimited ? 'Unlimited' : m.aic.toLocaleString()) + ' AI Credits / month',
        'Power AI captions, hashtags & analysis',
        ent ? ent.can_ai || m.aic > 0 : m.aic > 0,
        'creator_lite',
      ],
      [
        ' No watermark',
        'Your videos publish clean with no UploadM8 branding',
        noWatermark,
        'creator_lite',
      ],
      [
        ' Smart scheduling (' + m.queue_lane_label + ')',
        'Schedule posts up to ' + m.scheduling_window_label + ' ahead',
        true,
        'creator_lite',
      ],
      [
        ' Priority processing',
        'Your uploads jump the queue — faster turnaround',
        hasPriority,
        'creator_pro',
      ],
      [
        ' AI captions & hashtags',
        'Auto-generate platform-optimized captions and hashtag sets',
        ent ? ent.can_ai : tier !== 'free',
        'creator_lite',
      ],
      [
        ' HUD burn',
        'Burn dynamic overlays & stats directly into your video',
        ent ? ent.can_burn_hud : m.hud,
        'creator_pro',
      ],
      [
        ' Analytics',
        'Track performance across all platforms in one place',
        tier !== 'free',
        'creator_lite',
      ],
      [
        ' Export analytics',
        'Download your data as Excel / CSV',
        ent ? ent.can_excel : m.excel,
        'studio',
      ],
      [
        ' Team seats (' + teamSeatsPhrase(m) + ')',
        'Invite collaborators to your workspace',
        tier !== 'free',
        'creator_lite',
      ],
      [
        ' White-label',
        'Remove all UploadM8 branding for your clients',
        ent ? ent.can_white_label : m.white_label,
        'agency',
      ],
      [
        ' Webhooks',
        'Fire HTTP events to your own systems on upload events',
        ent ? ent.can_webhooks : m.webhooks,
        'creator_lite',
      ],
      [
        ' Flex transfers',
        'Transfer wallet credits between team members',
        ent ? ent.can_flex : m.flex,
        'agency',
      ],
    ];
  }

  function buildUpgradeModalTiers(bySlug) {
    var order = ['free', 'creator_lite', 'creator_pro', 'studio', 'agency'];
    var out = [];
    order.forEach(function (slug) {
      var m = bySlug[slug];
      if (!m) return;
      var row = {
        slug: slug,
        name: m.name,
        color: m.color,
        put: m.put,
        aic: m.aic,
        accounts: m.max_accounts_per_platform,
        lookupKey: m.stripe_lookup_key || null,
        trial: m.trial_days > 0 ? m.trial_days + '-day free trial' : null,
        feats: [],
        missing: [],
      };
      if (slug === 'free') {
        row.price = '$0';
        row.feats = [
          m.max_accounts_per_platform + ' accounts per platform',
          m.analytics_label || 'Basic analytics',
          'AI captions',
          'Smart scheduling',
        ];
        row.missing = ['HUD overlay', 'Priority queue', 'Webhooks'];
        row.lookupKey = null;
      } else {
        row.price = '$' + m.price.toFixed(2) + '/mo';
        if (slug === 'creator_lite') {
          row.feats = [
            m.max_accounts_per_platform + ' accounts per platform',
            aiCaptionLabel(m.ai_depth),
            'Scheduling',
            m.max_thumbnails + ' thumbnails',
            'Styled thumbnails',
            'Webhooks',
            m.analytics_label || 'Standard analytics',
          ];
          row.missing = [
            'HUD overlay',
            'AI thumbnail styling',
            'Priority queue',
            'Excel export',
          ];
        } else if (slug === 'creator_pro') {
          row.feats = [
            m.max_accounts_per_platform + ' accounts per platform',
            'Advanced AI',
            'HUD overlay',
            'Priority queue (' + m.priority_class + ')',
            m.max_thumbnails + ' thumbnails',
            'Styled + AI thumbnails',
            m.team_seats + ' team seats',
          ];
          row.missing = ['Excel export', 'White label', 'Flex transfers'];
        } else if (slug === 'studio') {
          row.feats = [
            m.max_accounts_per_platform + ' accounts per platform',
            'Max AI (' + m.max_caption_frames + ' frames)',
            'Excel export',
            'Turbo queue (' + m.priority_class + ')',
            m.team_seats + ' team seats',
            m.analytics_label || 'Full analytics',
          ];
          row.missing = ['White label', 'Flex transfers'];
        } else if (slug === 'agency') {
          row.feats = [
            m.max_accounts_per_platform + ' accounts per platform',
            'Max AI',
            'White label',
            'Top priority (' + m.priority_class + ')',
            'Flex transfers',
            m.team_seats + ' team seats',
          ];
          row.missing = [];
        }
      }
      out.push(row);
    });
    return out;
  }

  var _cache = null;
  var _loading = null;

  global.UploadM8TierCatalog = {
    load: function () {
      if (_cache) return Promise.resolve(_cache);
      if (_loading) return _loading;
      var self = this;
      var b = apiBase();
      _loading = Promise.all([
        typeof global.apiCall === 'function'
          ? global.apiCall('/api/entitlements/tiers')
          : fetch(b + '/api/entitlements/tiers').then(function (r) {
              if (!r.ok) {
                return r
                  .json()
                  .catch(function () {
                    return null;
                  })
                  .then(function (body) {
                    var msg =
                      (typeof global.uploadM8ApiErrorMessage === 'function' &&
                        global.uploadM8ApiErrorMessage(body)) ||
                      '';
                    throw new Error(msg || 'tiers HTTP ' + r.status);
                  });
              }
              return r.json();
            }),
        typeof global.apiFetch === 'function'
          ? global
              .apiFetch('/api/pricing', { method: 'GET', authRedirectOn401: false })
              .then(function (r) {
                return r.ok ? r.json() : {};
              })
              .catch(function () {
                return {};
              })
          : fetch(b + '/api/pricing')
              .then(function (r) {
                return r.ok ? r.json() : {};
              })
              .catch(function () {
                return {};
              }),
      ])
        .then(function (pair) {
          var tRes = pair[0] || {};
          var pRes = pair[1] || {};
          var priceBySlug = {};
          (pRes.tiers || []).forEach(function (t) {
            if (t && t.slug) priceBySlug[t.slug] = t;
          });
          var bySlug = {};
          (tRes.tiers || []).forEach(function (row) {
            var m = mergeRow(row, priceBySlug[row.slug]);
            if (m) bySlug[row.slug] = m;
          });
          _cache = { bySlug: bySlug, raw: tRes, pricing: pRes };
          _loading = null;
          return _cache;
        })
        .catch(function (err) {
          _loading = null;
          console.warn('[UploadM8TierCatalog]', err);
          _cache = { bySlug: {}, raw: null, pricing: null, error: err };
          return _cache;
        });
      return _loading;
    },

    /**
     * @param {string} slug
     * @param {string} [fallbackSlug]  e.g. 'creator_lite' on billing/success; default 'free'
     */
    getTierMeta: function (slug, fallbackSlug) {
      if (!_cache || !_cache.bySlug) return null;
      var s = String(slug || fallbackSlug || 'free');
      var fb = fallbackSlug || 'free';
      return (
        _cache.bySlug[s] ||
        _cache.bySlug[fb] ||
        _cache.bySlug.free ||
        _cache.bySlug.creator_lite ||
        null
      );
    },

    resolveTierSlug: function (slug) {
      if (!_cache || !_cache.bySlug) return slug || 'free';
      var s = slug || 'free';
      if (_cache.bySlug[s]) return s;
      return s;
    },

    billingSuccessFeats: function (slug) {
      var m = this.getTierMeta(slug);
      if (!m) return [];
      return buildBillingSuccessFeats(m);
    },

    dashboardBillingFeats: function (slug) {
      var m = this.getTierMeta(slug);
      if (!m) return [];
      return buildDashboardBillingFeats(m);
    },

    successModalFeatureList: function (tier, ent) {
      var self = this;
      return buildSuccessModalFeatureList(tier, ent, function (s) {
        return self.getTierMeta(s);
      });
    },

    upgradeModalTiers: function () {
      if (!_cache || !_cache.bySlug) return [];
      return buildUpgradeModalTiers(_cache.bySlug);
    },

    /** Settings / billing.js: map slug → { name, price, put, aic, color, trial_days } */
    settingsTierMeta: function (slug) {
      var m = this.getTierMeta(slug);
      if (!m) return null;
      return {
        name: m.name,
        price: m.price,
        put: m.put,
        aic: m.aic,
        color: m.color,
        trial_days: m.trial_days,
      };
    },
  };
})(typeof window !== 'undefined' ? window : globalThis);
