/**
 * Shared formatting + platform helper globals.
 * Loaded before app.js on authenticated app pages.
 */
(function () {
    function _escapeForStatus(value) {
        if (typeof window.escapeHTML === 'function') return window.escapeHTML(value);
        if (value === null || value === undefined) return '';
        return String(value)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function _parseTime(d) {
        if (d == null || d === '') return NaN;
        const t = new Date(d).getTime();
        return Number.isNaN(t) ? NaN : t;
    }

    function formatDate(d) {
        if (!d) return '-';
        const t = _parseTime(d);
        if (Number.isNaN(t)) return '—';
        return new Date(t).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    }

    function formatDateTime(d) {
        if (!d) return '-';
        const t = _parseTime(d);
        if (Number.isNaN(t)) return '—';
        return new Date(t).toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' });
    }

    function formatRelativeTime(d) {
        if (!d) return '-';
        const t = _parseTime(d);
        if (Number.isNaN(t)) return '—';
        const diff = Date.now() - t;
        if (diff < 0) {
            const abs = Math.abs(diff);
            const m = Math.floor(abs / 60000), h = Math.floor(abs / 3600000), days = Math.floor(abs / 86400000);
            if (m < 1) return 'Soon';
            if (m < 60) return `in ${m}m`;
            if (h < 24) return `in ${h}h`;
            if (days < 7) return `in ${days}d`;
            return formatDate(d);
        }
        const m = Math.floor(diff / 60000), h = Math.floor(diff / 3600000), days = Math.floor(diff / 86400000);
        if (m < 1) return 'Just now';
        if (m < 60) return `${m}m ago`;
        if (h < 24) return `${h}h ago`;
        if (days < 7) return `${days}d ago`;
        return formatDate(d);
    }

    function formatNumber(num) {
        if (num == null) return '-';
        const n = Number(num);
        if (isNaN(n)) return '-';
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return n.toLocaleString();
    }

    function formatFileSize(bytes) {
        if (!bytes) return '0 B';
        const k = 1024, sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function formatCurrency(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(amount);
    }

    function getPlatformInfo(platform) {
        const map = {
            tiktok: { name: 'TikTok', icon: 'fab fa-tiktok', color: '#00f2ea' },
            youtube: { name: 'YouTube', icon: 'fab fa-youtube', color: '#FF0000' },
            instagram: { name: 'Instagram', icon: 'fab fa-instagram', color: '#E1306C' },
            facebook: { name: 'Facebook', icon: 'fab fa-facebook', color: '#1877F2' },
            meta: { name: 'Meta', icon: 'fab fa-meta', color: '#0668E1' },
            google: { name: 'YouTube', icon: 'fab fa-youtube', color: '#FF0000' },
        };
        const p = typeof platform === 'string' ? platform : (platform?.platform || platform?.name || String(platform || ''));
        const key = String(p).toLowerCase().trim();
        return map[key] || { name: typeof p === 'string' ? p : 'Unknown', icon: 'fas fa-globe', color: '#666' };
    }

    function getPlatformIcon(platform) {
        const i = getPlatformInfo(platform);
        return `<i class="${i.icon}" style="color:${i.color};" title="${i.name}"></i>`;
    }

    function getPlatformBadge(platform) {
        const i = getPlatformInfo(platform);
        return `<span class="platform-badge" style="background:${i.color}20;color:${i.color};"><i class="${i.icon}"></i> ${i.name}</span>`;
    }

    /**
     * Flatten hashtag values from API/DB: arrays, JSON strings, comma-separated text.
     * Returns raw trimmed strings (may include #; callers often strip/normalize).
     */
    function um8CoerceHashtagList(raw) {
        const out = [];
        function pushOne(v) {
            if (v == null || v === '') return;
            const s = String(v).trim();
            if (!s) return;
            try {
                const j0 = JSON.parse(s);
                if (Array.isArray(j0)) {
                    j0.forEach(pushOne);
                    return;
                }
                if (typeof j0 === 'string') {
                    pushOne(j0);
                    return;
                }
                return;
            } catch (_) {
                /* not JSON */
            }
            if (s.startsWith('[')) {
                try {
                    const fixed = s.replace(/"\s+"/g, '", "').replace(/'\s+'/g, "', '");
                    const j = JSON.parse(fixed);
                    if (Array.isArray(j)) {
                        j.forEach(pushOne);
                        return;
                    }
                } catch (_) {
                    const quoted = s.match(/"([^"]{1,200})"/g);
                    if (quoted && quoted.length) {
                        quoted.forEach(function (q) {
                            pushOne(q.replace(/^"|"$/g, ''));
                        });
                        return;
                    }
                }
            }
            if (s.includes(',')) {
                s.split(',').forEach(function (p) {
                    pushOne(p);
                });
                return;
            }
            out.push(s);
        }
        if (!raw) return [];
        if (Array.isArray(raw)) raw.forEach(pushOne);
        else pushOne(raw);
        return out;
    }

    function um8HashtagsDisplayString(raw) {
        function stripBody(t) {
            const s0 = String(t || '')
                .trim()
                .replace(/^#+/, '');
            try {
                return s0.replace(/[^\p{L}\p{N}_]/gu, '').toLowerCase();
            } catch (_) {
                return s0.replace(/[^a-z0-9_]/g, '').toLowerCase();
            }
        }
        let parts = [];
        try {
            parts = um8CoerceHashtagList(raw);
        } catch (_) {
            parts = [];
        }
        const seen = new Set();
        const tags = [];
        for (let i = 0; i < parts.length; i++) {
            const b = stripBody(parts[i]);
            if (!b || seen.has(b)) continue;
            seen.add(b);
            tags.push('#' + b);
        }
        return tags.join(' ');
    }

    function getStatusBadge(uploadOrStatus) {
        const map = {
            pending: { label: 'Pending', color: 'yellow', icon: 'clock' },
            queued: { label: 'Queued', color: 'blue', icon: 'list' },
            staged: { label: 'Staged', color: 'yellow', icon: 'clock' },
            scheduled: { label: 'Scheduled', color: 'purple', icon: 'calendar-alt' },
            ready_to_publish: { label: 'Ready to Publish', color: 'purple', icon: 'calendar-check' },
            processing: { label: 'Processing', color: 'blue', icon: 'spinner fa-spin' },
            uploading: { label: 'Uploading', color: 'blue', icon: 'cloud-upload-alt' },
            completed: { label: 'Completed', color: 'green', icon: 'check-circle' },
            succeeded: { label: 'Succeeded', color: 'green', icon: 'check-circle' },
            failed: { label: 'Failed', color: 'red', icon: 'exclamation-circle' },
            partial: { label: 'Partial', color: 'orange', icon: 'exclamation-triangle' },
            cancelled: { label: 'Cancelled', color: 'gray', icon: 'ban' },
        };
        const isObj = uploadOrStatus && typeof uploadOrStatus === 'object';
        const status = isObj ? (uploadOrStatus.status || '') : (uploadOrStatus || '');
        const label = (isObj && uploadOrStatus.status_label ? String(uploadOrStatus.status_label).trim() : null)
            || map[status]?.label
            || (status ? String(status).replace(/_/g, ' ') : 'Unknown');
        const info = map[(status || '').toLowerCase()] || { color: 'gray', icon: 'circle' };
        return `<span class="status-badge status-${info.color}"><i class="fas fa-${info.icon}"></i> ${_escapeForStatus(label)}</span>`;
    }

    /** Use on img tags for YouTube ggpht / Meta fbcdn avatars — avoids Referer-based blocks from localhost. */
    window.UM8_EXTERNAL_AVATAR_IMG_ATTRS = 'referrerpolicy="no-referrer" loading="lazy" decoding="async"';

    window.formatDate = formatDate;
    window.formatDateTime = formatDateTime;
    window.formatRelativeTime = formatRelativeTime;
    window.formatNumber = formatNumber;
    window.formatFileSize = formatFileSize;
    window.formatCurrency = formatCurrency;
    window.getPlatformInfo = getPlatformInfo;
    window.getPlatformIcon = getPlatformIcon;
    window.getPlatformBadge = getPlatformBadge;
    window.getStatusBadge = getStatusBadge;
    window.um8CoerceHashtagList = um8CoerceHashtagList;
    window.um8HashtagsDisplayString = um8HashtagsDisplayString;
})();
