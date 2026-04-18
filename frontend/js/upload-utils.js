/**
 * Upload utility helpers shared by app pages.
 */
(function () {
    function toNum(v) {
        const n = Number(v);
        return isNaN(n) ? 0 : n;
    }

    function getUploadStats(upload) {
        let colV = toNum(upload?.views), colL = toNum(upload?.likes);
        let colC = toNum(upload?.comments), colS = toNum(upload?.shares);
        let aggV = 0, aggL = 0, aggC = 0, aggS = 0;
        const pr = Array.isArray(upload?.platform_results) ? upload.platform_results : [];
        const shortform = new Set(['tiktok', 'youtube', 'instagram', 'facebook']);
        const okStatus = new Set(['published', 'succeeded', 'success', 'completed', 'partial']);
        pr.forEach(r => {
            if (!r) return;
            const plat = String(r.platform || '').toLowerCase();
            if (plat && !shortform.has(plat)) return;
            const st = String(r.status || '').toLowerCase();
            if (!(r.success === true || okStatus.has(st))) return;
            aggV += toNum(r.view_count ?? r.views ?? r.viewCount ?? r.plays);
            aggL += toNum(r.like_count ?? r.likes ?? r.likeCount);
            aggC += toNum(r.comment_count ?? r.comments ?? r.commentCount);
            aggS += toNum(r.share_count ?? r.shares ?? r.shareCount);
        });
        return {
            views: Math.max(colV, aggV),
            likes: Math.max(colL, aggL),
            comments: Math.max(colC, aggC),
            shares: Math.max(colS, aggS),
        };
    }

    /**
     * Token row id on a platform_results element (matches platform_tokens.id when set).
     */
    function entryTokenRowId(entry) {
        if (!entry) return '';
        return String(entry.token_row_id || entry.token_id || entry.platform_token_id || '').trim();
    }

    function getTargetAccountIds(upload) {
        const raw = upload && (upload.target_accounts || upload.targetAccounts);
        if (!raw) return [];
        const arr = Array.isArray(raw) ? raw : [];
        return arr.map(function (id) { return String(id); }).filter(Boolean);
    }

    /**
     * Keep only platform_results rows that represent an actual publish attempt for this upload:
     * — platform must be in uploads.platforms (when that list is non-empty), and
     * — when target_accounts is set, token_row_id must be in that list (no “connected but not targeted” rows).
     * No chips from connection list alone; every chip maps to a real platform_results row.
     */
    function filterPrListForChips(prList, upload, normalizePlatformFn) {
        const np = normalizePlatformFn || function (x) {
            return String(x || '').toLowerCase().trim();
        };
        const platforms = (upload._platforms || upload.platforms || []).map(np).filter(Boolean);
        const platformSet = new Set(platforms);
        const targetIds = getTargetAccountIds(upload);
        let rows = (prList || []).filter(function (e) { return e && e.platform; });

        if (platformSet.size > 0) {
            rows = rows.filter(function (e) {
                const p = np(e.platform);
                return p && platformSet.has(p);
            });
        }

        if (targetIds.length > 0) {
            const tidSet = new Set(targetIds);
            const strict = rows.filter(function (e) {
                const tid = entryTokenRowId(e);
                return tid && tidSet.has(tid);
            });
            if (strict.length > 0) {
                const order = new Map(targetIds.map(function (id, i) { return [id, i]; }));
                strict.sort(function (a, b) {
                    const ia = order.get(entryTokenRowId(a));
                    const ib = order.get(entryTokenRowId(b));
                    return (ia !== undefined ? ia : 999) - (ib !== undefined ? ib : 999);
                });
                return strict;
            }
            return [];
        }
        return rows;
    }

    /**
     * One chip per qualifying platform_results row only (no placeholders from targets or platforms[]).
     */
    function buildChipEntries(prList, upload, normalizePlatformFn) {
        return filterPrListForChips(prList, upload, normalizePlatformFn);
    }

    function findAccountByTokenId(tokenId) {
        if (!tokenId || typeof window === 'undefined') return null;
        const map = window._platformAccountsByPlatform;
        if (!map) return null;
        const keys = Object.keys(map);
        for (let i = 0; i < keys.length; i++) {
            const k = keys[i];
            const arr = map[k] || [];
            const a = arr.find(function (x) { return String(x.id) === String(tokenId); });
            if (a) {
                const plat = String(a.platform || a.provider || k || '').toLowerCase() || k;
                return Object.assign({}, a, { platform: plat });
            }
        }
        return null;
    }

    function enrichEntryFromTokenRow(entry) {
        if (!entry) return entry;
        const tid = entryTokenRowId(entry);
        if (!tid) return entry;
        const acc = findAccountByTokenId(tid);
        if (!acc) return entry;
        const next = Object.assign({}, entry);
        if (!next.account_username && acc.username) next.account_username = acc.username;
        if (!next.username && acc.username) next.username = acc.username;
        if (!next.account_id && acc.account_id) next.account_id = acc.account_id;
        if (!next.account_name && (acc.name || acc.account_name)) {
            next.account_name = acc.name || acc.account_name;
        }
        return next;
    }

    function normalizeOutboundUrl(candidate) {
        const raw = String(candidate || '').trim();
        if (!raw) return null;
        if (/^https?:\/\//i.test(raw)) return raw;
        if (raw.startsWith('//')) return 'https:' + raw;
        if (/^[a-z0-9.-]+\.[a-z]{2,}(?:\/|$)/i.test(raw)) return 'https://' + raw;
        return null;
    }

    function isKnownSocialPostHost(urlStr) {
        try {
            const u = new URL(urlStr);
            const raw = u.hostname.toLowerCase();
            const h = raw.replace(/^www\./i, '');
            return (
                h === 'youtube.com' || h === 'youtu.be' ||
                h === 'tiktok.com' ||
                raw === 'vm.tiktok.com' || raw === 'vt.tiktok.com' ||
                h === 'instagram.com' ||
                h === 'facebook.com' || h === 'fb.watch' || h === 'm.facebook.com' ||
                h === 'l.facebook.com'
            );
        } catch (e) {
            return false;
        }
    }

    function isLikelySourceOrCdnVideoUrl(urlStr) {
        try {
            const h = new URL(urlStr).hostname.toLowerCase();
            return /r2\.cloudflarestorage\.|\.r2\.dev|amazonaws\.com|\.s3\.|digitaloceanspaces\.|blob\.core\.windows\.net|storage\.googleapis\.com/i.test(h);
        } catch (e) {
            return false;
        }
    }

    /**
     * Reject generic profile/home links when we need a watch/post URL.
     */
    function isPlausibleWatchUrl(urlStr, platformNorm) {
        if (!urlStr || !isKnownSocialPostHost(urlStr)) return false;
        try {
            const u = new URL(urlStr);
            const path = u.pathname.toLowerCase();
            const host = u.hostname.replace(/^www\./i, '').toLowerCase();
            const p = platformNorm || '';
            const rawHost = u.hostname.toLowerCase();
            if (p === 'tiktok') {
                if (rawHost === 'vm.tiktok.com' || rawHost === 'vt.tiktok.com') return true;
                return path.indexOf('/video/') !== -1 || path.indexOf('/t/') !== -1;
            }
            if (p === 'youtube' || host === 'youtu.be') {
                return path.indexOf('/watch') !== -1 || path.indexOf('/shorts/') !== -1 || host === 'youtu.be';
            }
            if (p === 'instagram') {
                return path.indexOf('/p/') !== -1 || path.indexOf('/reel/') !== -1 || path.indexOf('/reels/') !== -1 || path.indexOf('/tv/') !== -1;
            }
            if (p === 'facebook') {
                if (rawHost === 'fb.watch' || rawHost === 'l.facebook.com') return true;
                return (
                    path.indexOf('/watch') !== -1 ||
                    path.indexOf('/reel') !== -1 ||
                    path.indexOf('/videos/') !== -1 ||
                    path.indexOf('/video/') !== -1 ||
                    u.search.indexOf('v=') !== -1
                );
            }
            return true;
        } catch (e) {
            return false;
        }
    }

    function buildUrlFromEntry(entry, normalizePlatformFn) {
        const np = normalizePlatformFn || function (x) {
            return String(x || '').toLowerCase().trim();
        };
        let e = enrichEntryFromTokenRow(entry);
        if (!e) return null;
        const direct = e.platform_url || e.url || e.video_url || e.post_url || e.share_url || e.permalink;
        let directUrl = normalizeOutboundUrl(direct);
        if (directUrl && isLikelySourceOrCdnVideoUrl(directUrl)) directUrl = null;
        const p = np(e.platform) || '';
        if (directUrl && isKnownSocialPostHost(directUrl) && isPlausibleWatchUrl(directUrl, p)) {
            return directUrl;
        }
        if (directUrl && isKnownSocialPostHost(directUrl) && !isPlausibleWatchUrl(directUrl, p)) {
            directUrl = null;
        }
        const vidRaw = e.platform_video_id || e.video_id || e.videoId || e.media_id || e.post_id || e.share_id || e.item_id || e.id;
        let vid = vidRaw != null && vidRaw !== '' ? String(vidRaw).trim() : '';
        if (vid && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(vid)) vid = '';
        const igSc = e.shortcode || e.ig_shortcode;
        if (p === 'instagram' && igSc) return 'https://www.instagram.com/p/' + encodeURIComponent(igSc) + '/';
        if (!vid) return null;
        if (p === 'youtube') {
            return 'https://www.youtube.com/watch?v=' + encodeURIComponent(vid);
        }
        if (p === 'facebook') {
            const pageId = String(e.account_id || '').trim();
            if (/^\d+$/.test(vid) && pageId && /^\d+$/.test(pageId)) {
                return 'https://www.facebook.com/' + encodeURIComponent(pageId) + '/videos/' + encodeURIComponent(vid);
            }
            if (/^\d+$/.test(vid)) return 'https://www.facebook.com/watch/?v=' + encodeURIComponent(vid);
            if (vid.indexOf('_') !== -1) return 'https://www.facebook.com/' + encodeURIComponent(vid);
            return 'https://www.facebook.com/watch/?v=' + encodeURIComponent(vid);
        }
        if (p === 'tiktok') {
            let uname = String(e.account_username || e.username || e.author || '').replace(/^@+/g, '').trim();
            if (!uname) {
                const acc = entryTokenRowId(e) ? findAccountByTokenId(entryTokenRowId(e)) : null;
                if (acc && acc.username) uname = String(acc.username).replace(/^@+/g, '').trim();
            }
            if (uname) return 'https://www.tiktok.com/@' + encodeURIComponent(uname) + '/video/' + encodeURIComponent(vid);
            return 'https://www.tiktok.com/video/' + encodeURIComponent(vid);
        }
        if (p === 'instagram') {
            if (igSc) return 'https://www.instagram.com/p/' + encodeURIComponent(igSc) + '/';
            if (vid && !/^\d+$/.test(vid)) return 'https://www.instagram.com/p/' + encodeURIComponent(vid) + '/';
            return null;
        }
        return null;
    }

    function buildProfileUrl(platform, entry, normalizePlatformFn) {
        const np = normalizePlatformFn || function (x) {
            return String(x || '').toLowerCase().trim();
        };
        const p = np(platform) || '';
        let e = enrichEntryFromTokenRow(entry);
        if (!e) e = entry || {};
        let uname = String(e.account_username || e.username || '').replace(/^@+/g, '').trim();
        if (!uname && entryTokenRowId(e)) {
            const acc = findAccountByTokenId(entryTokenRowId(e));
            if (acc && acc.username) uname = String(acc.username).replace(/^@+/g, '').trim();
        }
        const aid = e.account_id || '';
        if (p === 'youtube' && aid) return 'https://www.youtube.com/channel/' + encodeURIComponent(aid);
        if (p === 'tiktok' && uname) return 'https://www.tiktok.com/@' + encodeURIComponent(uname);
        if (p === 'instagram' && uname) return 'https://www.instagram.com/' + encodeURIComponent(uname) + '/';
        if (p === 'facebook' && aid) return 'https://www.facebook.com/' + encodeURIComponent(aid);
        if (p === 'facebook' && uname) return 'https://www.facebook.com/' + encodeURIComponent(uname);
        return null;
    }

    function shouldPreferProfileOverPending(entry, upload) {
        const st = String(upload && upload.status || '').toLowerCase();
        if (entry && entry.success === false) return true;
        // Partial / terminal: prefer a real profile link over a missing or flaky watch URL.
        if (['failed', 'cancelled', 'canceled', 'partial'].indexOf(st) !== -1) return true;
        return false;
    }

    /**
     * Prefer watch URL; avoid profile-as-post when upload succeeded but permalink not synced yet.
     */
    function resolveChipHref(entry, upload, videoUrl, profileUrl) {
        if (videoUrl) {
            return { href: videoUrl, kind: 'video', tip: 'Watch video' };
        }
        if (shouldPreferProfileOverPending(entry, upload) && profileUrl) {
            return { href: profileUrl, kind: 'profile', tip: 'View profile' };
        }
        return {
            href: null,
            kind: 'pending',
            tip: 'Post link pending — sync from platform shortly',
        };
    }

    var UploadM8Urls = {
        findAccountByTokenId: findAccountByTokenId,
        enrichEntryFromTokenRow: enrichEntryFromTokenRow,
        normalizeOutboundUrl: normalizeOutboundUrl,
        isKnownSocialPostHost: isKnownSocialPostHost,
        isLikelySourceOrCdnVideoUrl: isLikelySourceOrCdnVideoUrl,
        isPlausibleWatchUrl: isPlausibleWatchUrl,
        buildUrlFromEntry: buildUrlFromEntry,
        buildProfileUrl: buildProfileUrl,
        resolveChipHref: resolveChipHref,
        shouldPreferProfileOverPending: shouldPreferProfileOverPending,
    };

    window.toNum = toNum;
    window.getUploadStats = getUploadStats;
    window.entryTokenRowId = entryTokenRowId;
    window.getTargetAccountIds = getTargetAccountIds;
    window.filterPrListForChips = filterPrListForChips;
    window.buildChipEntries = buildChipEntries;
    window.findAccountByTokenId = findAccountByTokenId;
    window.UploadM8Urls = UploadM8Urls;
})();
