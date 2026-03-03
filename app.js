/**
 * UploadM8 v3.1 - Production-Grade Application JavaScript
 * Complete with all requested features
 */

// ============================================================
// Configuration
// ============================================================
const API_BASE = 'https://auth.uploadm8.com';
const APP_VERSION = '3.1.0';
const TOKEN_KEY = 'uploadm8_access_token';
const REFRESH_KEY = 'uploadm8_refresh_token';

let currentUser = null;
let isAuthChecking = false;
let authCheckPromise = null;
let _cachedUser = null;
let _cachedUserAt = 0;
const _USER_CACHE_TTL = 30000; // 30 seconds — covers all sequential calls on a single page load

// ============================================================
// Theme Management (Dark/Light Mode)
// ============================================================
function getTheme() {
    return localStorage.getItem('uploadm8_theme') || 'dark';
}

function setTheme(theme) {
    localStorage.setItem('uploadm8_theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
    if (theme === 'light') {
        document.body.classList.add('light-mode');
        document.body.classList.remove('dark-mode');
    } else {
        document.body.classList.add('dark-mode');
        document.body.classList.remove('light-mode');
    }
    const icons = document.querySelectorAll('#themeToggleIcon, #themeToggleIconDesktop, #themeToggleIconMobile');
    icons.forEach(icon => {
        if (icon) icon.className = theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    });
}

function toggleTheme() {
    setTheme(getTheme() === 'dark' ? 'light' : 'dark');
}

// Initialize theme immediately
(function initTheme() {
    const theme = getTheme();
    document.documentElement.setAttribute('data-theme', theme);
})();

// ============================================================
// Request ID Generation
// ============================================================
function generateRequestId() {
    return 'req_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
}

// ============================================================
// Token Management
// ============================================================
function getAccessToken() {
    return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY);
}

function getRefreshToken() {
    // Primary (UploadM8 keys)
    const primary = localStorage.getItem(REFRESH_KEY) || sessionStorage.getItem(REFRESH_KEY);
    if (primary) return primary;

    // Compatibility keys (older builds / auth-core.js)
    return (
        localStorage.getItem('refreshToken') ||
        localStorage.getItem('refresh_token') ||
        sessionStorage.getItem('refreshToken') ||
        sessionStorage.getItem('refresh_token') ||
        ''
    );
}

function setTokens(accessToken, refreshToken, remember = true) {
    if (remember) {
        localStorage.setItem(TOKEN_KEY, accessToken);
        // Compatibility access token keys (used by some pages/auth-core)
        try {
            localStorage.setItem('accessToken', accessToken);
            localStorage.setItem('access_token', accessToken);
            localStorage.setItem('authToken', accessToken);
            localStorage.setItem('auth_token', accessToken);
            localStorage.setItem('token', accessToken);
        } catch (_) {}
        if (refreshToken) {
            // Primary storage
            localStorage.setItem(REFRESH_KEY, refreshToken);
            // Compatibility storage
            localStorage.setItem('refreshToken', refreshToken);
            localStorage.setItem('refresh_token', refreshToken);
        }
    } else {
        sessionStorage.setItem(TOKEN_KEY, accessToken);
        try {
            sessionStorage.setItem('accessToken', accessToken);
            sessionStorage.setItem('access_token', accessToken);
            sessionStorage.setItem('authToken', accessToken);
            sessionStorage.setItem('auth_token', accessToken);
            sessionStorage.setItem('token', accessToken);
        } catch (_) {}
        if (refreshToken) {
            sessionStorage.setItem(REFRESH_KEY, refreshToken);
            sessionStorage.setItem('refreshToken', refreshToken);
            sessionStorage.setItem('refresh_token', refreshToken);
        }
    }
}

function clearTokens() {
    _cachedUser = null;
    _cachedUserAt = 0;
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_KEY);
    sessionStorage.removeItem(TOKEN_KEY);
    currentUser = null;
    window.currentUser = null;
}

function isLoggedIn() {
    return !!getAccessToken();
}

// ============================================================
// Authentication - HARD GATE
// ============================================================
async function checkAuth(options = {}) {
    const { redirectOnFail = true, silent = false } = options;
    
    if (isAuthChecking && authCheckPromise) {
        return authCheckPromise;
    }

    // Return cached user if still fresh — prevents duplicate /api/me calls
    // during a single page load (initApp + applyTierOverride + auth-core each call this)
    if (_cachedUser && (Date.now() - _cachedUserAt) < _USER_CACHE_TTL) {
        return _cachedUser;
    }
    
    const token = getAccessToken();
    
    if (!token) {
        if (redirectOnFail && !isOnAuthPage()) {
            sessionStorage.setItem('uploadm8_auth_message', 'Please log in to continue.');
            window.location.href = 'login.html';
        }
        return null;
    }
    
    isAuthChecking = true;
    authCheckPromise = (async () => {
        try {
            const requestId = generateRequestId();
            
            // Add timeout to prevent hanging
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 15000);
            
            const resp = await fetch(`${API_BASE}/api/me`, {
                headers: { 
                    'Authorization': `Bearer ${token}`,
                    'X-Request-ID': requestId,
                    'Content-Type': 'application/json'
                },
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (resp.status === 401) {
                const refreshed = await tryRefreshToken();
                if (refreshed) {
                    isAuthChecking = false;
                    authCheckPromise = null;
                    return checkAuth(options);
                }
                clearTokens();
                if (redirectOnFail && !isOnAuthPage()) {
                    sessionStorage.setItem('uploadm8_auth_message', 'Session expired. Please log in again.');
                    window.location.href = 'login.html';
                }
                return null;
            }
            
            if (!resp.ok) {
                console.error(`Auth check failed: ${resp.status} [${requestId}]`);
                if (redirectOnFail && !isOnAuthPage()) {
                    clearTokens();
                    window.location.href = 'login.html';
                }
                return null;
            }
            
            currentUser = await resp.json();
            
            // Validate user object
            if (!currentUser || !currentUser.email || !currentUser.role) {
                throw new Error('Invalid user data received');
            }
            
            window.currentUser = currentUser;
            // Cache for this page load — avoids repeat /api/me calls within 30s
            _cachedUser = currentUser;
            _cachedUserAt = Date.now();
            return currentUser;
            
        } catch (e) {
            console.error('Auth check error:', e);
            if (!silent && redirectOnFail && !isOnAuthPage()) {
                if (e.name === 'AbortError') {
                    sessionStorage.setItem('uploadm8_auth_message', 'Connection timed out. Please try again.');
                } else {
                    sessionStorage.setItem('uploadm8_auth_message', 'Connection error. Please try again.');
                }
                window.location.href = 'login.html';
            }
            return null;
        } finally {
            isAuthChecking = false;
            authCheckPromise = null;
        }
    })();
    
    return authCheckPromise;
}

function isOnAuthPage() {
    const path = window.location.pathname;
    return path.includes('login.html') || 
           path.includes('signup.html') || 
           path.includes('forgot-password.html') ||
           path.includes('reset-password.html') ||
           path.endsWith('/') ||
           path.includes('index.html') ||
           path.includes('terms.html') ||
           path.includes('privacy.html') ||
           path.includes('support.html');
}

async function tryRefreshToken() {
    const refreshToken = getRefreshToken();
    if (!refreshToken) return false;
    
    try {
        // Some FastAPI models forbid extra fields (422 if we send both keys).
        // Attempt snake_case first, then camelCase fallback.
        const attempts = [
            { refresh_token: refreshToken },
            { refreshToken: refreshToken }
        ];

        let resp = null;
        for (const payload of attempts) {
            resp = await fetch(`${API_BASE}/api/auth/refresh`, {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Request-ID': generateRequestId()
                },
                body: JSON.stringify(payload)
            });
            if (resp.ok) break;
            // If validation fails, try the other payload shape.
            if (resp.status !== 422) break;
        }
        
        if (!resp.ok) {
            clearTokens();
            return false;
        }
        
        const data = await resp.json();
        setTokens(data.access_token, data.refresh_token, true);
        return true;
    } catch (e) {
        console.error('Token refresh failed:', e);
        return false;
    }
}

// ============================================================
// Login / Logout / Register
// ============================================================
async function login(email, password, remember = true) {
    const requestId = generateRequestId();
    
    try {
        const resp = await fetch(`${API_BASE}/api/auth/login`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-Request-ID': requestId
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await resp.json().catch(() => ({}));
        
        if (!resp.ok) {
            return { 
                success: false, 
                error: data.detail || 'Invalid email or password',
                requestId 
            };
        }
        
        setTokens(data.access_token, data.refresh_token, remember);
        return { success: true, requestId };
        
    } catch (e) {
        console.error(`Login error [${requestId}]:`, e);
        return { 
            success: false, 
            error: 'Connection failed. Please check your internet.',
            requestId 
        };
    }
}

async function logout() {
    const token = getAccessToken();
    const refreshToken = getRefreshToken();
    
    if (token && refreshToken) {
        fetch(`${API_BASE}/api/auth/logout`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
                'X-Request-ID': generateRequestId()
            },
            body: JSON.stringify({ refresh_token: refreshToken })
        }).catch(() => {});
    }
    
    clearTokens();
    window.location.href = 'login.html';
}

async function logoutAll() {
    const token = getAccessToken();
    
    if (token) {
        try {
            await fetch(`${API_BASE}/api/auth/logout-all`, {
                method: 'POST',
                headers: { 
                    'Authorization': `Bearer ${token}`,
                    'X-Request-ID': generateRequestId()
                }
            });
        } catch (e) {}
    }
    
    clearTokens();
    window.location.href = 'login.html';
}

async function register(name, email, password) {
    const requestId = generateRequestId();
    
    try {
        const resp = await fetch(`${API_BASE}/api/auth/register`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-Request-ID': requestId
            },
            body: JSON.stringify({ name, email, password })
        });
        
        const data = await resp.json().catch(() => ({}));
        
        if (!resp.ok) {
            return { success: false, error: data.detail || 'Registration failed', requestId };
        }
        
        if (data.access_token) {
            setTokens(data.access_token, data.refresh_token, true);
        }
        
        return { success: true, requestId };
        
    } catch (e) {
        console.error(`Registration error [${requestId}]:`, e);
        return { success: false, error: 'Connection failed. Please try again.', requestId };
    }
}

// ============================================================
// API Calls with Error Handling
// ============================================================
async function apiCall(endpoint, options = {}) {
    const token = getAccessToken();
    const requestId = generateRequestId();
    
    if (!token) {
        const error = new Error('Not authenticated');
        error.status = 401;
        error.requestId = requestId;
        throw error;
    }
    
    const config = {
        ...options,
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
            'X-Request-ID': requestId,
            ...options.headers
        }
    };
    
    if (options.body instanceof FormData) {
        delete config.headers['Content-Type'];
    }
    
    try {
        const resp = await fetch(`${API_BASE}${endpoint}`, config);
        
        if (resp.status === 401) {
            const refreshed = await tryRefreshToken();
            if (refreshed) {
                return apiCall(endpoint, options);
            }
            clearTokens();
            sessionStorage.setItem('uploadm8_auth_message', 'Session expired. Please log in again.');
            window.location.href = 'login.html';
            throw new Error('Session expired');
        }
        
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Request failed' }));
            const error = new Error(err.detail || `API Error: ${resp.status}`);
            error.status = resp.status;
            error.requestId = requestId;
            error.response = err;
            throw error;
        }
        
        const text = await resp.text();
        return text ? JSON.parse(text) : {};
        
    } catch (e) {
        if (e.message === 'Session expired') throw e;
        if (!e.requestId) e.requestId = requestId;
        console.error(`API Error [${requestId}] ${endpoint}:`, e);
        throw e;
    }
}

// ============================================================
// Upload with Progress
// ============================================================
let activeUploads = new Map();

async function uploadFile(file, metadata, onProgress, onStatusChange) {
    const requestId = generateRequestId();
    
    try {
        if (onStatusChange) onStatusChange('presigning');
        const presign = await apiCall('/api/uploads/presign', {
            method: 'POST',
            body: JSON.stringify({
                filename: file.name,
                file_size: file.size,
                content_type: file.type,
                ...metadata
            })
        });
        
        const uploadId = presign.upload_id;
        
        if (onStatusChange) onStatusChange('uploading');
        await uploadToR2WithProgress(presign.presigned_url, file, uploadId, onProgress);
        
        const uploadState = activeUploads.get(uploadId);
        if (uploadState?.cancelled) {
            return { success: false, error: 'Upload cancelled', uploadId };
        }
        
        if (onStatusChange) onStatusChange('completing');
        const result = await apiCall(`/api/uploads/${uploadId}/complete`, { method: 'POST' });
        
        activeUploads.delete(uploadId);
        return { success: true, upload: result, uploadId };
        
    } catch (e) {
        console.error(`Upload error [${requestId}]:`, e);
        return { success: false, error: e.message, requestId };
    }
}

function uploadToR2WithProgress(url, file, uploadId, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        
        activeUploads.set(uploadId, { xhr, cancelled: false });
        
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable && onProgress) {
                onProgress(Math.round((e.loaded / e.total) * 100));
            }
        });
        
        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(xhr);
            } else {
                reject(new Error(`Upload failed: ${xhr.status}`));
            }
        });
        
        xhr.addEventListener('error', () => reject(new Error('Upload failed: network error')));
        xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')));
        
        xhr.open('PUT', url);
        xhr.setRequestHeader('Content-Type', file.type);
        xhr.send(file);
    });
}

async function cancelUpload(uploadId) {
    try {
        const uploadState = activeUploads.get(uploadId);
        if (uploadState?.xhr) {
            uploadState.cancelled = true;
            uploadState.xhr.abort();
        }
        
        await apiCall(`/api/uploads/${uploadId}/cancel`, { method: 'POST' });
        activeUploads.delete(uploadId);
        return { success: true };
    } catch (e) {
        return { success: false, error: e.message };
    }
}

async function retryUpload(uploadId) {
    try {
        await apiCall(`/api/uploads/${uploadId}/retry`, { method: 'POST' });
        return { success: true };
    } catch (e) {
        return { success: false, error: e.message };
    }
}

// ============================================================
// User Info & Entitlements
// ============================================================
function updateUserUI() {
    if (!currentUser) return;
    
    const role = currentUser.role || 'user';
    const tier = currentUser.subscription_tier || 'free';
    const isAdmin = ['admin', 'master_admin'].includes(role);
    const isMasterAdmin = role === 'master_admin';
    
    // For sidebar display: show role for admins, tier for regular users
    let displayTier;
    if (isMasterAdmin) {
        displayTier = 'Master Admin';
    } else if (role === 'admin') {
        displayTier = 'Admin';
    } else {
        displayTier = getTierDisplayName(tier);
    }
    
    const els = {
        userName: currentUser.name || 'User',
        userEmail: currentUser.email || '',
        welcomeName: (currentUser.name || 'User').split(' ')[0],
        userTier: displayTier,
        userRole: role
    };
    
    Object.entries(els).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    });
    
    // Avatar
    const userAvatar = document.getElementById('userAvatar');
    if (userAvatar) {
        const avatarSrc =
            currentUser.avatarSignedUrl ||
            currentUser.avatar_signed_url ||
            currentUser.avatarUrl ||
            currentUser.avatar_url ||
            null;

        if (avatarSrc) {
            const base = String(avatarSrc).split('#')[0];
            const safeUrl = `${base}#v=${Date.now()}`;
            userAvatar.innerHTML = `<img src="${safeUrl}" alt="" style="width:100%;height:100%;border-radius:50%;object-fit:cover;">`;
        } else {
            userAvatar.textContent = (currentUser.name || currentUser.email || 'U')[0].toUpperCase();
        }
    }
    
    // Tier badge - show role badge for admins
    const tierBadge = document.getElementById('tierBadge');
    if (tierBadge) {
        if (isMasterAdmin) {
            tierBadge.innerHTML = '<span class="tier-badge bg-red">Master Admin</span>';
        } else if (role === 'admin') {
            tierBadge.innerHTML = '<span class="tier-badge bg-orange">Admin</span>';
        } else {
            tierBadge.innerHTML = getTierBadgeHTML(tier);
        }
    }
    
    // Quota display
    const isUnlimited = currentUser.unlimited_uploads || 
        ['lifetime', 'friends_family'].includes(tier) ||
        isAdmin;
    
    const quotaUsed = document.getElementById('quotaUsed');
    const quotaTotal = document.getElementById('quotaTotal');
    const quotaBar = document.getElementById('quotaBar');
    
    const used = currentUser.uploads_this_month || 0;
    const limit = currentUser.upload_quota || getUploadLimit(tier);
    
    if (quotaUsed) quotaUsed.textContent = used;
    if (quotaTotal) quotaTotal.textContent = isUnlimited ? '∞' : limit;
    
    if (quotaBar) {
        if (isUnlimited) {
            quotaBar.style.width = '10%';
            quotaBar.className = 'quota-bar bg-green';
        } else {
            const percent = Math.min(100, (used / limit) * 100);
            quotaBar.style.width = `${percent}%`;
            quotaBar.className = `quota-bar ${percent > 80 ? 'bg-red' : percent > 50 ? 'bg-orange' : 'bg-green'}`;
        }
    }
    
    // Admin visibility - handle both class-based and ID-based approaches
    // Handle class="admin-only hidden" approach
    document.querySelectorAll('.admin-only').forEach(el => {
        if (isAdmin) {
            el.classList.remove('hidden');
            el.style.display = '';
        } else {
            el.classList.add('hidden');
            el.style.display = 'none';
        }
    });
    
    // Handle id="adminSection" style="display:none" approach
    const adminSection = document.getElementById('adminSection');
    if (adminSection) {
        adminSection.style.display = isAdmin ? 'block' : 'none';
    }
    
    console.log('[Auth] User:', currentUser.email, '| Role:', role, '| Tier:', tier, '| isAdmin:', isAdmin);
    
    // Master admin visibility
    document.querySelectorAll('.master-admin-only').forEach(el => {
        if (isMasterAdmin) {
            el.classList.remove('hidden');
            el.style.display = '';
        } else {
            el.classList.add('hidden');
            el.style.display = 'none';
        }
    });
    
    // Store in window for other scripts
    window.isUserAdmin = isAdmin;
    window.isUserMasterAdmin = isMasterAdmin;
    window.userRole = role;
    window.userTier = tier;
}

// ============================================================
// Role & Tier Check Helpers
// ============================================================
function isAdmin() {
    if (!currentUser) return false;
    return ['admin', 'master_admin'].includes(currentUser.role);
}

function isMasterAdmin() {
    if (!currentUser) return false;
    return currentUser.role === 'master_admin';
}

function isPaidUser() {
    if (!currentUser) return false;
    const role = currentUser.role || 'user';
    const tier = currentUser.subscription_tier || 'free';
    
    // Admins count as paid
    if (['admin', 'master_admin'].includes(role)) return true;
    
    // These tiers are considered paid/premium
    const paidTiers = ['launch', 'creator', 'creator_pro', 'growth', 'studio', 'agency', 'lifetime', 'friends_family'];
    return paidTiers.includes(tier);
}

function isFreeUser() {
    if (!currentUser) return true;
    return !isPaidUser();
}

function isFriendsFamily() {
    if (!currentUser) return false;
    return currentUser.subscription_tier === 'friends_family';
}

function isLifetime() {
    if (!currentUser) return false;
    return currentUser.subscription_tier === 'lifetime';
}

function getUserAccessLevel() {
    if (!currentUser) return 'guest';
    const role = currentUser.role || 'user';
    const tier = currentUser.subscription_tier || 'free';
    
    if (role === 'master_admin') return 'master_admin';
    if (role === 'admin') return 'admin';
    if (['lifetime', 'friends_family'].includes(tier)) return 'premium';
    if (['agency', 'studio'].includes(tier)) return 'business';
    if (['creator_pro', 'creator', 'growth'].includes(tier)) return 'pro';
    if (['launch', 'starter'].includes(tier)) return 'basic';
    return 'free';
}

function hasEntitlement(feature) {
    if (!currentUser) return false;
    const ent = currentUser.entitlements || {};
    const role = currentUser.role || 'user';
    
    // Admin always has all entitlements
    if (['admin', 'master_admin'].includes(role)) return true;
    
    // Special tiers have all entitlements
    if (['lifetime', 'friends_family'].includes(currentUser.subscription_tier)) return true;
    
    return !!ent[feature];
}

function showsAds() {
    if (!currentUser) return true;
    const role = currentUser.role || 'user';
    const tier = currentUser.subscription_tier || 'free';
    
    if (['admin', 'master_admin'].includes(role)) return false;
    if (['lifetime', 'friends_family', 'launch', 'creator', 'creator_pro', 'growth', 'studio', 'agency'].includes(tier)) return false;
    
    return currentUser.entitlements?.show_ads !== false;
}

function hasWatermark() {
    if (!currentUser) return true;
    const role = currentUser.role || 'user';
    const tier = currentUser.subscription_tier || 'free';
    
    if (['admin', 'master_admin'].includes(role)) return false;
    if (['lifetime', 'friends_family', 'launch', 'creator', 'creator_pro', 'growth', 'studio', 'agency'].includes(tier)) return false;
    
    return currentUser.entitlements?.show_watermark !== false;
}

function getMaxAccounts() {
    if (!currentUser) return 1;
    const role = currentUser.role || 'user';
    
    if (['admin', 'master_admin'].includes(role)) return 999;
    
    // Try to get from entitlements first, then max_accounts field, then default
    return currentUser.entitlements?.max_accounts || currentUser.max_accounts || 1;
}

function getMaxHashtags() {
    if (!currentUser) return 2;
    const role = currentUser.role || 'user';
    
    if (['admin', 'master_admin'].includes(role)) return 9999;
    if (['lifetime', 'friends_family'].includes(currentUser.subscription_tier)) return 9999;
    
    return currentUser.entitlements?.max_hashtags || 2;
}

function getUploadLimit(tier) {
    const limits = {
        'free': 5, 'starter': 10, 'solo': 60, 'creator': 200,
        'growth': 500, 'studio': 1500, 'agency': 5000,
        'lifetime': 999999, 'friends_family': 999999
    };
    return limits[tier] || 5;
}

function getTierDisplayName(tier) {
    const names = {
        'free': 'Free', 'starter': 'Starter', 'solo': 'Solo',
        'creator': 'Creator', 'growth': 'Growth', 'studio': 'Studio',
        'agency': 'Agency', 'lifetime': 'Lifetime', 'friends_family': 'Friends & Family',
        'launch': 'Launch', 'creator_pro': 'Creator Pro', 'master_admin': 'Admin'
    };
    return names[tier] || tier || 'Free';
}

function getTierBadgeHTML(tier) {
    const colors = {
        'free': 'bg-gray', 'starter': 'bg-blue', 'solo': 'bg-blue',
        'creator': 'bg-orange', 'growth': 'bg-orange', 'studio': 'bg-purple',
        'agency': 'bg-purple', 'lifetime': 'bg-gradient', 'friends_family': 'bg-gradient',
        'launch': 'bg-blue', 'creator_pro': 'bg-orange', 'master_admin': 'bg-red'
    };
    const color = colors[tier] || 'bg-gray';
    return `<span class="tier-badge ${color}">${getTierDisplayName(tier)}</span>`;
}

function getUserStatusDot(status) {
    const colors = { 'active': 'green', 'trialing': 'yellow', 'canceled': 'red', 'past_due': 'red' };
    return `<span class="status-dot bg-${colors[status] || 'gray'}"></span>`;
}

function getUserStatusBadge(user) {
    const tier = user.subscription_tier || 'free';
    const status = user.subscription_status;
    const trialEnds = user.trial_ends_at;
    
    if (['lifetime', 'friends_family'].includes(tier)) {
        return '<span class="badge badge-purple">Lifetime</span>';
    }
    if (status === 'trialing' || (trialEnds && new Date(trialEnds) > new Date())) {
        return '<span class="badge badge-yellow">Trial</span>';
    }
    if (status === 'active') {
        return '<span class="badge badge-green">Active</span>';
    }
    if (status === 'canceled' || status === 'past_due') {
        return '<span class="badge badge-red">Canceled</span>';
    }
    return '<span class="badge badge-gray">Free</span>';
}

// ============================================================
// UI Helpers
// ============================================================
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toastContainer') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icons = { success: 'check-circle', error: 'exclamation-circle', warning: 'exclamation-triangle', info: 'info-circle' };
    toast.innerHTML = `
        <i class="fas fa-${icons[type] || 'info-circle'}"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()" class="toast-close">&times;</button>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => toast.classList.add('show'), 10);
    
    if (duration > 0) {
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
    
    return toast;
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toastContainer';
    document.body.appendChild(container);
    return container;
}

function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

function hideModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

function showConfirmModal(title, message, onConfirm) {
    const modal = document.getElementById('confirmModal');
    if (!modal) return;
    
    document.getElementById('confirmTitle').textContent = title;
    document.getElementById('confirmMessage').textContent = message;
    
    const confirmBtn = document.getElementById('confirmAction');
    confirmBtn.onclick = () => {
        hideModal('confirmModal');
        if (onConfirm) onConfirm();
    };
    
    showModal('confirmModal');
}

function showLoading(container, message = 'Loading...') {
    if (typeof container === 'string') {
        container = document.getElementById(container);
    }
    if (container) {
        container.innerHTML = `
            <div class="flex items-center justify-center py-8 text-secondary">
                <i class="fas fa-spinner fa-spin mr-2"></i> ${message}
            </div>
        `;
    }
}

function showEmptyState(container, icon, message, actionHtml = '') {
    if (typeof container === 'string') {
        container = document.getElementById(container);
    }
    if (container) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-${icon}"></i>
                <p>${message}</p>
                ${actionHtml}
            </div>
        `;
    }
}

function showError(container, message) {
    if (typeof container === 'string') {
        container = document.getElementById(container);
    }
    if (container) {
        container.innerHTML = `
            <div class="error-state">
                <i class="fas fa-exclamation-circle"></i>
                <p>${message}</p>
            </div>
        `;
    }
}

function toggleSidebar() {
    // Legacy-safe proxy. Real handler is wired on DOMContentLoaded.
    if (typeof window.toggleSidebar === 'function' && window.toggleSidebar !== toggleSidebar) {
        window.toggleSidebar();
    }
}

// ============================================================
// Formatting Helpers
// ============================================================
function formatDate(dateStr) {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatDateTime(dateStr) {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleString('en-US', { 
        month: 'short', day: 'numeric', year: 'numeric',
        hour: 'numeric', minute: '2-digit'
    });
}

function formatRelativeTime(dateStr) {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return formatDate(dateStr);
}

function formatNumber(num) {
    if (num === null || num === undefined) return '-';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toLocaleString();
}

function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatCurrency(amount, currency = 'USD') {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(amount);
}

// ============================================================
// Platform Helpers
// ============================================================
function getPlatformInfo(platform) {
    const platforms = {
        'tiktok': { name: 'TikTok', icon: 'fab fa-tiktok', color: '#00f2ea' },
        'youtube': { name: 'YouTube', icon: 'fab fa-youtube', color: '#FF0000' },
        'instagram': { name: 'Instagram', icon: 'fab fa-instagram', color: '#E1306C' },
        'facebook': { name: 'Facebook', icon: 'fab fa-facebook', color: '#1877F2' },
        'meta': { name: 'Meta', icon: 'fab fa-meta', color: '#0668E1' },
        'google': { name: 'YouTube', icon: 'fab fa-youtube', color: '#FF0000' }
    };
    return platforms[platform?.toLowerCase()] || { name: platform, icon: 'fas fa-globe', color: '#666' };
}

function getPlatformIcon(platform) {
    const info = getPlatformInfo(platform);
    return `<i class="${info.icon}" style="color: ${info.color};" title="${info.name}"></i>`;
}

function getPlatformBadge(platform) {
    const info = getPlatformInfo(platform);
    return `<span class="platform-badge" style="background: ${info.color}20; color: ${info.color};">
        <i class="${info.icon}"></i> ${info.name}
    </span>`;
}

function getStatusBadge(status) {
    const statuses = {
        'pending': { label: 'Pending', color: 'yellow', icon: 'clock' },
        'queued': { label: 'Queued', color: 'blue', icon: 'list' },
        'processing': { label: 'Processing', color: 'blue', icon: 'spinner fa-spin' },
        'uploading': { label: 'Uploading', color: 'blue', icon: 'cloud-upload-alt' },
        'completed': { label: 'Completed', color: 'green', icon: 'check-circle' },
        'failed': { label: 'Failed', color: 'red', icon: 'exclamation-circle' },
        'cancelled': { label: 'Cancelled', color: 'gray', icon: 'ban' },
        'scheduled': { label: 'Scheduled', color: 'purple', icon: 'calendar-alt' },
        'partial': { label: 'Partial', color: 'orange', icon: 'exclamation-triangle' }
    };
    const info = statuses[status] || { label: status, color: 'gray', icon: 'question' };
    return `<span class="status-badge status-${info.color}">
        <i class="fas fa-${info.icon}"></i> ${info.label}
    </span>`;
}

// ============================================================
// KPI Time Ranges
// ============================================================
const KPI_RANGES = [
    { value: '30m', label: '30 Minutes', minutes: 30 },
    { value: '1h', label: '1 Hour', minutes: 60 },
    { value: '6h', label: '6 Hours', minutes: 360 },
    { value: '12h', label: '12 Hours', minutes: 720 },
    { value: '1d', label: '1 Day', minutes: 1440 },
    { value: '7d', label: '7 Days', minutes: 10080 },
    { value: '30d', label: '30 Days', minutes: 43200 },
    { value: '6m', label: '6 Months', minutes: 262800 },
    { value: '1y', label: '1 Year', minutes: 525600 },
    { value: 'custom', label: 'Custom Range', minutes: 0 }
];

function getKpiRangeMinutes(range) {
    return KPI_RANGES.find(r => r.value === range)?.minutes || 43200;
}

function buildKpiRangeDropdown(selectId, onChange) {
    const select = document.getElementById(selectId);
    if (!select) return;
    
    select.innerHTML = KPI_RANGES.map(r => 
        `<option value="${r.value}">${r.label}</option>`
    ).join('');
    
    if (onChange) {
        select.addEventListener('change', (e) => onChange(e.target.value));
    }
}

// ============================================================
// Dashboard Card Drag & Drop
// ============================================================
function initDashboardCustomization() {
    const grid = document.getElementById('dashboardGrid');
    if (!grid) return;
    
    const saved = localStorage.getItem('uploadm8_dashboard_order');
    if (saved) {
        try {
            const order = JSON.parse(saved);
            const cards = Array.from(grid.children);
            order.forEach(id => {
                const card = cards.find(c => c.dataset.cardId === id);
                if (card) grid.appendChild(card);
            });
        } catch (e) {}
    }
    
    grid.querySelectorAll('[data-card-id]').forEach(card => {
        card.draggable = true;
        card.classList.add('draggable-card');
        
        card.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/plain', card.dataset.cardId);
            card.classList.add('dragging');
            setTimeout(() => card.style.opacity = '0.5', 0);
        });
        
        card.addEventListener('dragend', () => {
            card.classList.remove('dragging');
            card.style.opacity = '';
            saveDashboardOrder();
        });
        
        card.addEventListener('dragover', (e) => {
            e.preventDefault();
            const dragging = grid.querySelector('.dragging');
            if (dragging && dragging !== card) {
                const rect = card.getBoundingClientRect();
                const midY = rect.top + rect.height / 2;
                if (e.clientY < midY) {
                    card.parentNode.insertBefore(dragging, card);
                } else {
                    card.parentNode.insertBefore(dragging, card.nextSibling);
                }
            }
        });
    });
}

function saveDashboardOrder() {
    const grid = document.getElementById('dashboardGrid');
    if (!grid) return;
    const order = Array.from(grid.querySelectorAll('[data-card-id]')).map(c => c.dataset.cardId);
    localStorage.setItem('uploadm8_dashboard_order', JSON.stringify(order));
    showToast('Dashboard layout saved', 'success');
}

function resetDashboardOrder() {
    localStorage.removeItem('uploadm8_dashboard_order');
    location.reload();
}

// ============================================================
// Hide Figures (Financial Privacy)
// ============================================================
let figuresHidden = localStorage.getItem('uploadm8_hide_figures') === 'true';

function toggleHideFigures() {
    figuresHidden = !figuresHidden;
    localStorage.setItem('uploadm8_hide_figures', figuresHidden);
    updateHiddenFigures();
}

function updateHiddenFigures() {
    document.querySelectorAll('[data-hide-value]').forEach(el => {
        if (figuresHidden) {
            el.dataset.realValue = el.textContent;
            el.textContent = '••••••';
        } else if (el.dataset.realValue) {
            el.textContent = el.dataset.realValue;
        }
    });
    
    const icon = document.getElementById('hideFiguresIcon');
    if (icon) {
        icon.className = figuresHidden ? 'fas fa-eye' : 'fas fa-eye-slash';
    }
}

// ============================================================
// Navigation & Page Initialization
// ============================================================
function highlightCurrentNav() {
    const path = window.location.pathname.split('/').pop() || 'dashboard.html';
    document.querySelectorAll('.nav-link').forEach(link => {
        const href = link.getAttribute('href');
        link.classList.toggle('active', href === path);
    });
}

function initDragDrop(dropZoneId, fileInputId, onFileSelect) {
    const dropZone = document.getElementById(dropZoneId);
    const fileInput = document.getElementById(fileInputId);
    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(e => {
        dropZone.addEventListener(e, (ev) => { ev.preventDefault(); dropZone.classList.add('drag-over'); });
    });
    ['dragleave', 'drop'].forEach(e => {
        dropZone.addEventListener(e, (ev) => { ev.preventDefault(); dropZone.classList.remove('drag-over'); });
    });
    
    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length) onFileSelect(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) onFileSelect(e.target.files[0]);
    });
}

// ============================================================
// App Initialization
// ============================================================
async function initApp(pageName) {
    setTheme(getTheme());
    
    const authMessage = sessionStorage.getItem('uploadm8_auth_message');
    if (authMessage) {
        sessionStorage.removeItem('uploadm8_auth_message');
        setTimeout(() => showToast(authMessage, 'warning'), 100);
    }
    
    const user = await checkAuth({ redirectOnFail: true });
    if (!user) return null;
    
    updateUserUI();
    highlightCurrentNav();
    updateHiddenFigures();
    
    // Show admin section based on user role from database
    const isAdmin = user.role === 'admin' || user.role === 'master_admin';
    const adminSection = document.getElementById('adminSection');
    if (adminSection && isAdmin) {
        adminSection.style.display = 'block';
        console.log('Admin section enabled for role:', user.role);
    }
    
    // Dispatch event so other components know user is loaded
    window.dispatchEvent(new CustomEvent('userLoaded', { detail: user }));
    
    document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);
    // NOTE: menuToggle and sidebarOverlay are wired in the universal DOMContentLoaded
    // block at the bottom of this file. Do NOT add them here — it causes double-fire
    // (sidebar opens then instantly closes on the same click).
    
    document.querySelectorAll('[data-action="logout"], .logout-btn, #logoutBtn').forEach(btn => {
        btn.addEventListener('click', (e) => { e.preventDefault(); logout(); });
    });
    
    if (pageName === 'dashboard') {
        initDashboardCustomization();
    }
    
    if (!showsAds()) {
        document.querySelectorAll('.ad-container, .ad-banner').forEach(el => el.remove());
    }
    
    return user;
}

// ============================================================
// Global Exports
// ============================================================
if (typeof window !== 'undefined') {
    window.API_BASE = API_BASE;
    window.APP_VERSION = APP_VERSION;
    window.checkAuth = checkAuth;
    window.login = login;
    window.logout = logout;
    window.logoutAll = logoutAll;
    window.register = register;
    window.isLoggedIn = isLoggedIn;
    window.getAccessToken = getAccessToken;
    window.clearTokens = clearTokens;
    window.apiCall = apiCall;
    window.uploadFile = uploadFile;
    window.cancelUpload = cancelUpload;
    window.retryUpload = retryUpload;
    window.currentUser = currentUser;
    window.updateUserUI = updateUserUI;
    window.hasEntitlement = hasEntitlement;
    window.getMaxAccounts = getMaxAccounts;
    window.getMaxHashtags = getMaxHashtags;
    window.showsAds = showsAds;
    window.hasWatermark = hasWatermark;
    window.isAdmin = isAdmin;
    window.isMasterAdmin = isMasterAdmin;
    window.isPaidUser = isPaidUser;
    window.isFreeUser = isFreeUser;
    window.isFriendsFamily = isFriendsFamily;
    window.isLifetime = isLifetime;
    window.getUserAccessLevel = getUserAccessLevel;
    window.getTierBadgeHTML = getTierBadgeHTML;
    window.getUserStatusDot = getUserStatusDot;
    window.getUserStatusBadge = getUserStatusBadge;
    window.showToast = showToast;
    window.showModal = showModal;
    window.hideModal = hideModal;
    window.showConfirmModal = showConfirmModal;
    window.showLoading = showLoading;
    window.showEmptyState = showEmptyState;
    window.showError = showError;
    window.toggleSidebar = toggleSidebar;
    window.toggleTheme = toggleTheme;
    window.getTheme = getTheme;
    window.setTheme = setTheme;
    window.toggleHideFigures = toggleHideFigures;
    window.initApp = initApp;
    window.initDragDrop = initDragDrop;
    window.initDashboardCustomization = initDashboardCustomization;
    window.resetDashboardOrder = resetDashboardOrder;
    window.highlightCurrentNav = highlightCurrentNav;
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
    window.KPI_RANGES = KPI_RANGES;
    window.getKpiRangeMinutes = getKpiRangeMinutes;
    window.buildKpiRangeDropdown = buildKpiRangeDropdown;
}

// ============================================================
// MOBILE SIDEBAR — Universal DOMContentLoaded Init
// Fires on EVERY page regardless of whether initApp() is called.
// This is the single source of truth for burger menu + back buttons.
// ============================================================
document.addEventListener('DOMContentLoaded', function () {

    // ── 1. Sidebar toggle ──────────────────────────────────────
    var menuToggle   = document.getElementById('menuToggle');
    var sidebar      = document.getElementById('sidebar');
    var overlay      = document.getElementById('sidebarOverlay');

    var _sidebarIsOpen = false;

    function openSidebar() {
        if (!sidebar) return;
        _sidebarIsOpen = true;
        sidebar.classList.add('open');
        document.body.classList.add('sidebar-open');
        if (overlay) {
            overlay.classList.remove('hidden');
            overlay.classList.add('active');
            overlay.style.display = 'block';
        }
    }

    function closeSidebar() {
        if (!sidebar) return;
        _sidebarIsOpen = false;
        sidebar.classList.remove('open');
        document.body.classList.remove('sidebar-open');
        if (overlay) {
            overlay.classList.add('hidden');
            overlay.classList.remove('active');
            overlay.style.display = 'none';
        }
    }

    function _toggleSidebar() {
        if (!sidebar) return;
        if (_sidebarIsOpen || sidebar.classList.contains('open')) closeSidebar();
        else openSidebar();
    }

    // Wire burger button — replaces ANY existing onclick to avoid double-fire
    if (menuToggle) {
        menuToggle.removeAttribute('onclick'); // remove inline HTML onclick (prevents double-fire)
        menuToggle.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            _toggleSidebar();
        }, { passive: false });
    }

    // Wire overlay tap-to-close
    if (overlay) {
        overlay.removeAttribute('onclick'); // remove inline HTML onclick (prevents double-fire)
        overlay.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            closeSidebar();
        }, { passive: false });
    }

    // Close sidebar on any nav-link click (mobile UX — navigate and close)
    if (sidebar) {
        sidebar.querySelectorAll('.nav-link').forEach(function (link) {
            link.addEventListener('click', function () {
                if (window.innerWidth <= 1024 && sidebar.classList.contains('open')) {
                    closeSidebar();
                }
            }, { passive: true });
        });
    }

    // Also expose as window.toggleSidebar so inline onclicks still work
    window.toggleSidebar = _toggleSidebar;

    // Deterministic overlay visibility on load
    if (overlay) overlay.style.display = (sidebar && sidebar.classList.contains('open')) ? 'block' : 'none';

    // ── 2. Back button injection ───────────────────────────────
    // Inject a back button in the mobile top-bar on sub-pages.
    // Only on pages that are NOT the dashboard/home.
    var currentPage = window.location.pathname.split('/').pop() || 'index.html';
    var NO_BACK_PAGES = [
        'dashboard.html', 'index.html', 'login.html', 'signup.html', ''
    ];

    // Map: page → label + href for back button
    var BACK_MAP = {
        'upload.html':              { label: 'Dashboard', href: 'dashboard.html' },
        'queue.html':               { label: 'Dashboard', href: 'dashboard.html' },
        'scheduled.html':           { label: 'Dashboard', href: 'dashboard.html' },
        'platforms.html':           { label: 'Dashboard', href: 'dashboard.html' },
        'groups.html':              { label: 'Platforms', href: 'platforms.html' },
        'analytics.html':           { label: 'Dashboard', href: 'dashboard.html' },
        'settings.html':            { label: 'Dashboard', href: 'dashboard.html' },
        'color-preferences.html':   { label: 'Settings',  href: 'settings.html'  },
        'guide.html':               { label: 'Dashboard', href: 'dashboard.html' },
        'admin.html':               { label: 'Dashboard', href: 'dashboard.html' },
        'account-management.html':  { label: 'Admin',     href: 'admin.html'     },
        'admin-kpi.html':           { label: 'Admin',     href: 'admin.html'     },
        'admin-calculator.html':    { label: 'Admin',     href: 'admin.html'     },
        'admin-wallet.html':        { label: 'Admin',     href: 'admin.html'     },
        'billing.html':             { label: 'Settings',  href: 'settings.html'  },
        'success.html':             { label: 'Dashboard', href: 'dashboard.html' },
        'walkthrough.html':         { label: 'Home',      href: 'index.html'     },
        'kpi.html':                 { label: 'Admin',     href: 'admin.html'     },
    };

    if (!NO_BACK_PAGES.includes(currentPage) && BACK_MAP[currentPage]) {
        var backInfo = BACK_MAP[currentPage];
        var topBarActions = document.querySelector('.top-bar-actions');
        if (topBarActions && window.innerWidth <= 1024) {
            var backBtn = document.createElement('a');
            backBtn.href = backInfo.href;
            backBtn.className = 'back-btn-mobile back-btn';
            backBtn.innerHTML = '<i class="fas fa-arrow-left"></i><span>' + backInfo.label + '</span>';
            backBtn.style.cssText = 'display:inline-flex;align-items:center;gap:0.4rem;padding:0.4rem 0.75rem;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);border-radius:8px;color:var(--text-secondary);font-size:0.78rem;font-weight:500;text-decoration:none;min-height:36px;white-space:nowrap;';
            topBarActions.prepend(backBtn);
        }
    }

    // ── 3. Swipe-to-open sidebar (mobile gesture) ─────────────
    var touchStartX = 0;
    var touchStartY = 0;
    document.addEventListener('touchstart', function (e) {
        touchStartX = e.touches[0].clientX;
        touchStartY = e.touches[0].clientY;
    }, { passive: true });

    document.addEventListener('touchend', function (e) {
        if (!sidebar) return;
        var dx = e.changedTouches[0].clientX - touchStartX;
        var dy = Math.abs(e.changedTouches[0].clientY - touchStartY);
        // Swipe right from left edge → open
        if (touchStartX < 25 && dx > 60 && dy < 80 && !sidebar.classList.contains('open')) {
            _toggleSidebar();
        }
        // Swipe left while open → close
        if (sidebar.classList.contains('open') && dx < -60 && dy < 80) {
            _toggleSidebar();
        }
    }, { passive: true });

});
