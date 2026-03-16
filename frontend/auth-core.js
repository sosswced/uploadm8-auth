/**
 * UploadM8 Auth Core v3.0
 * Self-contained authentication module
 * Include on EVERY page that requires authentication
 */

(function() {
    'use strict';
    if (!window.API_BASE && /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(location.origin))
        window.API_BASE = 'http://127.0.0.1:8000';
    const API_BASE = window.API_BASE || 'https://auth.uploadm8.com';
    const TOKEN_KEY = 'uploadm8_access_token';
    const REFRESH_KEY = 'uploadm8_refresh_token';
    
    // Public pages that don't require auth
    const PUBLIC_PAGES = [
        'index.html', 'login.html', 'signup.html', 'forgot-password.html',
        'terms.html', 'privacy.html', 'about.html', 'contact.html',
        'support.html', 'blog.html', 'how-it-works.html', 'data-deletion.html',
        'walkthrough.html', ''
    ];
    
    function getCurrentPage() {
        return window.location.pathname.split('/').pop() || 'index.html';
    }
    
    function isPublicPage() {
        return PUBLIC_PAGES.includes(getCurrentPage());
    }
    
    function getToken() {
        return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY);
    }
    
    function clearTokens() {
        const accessKeys = [
            TOKEN_KEY,
            'accessToken', 'access_token',
            'authToken', 'auth_token',
            'token'
        ];
        const refreshKeys = [
            REFRESH_KEY,
            'refreshToken', 'refresh_token'
        ];
        for (const k of accessKeys.concat(refreshKeys)) {
            try { localStorage.removeItem(k); } catch (_) {}
            try { sessionStorage.removeItem(k); } catch (_) {}
        }
        window.currentUser = null;
    }
    
    function redirectToLogin(message) {
        clearTokens();
        if (message) sessionStorage.setItem('uploadm8_auth_message', message);
        window.location.href = 'login.html';
    }
    
    function formatTier(tier) {
        const tiers = {
            'free': 'Free', 'launch': 'Launch', 'creator_pro': 'Creator Pro',
            'studio': 'Studio', 'agency': 'Agency', 'master_admin': 'Master Admin',
            'friends_family': 'Friends & Family', 'lifetime': 'Lifetime'
        };
        return tiers[tier] || tier || 'Free';
    }
    
    function isAdmin(user) {
        return user && (user.role === 'admin' || user.role === 'master_admin');
    }
    
    function isMasterAdmin(user) {
        return user && user.role === 'master_admin';
    }
    
    function updateSidebar(user) {
        if (!user) return;
        
        const name = user.name || user.email?.split('@')[0] || 'User';
        const tier = formatTier(user.subscription_tier);
        const initial = name.charAt(0).toUpperCase();
        
        // Update name elements
        document.querySelectorAll('#sidebarUserName, #userName, .sidebar-user-name').forEach(el => {
            el.textContent = name;
        });
        
        // Update tier elements
        document.querySelectorAll('#sidebarUserTier, #userTier, .sidebar-user-tier').forEach(el => {
            el.textContent = tier;
        });
        
        // Update avatar
        document.querySelectorAll('#sidebarAvatar, #userAvatar, .sidebar-avatar').forEach(el => {
            if (!el.querySelector('img')) el.textContent = initial;
        });
        
        // Update welcome messages
        const welcomeEl = document.getElementById('welcomeName');
        if (welcomeEl) welcomeEl.textContent = name;
        
        // Show/hide admin section
        const adminSection = document.getElementById('adminSection');
        if (adminSection) {
            if (isAdmin(user)) {
                adminSection.style.display = 'block';
                adminSection.classList.remove('hidden');
            } else {
                adminSection.style.display = 'none';
            }
        }
        
        // Also handle any elements with admin-only class
        document.querySelectorAll('.admin-only').forEach(el => {
            if (isAdmin(user)) {
                el.style.display = 'block';
                el.classList.remove('hidden');
            } else {
                el.style.display = 'none';
            }
        });
    }
    
    /**
     * Main authentication function
     * @param {Object} options - { requireAdmin: false, requireMasterAdmin: false, silent: false }
     * @returns {Promise<Object|null>} User object or null
     */
    async function authenticate(options = {}) {
        const { requireAdmin = false, requireMasterAdmin = false, silent = false } = options;
        
        // Skip auth for public pages
        if (isPublicPage()) {
            console.log('[AuthCore] Public page, skipping auth');
            return null;
        }
        
        const token = getToken();
        
        // No token - redirect to login
        if (!token) {
            console.log('[AuthCore] No token found');
            if (!silent) redirectToLogin('Please log in to continue.');
            return null;
        }
        
        try {
            console.log('[AuthCore] Calling /api/me...');
            
            // Add timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 15000);
            
            const response = await fetch(`${API_BASE}/api/me`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (response.status === 401) {
                console.log('[AuthCore] Token expired (401)');
                if (!silent) redirectToLogin('Session expired. Please log in again.');
                return null;
            }
            
            if (!response.ok) {
                throw new Error(`API returned ${response.status}`);
            }
            
            const user = await response.json();
            
            // Validate user object
            if (!user || !user.email || !user.role) {
                throw new Error('Invalid user data received');
            }
            
            // Check role requirements
            if (requireMasterAdmin && !isMasterAdmin(user)) {
                console.log('[AuthCore] Master admin required, user is:', user.role);
                window.dispatchEvent(new CustomEvent('authComplete', { detail: { user, accessDenied: true, reason: 'master_admin_required' } }));
                return null;
            }
            
            if (requireAdmin && !isAdmin(user)) {
                console.log('[AuthCore] Admin required, user is:', user.role);
                window.dispatchEvent(new CustomEvent('authComplete', { detail: { user, accessDenied: true, reason: 'admin_required' } }));
                return null;
            }
            
            // Store globally
            window.currentUser = user;
            window.userRole = user.role;
            window.isUserAdmin = isAdmin(user);
            window.isUserMasterAdmin = isMasterAdmin(user);
            
            console.log('[AuthCore] Authenticated:', {
                email: user.email,
                role: user.role,
                tier: user.subscription_tier,
                isAdmin: window.isUserAdmin
            });
            
            // Update UI
            updateSidebar(user);
            
            // Dispatch success event
            window.dispatchEvent(new CustomEvent('authComplete', { detail: { user, accessDenied: false } }));
            window.dispatchEvent(new CustomEvent('userLoaded', { detail: user }));
            
            return user;
            
        } catch (error) {
            console.error('[AuthCore] Authentication error:', error);
            
            // Dispatch failure event
            window.dispatchEvent(new CustomEvent('authComplete', { detail: { user: null, error: error.message } }));
            window.dispatchEvent(new CustomEvent('authFailed', { detail: error }));
            
            return null;
        }
    }
    
    // Export globally
    window.AuthCore = {
        authenticate,
        isAdmin,
        isMasterAdmin,
        getToken,
        clearTokens,
        redirectToLogin,
        formatTier,
        updateSidebar,
        getCurrentUser: () => window.currentUser
    };
    
    // Backwards compatibility
    window.initAuth = authenticate;
    
})();

// ================================
// PATCH: Avatar + API helpers bridge
// - Adds non-breaking helpers used by settings.html.
// - Does NOT change existing auth flows; only augments.
// ================================
(function () {
  try {
    if (typeof window.API_BASE !== "string") window.API_BASE = "";

    // Non-breaking: expose getToken() if not already present.
    if (typeof window.getToken !== "function") {
      window.getToken = function getToken() {
        return (
          localStorage.getItem("uploadm8_access_token") ||
          sessionStorage.getItem("uploadm8_access_token") ||
          localStorage.getItem("accessToken") ||
          localStorage.getItem("access_token") ||
          localStorage.getItem("authToken") ||
          localStorage.getItem("auth_token") ||
          localStorage.getItem("token") ||
          ""
        );
      };
    }

    // Non-breaking: standard unauthorized handler (attempt refresh once if available).
    if (typeof window.handleUnauthorized !== "function") {
      window.handleUnauthorized = async function handleUnauthorized() {
        // If auth-core already has refresh logic, do nothing here.
        if (typeof window.refreshAccessToken === "function") {
          return window.refreshAccessToken();
        }
        // Best-effort refresh using stored refresh token, if your system uses it.
        const refresh =
          localStorage.getItem("uploadm8_refresh_token") ||
          sessionStorage.getItem("uploadm8_refresh_token") ||
          localStorage.getItem("refreshToken") ||
          localStorage.getItem("refresh_token") ||
          "";
        if (!refresh) return;

        // Try snake_case first (most likely backend contract). If backend forbids extras, avoid sending both.
        let res = await fetch(`${window.API_BASE || ""}/api/auth/refresh`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refresh })
        });

        // If validation failed, attempt camelCase payload.
        if (res.status === 422) {
          res = await fetch(`${window.API_BASE || ""}/api/auth/refresh`, {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ refreshToken: refresh })
          });
        }

        if (!res.ok) return;
        const data = await res.json().catch(() => null);

        // Normalize token fields across the entire frontend.
        // Backend contract: { access_token, refresh_token, token_type }
        const access = data && (data.access_token || data.accessToken);
        const refreshNew = data && (data.refresh_token || data.refreshToken);
        if (access) {
          // Primary UploadM8 key
          try { localStorage.setItem("uploadm8_access_token", access); } catch (_) {}
          // Compatibility keys
          try {
            localStorage.setItem("accessToken", access);
            localStorage.setItem("access_token", access);
            localStorage.setItem("authToken", access);
            localStorage.setItem("auth_token", access);
            localStorage.setItem("token", access);
          } catch (_) {}
        }
        if (refreshNew) {
          try { localStorage.setItem("uploadm8_refresh_token", refreshNew); } catch (_) {}
          try {
            localStorage.setItem("refreshToken", refreshNew);
            localStorage.setItem("refresh_token", refreshNew);
          } catch (_) {}
        }
      };
    }

    // Helper: apply avatar to common UI nodes (sidebar + settings page)
    window.applyAvatarToUI = function applyAvatarToUI(user) {
      if (!user) return;
      const url = user.avatar_url || user.avatarUrl || null;
      if (!url || typeof url !== 'string') return;
      const base = url.split('#')[0];
      if (!/^https?:/i.test(base)) return;
      const busted = base + '#v=' + Date.now();

      const sidebarAvatar = document.getElementById("sidebarAvatar");
      if (sidebarAvatar) {
        sidebarAvatar.textContent = '';
        const img = document.createElement('img');
        img.src = busted;
        img.style.cssText = 'width:100%;height:100%;object-fit:cover;border-radius:50%;';
        sidebarAvatar.appendChild(img);
      }

      const avatarImage = document.getElementById("avatarImage");
      const avatarInitial = document.getElementById("avatarInitial");
      if (avatarImage) {
        avatarImage.src = busted;
        avatarImage.style.display = "block";
      }
      if (avatarInitial) {
        avatarInitial.style.display = "none";
      }
    };
  } catch (e) {
    // no-op
  }
})();
