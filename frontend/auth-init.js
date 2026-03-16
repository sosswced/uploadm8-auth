/**
 * UploadM8 Auth Initialization Script
 * Include this script on EVERY authenticated page AFTER app.js
 * This ensures /api/me is called and admin sections are properly shown
 */

(function() {
    'use strict';
    
    const API_BASE = window.API_BASE || (typeof location !== 'undefined' && /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(location.origin) ? 'http://127.0.0.1:8000' : 'https://auth.uploadm8.com');
    const TOKEN_KEY = 'uploadm8_access_token';
    const REFRESH_KEY = 'uploadm8_refresh_token';
    
    // Get token
    function getToken() {
        return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY);
    }
    
    // Clear tokens and redirect to login
    function clearAndRedirect(message) {
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
        if (message) {
            sessionStorage.setItem('uploadm8_auth_message', message);
        }
        window.location.href = 'login.html';
    }
    
    // Format tier name for display
    function formatTier(tier) {
        const tiers = {
            'free': 'Free',
            'launch': 'Launch',
            'creator_pro': 'Creator Pro',
            'studio': 'Studio',
            'agency': 'Agency',
            'master_admin': 'Master Admin',
            'friends_family': 'Friends & Family',
            'lifetime': 'Lifetime'
        };
        return tiers[tier] || tier || 'Free';
    }
    
    // Update sidebar with user info and show admin section if needed
    function updateSidebarWithUser(user) {
        if (!user) return;
        
        // Update user info in sidebar
        const nameEl = document.getElementById('sidebarUserName');
        const tierEl = document.getElementById('sidebarUserTier');
        const avatarEl = document.getElementById('sidebarAvatar');
        
        if (nameEl) nameEl.textContent = user.name || user.email || 'User';
        if (tierEl) tierEl.textContent = formatTier(user.subscription_tier);
        if (avatarEl) {
            if (typeof window.applyAvatarToUI === 'function') {
                window.applyAvatarToUI(user);
            } else {
                avatarEl.textContent = (user.name || user.email || 'U').charAt(0).toUpperCase();
            }
        }
        
        // Check if user is admin based on role from database
        const isAdmin = user.role === 'admin' || user.role === 'master_admin';
        
        console.log('[AuthInit] User:', user.name, '| Role:', user.role, '| isAdmin:', isAdmin);
        
        // Show admin section - try multiple selectors for compatibility
        if (isAdmin) {
            // Method 1: By ID
            const adminSectionById = document.getElementById('adminSection');
            if (adminSectionById) {
                adminSectionById.style.display = 'block';
                adminSectionById.classList.remove('hidden');
            }
            
            // Method 2: By class
            document.querySelectorAll('.admin-only, .admin-section, .nav-admin-section').forEach(el => {
                el.style.display = 'block';
                el.classList.remove('hidden');
            });
            
            console.log('[AuthInit] Admin section enabled');
        }
    }
    
    // Main initialization function
    async function initAuth() {
        const token = getToken();
        
        // Check if we're on a public page that doesn't need auth
        const publicPages = ['index.html', 'login.html', 'signup.html', 'forgot-password.html', 'terms.html', 'privacy.html', 'about.html', 'contact.html', 'support.html', 'blog.html', 'how-it-works.html', 'data-deletion.html'];
        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        
        if (publicPages.includes(currentPage)) {
            console.log('[AuthInit] Public page, skipping auth check');
            return null;
        }
        
        // No token - redirect to login
        if (!token) {
            console.log('[AuthInit] No token found, redirecting to login');
            clearAndRedirect('Please log in to continue.');
            return null;
        }
        
        try {
            // Fetch user from /api/me
            console.log('[AuthInit] Fetching /api/me...');
            const response = await fetch(`${API_BASE}/api/me`, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.status === 401) {
                console.log('[AuthInit] Token expired (401), redirecting to login');
                clearAndRedirect('Session expired. Please log in again.');
                return null;
            }
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const user = await response.json();
            
            // Store user globally
            window.currentUser = user;
            
            // Update sidebar
            updateSidebarWithUser(user);
            
            // Dispatch event for other scripts to listen
            window.dispatchEvent(new CustomEvent('userLoaded', { detail: user }));
            
            console.log('[AuthInit] User loaded successfully:', user.email);
            return user;
            
        } catch (error) {
            console.error('[AuthInit] Error fetching user:', error);
            // Don't redirect on network errors - just log
            return null;
        }
    }
    
    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAuth);
    } else {
        // Small delay to ensure app.js has loaded
        setTimeout(initAuth, 50);
    }
    
    // Export for manual use
    window.initAuth = initAuth;
    window.updateSidebarWithUser = updateSidebarWithUser;
    
})();