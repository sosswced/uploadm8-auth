/**
 * UploadM8 Shared Sidebar Component
 * Include this on every dashboard page for consistent navigation
 * Properly reads user role from /api/me to show/hide admin section
 */

const SIDEBAR_HTML = `
<aside class="sidebar" id="sidebar">
    <div class="sidebar-header">
        <img src="images/logo.png" alt="UploadM8" class="sidebar-logo">
        <span class="sidebar-title">Upload<span style="color: #f97316;">M8</span></span>
    </div>
    
    <nav class="sidebar-nav">
        <div class="nav-section">
            <span class="nav-section-title">MAIN</span>
            <a href="dashboard.html" class="nav-link" data-page="dashboard">
                <i class="fas fa-home"></i>
                <span>Dashboard</span>
            </a>
            <a href="upload.html" class="nav-link" data-page="upload">
                <i class="fas fa-cloud-upload-alt"></i>
                <span>Upload</span>
            </a>
            <a href="queue.html" class="nav-link" data-page="queue">
                <i class="fas fa-list"></i>
                <span>Queue</span>
            </a>
            <a href="scheduled.html" class="nav-link" data-page="scheduled">
                <i class="fas fa-calendar"></i>
                <span>Scheduled</span>
            </a>
        </div>
        
        <div class="nav-section">
            <span class="nav-section-title">PLATFORMS</span>
            <a href="platforms.html" class="nav-link" data-page="platforms">
                <i class="fas fa-plug"></i>
                <span>Connected Accounts</span>
            </a>
            <a href="groups.html" class="nav-link" data-page="groups">
                <i class="fas fa-layer-group"></i>
                <span>Account Groups</span>
            </a>
        </div>
        
        <div class="nav-section">
            <span class="nav-section-title">INSIGHTS</span>
            <a href="analytics.html" class="nav-link" data-page="analytics">
                <i class="fas fa-chart-line"></i>
                <span>Analytics</span>
            </a>
        </div>
        
        <div class="nav-section nav-admin-section" id="adminSection" style="display: none;">
            <span class="nav-section-title">ADMIN</span>
            <a href="admin.html" class="nav-link" data-page="admin">
                <i class="fas fa-shield-alt"></i>
                <span>Admin Panel</span>
            </a>
            <a href="account-management.html" class="nav-link" data-page="account-management">
                <i class="fas fa-users-cog"></i>
                <span>Account Mgmt</span>
            </a>
            <a href="admin-kpi.html" class="nav-link" data-page="admin-kpi">
                <i class="fas fa-chart-pie"></i>
                <span>KPI Dashboard</span>
            </a>
        </div>
        
        <div class="nav-section">
            <span class="nav-section-title">ACCOUNT</span>
            <a href="settings.html" class="nav-link" data-page="settings">
                <i class="fas fa-cog"></i>
                <span>Settings</span>
            </a>
        </div>
    </nav>
    
    <div class="sidebar-footer">
        <div class="user-info" id="sidebarUserInfo">
            <div class="user-avatar" id="sidebarAvatar">U</div>
            <div class="user-details">
                <div class="user-name" id="sidebarUserName">User</div>
                <div class="user-tier" id="sidebarUserTier">Free</div>
            </div>
            <button class="logout-btn" onclick="handleLogout()" title="Logout">
                <i class="fas fa-sign-out-alt"></i>
            </button>
        </div>
    </div>
</aside>
`;


// ------------------------------
// Avatar helpers (non-breaking)
// ------------------------------
function _resolveAvatarUrl(user) {
  if (!user) return null;
  return user.avatarSignedUrl || user.avatar_signed_url || user.avatarUrl || user.avatar_url || null;
}
function _cacheBustFragment(url) {
  if (!url) return url;
  return String(url).split("#")[0] + "#v=" + Date.now();
}
function _applyAvatarToSidebar(user) {
  const sidebarAvatar = document.getElementById("sidebarAvatar");
  if (!sidebarAvatar) return;
  const url = _resolveAvatarUrl(user);
  if (!url) return;
  const busted = _cacheBustFragment(url);
  // Replace letter badge with an <img> but keep same container/design
  sidebarAvatar.innerHTML = `<img src="${busted}" alt="Avatar" style="width:100%;height:100%;border-radius:50%;object-fit:cover;display:block;" />`;
}

function initSharedSidebar() {
    // Find sidebar container or create one
    let container = document.getElementById('sidebarContainer');
    if (!container) {
        container = document.querySelector('.sidebar');
        if (container) {
            container.outerHTML = SIDEBAR_HTML;
        }
    } else {
        container.innerHTML = SIDEBAR_HTML;
    }
    
    // Highlight current page
    const currentPage = window.location.pathname.split('/').pop().replace('.html', '') || 'dashboard';
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.dataset.page === currentPage) {
            link.classList.add('active');
        }
    });
    
    // Show admin section if user is admin (check role from database)
    // This checks window.currentUser which should be set by /api/me
    const userRole = window.currentUser?.role;
    const isAdmin = userRole === 'admin' || userRole === 'master_admin';
    
    if (isAdmin) {
        const adminSection = document.getElementById('adminSection');
        if (adminSection) {
            adminSection.style.display = 'block';
            console.log('Admin section shown for role:', userRole);
        }
    }
    
    // Update user info in sidebar
    if (window.currentUser) {
        const nameEl = document.getElementById('sidebarUserName');
        const tierEl = document.getElementById('sidebarUserTier');
        const avatarEl = document.getElementById('sidebarAvatar');
        
        if (nameEl) nameEl.textContent = window.currentUser.name || 'User';
        if (tierEl) {
            const u = window.currentUser;
            tierEl.textContent = u?.tier_display || (typeof getTierDisplayName === 'function' ? getTierDisplayName(u) : (u?.tier || u?.subscription_tier || 'Free'));
        }
        if (avatarEl) {
            if (typeof window.applyAvatarToUI === 'function') {
                window._applyAvatarToSidebar(window.currentUser);
            } else {
                avatarEl.textContent = (window.currentUser.name || window.currentUser.email || 'U').charAt(0).toUpperCase();
            }
        }
    }
}

// Handle logout
function handleLogout() {
    if (typeof logout === 'function') {
        logout();
    } else {
        // Fallback logout
        localStorage.removeItem('uploadm8_access_token');
        localStorage.removeItem('uploadm8_refresh_token');
        sessionStorage.removeItem('uploadm8_access_token');
        window.location.href = 'login.html';
    }
}

// Auto-init when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSharedSidebar);
} else {
    initSharedSidebar();
}

// Re-init after auth check
window.addEventListener('userLoaded', initSharedSidebar);
