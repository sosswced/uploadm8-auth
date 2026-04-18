/**
 * UploadM8 Shared Sidebar Component
 *
 * Injects the canonical sidebar HTML into an empty <aside id="sidebar"> stub, then calls
 * window.__um8ApplyTrustedSessionChrome() (from session-user-hydrate.js) to paint name/tier/avatar
 * from the trusted session snapshot before app.js runs. app.js + /api/me reconcile afterward.
 *
 * Canonical load order (app-shell pages — match dashboard.html):
 *   js/api-base.js → js/auth-stack.js → js/session-user-hydrate.js → js/public-shell-hydrate.js → shared-sidebar.js
 *   → js/helpers-formatting.js → js/upload-utils.js → app.js → wallet-tokens.js
 *
 * IDs used here MUST match what app.js targets:
 *   userAvatar, userName, userTier, adminSection
 */
(function () {
    'use strict';

    var SIDEBAR_INNER = [
        '<div class="sidebar-header">',
        '    <a href="dashboard.html" class="sidebar-logo">',
        '        <img src="images/logo.svg" alt="UploadM8">',
        '        <span>UploadM8</span>',
        '    </a>',
        '</div>',
        '<nav class="sidebar-nav">',
        '    <div class="nav-section">',
        '        <div class="nav-section-title">Main</div>',
        '        <a href="dashboard.html" class="nav-link"><i class="fas fa-home"></i><span>Dashboard</span></a>',
        '        <a href="upload.html" class="nav-link"><i class="fas fa-cloud-upload-alt"></i><span>Upload</span></a>',
        '        <a href="thumbnail-studio.html" class="nav-link"><i class="fas fa-images"></i><span>Thumbnail Studio</span></a>',
        '        <a href="queue.html" class="nav-link"><i class="fas fa-list"></i><span>Queue</span></a>',
        '        <a href="scheduled.html" class="nav-link"><i class="fas fa-calendar-alt"></i><span>Scheduled</span></a>',
        '    </div>',
        '    <div class="nav-section">',
        '        <div class="nav-section-title">Platforms</div>',
        '        <a href="platforms.html" class="nav-link"><i class="fas fa-plug"></i><span>Connected Accounts</span></a>',
        '        <a href="groups.html" class="nav-link"><i class="fas fa-layer-group"></i><span>Account Groups</span></a>',
        '    </div>',
        '    <div class="nav-section">',
        '        <div class="nav-section-title">Insights</div>',
        '        <a href="analytics.html" class="nav-link"><i class="fas fa-chart-line"></i><span>Analytics</span></a>',
        '        <a href="kpi.html" class="nav-link"><i class="fas fa-chart-bar"></i><span>Upload KPIs</span></a>',
        '    </div>',
        '    <div class="nav-section admin-only" id="adminSection" style="display:none;">',
        '        <div class="nav-section-title">Admin</div>',
        '        <a href="admin.html" class="nav-link"><i class="fas fa-shield-alt"></i><span>Admin Panel</span></a>',
        '        <a href="account-management.html" class="nav-link"><i class="fas fa-users-cog"></i><span>Account Mgmt</span></a>',
        '        <a href="admin-kpi.html" class="nav-link"><i class="fas fa-chart-pie"></i><span>Admin KPIs</span></a>',
        '        <a href="admin-marketing.html" class="nav-link"><i class="fas fa-bullhorn"></i><span>Marketing Ops</span></a>',
        '        <a href="admin-calculator.html" class="nav-link"><i class="fas fa-calculator"></i><span>Biz Calculator</span></a>',
        '        <a href="admin-wallet.html" class="nav-link"><i class="fas fa-coins"></i><span>Wallet Manager</span></a>',
        '        <a href="admin-incidents.html" class="nav-link"><i class="fas fa-ambulance"></i><span>Ops incidents</span></a>',
        '    </div>',
        '    <div class="nav-section">',
        '        <div class="nav-section-title">Account</div>',
        '        <a href="settings.html" class="nav-link"><i class="fas fa-sliders-h"></i><span>Settings</span></a>',
        '        <a href="guide.html" class="nav-link"><i class="fas fa-book-open"></i><span>Feature Guide</span></a>',
        '        <a href="guide.html#feat-settings-playbook" class="nav-link"><i class="fas fa-hand-holding-heart"></i><span>Setup Handbook</span></a>',
        '        <a href="https://discord.gg/TVDAc8fnwu" target="_blank" rel="noopener" class="nav-link"><i class="fab fa-discord"></i><span>Discord community</span></a>',
        '        <a href="report-bug.html" class="nav-link"><i class="fas fa-bug"></i><span>Report a bug</span></a>',
        '    </div>',
        '</nav>',
        '<div class="sidebar-footer">',
        '    <div class="user-info">',
        '        <a href="settings.html" class="sidebar-footer-profile">',
        '        <div class="user-avatar" id="userAvatar">U</div>',
        '        <div class="user-details">',
        '            <div class="user-name" id="userName">User</div>',
        '            <div class="user-tier" id="userTier">Free</div>',
        '        </div>',
        '        </a>',
        '        <button type="button" class="btn btn-ghost btn-icon" id="sidebarLogoutBtn" data-um8-fn="logout" title="Sign Out" aria-label="Sign out">',
        '            <i class="fas fa-sign-out-alt"></i>',
        '        </button>',
        '    </div>',
        '</div>'
    ].join('\n');

    function navHrefIsActive(href) {
        if (!href || href.indexOf('http') === 0) return false;
        var path = window.location.pathname.split('/').pop() || 'dashboard.html';
        var pageHash = (window.location.hash || '').split('?')[0];
        var i = href.indexOf('#');
        var base = i >= 0 ? href.slice(0, i) : href;
        var linkHash = i >= 0 ? href.slice(i) : '';
        if (base !== path) return false;
        if (path === 'guide.html') {
            if (linkHash === '#feat-settings-playbook') return pageHash === linkHash;
            return pageHash !== '#feat-settings-playbook';
        }
        if (!linkHash) return true;
        return pageHash === linkHash;
    }

    function initSharedSidebar() {
        var sidebar = document.getElementById('sidebar');
        if (!sidebar) return;

        // Guard: already injected (phased rollout or double-call)
        if (sidebar.querySelector('.sidebar-nav')) return;

        sidebar.innerHTML = SIDEBAR_INNER;

        if (typeof window.__um8ApplyTrustedSessionChrome === 'function') {
            window.__um8ApplyTrustedSessionChrome();
        }

        // Highlight current page immediately so the user sees the correct
        // active link before app.js runs. app.js highlightCurrentNav()
        // will reinforce this later.
        sidebar.querySelectorAll('.nav-link').forEach(function (link) {
            var h = link.getAttribute('href');
            if (navHrefIsActive(h)) link.classList.add('active');
        });
    }

    // Inject as soon as this script runs. The empty <aside id="sidebar"> stub is
    // always above this script in the document, so it exists here — even when
    // document.readyState is still "loading". Waiting only for DOMContentLoaded
    // races async initApp()/checkAuth microtasks: settings.html and admin.html
    // then hit getElementById('adminSection') before the sidebar exists and throw.
    if (document.getElementById('sidebar')) {
        initSharedSidebar();
    } else if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSharedSidebar);
    } else {
        initSharedSidebar();
    }

    // Minimal fallback sidebar toggle for pages where app.js binds late/fails.
    // app.js will override this with the richer implementation when ready.
    if (typeof window.toggleSidebar !== 'function') {
        window.toggleSidebar = function () {
            var sidebar = document.getElementById('sidebar');
            var overlay = document.getElementById('sidebarOverlay');
            if (!sidebar) return;
            var open = sidebar.classList.contains('open');
            if (open) {
                sidebar.classList.remove('open');
                document.body.classList.remove('sidebar-open');
                if (overlay) overlay.classList.add('hidden');
            } else {
                sidebar.classList.add('open');
                document.body.classList.add('sidebar-open');
                if (overlay) overlay.classList.remove('hidden');
            }
        };
    }

    // Re-highlight on popstate (browser back/forward) for soft-nav support
    window.addEventListener('popstate', function () {
        document.querySelectorAll('.sidebar-nav .nav-link').forEach(function (link) {
            link.classList.toggle('active', navHrefIsActive(link.getAttribute('href')));
        });
    });
})();
