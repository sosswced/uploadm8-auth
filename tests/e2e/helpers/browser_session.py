"""Human-like Playwright session: login once, cookie-primary auth, sidebar navigation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from tests.e2e.helpers.auth import api_login_with_cookies, require_master_credentials
from tests.e2e.helpers.human_pace import CHROME_UA, click_delay_ms, pause_between_requests, slow_mo_ms
from tests.e2e.helpers.pages import page_url, page_url_matches, canonical_page_path, NO_APP_SHELL_PAGES

from tests.e2e.helpers.page_monitors import reset_page_monitors, settle_page_monitors
T = TypeVar("T")

_SINGLE_TAB_INIT_SCRIPT = """
(() => {
  window.open = function(url) {
    if (url) window.location.assign(url);
    return null;
  };
  document.addEventListener('click', (ev) => {
    const a = ev.target && ev.target.closest ? ev.target.closest('a[target="_blank"]') : null;
    if (!a || !a.href) return;
    if (a.href.startsWith('http') || a.href.endsWith('.html') || a.getAttribute('href')?.startsWith('/')) {
      ev.preventDefault();
      window.location.assign(a.href);
    }
  }, true);
})();
"""


def goto_page_resilient(page: Page, url: str, *, timeout_ms: int = 90_000) -> None:
    """Navigate with retry — slow admin pages may miss domcontentloaded under API load."""
    last: Exception | None = None
    for wait_until in ("domcontentloaded", "commit"):
        try:
            page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            return
        except Exception as e:
            last = e
            page.wait_for_timeout(600)
    if last is not None:
        raise last


def wait_for_authenticated_shell(page: Page, *, timeout_ms: int = 90_000) -> None:
    """Wait until app shell has validated /api/me and painted chrome."""
    if "login.html" in (page.url or ""):
        raise AssertionError("wait_for_authenticated_shell: still on login.html")
    page.wait_for_function(
        """() => {
            if (location.href.includes('login.html')) return false;
            const root = document.documentElement;
            if (root.classList.contains('um8-shell-ready') || root.classList.contains('um8-user-ready')) {
                return true;
            }
            // Settings page often paints tabs before sidebar tier hydrate.
            if (document.querySelector('a.settings-tab')) return true;
            const tier = (document.getElementById('userTier')?.textContent || '').trim().toLowerCase();
            if (/admin|master|starter|creator|studio|agency|lite|pro|friends|family|free|launch/.test(tier)) {
                return true;
            }
            // Sidebar chrome present + not on login is enough under slow hydrate.
            const name = (document.getElementById('userName')?.textContent || '').trim();
            const nav = document.querySelector('nav.sidebar-nav');
            return Boolean(nav && name && name.toLowerCase() !== 'user');
        }""",
        timeout=timeout_ms,
    )


def ensure_authed(page: Page, base_url: str) -> None:
    """Re-login via login.html when the tab landed on an auth page."""
    if "login.html" in page.url:
        human_login_via_form(page, base_url)


def assert_not_on_login(page: Page, *, context: str = "") -> None:
    if "login.html" in page.url:
        suffix = f" ({context})" if context else ""
        raise AssertionError(f"Session expired — redirected to login{suffix}")


def human_login_via_form(page: Page, base_url: str) -> None:
    """Fill login.html and submit — same path a user takes (HttpOnly cookies, no bearer)."""
    email, password = require_master_credentials()
    page.goto(page_url(base_url, "login.html"), wait_until="domcontentloaded")
    pause_between_requests()
    page.locator("#email").fill(email)
    page.locator("#password").fill(password)
    page.locator("#loginForm button[type=submit]").click()
    page.wait_for_url("**/dashboard.html**", timeout=90_000)
    wait_for_authenticated_shell(page)
    page.wait_for_timeout(click_delay_ms())


def bootstrap_human_session(page: Page, base_url: str, *, force_form_login: bool = False) -> None:
    """
    Establish one authenticated session in this tab.

    Prefer the real login form when cookies are missing or stale; otherwise reuse
    cookies from storage state and only confirm the shell on dashboard.
    """
    if force_form_login:
        human_login_via_form(page, base_url)
        return

    goto_page_resilient(page, page_url(base_url, "dashboard.html"))
    pause_between_requests()

    if "login.html" in page.url:
        human_login_via_form(page, base_url)
        return

    try:
        wait_for_authenticated_shell(page, timeout_ms=60_000)
    except Exception:
        human_login_via_form(page, base_url)


def navigate_to_page_human(page: Page, base_url: str, rel_path: str) -> None:
    """
    Navigate like a user: sidebar click when the link exists, otherwise direct URL.

    Always waits for authenticated shell before returning.
    """
    reset_page_monitors(page)

    target = rel_path.split("/")[-1]
    rel_norm = rel_path if rel_path.endswith(".html") else f"{rel_path}.html"
    canonical = canonical_page_path(rel_norm)

    if target == "dashboard.html":
        ensure_authed(page, base_url)
        goto_page_resilient(page, page_url(base_url, "dashboard.html"))
        if "login.html" in page.url:
            human_login_via_form(page, base_url)
        wait_for_authenticated_shell(page)
        page.wait_for_timeout(click_delay_ms())
        settle_page_monitors(page)
        return

    ensure_authed(page, base_url)

    if page.locator("nav.sidebar-nav").count() == 0:
        goto_page_resilient(page, page_url(base_url, "dashboard.html"))
        wait_for_authenticated_shell(page)

    link = page.locator(
        f'nav.sidebar-nav a.nav-link[href="{rel_norm}"], '
        f'nav.sidebar-nav a.nav-link[href="{rel_path}"], '
        f'nav.sidebar-nav a.nav-link[href$="{target}"]'
    )
    if link.count() > 0 and link.first.is_visible():
        try:
            link.first.click(no_wait_after=True)
            page.wait_for_url(f"**/{canonical}**", timeout=90_000)
        except Exception:
            goto_page_resilient(page, page_url(base_url, rel_path))
        page.wait_for_load_state("domcontentloaded")
        pause_between_requests()
    else:
        goto_page_resilient(page, page_url(base_url, rel_path))
        pause_between_requests()

    if "login.html" in page.url:
        human_login_via_form(page, base_url)
        goto_page_resilient(page, page_url(base_url, rel_path))
        pause_between_requests()
    if rel_norm in NO_APP_SHELL_PAGES or target in NO_APP_SHELL_PAGES:
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(click_delay_ms())
        settle_page_monitors(page)
        return
    if target == "settings.html" or rel_norm == "settings.html":
        # Settings tabs are static HTML; don't block on sidebar tier hydrate.
        page.locator("a.settings-tab").first.wait_for(state="attached", timeout=90_000)
        page.wait_for_timeout(click_delay_ms())
        settle_page_monitors(page)
        return
    wait_for_authenticated_shell(page)
    page.wait_for_timeout(click_delay_ms())
    settle_page_monitors(page)


def human_scroll(page: Page, *, passes: int = 3) -> None:
    """Scroll like a user reading the page."""
    for i in range(passes):
        page.mouse.wheel(0, 420 + (i * 80))
        page.wait_for_timeout(click_delay_ms())
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(click_delay_ms() // 2)


class SingleBrowserSession:
    """
    Exactly one Chromium window + one tab for the whole live demo.

    Closes accidental popups and forces same-tab navigation.
    """

    def __init__(self, *, headed: bool = True, timeout_ms: int = 120_000) -> None:
        self.headed = headed
        self.timeout_ms = timeout_ms
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None

    def start(self) -> Page:
        if self.page is not None:
            return self.page
        self._pw = sync_playwright().start()
        smo = slow_mo_ms() if self.headed else 0
        self._browser = self._pw.chromium.launch(
            headless=not self.headed,
            slow_mo=smo,
            args=["--disable-popup-blocking"],
        )
        self._context = self._browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=CHROME_UA,
            ignore_https_errors=True,
        )
        # HttpOnly cookies from API login — one session for the whole demo (no per-page re-login).
        try:
            _tokens, cookie_jar = api_login_with_cookies()
            if cookie_jar:
                self._context.add_cookies(cookie_jar)
        except Exception:
            pass
        self._context.add_init_script(_SINGLE_TAB_INIT_SCRIPT)
        main_holder: list[Page] = []

        def _on_page(new_page: Page) -> None:
            if not main_holder or new_page == main_holder[0]:
                return
            try:
                url = new_page.url
                new_page.close()
                if url and url not in ("about:blank", "") and main_holder[0]:
                    main_holder[0].goto(url, wait_until="domcontentloaded")
            except Exception:
                try:
                    new_page.close()
                except Exception:
                    pass

        self._context.on("page", _on_page)
        self.page = self._context.new_page()
        main_holder.append(self.page)
        self.page.set_default_timeout(self.timeout_ms)
        self._attach_human_guards(self.page)
        return self.page

    @staticmethod
    def _attach_human_guards(page: Page) -> None:
        """Keep one tab: dismiss alert/confirm/prompt so the tour never blocks on popups."""

        def _on_dialog(dialog) -> None:
            try:
                dialog.dismiss()
            except Exception:
                try:
                    dialog.accept()
                except Exception:
                    pass

        page.on("dialog", _on_dialog)

    def stop(self) -> None:
        try:
            if self._context:
                self._context.close()
        finally:
            try:
                if self._browser:
                    self._browser.close()
            finally:
                if self._pw:
                    self._pw.stop()
        self.page = None
        self._context = None
        self._browser = None
        self._pw = None

    def __enter__(self) -> Page:
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


@contextmanager
def single_browser_session(*, headed: bool = True, timeout_ms: int = 120_000) -> Iterator[Page]:
    """Context manager: one window, one tab, for the full live demo."""
    session = SingleBrowserSession(headed=headed, timeout_ms=timeout_ms)
    try:
        yield session.start()
    finally:
        session.stop()


def run_in_single_browser(fn: Callable[[Page], T], *, headed: bool = True, timeout_ms: int = 120_000) -> T:
    """Run ``fn(page)`` in a single headed Chromium window."""
    with single_browser_session(headed=headed, timeout_ms=timeout_ms) as page:
        return fn(page)
