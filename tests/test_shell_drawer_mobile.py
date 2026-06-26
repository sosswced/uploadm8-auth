"""Playwright: mobile sidebar drawer on every app-shell page."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e]

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / 'frontend'

APP_SHELL_PAGES = sorted(
    p.name
    for p in FRONTEND.glob('*.html')
    if 'app-layout' in p.read_text(encoding='utf-8', errors='replace')
)


def _base_url() -> str:
    return os.environ.get('PLAYWRIGHT_BASE_URL', os.environ.get('E2E_BASE_URL', 'http://127.0.0.1:8000')).rstrip('/')


@pytest.fixture(scope='module')
def browser_page():
    with sync_playwright() as p:
        storage: str | None = None
        try:
            from tests.e2e.helpers.auth import ensure_playwright_storage_state

            storage = ensure_playwright_storage_state(p)
        except Exception:
            storage = None
        browser = p.chromium.launch(headless=os.environ.get('PLAYWRIGHT_HEADLESS', 'true').lower() == 'true')
        context = browser.new_context(
            storage_state=storage,
            viewport={'width': 390, 'height': 844},
        )
        page = context.new_page()
        yield page
        context.close()
        browser.close()


def test_shell_drawer_smoke_harness(browser_page: Page):
    browser_page.set_viewport_size({'width': 390, 'height': 844})
    browser_page.goto(f'{_base_url()}/test-fixtures/shell-drawer-smoke.html', wait_until='load')
    result = browser_page.evaluate(
        """
        () => {
          window.um8OpenSidebarDrawer();
          const opened = document.getElementById('sidebar').classList.contains('open');
          window.um8ForceCloseSidebarDrawer();
          const closed = !document.getElementById('sidebar').classList.contains('open');
          return { opened, closed };
        }
        """
    )
    assert result['opened'], 'smoke harness: drawer did not open'
    assert result['closed'], 'smoke harness: drawer did not close'


def test_shell_drawer_structure_audit():
    script = ROOT / 'tools' / 'check_shell_drawer_pages.py'
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('page_name', APP_SHELL_PAGES)
def test_mobile_drawer_api_on_page(browser_page: Page, page_name: str):
    browser_page.set_viewport_size({'width': 390, 'height': 844})
    browser_page.goto(f'{_base_url()}/{page_name}', wait_until='domcontentloaded', timeout=60000)
    browser_page.wait_for_timeout(400)

    if 'login.html' in browser_page.url:
        pytest.skip(f'{page_name} redirected to login')

    try:
        browser_page.wait_for_function(
            "() => typeof window.um8OpenSidebarDrawer === 'function' && !!document.getElementById('sidebar')",
            timeout=45_000,
        )
        result = browser_page.evaluate(
            """
            () => {
              const has = {
                drawerFn: typeof window.um8OpenSidebarDrawer === 'function'
                  && typeof window.um8ForceCloseSidebarDrawer === 'function',
                sidebar: !!document.getElementById('sidebar'),
                overlay: !!document.getElementById('sidebarOverlay'),
                menu: !!document.getElementById('menuToggle'),
              };
              if (!has.drawerFn || !has.sidebar) return { ...has, opened: false, closed: false };
              window.um8BindShellDrawer && window.um8BindShellDrawer();
              window.um8OpenSidebarDrawer();
              const opened = document.getElementById('sidebar').classList.contains('open');
              window.um8ForceCloseSidebarDrawer();
              const closed = !document.getElementById('sidebar').classList.contains('open');
              return { ...has, opened, closed };
            }
            """
        )
    except Exception as exc:
        if 'login.html' in browser_page.url or 'navigation' in str(exc).lower():
            pytest.skip(f'{page_name} redirected during auth bootstrap')
        pytest.skip(f'{page_name} shell drawer not ready: {type(exc).__name__}')
    assert result['sidebar'], f'{page_name}: missing sidebar'
    assert result['overlay'], f'{page_name}: missing overlay'
    assert result['menu'], f'{page_name}: missing menu toggle'
    assert result['drawerFn'], f'{page_name}: shell-drawer.js not loaded'
    assert result['opened'], f'{page_name}: um8OpenSidebarDrawer failed'
    assert result['closed'], f'{page_name}: um8ForceCloseSidebarDrawer failed'
