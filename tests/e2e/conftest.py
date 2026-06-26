"""Playwright + API fixtures for overnight E2E."""



from __future__ import annotations



import os



import httpx

import pytest



from tests.e2e.helpers.auth import (

    E2EAuthError,

    api_client,

    ensure_playwright_storage_state,

    fetch_me,

)

from tests.e2e.helpers.browser_session import SingleBrowserSession, bootstrap_human_session

from tests.e2e.helpers.config import (

    ROOT,

    e2e_base_url,

    e2e_page_timeout_ms,

    e2e_test_telemetry_map,

    e2e_test_video,

)

from tests.e2e.helpers.human_pace import CHROME_UA, slow_mo_ms



# Load project .env when tests run directly (run_tests.py also loads it).

try:

    from dotenv import load_dotenv



    load_dotenv(ROOT / ".env")

except Exception:

    pass





def pytest_configure(config):

    config.addinivalue_line("markers", "e2e: browser or live-API end-to-end test")

    config.addinivalue_line("markers", "api_smoke: OpenAPI GET sweep against running API")

    config.addinivalue_line("markers", "ui_smoke: static page load + navigation")

    config.addinivalue_line("markers", "ui_clicks: in-page tab/toggle click sweep")

    config.addinivalue_line("markers", "upload_ui: full browser upload (slow)")

    config.addinivalue_line("markers", "overnight: included in default overnight suite")





def _attach_page_monitors(page):

    page.set_default_timeout(e2e_page_timeout_ms())

    console_errors: list[str] = []

    failed_requests: list[str] = []



    def on_console(msg):

        if msg.type == "error":

            text = msg.text or ""

            if "favicon" in text.lower():

                return

            console_errors.append(text)



    def on_response(resp):

        if resp.status >= 500 and "/api/" in resp.url:

            failed_requests.append(f"{resp.status} {resp.url}")



    page.on("console", on_console)

    page.on("response", on_response)

    page._e2e_console_errors = console_errors  # type: ignore[attr-defined]

    page._e2e_failed_requests = failed_requests  # type: ignore[attr-defined]

    return page


from tests.e2e.helpers.page_monitors import reset_page_monitors, settle_page_monitors





@pytest.fixture(scope="session")

def base_url() -> str:

    return e2e_base_url()





@pytest.fixture(scope="session")

def api_session() -> httpx.Client:

    try:

        client = api_client()

    except E2EAuthError as e:

        pytest.skip(str(e))

    try:

        r = client.get("/api/auth/session-probe")

        if r.status_code >= 500:

            pytest.skip(f"API unreachable at {e2e_base_url()}: {r.status_code}")

    except httpx.HTTPError as e:

        pytest.skip(f"API unreachable at {e2e_base_url()}: {e}")

    yield client

    client.close()





@pytest.fixture(scope="session")

def master_user(api_session: httpx.Client) -> dict:

    return fetch_me(api_session)





@pytest.fixture(scope="session")

def playwright_browser_type_launch_args():

    headed = os.environ.get("E2E_HEADED", "").lower() in ("1", "true", "yes")

    args: dict = {"headless": not headed}

    smo = slow_mo_ms()

    if smo > 0:

        args["slow_mo"] = smo

    return args





@pytest.fixture(scope="session")

def e2e_storage_state(playwright) -> str:

    try:

        return ensure_playwright_storage_state(playwright)

    except E2EAuthError as e:

        pytest.skip(str(e))





@pytest.fixture(scope="session")

def browser_context_args(e2e_storage_state: str, base_url: str):

    return {

        "storage_state": e2e_storage_state,

        "base_url": base_url,

        "viewport": {"width": 1440, "height": 900},

        "ignore_https_errors": True,

        "user_agent": CHROME_UA,

    }





@pytest.fixture(scope="session")

def authenticated_context(browser, browser_context_args):

    """One logged-in browser context for the whole UI session (cookie-primary)."""

    ctx = browser.new_context(**browser_context_args)

    yield ctx

    ctx.close()





@pytest.fixture(scope="session")

def human_session_page(authenticated_context, base_url: str):

    """

    Single browser tab for all authenticated UI tests — login once, stay logged in.

    Re-logs via login.html when navigate_to_page_human detects session expiry.

    """

    page = authenticated_context.new_page()

    _attach_page_monitors(page)

    force_form = os.environ.get("E2E_FORCE_RELOGIN", "").lower() in ("1", "true", "yes")

    bootstrap_human_session(page, base_url, force_form_login=force_form)

    yield page





@pytest.fixture(scope="session")
def live_demo_page(base_url: str):
    """
    Dedicated single-window session for test_live_demo_journey only.

    Does not share pytest-playwright's browser or storage-state preflight.
    """
    headed = os.environ.get("E2E_HEADED", "").lower() in ("1", "true", "yes")
    session = SingleBrowserSession(headed=headed)
    page = session.start()
    _attach_page_monitors(page)
    yield page
    session.stop()


@pytest.fixture(scope="session")

def public_context(browser, base_url: str):

    ctx = browser.new_context(

        base_url=base_url,

        viewport={"width": 1440, "height": 900},

        ignore_https_errors=True,

        user_agent=CHROME_UA,

    )

    yield ctx

    ctx.close()





@pytest.fixture

def public_page(public_context):

    page = public_context.new_page()

    _attach_page_monitors(page)

    yield page





@pytest.fixture

def authed_page(human_session_page):

    """Alias — reuse the shared human session tab."""

    yield human_session_page





@pytest.fixture(scope="session")

def test_video_path():

    return e2e_test_video()





@pytest.fixture(scope="session")

def test_telemetry_map_path():

    return e2e_test_telemetry_map()





def pytest_report_header(config):

    return f"E2E base URL: {e2e_base_url()}"


