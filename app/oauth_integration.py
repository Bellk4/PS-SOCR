import html
import logging
from importlib import import_module
import secrets
from pathlib import Path
from typing import Any, Optional, cast

from auth0_server_python.auth_types import LogoutOptions
from auth0_server_python.error import MissingTransactionError
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

OAUTH_BASE_PATH = "/oauth"
OAUTH_COMPAT_PATHS = {"/login", "/callback", "/logout", "/profile"}
LOCAL_AUTH_SESSION_COOKIE = "ps_socr_auth"
_ACTIVE_AUTH_SESSIONS: set[str] = set()


def is_oauth_public_path(path: str) -> bool:
    normalized = (path or "/").rstrip("/") or "/"
    return normalized.startswith(OAUTH_BASE_PATH) or normalized in OAUTH_COMPAT_PATHS


def has_local_oauth_session(request: Request) -> bool:
    token = request.cookies.get(LOCAL_AUTH_SESSION_COOKIE)
    return bool(token and token in _ACTIVE_AUTH_SESSIONS)


def issue_local_oauth_session(request: Request, response: Response) -> None:
    token = secrets.token_urlsafe(32)
    _ACTIVE_AUTH_SESSIONS.add(token)
    response.set_cookie(
        key=LOCAL_AUTH_SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )


def clear_local_oauth_session(request: Request, response: Response) -> None:
    token = request.cookies.get(LOCAL_AUTH_SESSION_COOKIE)
    if token:
        _ACTIVE_AUTH_SESSIONS.discard(token)
    response.delete_cookie(LOCAL_AUTH_SESSION_COOKIE, path="/")


OAUTH_INLINE_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=M+PLUS+1p:wght@400;500;700&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'M PLUS 1p', 'Hiragino Kaku Gothic ProN', sans-serif; background: #f5f5f5; min-height: 100vh; display: flex; flex-direction: column; color: #333333; }
.page-header { background: #4a90e2; padding: 8px 24px; display: flex; align-items: center; gap: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.page-header-title { color: #ffffff; font-size: 20px; font-weight: 700; letter-spacing: 0.03em; }
.page-body { flex: 1; display: flex; justify-content: center; align-items: center; padding: 40px 20px; }
.container { width: 100%; max-width: 480px; }
.card { background: #ffffff; border: 1px solid #d0d0d0; border-radius: 8px; padding: 2.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
h2 { font-size: 1.4rem; font-weight: 700; margin: 0 0 1.5rem; color: #333333; text-align: center; }
h3 { font-size: 1rem; font-weight: 700; margin: 1.5rem 0 0.75rem; color: #555555; }
.logged-in, .logged-out, .profile-details { text-align: center; }
.logged-out p { font-size: 1rem; color: #666666; margin-bottom: 1.5rem; line-height: 1.7; }
.success { font-size: 1rem; color: #2e7d32; font-weight: 700; margin-bottom: 1rem; padding: 8px 12px; background: #e8f5e9; border-radius: 4px; }
.user-info, .profile-info { background: #f9f9f9; border: 1px solid #d0d0d0; border-radius: 6px; padding: 1.25rem; margin: 1rem 0 0; }
.profile-pic, .profile-pic-large { border-radius: 50%; object-fit: cover; }
.profile-pic { width: 64px; height: 64px; margin-bottom: 0.75rem; border: 2px solid #4a90e2; }
.profile-pic-large { width: 96px; height: 96px; margin-bottom: 0.75rem; border: 3px solid #4a90e2; }
.user-info p, .email { color: #666666; font-size: 0.95rem; }
.user-name { font-size: 1.1rem; font-weight: 700; color: #333333; margin: 4px 0; }
.button { display: inline-block; padding: 0.6rem 1.8rem; font-size: 0.95rem; font-weight: 700; border-radius: 4px; text-decoration: none; margin: 0.4rem; background: #4a90e2; color: #ffffff; border: none; cursor: pointer; }
.button:hover { background: #357abd; }
.button.secondary { background: #ffffff; color: #4a90e2; border: 1px solid #4a90e2; }
.button.secondary:hover { background: #eaf1fb; }
.button.logout { background: #ffffff; color: #c0392b; border: 1px solid #e74c3c; }
.button.logout:hover { background: #fdf2f2; }
.profile-data { margin-top: 1rem; text-align: left; }
.profile-data dl { display: grid; grid-template-columns: 130px 1fr; gap: 0.6rem 1rem; margin-top: 0.5rem; font-size: 0.9rem; }
.profile-data dt { font-weight: 700; color: #555555; }
.profile-data dd { color: #666666; word-break: break-all; }
.actions { margin-top: 1.5rem; display: flex; justify-content: center; flex-wrap: wrap; gap: 0.25rem; }
@media (max-width: 520px) { .card { padding: 1.5rem; } .profile-data dl { grid-template-columns: 1fr; } .profile-data dt { margin-top: 0.5rem; } }
"""


def load_oauth_client(root_dir: Path, logger: logging.Logger) -> Optional[Any]:
    auth_module_path = root_dir / "app" / "auth.py"
    if not auth_module_path.exists():
        logger.info("OAuth設定が見つからないため、統合をスキップします: %s", auth_module_path)
        return None

    try:
        module = import_module("app.auth")
        oauth_client = getattr(module, "auth0", None)
        if oauth_client is None:
            logger.warning("OAuthクライアント auth0 が見つかりません")
            return None
        logger.info("OAuthクライアントを統合しました: %s", auth_module_path)
        return oauth_client
    except Exception as exc:
        logger.warning("OAuthクライアントの統合をスキップします: %s", exc)
        return None


def render_oauth_page(title: str, body: str) -> HTMLResponse:
    document = f"""<!DOCTYPE html>
<html lang=\"ja\">
<head>
<meta charset=\"UTF-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
<title>{html.escape(title)}</title>
<style>{OAUTH_INLINE_STYLE}</style>
</head>
<body>
<header class=\"page-header\">
    <span class=\"page-header-title\">PS SOCR</span>
</header>
<div class=\"page-body\">
<div class=\"container\">
    <div class=\"card\">
        {body}
    </div>
</div>
</div>
</body>
</html>"""
    return HTMLResponse(document)


def render_oauth_home(user: Optional[dict[str, Any]]) -> HTMLResponse:
    if user:
        picture_html = ""
        if user.get("picture"):
            picture_html = (
                f'<img src="{html.escape(str(user.get("picture")))}" '
                'alt="Profile" class="profile-pic">'
            )
        body = f"""
        <div class=\"logged-in\">
            <p class=\"success\">ログインしました</p>
            <div class=\"user-info\">
                {picture_html}
                <p class=\"user-name\">{html.escape(str(user.get("name") or "Unknown User"))}</p>
                <p>{html.escape(str(user.get("email") or ""))}</p>
            </div>
            <div class=\"actions\">
                <a href=\"/\" class=\"button\">OCRへ戻る</a>
                <a href=\"{OAUTH_BASE_PATH}/profile\" class=\"button secondary\">プロフィール</a>
                <a href=\"{OAUTH_BASE_PATH}/logout\" class=\"button logout\">ログアウト</a>
            </div>
        </div>
        """
    else:
        body = f"""
        <h2>ログイン</h2>
        <div class=\"logged-out\">
            <p>PS SOCR を使用するにはログインが必要です。</p>
            <a href=\"{OAUTH_BASE_PATH}/login\" class=\"button\">ログイン</a>
        </div>
        """
    return render_oauth_page("ログイン - PS SOCR", body)


def render_oauth_profile(user: dict[str, Any]) -> HTMLResponse:
    picture_html = ""
    if user.get("picture"):
        picture_html = (
            f'<img src="{html.escape(str(user.get("picture")))}" '
            'alt="Profile" class="profile-pic-large">'
        )
    nickname_html = ""
    if user.get("nickname"):
        nickname_html = (
            f"<dt>Nickname:</dt><dd>{html.escape(str(user.get('nickname')))}</dd>"
        )
    updated_html = ""
    if user.get("updated_at"):
        updated_html = (
            f"<dt>Last Updated:</dt><dd>{html.escape(str(user.get('updated_at')))}</dd>"
        )
    body = f"""
    <h2>プロフィール</h2>
    <div class=\"profile-details\">
        {picture_html}
        <div class=\"profile-info\">
            <p class=\"user-name\">{html.escape(str(user.get("name") or "Unknown User"))}</p>
            <p class=\"email\">{html.escape(str(user.get("email") or ""))}</p>
            <div class=\"profile-data\">
                <h3>アカウント情報</h3>
                <dl>
                    <dt>ユーザーID:</dt>
                    <dd>{html.escape(str(user.get("sub") or ""))}</dd>
                    {nickname_html}
                    {updated_html}
                </dl>
            </div>
        </div>
    </div>
    <div class=\"actions\">
        <a href=\"/\" class=\"button\">OCRへ戻る</a>
        <a href=\"{OAUTH_BASE_PATH}/logout\" class=\"button logout\">ログアウト</a>
    </div>
    """
    return render_oauth_page("プロフィール - PS SOCR", body)


def register_oauth_routes(
    app: FastAPI,
    root_dir: Path,
    logger: logging.Logger,
) -> Optional[Any]:
    oauth_client = load_oauth_client(root_dir, logger)
    router = APIRouter()

    def require_oauth_client() -> Any:
        if oauth_client is None:
            raise HTTPException(status_code=503, detail="OAuth機能は有効化されていません")
        return oauth_client

    @router.get("/login")
    async def oauth_login_compat() -> RedirectResponse:
        require_oauth_client()
        return RedirectResponse(url=f"{OAUTH_BASE_PATH}/login", status_code=307)

    async def complete_login_callback(request: Request) -> RedirectResponse:
        client = require_oauth_client()
        try:
            await client.complete_interactive_login(str(request.url), {"request": request})
        except MissingTransactionError:
            return RedirectResponse(url=f"{OAUTH_BASE_PATH}/login", status_code=307)
        response = RedirectResponse(url="/", status_code=307)
        issue_local_oauth_session(request, response)
        return response

    @router.get("/callback")
    async def oauth_callback_compat(request: Request) -> RedirectResponse:
        return await complete_login_callback(request)

    # Note: クエリを保持するのは、OAuthプロバイダからのコールバックでクエリパラメータが必要な場合があるためです。
    @router.get("/logout")
    async def oauth_logout_compat() -> RedirectResponse:
        require_oauth_client()
        return RedirectResponse(url=f"{OAUTH_BASE_PATH}/logout", status_code=307)

    # Note: /profile はOAuthクライアントの保護されたルートにリダイレクトするだけですが、
    # これもOAuth統合が有効な場合は常に存在するようにしておくと、フロントエンドでログイン状態の判定がしやすくなります。
    @router.get("/profile")
    async def oauth_profile_compat() -> RedirectResponse:
        require_oauth_client()
        return RedirectResponse(url=f"{OAUTH_BASE_PATH}/profile", status_code=307)

    @router.get(f"{OAUTH_BASE_PATH}/", response_class=HTMLResponse)
    async def oauth_home(request: Request) -> HTMLResponse:
        client = require_oauth_client()
        user: Optional[dict[str, Any]] = None
        if has_local_oauth_session(request):
            user = cast(Optional[dict[str, Any]], await client.get_user({"request": request}))
        return render_oauth_home(user)

    @router.get(f"{OAUTH_BASE_PATH}/login")
    async def oauth_login(request: Request) -> RedirectResponse:
        client = require_oauth_client()
        authorization_url = await client.start_interactive_login(store_options={"request": request})
        return RedirectResponse(url=authorization_url, status_code=307)

    @router.get(f"{OAUTH_BASE_PATH}/callback")
    async def oauth_callback(request: Request) -> RedirectResponse:
        return await complete_login_callback(request)

    @router.get(f"{OAUTH_BASE_PATH}/profile", response_class=HTMLResponse)
    async def oauth_profile(request: Request) -> Response:
        client = require_oauth_client()
        if not has_local_oauth_session(request):
            return RedirectResponse(url=f"{OAUTH_BASE_PATH}/login", status_code=307)
        user = await client.get_user({"request": request})
        if not user:
            return RedirectResponse(url=f"{OAUTH_BASE_PATH}/login", status_code=307)
        return render_oauth_profile(cast(dict[str, Any], user))

    @router.get(f"{OAUTH_BASE_PATH}/logout")
    async def oauth_logout(request: Request) -> RedirectResponse:
        client = require_oauth_client()
        base_url = str(request.base_url).rstrip("/")
        return_to = f"{base_url}{OAUTH_BASE_PATH}/login"
        logout_url = await client.logout(
            options=LogoutOptions(return_to=return_to),
            store_options={"request": request},
        )
        response = RedirectResponse(url=logout_url, status_code=307)
        clear_local_oauth_session(request, response)
        return response

    app.include_router(router)
    return oauth_client
