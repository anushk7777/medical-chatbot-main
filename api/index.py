from deployment_api import app

def api_stripper_asgi(app):
    async def asgi_app(scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/api"):
            scope = dict(scope)
            scope["path"] = scope["path"][4:] or "/"
        return await app(scope, receive, send)
    return asgi_app

app = api_stripper_asgi(app)
