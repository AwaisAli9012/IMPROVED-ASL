"""FastAPI app: the JSON API, the static assets and the SvelteKit single-page app."""
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config, services
from .routes import language, learning, meta, recognition

load_dotenv(config.ROOT / ".env")
BUILD = config.WEB / "app" / "build"


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(config.TORCH_THREADS)
    services.warm_up()  # load every model once so the first request is fast
    yield


def spa(path: str):
    """Any other URL gets the SvelteKit app, which routes in the browser."""
    if path == "api" or path.startswith("api/"):
        raise HTTPException(404, "not found")
    file = (BUILD / path).resolve()
    if path and file.is_file() and BUILD.resolve() in file.parents:
        return FileResponse(file)
    index = BUILD / "index.html"
    if not index.exists():
        return JSONResponse({"error": "frontend not built: run npm run build in web/app"}, status_code=503)
    return FileResponse(index)


def create_app() -> FastAPI:
    app = FastAPI(title="SignBridge", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def limits_and_caching(request: Request, call_next):
        # shortcut: bodies sent without a content-length are not capped, add a streaming check if this is ever public
        if int(request.headers.get("content-length") or 0) > config.MAX_BODY_BYTES:
            return JSONResponse({"error": "request too large"}, status_code=413)
        response = await call_next(request)
        if request.url.path.startswith("/_app/immutable/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        err = exc.errors()[0]
        if err["loc"][0] == "path":  # /api/learn/abc: no such route, as before
            return JSONResponse({"error": "not found"}, status_code=404)
        where = ".".join(str(p) for p in err["loc"][1:]) or "body"
        return JSONResponse({"error": "expected a JSON object" if where == "body" else f"{where}: {err['msg']}"}, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(ValueError)
    async def bad_value(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing_model(request, exc):
        return JSONResponse({"error": "model not trained yet"}, status_code=503)

    for router in (meta.router, recognition.router, language.router, learning.router):
        app.include_router(router, prefix="/api")
    app.mount("/static", StaticFiles(directory=config.WEB / "static"), name="static")
    app.mount("/reports", StaticFiles(directory=config.REPORTS), name="reports")
    app.add_api_route("/{path:path}", spa, include_in_schema=False)
    return app


app = create_app()
