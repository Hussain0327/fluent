"""Entry point — starts the aiohttp gateway server."""

from __future__ import annotations

from aiohttp import web

from gateway.config import settings
from gateway.db.connection import close_pool, get_pool
from gateway.telephony.webhooks import (
    handle_sms_incoming,
    handle_voice_incoming,
    handle_voice_stream,
)
from gateway.utils.logging import get_logger, setup_logging


async def health_check(request: web.Request) -> web.Response:
    return web.Response(text="ok")


async def on_startup(app: web.Application) -> None:
    log = get_logger(__name__)
    await get_pool()
    log.info("gateway_started", port=settings.gateway_port)


async def on_shutdown(app: web.Application) -> None:
    log = get_logger(__name__)
    await close_pool()
    log.info("gateway_stopped")


def create_app() -> web.Application:
    setup_logging()
    app = web.Application()

    # Lifecycle
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    # Routes
    app.router.add_get("/health", health_check)
    app.router.add_post("/voice/incoming", handle_voice_incoming)
    app.router.add_post("/sms/incoming", handle_sms_incoming)
    app.router.add_get("/voice/stream", handle_voice_stream)

    return app


def main() -> None:
    app = create_app()
    web.run_app(app, host=settings.gateway_host, port=settings.gateway_port)


if __name__ == "__main__":
    main()
