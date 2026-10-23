import uvicorn

from . import config

uvicorn.run("signbridge.app:app", host=config.HOST, port=config.PORT, log_level="info")
