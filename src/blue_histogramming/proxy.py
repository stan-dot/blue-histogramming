from functools import lru_cache

import httpx


@lru_cache
def get_davidia_client():
    return httpx.AsyncClient(base_url="http://localhost:8002/davidia")


@lru_cache
def get_blueapi_client():
    return httpx.AsyncClient(base_url="https://b01-1-blueapi.diamond.ac.uk")
