from __future__ import annotations

import time

import docker
import httpx


class BotLifecycle:
    def __init__(self):
        self.client = docker.from_env()

    def ensure_ready(self, container_name: str, ready_url: str, timeout: float = 30.0) -> None:
        try:
            container = self.client.containers.get(container_name)
        except docker.errors.NotFound as e:
            raise RuntimeError(
                f"Container '{container_name}' does not exist. Please run 'docker compose up -d --no-start' or similar."
            ) from e

        if container.status != "running":
            container.start()

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                res = httpx.get(ready_url, timeout=2.0)
                if res.status_code == 200:
                    return
            except httpx.RequestError:
                pass
            time.sleep(0.5)

        raise RuntimeError(f"Container '{container_name}' failed to become ready at {ready_url} within {timeout}s.")

    def is_running(self, container_name: str) -> bool:
        try:
            container = self.client.containers.get(container_name)
            container.reload()
            return container.status == "running"
        except docker.errors.NotFound:
            return False

    def stop_if_running(self, container_name: str) -> None:
        try:
            container = self.client.containers.get(container_name)
            if container.status == "running":
                container.stop()
        except docker.errors.NotFound:
            pass
