from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def test_health_without_pipeline_loaded():
    s = Settings(db_url="sqlite://")
    with TestClient(create_app(pipeline=object(), settings=s)) as c:
        r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["pipeline_loaded"] is True
