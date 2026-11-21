"""HTTP tests for the answer stream: announcing, resuming and cancelling.

The engine is replaced by a scripted answer, so these tests exercise routing,
framing and access control without retrieval or a language model.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.authentication import AuthenticationService, PasswordHasher
from app.rag import QuestionRequest, RagEngine, StreamEvent
from app.server import RagWebApplication

#: Password given to the guest account for the duration of these tests only.
TEST_GUEST_PASSWORD = "sandi-uji-aliran"


def scripted_stream_answer(engine: RagEngine, request: QuestionRequest) -> Iterator[StreamEvent]:
    """Stands in for ``RagEngine.stream_answer``: a fixed four-event answer."""
    yield StreamEvent("sources", {"citations": []})
    yield StreamEvent("token", {"text": "Jawaban untuk: "})
    yield StreamEvent("token", {"text": request.question})
    yield StreamEvent("done", {"elapsed_seconds": 0.01})


def parse_frames(body: str) -> list[dict]:
    """Split an SSE body into frames of ``{"id", "event", "data"}``."""
    frames = []
    for raw_frame in body.split("\n\n"):
        if not raw_frame.strip():
            continue
        frame = {}
        for line in raw_frame.split("\n"):
            field_name, _, value = line.partition(": ")
            frame[field_name] = value
        frames.append(frame)
    return frames


@pytest.fixture
def signed_out_client(isolated_settings, monkeypatch) -> TestClient:
    """A client for an application whose engine streams a scripted answer."""
    isolated_settings.guest_username = "guest"
    isolated_settings.guest_password_hash = PasswordHasher().hash(TEST_GUEST_PASSWORD)
    isolated_settings.session_secret = "rahasia-uji"
    monkeypatch.setattr(RagEngine, "stream_answer", scripted_stream_answer)

    application = RagWebApplication(
        isolated_settings, authentication=AuthenticationService(isolated_settings)
    )
    return TestClient(application.api)


@pytest.fixture
def client(signed_out_client) -> TestClient:
    """The same client, signed in as a guest."""
    response = signed_out_client.post(
        "/api/auth/login", json={"username": "guest", "password": TEST_GUEST_PASSWORD}
    )
    assert response.status_code == 200
    return signed_out_client


def answer_identifier(frames: list[dict]) -> str:
    """The answer identifier announced in the ``accepted`` frame."""
    return frames[0]["data"].split('"answer_id": "')[1].split('"')[0]


def ask(client: TestClient, question: str = "Apa itu EWMP?") -> list[dict]:
    """Ask a question and return every frame of the answer stream."""
    response = client.post("/api/ask", json={"question": question})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_frames(response.text)


class TestAskStream:
    def test_the_first_frame_announces_the_answer_identifier(self, client):
        frames = ask(client)
        assert frames[0]["event"] == "accepted"
        assert "id" not in frames[0]
        assert '"answer_id"' in frames[0]["data"]

    def test_every_later_frame_is_numbered_in_order(self, client):
        frames = ask(client)[1:]
        assert [frame["id"] for frame in frames] == ["1", "2", "3", "4"]
        assert [frame["event"] for frame in frames] == ["sources", "token", "token", "done"]

    def test_asking_requires_signing_in(self, signed_out_client):
        response = signed_out_client.post("/api/ask", json={"question": "Apa itu EWMP?"})
        assert response.status_code == 401


class TestResumingAnAnswer:
    def test_resuming_returns_only_the_missed_events(self, client):
        answer_id = answer_identifier(ask(client, "EWMP"))
        response = client.get(f"/api/answers/{answer_id}/events", params={"after": 2})
        frames = parse_frames(response.text)
        assert [frame["id"] for frame in frames] == ["3", "4"]
        assert '"EWMP"' in frames[0]["data"]

    def test_the_last_event_id_header_is_honoured(self, client):
        answer_id = answer_identifier(ask(client))
        response = client.get(
            f"/api/answers/{answer_id}/events", headers={"Last-Event-ID": "3"}
        )
        assert [frame["id"] for frame in parse_frames(response.text)] == ["4"]

    def test_an_invalid_last_event_id_replays_from_the_start(self, client):
        answer_id = answer_identifier(ask(client))
        response = client.get(
            f"/api/answers/{answer_id}/events", headers={"Last-Event-ID": "bukan-angka"}
        )
        assert len(parse_frames(response.text)) == 4

    def test_a_resumed_stream_never_repeats_the_announcement(self, client):
        answer_id = answer_identifier(ask(client))
        frames = parse_frames(client.get(f"/api/answers/{answer_id}/events").text)
        assert all(frame["event"] != "accepted" for frame in frames)

    def test_an_unknown_answer_is_not_found(self, client):
        assert client.get("/api/answers/tidak-ada/events").status_code == 404

    def test_resuming_requires_signing_in(self, client):
        answer_id = answer_identifier(ask(client))
        client.cookies.clear()
        assert client.get(f"/api/answers/{answer_id}/events").status_code == 401


class TestCancellingAnAnswer:
    def test_a_known_answer_accepts_a_cancel_request(self, client):
        answer_id = answer_identifier(ask(client))
        response = client.post(f"/api/answers/{answer_id}/cancel")
        assert response.status_code == 200
        assert response.json()["cancel_requested"] is True

    def test_cancelling_an_unknown_answer_is_not_found(self, client):
        assert client.post("/api/answers/tidak-ada/cancel").status_code == 404


def test_server_indexes_with_the_question_encoder(isolated_settings, monkeypatch):
    """Indexing inside the server must not put a model on the GPU: a CUDA
    context stays in the process after the run and crowds out the answering
    model. The server therefore hands the pipeline its question encoder."""
    from app.retrieval import QUERY_EMBEDDER

    class RecordingPipeline:
        def __init__(self) -> None:
            self.embedders = []

        def use_embedder(self, embedder) -> None:
            self.embedders.append(embedder)

    pipeline = RecordingPipeline()
    monkeypatch.setattr("app.server.INGESTION_PIPELINE", pipeline)
    RagWebApplication(isolated_settings, authentication=AuthenticationService(isolated_settings))
    assert pipeline.embedders == [QUERY_EMBEDDER]
