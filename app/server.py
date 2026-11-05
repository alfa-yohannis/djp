"""HTTP interface: JSON endpoints plus the streaming answer channel.

:class:`RagWebApplication` owns the FastAPI instance and registers every route
as a bound method, which keeps request handling inside the object graph rather
than in module-level functions closing over globals.

Answers are delivered as Server-Sent Events. SSE is used in preference to a
WebSocket because the traffic is one-directional and short-lived: the browser
posts a question and reads a stream until it closes. It also survives a proxy or
a reload without connection bookkeeping.

Generation itself runs apart from the response (see :mod:`app.answer_streams`),
so a reader whose connection drops can resume the same answer rather than lose
it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer_streams import AnswerFrameStreamer, AnswerJobRegistry, ServerSentEventEncoder
from .authentication import AUTHENTICATION_SERVICE, AuthenticationService, Principal
from .config import SETTINGS, Settings
from .conversations import CONVERSATION_STORE
from .embeddings import ComputeDeviceResolver
from .extractors import EXTRACTOR_REGISTRY
from .ingestion import AUTOMATIC_INGESTION, INGESTION_RUNNER, TEXT_EMBEDDER, AutomaticIngestion
from .rag import LANGUAGE_MODEL_FACTORY, QuestionRequest, RagEngine
from .retrieval import CROSS_ENCODER_RERANKER, HybridRetriever
from .safety import SOURCE_GUARD
from .storage import DOCUMENT_CATALOGUE, VECTOR_INDEX

LOGGER = logging.getLogger(__name__)


class LoginPayload(BaseModel):
    """Body of a sign-in attempt, for either role."""

    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class AskPayload(BaseModel):
    """Body of a question submitted to the streaming answer endpoint."""

    question: str = Field(min_length=1, max_length=4000)
    folder_filter: str | None = None
    backend_name: str | None = None
    passage_count: int | None = Field(default=None, ge=1, le=20)
    #: Conversation to continue. Omitted for a one-off question that is not saved.
    session_id: str | None = Field(default=None, max_length=64)


class SessionCreatePayload(BaseModel):
    """Body of a request to start a new conversation."""

    title: str | None = Field(default=None, max_length=200)


class SessionRenamePayload(BaseModel):
    """Body of a request to retitle an existing conversation."""

    title: str = Field(min_length=1, max_length=200)


class SearchPayload(BaseModel):
    """Body of a retrieval-only request, used by the passage browser."""

    question: str = Field(min_length=1, max_length=4000)
    folder_filter: str | None = None
    passage_count: int | None = Field(default=None, ge=1, le=50)


class IngestPayload(BaseModel):
    """Body of a request to start an indexing run."""

    force_full_rebuild: bool = False


class FileActionPayload(BaseModel):
    """Body of a request to open or reveal an indexed document locally."""

    document_id: str = Field(min_length=1, max_length=64)
    #: Either ``"reveal"`` (highlight in Explorer) or ``"open"`` (default app).
    action: str = Field(default="reveal", pattern="^(reveal|open)$")


class LocalFileLauncher:
    """Opens an indexed document in Explorer or its default application.

    Only paths that are present in the index can be launched, so the endpoint
    cannot be used to reach arbitrary files on the machine. Revealing in
    Explorer is the default action because it cannot modify the document, which
    matters for a corpus that must stay untouched.
    """

    def __init__(self, settings: Settings) -> None:
        """Bind the launcher to the configured permission flag."""
        self._settings = settings

    @property
    def is_enabled(self) -> bool:
        """True when the interface may ask the server to launch files."""
        return self._settings.file_opening_enabled and sys.platform == "win32"

    def reveal_in_explorer(self, file_path: Path) -> None:
        """Open Explorer with the file selected, without opening the file."""
        subprocess.Popen(["explorer", f"/select,{file_path}"])

    def open_in_default_application(self, file_path: Path) -> None:
        """Hand the file to whichever application is registered for its type."""
        os.startfile(str(file_path))


class RevalidatingStaticFiles(StaticFiles):
    """Serves static files that browsers must re-check on every load.

    Without this the interface is edited, the server restarted, and the browser
    keeps running the previous `app.js` from cache — so a fix appears not to
    have worked at all. `no-cache` does not mean "do not store": the file is
    still cached, but revalidated each time, so an unchanged asset costs one
    conditional request answered with 304 rather than a full download.
    """

    def file_response(self, *args, **kwargs):
        """Attach revalidation headers to every static file served."""
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


class RagWebApplication:
    """Builds and owns the FastAPI application serving the RAG interface."""

    #: Paths reachable without signing in: the landing page, its assets, and
    #: the endpoints that perform the sign-in itself.
    PUBLIC_PATHS = frozenset({"/", "/favicon.ico"})
    PUBLIC_PREFIXES = ("/static/", "/api/auth/", "/api/public/")

    #: Shown when a guest attempts an action reserved for the administrator.
    ADMIN_ONLY_MESSAGE = (
        "Anda tidak memiliki hak untuk memperbarui dan/atau membangun ulang indeks."
    )

    #: Longest silence on an answer stream before a keep-alive frame is sent.
    #: Well under the idle limits of phones and proxies, and cheap: one short
    #: comment line per interval, only while nothing else is being sent.
    HEARTBEAT_SECONDS = 10.0

    #: Headers for every answer stream. `X-Accel-Buffering` stops a reverse
    #: proxy such as nginx from holding frames back until the stream ends.
    STREAM_HEADERS = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    def __init__(
        self,
        settings: Settings,
        authentication: AuthenticationService = AUTHENTICATION_SERVICE,
        automatic_ingestion: AutomaticIngestion = AUTOMATIC_INGESTION,
    ) -> None:
        """Compose every collaborator and register the HTTP routes."""
        self._settings = settings
        self._authentication = authentication
        self._automatic_ingestion = automatic_ingestion
        self._device_resolver = ComputeDeviceResolver()
        self._file_launcher = LocalFileLauncher(settings)
        self._answer_jobs = AnswerJobRegistry()
        self._frame_streamer = AnswerFrameStreamer(
            ServerSentEventEncoder(), self.HEARTBEAT_SECONDS
        )

        self._retriever = HybridRetriever(
            settings=settings,
            catalogue=DOCUMENT_CATALOGUE,
            vector_index=VECTOR_INDEX,
            embedder=TEXT_EMBEDDER,
            reranker=CROSS_ENCODER_RERANKER,
        )
        self._engine = RagEngine(
            settings=settings,
            retriever=self._retriever,
            model_factory=LANGUAGE_MODEL_FACTORY,
            conversation_store=CONVERSATION_STORE,
        )

        self.api = FastAPI(
            title="DJP RAG",
            description="Pencarian dan tanya-jawab atas salinan informasi perpajakan pajak.go.id",
            version="1.0.0",
            lifespan=self._lifespan,
        )
        self.api.middleware("http")(self._require_session)
        self._register_routes()
        self._mount_static_assets()

    def _register_routes(self) -> None:
        """Attach every endpoint as a bound method of this application."""
        self.api.add_api_route("/", self.serve_landing, methods=["GET"], include_in_schema=False)
        self.api.add_api_route("/app", self.serve_interface, methods=["GET"], include_in_schema=False)
        self.api.add_api_route("/api/auth/login", self.post_login, methods=["POST"])
        self.api.add_api_route("/api/auth/logout", self.post_logout, methods=["POST"])
        self.api.add_api_route("/api/auth/me", self.get_current_user, methods=["GET"])
        self.api.add_api_route("/api/public/overview", self.get_public_overview, methods=["GET"])
        self.api.add_api_route("/api/status", self.get_status, methods=["GET"])
        self.api.add_api_route("/api/folders", self.get_folders, methods=["GET"])
        self.api.add_api_route("/api/documents", self.get_documents, methods=["GET"])
        self.api.add_api_route(
            "/api/documents/{document_id}/download",
            self.get_document_download,
            methods=["GET"],
        )
        self.api.add_api_route("/api/ask", self.post_question, methods=["POST"])
        self.api.add_api_route(
            "/api/answers/{answer_id}/events", self.get_answer_events, methods=["GET"]
        )
        self.api.add_api_route(
            "/api/answers/{answer_id}/cancel", self.post_answer_cancel, methods=["POST"]
        )
        self.api.add_api_route("/api/search", self.post_search, methods=["POST"])
        self.api.add_api_route("/api/ingest/start", self.post_ingest_start, methods=["POST"])
        self.api.add_api_route("/api/ingest/status", self.get_ingest_status, methods=["GET"])
        self.api.add_api_route("/api/file-action", self.post_file_action, methods=["POST"])
        self.api.add_api_route("/api/sessions", self.get_sessions, methods=["GET"])
        self.api.add_api_route("/api/sessions", self.post_session, methods=["POST"])
        self.api.add_api_route(
            "/api/sessions/{session_id}", self.get_session, methods=["GET"]
        )
        self.api.add_api_route(
            "/api/sessions/{session_id}", self.patch_session, methods=["PATCH"]
        )
        self.api.add_api_route(
            "/api/sessions/{session_id}", self.delete_session, methods=["DELETE"]
        )

    @asynccontextmanager
    async def _lifespan(self, _api: FastAPI):
        """Keep the automatic indexer running for as long as the server is up."""
        self._automatic_ingestion.start()
        try:
            yield
        finally:
            self._automatic_ingestion.stop()

    def _mount_static_assets(self) -> None:
        """Serve the browser interface's static files."""
        web_directory = self._settings.web_directory
        if web_directory.exists():
            self.api.mount(
                "/static", RevalidatingStaticFiles(directory=str(web_directory)), name="static"
            )

    # --- Sessions -----------------------------------------------------------

    def _is_public_path(self, path: str) -> bool:
        """True for paths a visitor may reach before signing in."""
        return path in self.PUBLIC_PATHS or path.startswith(self.PUBLIC_PREFIXES)

    async def _require_session(self, request: Request, call_next):
        """Attach the signed-in principal, refusing protected paths without one.

        The check runs here, once, rather than in each handler, so a newly added
        endpoint is protected by default instead of by remembering to be.
        """
        principal = self._authentication.identify(
            request.cookies.get(AuthenticationService.COOKIE_NAME)
        )
        request.state.principal = principal
        path = request.url.path

        if principal is None and not self._is_public_path(path):
            if path.startswith("/api/"):
                return JSONResponse(
                    {"detail": "Silakan masuk terlebih dahulu."}, status_code=401
                )
            # Pages (the app itself, the API docs) send the visitor to sign in.
            return RedirectResponse("/#masuk", status_code=303)
        return await call_next(request)

    def _principal(self, request: Request) -> Principal | None:
        """The principal the session middleware attached to ``request``."""
        return getattr(request.state, "principal", None)

    def _session_response(self, token: str, body: dict) -> JSONResponse:
        """Return ``body`` with the session cookie set."""
        response = JSONResponse(body)
        response.set_cookie(
            AuthenticationService.COOKIE_NAME,
            token,
            max_age=self._authentication.session_lifetime_seconds,
            httponly=True,
            # Lax keeps the cookie off cross-site POSTs, which is what stops
            # another site from triggering an index rebuild through a visitor.
            samesite="lax",
            path="/",
        )
        return response

    async def post_login(self, payload: LoginPayload, request: Request) -> JSONResponse:
        """Sign in; the matching account decides whether the role is admin or guest."""
        client_address = request.client.host if request.client else "unknown"
        seconds_locked = self._authentication.throttle.seconds_locked(client_address)
        if seconds_locked:
            raise HTTPException(
                status_code=429,
                detail=f"Terlalu banyak percobaan. Coba lagi dalam {seconds_locked} detik.",
            )

        token = self._authentication.authenticate(payload.username, payload.password)
        if token is None:
            self._authentication.throttle.record_failure(client_address)
            LOGGER.warning("Login gagal dari %s", client_address)
            raise HTTPException(status_code=401, detail="Nama pengguna atau kata sandi salah.")

        self._authentication.throttle.record_success(client_address)
        principal = self._authentication.identify(token)
        LOGGER.info("%s (%s) masuk dari %s", principal.username, principal.role, client_address)
        return self._session_response(token, principal.as_dictionary())

    async def post_logout(self) -> JSONResponse:
        """End the current session."""
        response = JSONResponse({"signed_out": True})
        response.delete_cookie(AuthenticationService.COOKIE_NAME, path="/")
        return response

    async def get_current_user(self, request: Request) -> JSONResponse:
        """Report who is signed in, or ``{"signed_in": false}``."""
        principal = self._principal(request)
        if principal is None:
            return JSONResponse({"signed_in": False})
        return JSONResponse({"signed_in": True, **principal.as_dictionary()})

    async def get_public_overview(self) -> JSONResponse:
        """Headline figures for the landing page: counts only, no content."""
        statistics = DOCUMENT_CATALOGUE.collect_statistics()
        return JSONResponse(
            {
                "document_count": statistics["document_count"],
                "chunk_count": statistics["chunk_count"],
                "folder_count": len(statistics["folders"]),
                "total_megabytes": statistics["total_megabytes"],
                "last_indexed_at": statistics["last_indexed_at"],
            }
        )

    # --- Interface ----------------------------------------------------------

    def _serve_page(self, file_name: str) -> FileResponse:
        """Serve one HTML page from the web directory."""
        page_file = self._settings.web_directory / file_name
        if not page_file.exists():
            raise HTTPException(status_code=500, detail="Berkas antarmuka tidak ditemukan")
        # Revalidated for the same reason as the static assets: a cached page
        # would keep loading the old markup after an update.
        return FileResponse(page_file, headers={"Cache-Control": "no-cache, must-revalidate"})

    async def serve_landing(self) -> FileResponse:
        """Serve the public landing page with the sign-in form."""
        return self._serve_page("landing.html")

    async def serve_interface(self) -> FileResponse:
        """Serve the single-page interface; the middleware has checked the session."""
        return self._serve_page("index.html")

    # --- Status -------------------------------------------------------------

    def _is_local_client(self, request: Request) -> bool:
        """True when the request came from this machine rather than the network.

        Revealing or opening a file acts on the server's own desktop, so those
        actions are meaningful only for a visitor sitting at this computer. The
        interface uses this flag to hide them from everyone else.
        """
        client = request.client
        return bool(client) and client.host in {"127.0.0.1", "::1", "localhost"}

    async def get_status(self, request: Request) -> JSONResponse:
        """Report index size, hardware, backend availability and corpus settings."""
        index_statistics = DOCUMENT_CATALOGUE.collect_statistics()
        ocr_recogniser = EXTRACTOR_REGISTRY.ocr_recogniser
        principal = self._principal(request)

        # Where the corpus lives on this machine is operator detail; guests get
        # null, so hiding the panel is not merely cosmetic.
        source = None
        if principal is not None and principal.is_admin:
            source = {
                "directory": str(self._settings.source_directory),
                "exists": Path(self._settings.source_directory).exists(),
                "access_mode": "baca-saja (read-only)",
                "excluded_directories": sorted(self._settings.excluded_directories),
            }

        return JSONResponse(
            {
                "source": source,
                "index": index_statistics,
                "vector_index": {
                    "vector_count": VECTOR_INDEX.size,
                    "dimensions": VECTOR_INDEX.dimensions,
                },
                "models": {
                    "embedding_model": self._settings.embedding_model_name,
                    "embedding_device": TEXT_EMBEDDER.device,
                    "embedding_loaded": TEXT_EMBEDDER.is_loaded,
                    "reranking_model": self._settings.reranking_model_name,
                    "reranking_enabled": CROSS_ENCODER_RERANKER.is_enabled,
                    "reranking_loaded": CROSS_ENCODER_RERANKER.is_loaded,
                    "reranking_unavailable_reason": CROSS_ENCODER_RERANKER.unavailable_reason,
                },
                "hardware": {
                    "accelerator": self._device_resolver.describe_hardware(),
                    "cuda_available": self._device_resolver.is_cuda_available(),
                },
                "language_models": {
                    "active_backend": self._settings.language_model_backend,
                    "backends": LANGUAGE_MODEL_FACTORY.check_all_statuses(),
                },
                "conversations": CONVERSATION_STORE.collect_statistics(),
                "client": {
                    #: Drives whether the interface offers reveal/open, which
                    #: only make sense at the machine running the server.
                    "is_local": self._is_local_client(request),
                    "address": request.client.host if request.client else "",
                },
                "ocr": {
                    "enabled": self._settings.ocr_enabled,
                    "available": ocr_recogniser.is_available,
                    "reason": ocr_recogniser.unavailable_reason,
                },
                "retrieval": {
                    "answer_passage_count": self._settings.answer_passage_count,
                    "dense_candidate_count": self._settings.dense_candidate_count,
                    "sparse_candidate_count": self._settings.sparse_candidate_count,
                    "chunk_token_count": self._settings.chunk_token_count,
                },
            }
        )

    async def get_folders(self) -> JSONResponse:
        """List the top-level folders available as retrieval filters."""
        return JSONResponse({"folders": DOCUMENT_CATALOGUE.list_folders()})

    async def get_documents(self, status: str | None = None, limit: int = 1000) -> JSONResponse:
        """List indexed documents, optionally filtered by ingest status."""
        documents = DOCUMENT_CATALOGUE.list_documents(status_filter=status, limit=limit)
        return JSONResponse({"documents": documents, "count": len(documents)})

    async def get_document_download(self, document_id: str) -> FileResponse:
        """Send an indexed document to the browser as a download.

        This is the only file action that works for a colleague on another
        machine: revealing or opening a file acts on the *server's* desktop, not
        the visitor's. Only documents present in the index can be fetched, so
        the endpoint cannot be used to read arbitrary paths on this computer.
        """
        stored_path = DOCUMENT_CATALOGUE.find_document_path(document_id)
        if stored_path is None:
            raise HTTPException(status_code=404, detail="Dokumen tidak ada di indeks")

        file_path = Path(stored_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Berkas tidak lagi ada di sumber: {file_path.name}"
            )
        return FileResponse(
            file_path,
            filename=file_path.name,
            media_type="application/octet-stream",
        )

    # --- Question answering -------------------------------------------------

    async def post_question(self, payload: AskPayload) -> StreamingResponse:
        """Answer a question, streaming citations then text as SSE.

        The first frame, ``accepted``, carries the answer's identifier. Every
        later frame is numbered, so a reader that loses the connection can pick
        up from ``/api/answers/{answer_id}/events`` where it left off.
        """
        request = QuestionRequest(
            question=payload.question.strip(),
            folder_filter=payload.folder_filter or None,
            backend_name=payload.backend_name or None,
            passage_count=payload.passage_count,
            session_id=payload.session_id or None,
        )
        job = self._answer_jobs.start(self._engine.stream_answer(request))
        return self._build_stream_response(
            self._frame_streamer.iterate_frames(job, announce=True)
        )

    async def get_answer_events(
        self, answer_id: str, request: Request, after: int = 0
    ) -> StreamingResponse:
        """Resume an answer's stream after event number ``after``.

        The standard ``Last-Event-ID`` header is honoured too, so a native
        ``EventSource`` reconnecting on its own resumes correctly as well.
        """
        job = self._answer_jobs.find(answer_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail="Jawaban tidak ditemukan atau sudah kedaluwarsa."
            )
        after_event_number = max(after, self._read_last_event_number(request))
        return self._build_stream_response(
            self._frame_streamer.iterate_frames(job, after_event_number=after_event_number)
        )

    async def post_answer_cancel(self, answer_id: str) -> JSONResponse:
        """Stop generating an answer, which no longer ends when its reader leaves."""
        job = self._answer_jobs.find(answer_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail="Jawaban tidak ditemukan atau sudah kedaluwarsa."
            )
        job.request_cancel()
        return JSONResponse({"cancel_requested": True, "answer_id": answer_id})

    def _read_last_event_number(self, request: Request) -> int:
        """The event number in a ``Last-Event-ID`` header, or 0 when absent or invalid."""
        header_value = request.headers.get("last-event-id", "").strip()
        return int(header_value) if header_value.isdigit() else 0

    def _build_stream_response(self, frames: Iterator[str]) -> StreamingResponse:
        """Wrap SSE frames in a response that nothing along the way will buffer."""
        return StreamingResponse(
            frames, media_type="text/event-stream", headers=self.STREAM_HEADERS
        )

    async def post_search(self, payload: SearchPayload) -> JSONResponse:
        """Return retrieved passages without generating an answer."""
        passages = self._retriever.retrieve(
            payload.question.strip(), payload.folder_filter or None, payload.passage_count
        )
        return JSONResponse(
            {
                "passages": [
                    {
                        "citation_number": passage.citation_number,
                        "document_id": passage.document_id,
                        "document_name": passage.document_name,
                        "relative_path": passage.relative_path,
                        "section_label": passage.section_label,
                        "text": passage.text,
                        "produced_by_ocr": passage.produced_by_ocr,
                        "retrieval_methods": passage.retrieval_methods,
                        "dense_score": passage.dense_score,
                        "sparse_score": passage.sparse_score,
                        "reranked_score": passage.reranked_score,
                        "relevance_score": round(passage.ranking_score, 4),
                    }
                    for passage in passages
                ],
                "count": len(passages),
            }
        )

    # --- Indexing -----------------------------------------------------------

    async def post_ingest_start(self, payload: IngestPayload, request: Request) -> JSONResponse:
        """Start a background indexing run, refusing if one is already active."""
        principal = self._principal(request)
        if principal is None or not principal.is_admin:
            raise HTTPException(status_code=403, detail=self.ADMIN_ONLY_MESSAGE)
        if not Path(self._settings.source_directory).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Direktori sumber tidak ditemukan: {self._settings.source_directory}",
            )
        started = INGESTION_RUNNER.start(force_full_rebuild=payload.force_full_rebuild)
        if not started:
            return JSONResponse(
                {"started": False, "message": "Proses indexing sedang berjalan."},
                status_code=409,
            )
        return JSONResponse({"started": True, "message": "Proses indexing dimulai."})

    async def get_ingest_status(self) -> JSONResponse:
        """Report progress of the current or most recent indexing run."""
        return JSONResponse(INGESTION_RUNNER.current_progress())

    # --- Conversations ------------------------------------------------------

    async def get_sessions(self, limit: int = 100) -> JSONResponse:
        """List saved conversations, most recently used first."""
        sessions = CONVERSATION_STORE.list_sessions(limit=limit)
        return JSONResponse(
            {
                "sessions": [session.as_dictionary() for session in sessions],
                "count": len(sessions),
            }
        )

    async def post_session(self, payload: SessionCreatePayload) -> JSONResponse:
        """Start a new conversation and return it."""
        session = CONVERSATION_STORE.create_session(payload.title)
        return JSONResponse(session.as_dictionary(), status_code=201)

    async def get_session(self, session_id: str) -> JSONResponse:
        """Return one conversation together with every turn in it."""
        session = CONVERSATION_STORE.load_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Percakapan tidak ditemukan")
        return JSONResponse(session.as_dictionary(include_turns=True))

    async def patch_session(
        self, session_id: str, payload: SessionRenamePayload
    ) -> JSONResponse:
        """Rename a conversation."""
        if not CONVERSATION_STORE.rename_session(session_id, payload.title):
            raise HTTPException(status_code=404, detail="Percakapan tidak ditemukan")
        return JSONResponse({"renamed": True, "session_id": session_id})

    async def delete_session(self, session_id: str) -> JSONResponse:
        """Delete a conversation and all of its turns."""
        if not CONVERSATION_STORE.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Percakapan tidak ditemukan")
        return JSONResponse({"deleted": True, "session_id": session_id})

    # --- Files --------------------------------------------------------------

    async def post_file_action(
        self, payload: FileActionPayload, request: Request
    ) -> JSONResponse:
        """Reveal or open an indexed document on the machine running the server."""
        if not self._file_launcher.is_enabled:
            raise HTTPException(
                status_code=403, detail="Membuka berkas dinonaktifkan (ALLOW_OPEN_FILE=0)"
            )
        # Refused for remote visitors: it would pop a window on someone else's
        # desktop, which is both useless to them and startling for whoever is
        # sitting here. They get the download endpoint instead.
        if not self._is_local_client(request):
            raise HTTPException(
                status_code=403,
                detail="Aksi ini hanya tersedia di komputer yang menjalankan server. "
                       "Gunakan tombol Unduh.",
            )

        stored_path = DOCUMENT_CATALOGUE.find_document_path(payload.document_id)
        if stored_path is None:
            raise HTTPException(status_code=404, detail="Dokumen tidak ada di indeks")

        file_path = Path(stored_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Berkas tidak ditemukan di sumber: {file_path}"
            )

        try:
            if payload.action == "open":
                self._file_launcher.open_in_default_application(file_path)
            else:
                self._file_launcher.reveal_in_explorer(file_path)
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Gagal membuka berkas: {error}")

        return JSONResponse({"opened": True, "action": payload.action, "path": str(file_path)})


class ApplicationBootstrapper:
    """Prepares logging and safety checks, then builds the web application."""

    def __init__(self, settings: Settings) -> None:
        """Bind the bootstrapper to the configuration it will validate."""
        self._settings = settings

    def configure_logging(self) -> None:
        """Send readable logs to the console for the lifetime of the process."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        # These libraries log a line per HTTP call and per batch; quieten them.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    def verify_safety_invariants(self) -> None:
        """Fail fast if generated data would be written inside the corpus."""
        SOURCE_GUARD.verify_data_directory_isolation()
        LOGGER.info(
            "Direktori sumber (baca-saja): %s", self._settings.source_directory
        )
        LOGGER.info("Direktori data (tulis): %s", self._settings.data_directory)

    def build(self) -> FastAPI:
        """Return a fully configured FastAPI application."""
        self.configure_logging()
        self.verify_safety_invariants()
        application = RagWebApplication(self._settings)
        return application.api


def create_application() -> FastAPI:
    """Module-level factory used by uvicorn's import string."""
    return ApplicationBootstrapper(SETTINGS).build()
