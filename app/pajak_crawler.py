"""Crawling pajak.go.id into the local corpus.

The corpus of this system is the public website of Direktorat Jenderal Pajak.
This module walks the site section by section — regulations, tax basics, FAQ,
articles, press releases, profile and service pages — and writes every page
into ``corpus-pajak/<section>/`` as a dated plain-text snapshot, in the same
format :mod:`app.web_sources` uses. Linked PDFs (forms, regulation
attachments) are downloaded beside them as the original files.

**Why snapshots rather than live fetches.** A tax answer must be checkable:
the page a citation points to can change after it was quoted, so every
snapshot records its URL and the moment it was taken, and the answer cites
that frozen copy while still linking to the live page.

**The one writer of the corpus.** Everywhere else the corpus is read-only —
ingest and the web server go through :class:`~app.safety.SourceTreeGuard`
and can never modify it. The crawl is the step that *produces* the corpus,
so :class:`CorpusPageWriter` is the single component allowed to write there,
and it refuses any path outside ``SOURCE_DIR``.

**Courtesy.** Only ``www.pajak.go.id`` is fetched, the paths robots.txt
disallows (``/search/``, ``/admin/``, ``/user/…``) are never requested, and
requests are spaced by ``PAJAK_CRAWL_DELAY`` seconds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

from .config import SETTINGS, Settings
from .safety import SOURCE_GUARD, SourceTreeGuard
from .web_sources import ExternalPage, SnapshotWriter, WebPageReader

LOGGER = logging.getLogger(__name__)

SITE_ROOT = "https://www.pajak.go.id"


@dataclass(frozen=True)
class CrawlSection:
    """One part of the site, crawled into its own corpus folder.

    A section is either a *listing* — a paginated index whose links matching
    ``detail_pattern`` are followed — or a fixed set of pages, or both.
    The folder name doubles as the folder filter in the interface.
    """

    folder: str
    #: Index pages whose matching links are followed, each paginated with
    #: ``?page=N`` when ``paginate`` is set.
    listing_urls: tuple[str, ...] = ()
    #: Regex on the URL path selecting which links on a listing are pages.
    detail_pattern: str = r"^$"
    #: Pages fetched as they are, regardless of any listing.
    fixed_urls: tuple[str, ...] = ()
    paginate: bool = True
    #: Detail pages taken from *each* listing in a sample crawl.
    sample_limit_per_listing: int = 20
    #: Attachments (PDF and the like) downloaded in a sample crawl.
    sample_attachment_limit: int = 10
    #: Optional sections are crawled only when named explicitly.
    optional: bool = False


def _peraturan_listing(document_type_id: str) -> str:
    """Return the regulation index filtered to one document type."""
    return f"{SITE_ROOT}/id/peraturan?field_jenis_dokumen_target_id={document_type_id}"


#: The crawl plan. Order matters only for progress output.
CRAWL_PLAN: tuple[CrawlSection, ...] = (
    CrawlSection(
        folder="Belajar Pajak",
        listing_urls=(f"{SITE_ROOT}/index-belajar-pajak/",),
        detail_pattern=(
            r"^/id/(?!artikel|berita|siaran-pers|pengumuman|peraturan|index-|faq-page)"
            r"[^/]+$"
        ),
        paginate=False,
        sample_limit_per_listing=120,
    ),
    CrawlSection(
        folder="FAQ",
        listing_urls=(f"{SITE_ROOT}/id/faq-page",),
        detail_pattern=r"^/id/(?!faq-page)[^/]+$",
        sample_limit_per_listing=40,
    ),
    CrawlSection(
        folder="Peraturan",
        # Undang-Undang, Peraturan Pemerintah, PMK, Peraturan Dirjen Pajak.
        # Unfiltered, the newest entries are almost all weekly exchange-rate
        # decrees, which would crowd out the regulations people ask about.
        listing_urls=(
            _peraturan_listing("13926"),
            _peraturan_listing("13902"),
            _peraturan_listing("13899"),
            _peraturan_listing("13894"),
        ),
        detail_pattern=r"^/id/peraturan/[^/]+$",
        sample_limit_per_listing=18,
        sample_attachment_limit=20,
    ),
    CrawlSection(
        folder="Artikel",
        listing_urls=(f"{SITE_ROOT}/id/artikel-page",),
        detail_pattern=r"^/id/artikel/[^/]+$",
        sample_limit_per_listing=40,
    ),
    CrawlSection(
        folder="Siaran Pers",
        listing_urls=(f"{SITE_ROOT}/id/siaran-pers-page",),
        detail_pattern=r"^/id/siaran-pers/[^/]+$",
        sample_limit_per_listing=30,
    ),
    CrawlSection(
        folder="Profil dan Layanan",
        fixed_urls=tuple(
            f"{SITE_ROOT}{path}"
            for path in (
                "/id/selayang-pandang",
                "/id/visi-misi-tujuan-dan-maklumat-pelayanan",
                "/id/tugas-dan-fungsi-0",
                "/id/struktur-organisasi",
                "/id/kode-etik-dan-kode-perilaku",
                "/id/standar-pelayanan-di-lingkungan-direktorat-jenderal-pajak",
                "/id/jalur-dan-waktu-layanan",
                "/id/saluran-pengaduan",
                "/id/penegakan-hukum",
                "/id/kerja-sama-dan-kemitraan",
                "/id/rencana-strategis-direktorat-jenderal-pajak-2025-2029",
                "/id/uu-hpp",
                "/id/klaster-kemudahan-berusaha-bidang-perpajakan",
                "/id/pajakdigital",
                "/id/reformdjp",
                "/id/reformdjp/coretax",
                "/id/tax-treaty",
                "/id/taxtreaty-mli",
                "/id/eoi",
                "/id/eoi/aeoi",
                "/id/eoi/aeoi-wht",
                "/id/eoi/carf",
                "/id/eoi/cbcr",
                "/id/eoi/crs",
                "/id/eoi/eoir",
                "/id/eoi/fatca",
                "/id/eoi/seoi",
                "/id/akses-informasi-publik-melalui-layanan-ppid",
                "/id/prosedur-permohonan",
                "/id/informasi-publik-tersedia-setiap-saat",
                "/id/regulasi-ppid",
                "/id/unit-kerja",
                "/id/index-pjap",
                "/id/index-konsultan",
                "/id/PPS",
                "/id/tutorial-penggunaan-virtual-assistant-chatbot",
                "/prasyarat-pengguna",
                "/hubungi-kami",
                "/tax-refund-tourists",
                "/portal-layanan-wp/",
                "/layanan-digital/",
            )
        ),
        sample_attachment_limit=10,
    ),
    CrawlSection(
        folder="Formulir",
        fixed_urls=(f"{SITE_ROOT}/id/formulir-page",),
        sample_attachment_limit=40,
    ),
    CrawlSection(
        folder="Pengumuman",
        listing_urls=(f"{SITE_ROOT}/id/pengumuman-page",),
        detail_pattern=r"^/id/pengumuman/[^/]+$",
        sample_limit_per_listing=0,
        optional=True,
    ),
    CrawlSection(
        # Some 54,000 regional-office news items: many hours to crawl and
        # roughly five times the index, for little a tax question needs.
        folder="Berita",
        listing_urls=(f"{SITE_ROOT}/id/berita-page",),
        detail_pattern=r"^/id/berita/[^/]+$",
        sample_limit_per_listing=0,
        optional=True,
    ),
)


class PajakUrl:
    """Normalises links found on pajak.go.id and decides which may be fetched."""

    ALLOWED_HOSTS = {"www.pajak.go.id", "pajak.go.id"}

    #: Paths robots.txt disallows; never requested.
    DISALLOWED_PREFIXES = (
        "/core/", "/profiles/", "/admin/", "/comment/reply/", "/filter/tips",
        "/node/add/", "/search/", "/user/", "/id/search/", "/en/search/",
    )

    ATTACHMENT_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".pptx")

    @classmethod
    def normalise(cls, href: str, base_url: str) -> str | None:
        """Return an absolute, canonical ``www.pajak.go.id`` URL, or None.

        The site's own markup is inconsistent: links arrive as ``/index.php/id/…``,
        as bare ``pajak.go.id`` and even as ``.\\artikel-page\\`` with Windows
        separators. All of these collapse to one form so a page is fetched once.
        """
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            return None
        absolute = urljoin(base_url, href.strip().replace("\\", "/"))
        parsed = urlparse(absolute)
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or host not in cls.ALLOWED_HOSTS:
            return None
        path = re.sub(r"^/index\.php(?=/)", "", parsed.path) or "/"
        if any(path.startswith(prefix) for prefix in cls.DISALLOWED_PREFIXES):
            return None
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{SITE_ROOT}{path}{query}"

    @classmethod
    def path_of(cls, url: str) -> str:
        """Return the path of ``url`` without a trailing slash."""
        path = urlparse(url).path
        return path.rstrip("/") or "/"

    @classmethod
    def is_attachment(cls, url: str) -> bool:
        """True when ``url`` points at a downloadable document file."""
        return urlparse(url).path.lower().endswith(cls.ATTACHMENT_EXTENSIONS)

    @classmethod
    def with_page(cls, url: str, page_number: int) -> str:
        """Return ``url`` asking for one page of a paginated listing."""
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}page={page_number}"


@dataclass
class ParsedPage:
    """What one fetched page contributes: its title, text and outbound links."""

    title: str
    text: str
    links: list[str] = field(default_factory=list)
    attachments: list[tuple[str, str]] = field(default_factory=list)


class PajakPageParser:
    """Reduces a pajak.go.id page to the text of its main content block.

    Every page shares a header menu, a footer and a sidebar of recent
    articles. Indexed as-is, that boilerplate would appear in every chunk and
    match every question, so only the content block is kept.
    """

    #: Content containers in order of preference.
    CONTENT_SELECTORS = ("#block-porto-content", "div.region-content", "main", "body")

    #: Page furniture inside the content block: share buttons and a view
    #: counter that changes on every visit.
    FURNITURE_SELECTORS = (".addtoany_list", ".statistics-counter")

    TITLE_SUFFIX = re.compile(r"\s*\|\s*Direktorat Jenderal Pajak\s*$", re.IGNORECASE)

    def parse(self, html: str, url: str) -> ParsedPage:
        """Return the title, readable text, links and attachments of ``html``."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        title = self._title_of(soup)

        # Links are read from the whole content region first: on index pages
        # such as "Belajar Pajak" the entries sit in a block beside the main one.
        region = soup.select_one("div.region-content") or soup.body or soup
        links = self._links_in(region, url)

        content = next(
            (soup.select_one(selector) for selector in self.CONTENT_SELECTORS
             if soup.select_one(selector) is not None),
            soup,
        )
        attachments = [
            (label, link) for label, link in self._labelled_links_in(content, url)
            if PajakUrl.is_attachment(link)
        ]
        for tag_name in WebPageReader.NON_CONTENT_TAGS:
            for element in content.find_all(tag_name):
                element.decompose()
        for selector in self.FURNITURE_SELECTORS:
            for element in content.select(selector):
                element.decompose()
        return ParsedPage(
            title=title, text=self._readable_text(content), links=links, attachments=attachments
        )

    #: Elements that start a new line; everything else flows inline.
    BLOCK_TAGS = (
        "p", "div", "li", "ul", "ol", "br", "h1", "h2", "h3", "h4", "h5", "h6",
        "tr", "table", "section", "article", "blockquote", "pre", "dt", "dd",
    )

    def _readable_text(self, content) -> str:
        """Return the text of ``content`` with one line per block element.

        ``get_text(separator="\\n")`` would put every inline element on a line
        of its own — "wajib pajak (", "NPWP", ")" — which fragments sentences
        across chunks and buries them among provenance markers. Line breaks
        are inserted only around block elements instead, and table cells are
        separated by " | " so a row stays readable as one line.
        """
        for element in content.find_all(self.BLOCK_TAGS):
            element.insert_before("\n")
            element.insert_after("\n")
        for cell in content.find_all(["td", "th"]):
            cell.insert_after(" | ")
        lines = (" ".join(line.split()) for line in content.get_text().splitlines())
        return "\n".join(line for line in lines if line)

    def _title_of(self, soup) -> str:
        """Return the page title without the site-name suffix."""
        raw_title = soup.title.get_text(strip=True) if soup.title else ""
        title = self.TITLE_SUFFIX.sub("", raw_title).strip()
        if not title or title == "Direktorat Jenderal Pajak":
            heading = soup.find("h1")
            title = heading.get_text(" ", strip=True) if heading else title
        return title or "Halaman"

    def _links_in(self, element, base_url: str) -> list[str]:
        """Return the unique normalised links inside ``element``, in page order."""
        return list(dict.fromkeys(link for _label, link in self._labelled_links_in(element, base_url)))

    def _labelled_links_in(self, element, base_url: str) -> list[tuple[str, str]]:
        """Return ``(anchor text, normalised URL)`` for each fetchable link."""
        found: list[tuple[str, str]] = []
        for anchor in element.select("a[href]"):
            link = PajakUrl.normalise(anchor["href"], base_url)
            if link is not None:
                found.append((anchor.get_text(" ", strip=True), link))
        return found


class PajakFetcher:
    """Polite HTTP access to pajak.go.id: one session, spaced requests, retries."""

    RETRY_ATTEMPTS = 3

    def __init__(self, settings: Settings) -> None:
        """Open a session carrying browser-like headers."""
        import requests

        self._settings = settings
        self._session = requests.Session()
        self._session.headers.update(WebPageReader.BROWSER_HEADERS)
        self._last_request_at = 0.0

    def get_text(self, url: str) -> str:
        """Return the HTML of ``url``.

        Raises:
            RuntimeError: when the page still fails after the retries.
        """
        response = self._request(url, stream=False)
        response.encoding = response.encoding or "utf-8"
        return response.text

    def download(self, url: str, target: Path, maximum_bytes: int) -> int:
        """Stream ``url`` into ``target`` and return the number of bytes written.

        The file is written under a temporary name and renamed only when
        complete, so an interrupted download never leaves a truncated PDF that
        the next ingest would try to read.

        Raises:
            RuntimeError: on failure, or when the file exceeds ``maximum_bytes``.
        """
        response = self._request(url, stream=True)
        partial = target.with_name(target.name + ".part")
        written = 0
        try:
            with open(partial, "wb") as handle:
                for block in response.iter_content(chunk_size=256 * 1024):
                    written += len(block)
                    if written > maximum_bytes:
                        raise RuntimeError(
                            f"lebih besar dari {maximum_bytes // (1024 * 1024)} MB"
                        )
                    handle.write(block)
            partial.replace(target)
        finally:
            response.close()
            partial.unlink(missing_ok=True)
        return written

    def _request(self, url: str, stream: bool):
        """Perform one GET with courtesy spacing and retries on transient errors."""
        import requests

        last_error: Exception | None = None
        for attempt in range(1, self.RETRY_ATTEMPTS + 1):
            self._wait_for_turn()
            try:
                response = self._session.get(
                    url, timeout=self._settings.pajak_crawl_timeout_seconds, stream=stream
                )
                if response.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                last_error = error
                status = getattr(getattr(error, "response", None), "status_code", None)
                if status is not None and status < 500:
                    break
                LOGGER.info("Percobaan %d gagal untuk %s: %s", attempt, url, error)
                time.sleep(5 * attempt)
        raise RuntimeError(f"{type(last_error).__name__}: {last_error}")

    def _wait_for_turn(self) -> None:
        """Sleep until the configured delay has passed since the last request."""
        elapsed = time.monotonic() - self._last_request_at
        remaining = self._settings.pajak_crawl_delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


class CorpusPageWriter(SnapshotWriter):
    """Writes crawled pages and attachments into ``SOURCE_DIR/<section>/``.

    The snapshot format is inherited unchanged: a provenance header and a
    source marker repeated through the body, so every chunk carries its URL.
    """

    _PERATURAN_DATE_LINE = "Tanggal Peraturan"

    def __init__(self, settings: Settings) -> None:
        """Bind the writer to the corpus directory it produces."""
        self._settings = settings
        self._directory = Path(settings.source_directory).resolve()

    def section_directory(self, section: CrawlSection) -> Path:
        """Return (creating it) the folder one section is written to."""
        directory = self._inside_corpus(self._directory / section.folder)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_page(self, section: CrawlSection, page: ExternalPage, parsed: ParsedPage) -> Path:
        """Write one page snapshot and return its path."""
        text = parsed.text
        if parsed.attachments:
            listed = "\n".join(
                f"- {label or unquote(Path(urlparse(link).path).name)}: {link}"
                for label, link in parsed.attachments
            )
            text = f"{text}\nLampiran dan dokumen terkait:\n{listed}"

        target = self.section_directory(section) / self._page_filename(page, parsed)
        target.write_text(self._build_document(page, parsed.title, text), encoding="utf-8")
        return target

    def attachment_target(self, section: CrawlSection, url: str) -> Path:
        """Return where an attachment from ``url`` is saved.

        The original filename is kept so the file is recognisable, followed by
        a short hash of the URL so two uploads that share a name never collide.
        """
        original = unquote(Path(urlparse(url).path).name)
        stem, suffix = Path(original).stem, Path(original).suffix.lower()
        cleaned = self._clean(stem)[:100] or "lampiran"
        directory = self.section_directory(section) / "Lampiran"
        directory.mkdir(parents=True, exist_ok=True)
        return self._inside_corpus(directory / f"{cleaned} [{self._short_hash(url)}]{suffix}")

    def _page_filename(self, page: ExternalPage, parsed: ParsedPage) -> str:
        """Return a readable, unique filename for a page snapshot.

        A regulation is prefixed with its number — "PER-8/PJ/2026" — because
        that is how people search for and refer to it; the long all-caps title
        alone is hard to tell apart in a file list.
        """
        name = parsed.title
        lines = parsed.text.splitlines()
        if len(lines) > 2 and lines[2].strip() == self._PERATURAN_DATE_LINE:
            document_type, number = lines[0].strip(), lines[1].strip()
            # "PER-8/PJ/2026" names itself; "7 TAHUN 2021" needs its type.
            if number[:1].isdigit():
                number = f"{self._TYPE_ABBREVIATIONS.get(document_type, document_type)} {number}"
            name = f"{number} - {parsed.title}"
        return f"{self._clean(name)[:110] or 'Halaman'} [{self._short_hash(page.url)}].txt"

    #: Customary short forms of regulation types, as people search for them.
    _TYPE_ABBREVIATIONS = {
        "Undang-Undang": "UU",
        "Peraturan Pemerintah": "PP",
        "Peraturan Menteri Keuangan": "PMK",
        "Keputusan Menteri Keuangan": "KMK",
        "Peraturan Dirjen Pajak": "PER",
        "Keputusan Dirjen Pajak": "KEP",
        "Peraturan Presiden": "Perpres",
    }

    def _clean(self, name: str) -> str:
        """Make ``name`` safe as a Windows filename."""
        cleaned = self._UNSAFE_FILENAME_CHARACTERS.sub("-", name)
        return " ".join(cleaned.split()).strip(" .-")

    def _short_hash(self, url: str) -> str:
        """Return eight hex characters identifying ``url``."""
        return hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]

    def _inside_corpus(self, path: Path) -> Path:
        """Refuse any write target that is not inside the corpus directory."""
        resolved = path.resolve()
        if resolved != self._directory and self._directory not in resolved.parents:
            raise RuntimeError(f"Menolak menulis di luar korpus: {resolved}")
        return resolved


@dataclass
class CrawlOutcome:
    """Result of one crawl, for reporting to the caller."""

    pages_written: int = 0
    pages_unchanged: int = 0
    #: Pages with no text worth indexing — embedded videos, bare file links.
    pages_skipped: int = 0
    attachments_written: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class CrawlManifest:
    """Remembers which URL was saved to which file, across crawls.

    Kept in the data directory rather than the corpus, where a JSON file
    would itself be indexed. It is what lets a second crawl skip pages it
    already has instead of re-downloading the whole site.
    """

    def __init__(self, settings: Settings, guard: SourceTreeGuard) -> None:
        """Load the manifest, verifying its location is outside the corpus."""
        self._path = guard.assert_writable(settings.data_directory / "pajak-crawl.json")
        self._corpus = Path(settings.source_directory).resolve()
        self._entries: dict[str, dict] = {}
        if self._path.exists():
            self._entries = json.loads(self._path.read_text(encoding="utf-8"))

    def has_file_for(self, url: str) -> bool:
        """True when ``url`` was saved before and the file is still present."""
        entry = self._entries.get(url)
        return bool(entry) and (self._corpus / entry["file"]).exists()

    def record(
        self, url: str, written: Path, attachments: list[tuple[str, str]] | None = None
    ) -> None:
        """Remember that ``url`` was saved to ``written``, with its attachment links."""
        entry = {
            "file": str(written.relative_to(self._corpus)),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        if attachments:
            entry["attachments"] = [[label, link] for label, link in attachments]
        self._entries[url] = entry

    def attachments_of(self, url: str) -> list[tuple[str, str]]:
        """Return the ``(label, link)`` attachments recorded for page ``url``.

        A page that is not fetched again still hands its attachments to the
        download queue this way, so one that failed on a dropped connection
        is retried by the next crawl instead of being lost.
        """
        entry = self._entries.get(url) or {}
        return [(label, link) for label, link in entry.get("attachments", [])]

    def save(self) -> None:
        """Persist the manifest."""
        self._path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=1), encoding="utf-8"
        )


class PajakCrawler:
    """Walks the crawl plan and fills the corpus."""

    #: A content block shorter than this is navigation, not a page worth indexing.
    MINIMUM_TEXT_CHARACTERS = 150

    #: Pages between manifest saves, so an interrupted full crawl resumes
    #: close to where it stopped rather than at the start of the section.
    MANIFEST_SAVE_INTERVAL = 50

    def __init__(self, settings: Settings, guard: SourceTreeGuard) -> None:
        """Compose the fetcher, parser, writer and manifest a crawl needs."""
        self._settings = settings
        self._guard = guard
        self._page_limit: int | None = None

    def run(
        self,
        full: bool = False,
        section_names: list[str] | None = None,
        refresh: bool = False,
        page_limit: int | None = None,
        progress_callback=None,
    ) -> CrawlOutcome:
        """Crawl the selected sections and return what was written.

        Args:
            full: lift the sample limits and crawl every page of each listing.
            section_names: folders to crawl; default is every non-optional one.
            refresh: fetch pages again even when a snapshot already exists.
            page_limit: overrides the sample limits — at most this many pages
                per listing, fixed pages per section, and attachments per
                section. Useful for a quick first corpus.
        """
        self._page_limit = None if full else page_limit
        fetcher = PajakFetcher(self._settings)
        parser = PajakPageParser()
        writer = CorpusPageWriter(self._settings)
        manifest = CrawlManifest(self._settings, self._guard)
        outcome = CrawlOutcome()

        try:
            for section in self._select_sections(section_names):
                LOGGER.info("=== %s ===", section.folder)
                pages = self._collect_pages(section, fetcher, parser, full)
                LOGGER.info("%s: %d halaman", section.folder, len(pages))
                attachments: dict[str, str] = {}

                for index, page in enumerate(pages, start=1):
                    if not refresh and manifest.has_file_for(page.url):
                        outcome.pages_unchanged += 1
                        for label, link in manifest.attachments_of(page.url):
                            attachments.setdefault(link, label)
                        continue
                    try:
                        parsed = parser.parse(fetcher.get_text(page.url), page.url)
                        # Many FAQ entries are only a PDF, so attachments are
                        # queued even when the page itself has too little text.
                        for label, link in parsed.attachments:
                            attachments.setdefault(link, label)
                        if len(parsed.text) < self.MINIMUM_TEXT_CHARACTERS:
                            # A video page or a bare file link: nothing to index.
                            outcome.pages_skipped += 1
                            LOGGER.info("Dilewati (tanpa teks): %s", page.url)
                        else:
                            written = writer.write_page(section, page, parsed)
                            manifest.record(page.url, written, parsed.attachments)
                            outcome.pages_written += 1
                    except Exception as error:
                        outcome.failed += 1
                        outcome.errors.append(f"{page.url}: {error}")
                        LOGGER.warning("Gagal mengambil %s: %s", page.url, error)
                    if progress_callback is not None:
                        progress_callback(section.folder, index, len(pages), page.url)
                    if index % self.MANIFEST_SAVE_INTERVAL == 0:
                        manifest.save()

                self._download_attachments(
                    section, attachments, fetcher, writer, manifest, outcome, full, refresh
                )
                manifest.save()
        finally:
            manifest.save()
        return outcome

    def _select_sections(self, section_names: list[str] | None) -> list[CrawlSection]:
        """Return the sections to crawl, validating any names given."""
        if not section_names:
            return [section for section in CRAWL_PLAN if not section.optional]
        wanted = {name.strip().lower() for name in section_names}
        selected = [section for section in CRAWL_PLAN if section.folder.lower() in wanted]
        unknown = wanted - {section.folder.lower() for section in selected}
        if unknown:
            known = ", ".join(section.folder for section in CRAWL_PLAN)
            raise ValueError(f"Bagian tidak dikenal: {', '.join(sorted(unknown))}. Pilihan: {known}")
        return selected

    def _collect_pages(
        self, section: CrawlSection, fetcher: PajakFetcher, parser: PajakPageParser, full: bool
    ) -> list[ExternalPage]:
        """Return the pages of one section: its fixed pages plus listed details."""
        fixed_urls = section.fixed_urls
        if self._page_limit is not None:
            fixed_urls = fixed_urls[: self._page_limit]
        pages = {
            url: ExternalPage(url=url, label=f"pajak.go.id · {section.folder}", referenced_by="(tetap)")
            for url in fixed_urls
        }
        detail_pattern = re.compile(section.detail_pattern)
        limit = None if full else section.sample_limit_per_listing
        if self._page_limit is not None and limit:
            limit = min(limit, self._page_limit)

        for listing_url in section.listing_urls:
            if limit == 0:
                break
            for detail_url in self._walk_listing(
                listing_url, detail_pattern, section.paginate, limit, fetcher, parser
            ):
                pages.setdefault(
                    detail_url,
                    ExternalPage(
                        url=detail_url,
                        label=f"pajak.go.id · {section.folder}",
                        referenced_by=listing_url,
                    ),
                )
        return list(pages.values())

    def _walk_listing(
        self,
        listing_url: str,
        detail_pattern: re.Pattern,
        paginate: bool,
        limit: int | None,
        fetcher: PajakFetcher,
        parser: PajakPageParser,
    ) -> list[str]:
        """Follow one listing page by page until ``limit`` or the last page."""
        found: list[str] = []
        page_number = 0
        while limit is None or len(found) < limit:
            url = PajakUrl.with_page(listing_url, page_number) if paginate else listing_url
            try:
                links = parser.parse(fetcher.get_text(url), url).links
            except Exception as error:
                LOGGER.warning("Gagal membaca daftar %s: %s", url, error)
                break

            new_links = [
                link for link in links
                if detail_pattern.match(PajakUrl.path_of(link))
                and not PajakUrl.is_attachment(link)
                and link not in found
            ]
            # Detail links carry no query; a "?page=" link is the pager itself.
            new_links = [link for link in new_links if "?" not in link]
            if not new_links:
                break
            found.extend(new_links)
            if not paginate:
                break
            page_number += 1
        return found if limit is None else found[:limit]

    def _download_attachments(
        self,
        section: CrawlSection,
        attachments: dict[str, str],
        fetcher: PajakFetcher,
        writer: CorpusPageWriter,
        manifest: CrawlManifest,
        outcome: CrawlOutcome,
        full: bool,
        refresh: bool,
    ) -> None:
        """Download the attachments linked from a section's pages."""
        links = list(attachments)
        if not full:
            limit = section.sample_attachment_limit
            if self._page_limit is not None:
                limit = min(limit, self._page_limit)
            links = links[:limit]
        maximum_bytes = int(self._settings.pajak_attachment_maximum_megabytes * 1024 * 1024)

        for link in links:
            if not refresh and manifest.has_file_for(link):
                continue
            target = writer.attachment_target(section, link)
            try:
                size = fetcher.download(link, target, maximum_bytes)
                manifest.record(link, target)
                outcome.attachments_written += 1
                LOGGER.info("Lampiran tersimpan (%d KB): %s", size // 1024, target.name)
            except Exception as error:
                outcome.failed += 1
                outcome.errors.append(f"{link}: {error}")
                LOGGER.warning("Gagal mengunduh %s: %s", link, error)


#: Shared crawler bound to the process-wide settings and guard.
PAJAK_CRAWLER = PajakCrawler(SETTINGS, SOURCE_GUARD)
