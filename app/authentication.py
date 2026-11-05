"""Sign-in, roles and session cookies for the web interface.

Two roles exist, and only two:

* ``admin`` — may do everything, including updating and rebuilding the index.
* ``guest`` — may ask questions and browse, but the server refuses any request
  that would change the index.

Each role has one account with its own username and password. The login form
is shared: the credentials alone decide which role a visitor receives.

Sessions are stateless signed tokens carried in an ``HttpOnly`` cookie. A token
is ``role|username|expiry`` followed by an HMAC-SHA256 signature, so the server
needs no session table and a restart does not sign anyone out — the signing
secret is persisted under the data directory. The standard library is enough
for this; no extra dependency is pulled in.

Passwords are stored only as PBKDF2 hashes. Permission checks live on
the server: hiding a button in the browser is a courtesy, not a control.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import SETTINGS, Settings

LOGGER = logging.getLogger(__name__)


class Role:
    """The two roles the interface recognises."""

    ADMIN = "admin"
    GUEST = "guest"

    ALL = frozenset({ADMIN, GUEST})


@dataclass(frozen=True)
class Principal:
    """Who is making a request, as proven by a valid session cookie."""

    username: str
    role: str
    expires_at: int

    @property
    def is_admin(self) -> bool:
        """True when this principal may change the index."""
        return self.role == Role.ADMIN

    def as_dictionary(self) -> dict:
        """JSON-ready view used by ``/api/auth/me``."""
        return {
            "username": self.username,
            "role": self.role,
            "is_admin": self.is_admin,
            "expires_at": self.expires_at,
        }


class PasswordHasher:
    """PBKDF2-SHA256 hashing in the ``pbkdf2_sha256$iterations$salt$digest`` form."""

    ALGORITHM = "pbkdf2_sha256"
    ITERATIONS = 600_000

    def hash(self, password: str) -> str:
        """Return a new salted hash of ``password``."""
        salt = secrets.token_hex(16)
        digest = self._derive(password, salt, self.ITERATIONS)
        return f"{self.ALGORITHM}${self.ITERATIONS}${salt}${digest}"

    def verify(self, password: str, encoded_hash: str) -> bool:
        """True when ``password`` matches ``encoded_hash``; malformed hashes fail."""
        try:
            algorithm, iterations_text, salt, expected_digest = encoded_hash.split("$")
            iterations = int(iterations_text)
        except ValueError:
            return False
        if algorithm != self.ALGORITHM:
            return False
        actual_digest = self._derive(password, salt, iterations)
        return hmac.compare_digest(actual_digest, expected_digest)

    def _derive(self, password: str, salt_hex: str, iterations: int) -> str:
        """Run PBKDF2 and return the digest as hex."""
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), iterations
        ).hex()


class SessionTokenSigner:
    """Creates and verifies HMAC-signed session tokens."""

    def __init__(self, secret: bytes) -> None:
        """Bind the signer to the key every token is signed with."""
        self._secret = secret

    def issue(self, username: str, role: str, lifetime_seconds: int) -> str:
        """Return a signed token for ``username`` acting as ``role``."""
        expires_at = int(time.time()) + lifetime_seconds
        body = f"{role}|{username}|{expires_at}"
        encoded_body = base64.urlsafe_b64encode(body.encode("utf-8")).decode("ascii")
        return f"{encoded_body}.{self._sign(encoded_body)}"

    def verify(self, token: str | None) -> Principal | None:
        """Return the principal a token proves, or None if it is forged or stale."""
        if not token or token.count(".") != 1:
            return None
        encoded_body, signature = token.split(".")
        if not hmac.compare_digest(signature, self._sign(encoded_body)):
            return None
        try:
            body = base64.urlsafe_b64decode(encoded_body.encode("ascii")).decode("utf-8")
            role, username, expires_text = body.split("|")
            expires_at = int(expires_text)
        except (ValueError, UnicodeDecodeError):
            return None
        if role not in Role.ALL or expires_at < time.time():
            return None
        return Principal(username=username, role=role, expires_at=expires_at)

    def _sign(self, encoded_body: str) -> str:
        """Return the URL-safe HMAC-SHA256 signature of ``encoded_body``."""
        mac = hmac.new(self._secret, encoded_body.encode("ascii"), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")


class LoginThrottle:
    """Slows password guessing by locking an address out after repeated failures."""

    def __init__(self, maximum_failures: int = 5, lockout_seconds: int = 60) -> None:
        """Configure how many failures are tolerated and for how long to lock."""
        self._maximum_failures = maximum_failures
        self._lockout_seconds = lockout_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def seconds_locked(self, client_address: str) -> int:
        """Seconds remaining before ``client_address`` may try again (0 = allowed)."""
        with self._lock:
            recent = self._recent_failures(client_address)
            if len(recent) < self._maximum_failures:
                return 0
            return max(1, int(recent[-1] + self._lockout_seconds - time.time()))

    def record_failure(self, client_address: str) -> None:
        """Count one failed attempt from ``client_address``."""
        with self._lock:
            recent = self._recent_failures(client_address)
            recent.append(time.time())
            self._failures[client_address] = recent

    def record_success(self, client_address: str) -> None:
        """Forget earlier failures once the address signs in correctly."""
        with self._lock:
            self._failures.pop(client_address, None)

    def _recent_failures(self, client_address: str) -> list[float]:
        """Failures from ``client_address`` still inside the lockout window."""
        cutoff = time.time() - self._lockout_seconds
        return [moment for moment in self._failures.get(client_address, []) if moment > cutoff]


class AuthenticationService:
    """Checks credentials and turns them into session tokens."""

    #: Name of the cookie carrying the session token.
    COOKIE_NAME = "djp_session"

    def __init__(self, settings: Settings) -> None:
        """Load the signing secret and the account credentials from configuration."""
        self._settings = settings
        self._hasher = PasswordHasher()
        self._signer = SessionTokenSigner(self._load_or_create_secret())
        self.throttle = LoginThrottle()

    @property
    def session_lifetime_seconds(self) -> int:
        """How long a sign-in lasts before the visitor must sign in again."""
        return self._settings.session_lifetime_hours * 3600

    def _accounts(self) -> list[tuple[str, str, str]]:
        """Every account as ``(username, password_hash, role)``."""
        return [
            (self._settings.admin_username, self._settings.admin_password_hash, Role.ADMIN),
            (self._settings.guest_username, self._settings.guest_password_hash, Role.GUEST),
        ]

    def authenticate(self, username: str, password: str) -> str | None:
        """Return a session token for correct credentials, else None.

        The role comes from whichever account the credentials match.
        """
        submitted_username = username.strip().lower()
        matched: tuple[str, str] | None = None
        # Every account's password is checked, so the response time does not
        # reveal which usernames exist or which part was wrong.
        for account_username, password_hash, role in self._accounts():
            username_matches = hmac.compare_digest(submitted_username, account_username.lower())
            password_matches = self._hasher.verify(password, password_hash)
            if username_matches and password_matches and matched is None:
                matched = (account_username, role)
        if matched is None:
            return None
        return self._signer.issue(matched[0], matched[1], self.session_lifetime_seconds)

    def identify(self, token: str | None) -> Principal | None:
        """Return the principal behind a session cookie, if it is valid."""
        return self._signer.verify(token)

    def _load_or_create_secret(self) -> bytes:
        """Use the configured secret, or one persisted under the data directory.

        Persisting it means restarting the server keeps everyone signed in; the
        file lives in ``data/``, which is already outside version control.
        """
        if self._settings.session_secret:
            return self._settings.session_secret.encode("utf-8")

        secret_path: Path = self._settings.data_directory / "session-secret.key"
        try:
            if secret_path.exists():
                stored = secret_path.read_text(encoding="ascii").strip()
                if stored:
                    return stored.encode("ascii")
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            generated = secrets.token_urlsafe(48)
            secret_path.write_text(generated, encoding="ascii")
            return generated.encode("ascii")
        except OSError:
            LOGGER.warning(
                "Tidak dapat menyimpan kunci sesi; semua pengguna akan keluar saat server dimulai ulang.",
                exc_info=True,
            )
            return secrets.token_bytes(48)


#: Process-wide authentication service shared by the web application.
AUTHENTICATION_SERVICE = AuthenticationService(SETTINGS)
