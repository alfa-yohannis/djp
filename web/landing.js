/* ==========================================================================
   DJP RAG — landing page
   ==========================================================================
     SessionState      asks the server who is signed in and swaps the CTAs
     SignInCard        the shared admin/guest login form and its feedback
     OverviewStats     live index figures with a count-up
     AnswerDemo        types the sample answer once it scrolls into view
     ScrollEffects     sticky-nav tint and on-scroll reveals
   ========================================================================== */

'use strict';

document.documentElement.classList.remove('no-js');

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Reads a JSON response, raising the server's `detail` message on failure. */
async function readJson(response) {
  const parsed = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(parsed.detail || `Terjadi kesalahan (HTTP ${response.status}).`);
  }
  return parsed;
}

/* ------------------------------------------------------------ session */

class SessionState {
  /** Returns the current user, or null when nobody is signed in. */
  async load() {
    try {
      const user = await readJson(await fetch('/api/auth/me'));
      return user.signed_in ? user : null;
    } catch {
      return null;
    }
  }

  /** Shows the calls to action that match whether a session exists. */
  applyToPage(user) {
    const signedIn = Boolean(user);
    document.querySelectorAll('[data-signed-in]').forEach((element) => {
      element.classList.toggle('hidden', !signedIn);
    });
    document.querySelectorAll('[data-signed-out]').forEach((element) => {
      element.classList.toggle('hidden', signedIn);
    });
  }
}

/* ------------------------------------------------------------ sign-in */

class SignInCard {
  constructor() {
    this.card = document.getElementById('auth-card');
    this.form = document.getElementById('sign-in-form');
    this.usernameInput = document.getElementById('input-username');
    this.passwordInput = document.getElementById('input-password');

    this.form.addEventListener('submit', this.#handleSubmit.bind(this));
    document.getElementById('button-toggle-password')
      .addEventListener('click', this.#togglePasswordVisibility.bind(this));
    document.getElementById('button-sign-out')
      .addEventListener('click', this.#handleSignOut.bind(this));
    this.passwordInput.addEventListener('keyup', this.#updateCapsLockHint.bind(this));
    this.passwordInput.addEventListener('keydown', this.#updateCapsLockHint.bind(this));

    // Any "Masuk" link draws attention to the card instead of just scrolling.
    document.querySelectorAll('a[href="#masuk"]').forEach((link) => {
      link.addEventListener('click', () => this.focusCard());
    });
    if (window.location.hash === '#masuk') {
      window.setTimeout(() => this.focusCard(), 350);
    }
  }

  /** Paints the "already signed in" state. */
  showSignedIn(user) {
    const isAdmin = Boolean(user.is_admin);
    const name = user.username;
    document.getElementById('auth-signed-out').classList.add('hidden');
    document.getElementById('auth-signed-in').classList.remove('hidden');
    document.getElementById('auth-current-name').textContent = name;

    const avatar = document.getElementById('auth-avatar');
    avatar.textContent = name.charAt(0).toUpperCase();
    avatar.classList.toggle('guest', !isAdmin);

    const rolePill = document.getElementById('auth-current-role');
    rolePill.textContent = isAdmin ? 'Admin' : 'Guest';
    rolePill.classList.toggle('guest', !isAdmin);
  }

  /** Pulses the card and focuses its first field. */
  focusCard() {
    this.card.classList.remove('flash');
    void this.card.offsetWidth;
    this.card.classList.add('flash');
    if (this.usernameInput.offsetParent) {
      this.usernameInput.focus({ preventScroll: true });
    }
  }

  /** Signs in; the server decides from the account whether it is admin or guest. */
  async #handleSubmit(event) {
    event.preventDefault();
    const errorElement = document.getElementById('sign-in-error');
    const username = this.usernameInput.value.trim();
    const password = this.passwordInput.value;

    this.usernameInput.setAttribute('aria-invalid', String(!username));
    this.passwordInput.setAttribute('aria-invalid', String(!password));
    if (!username || !password) {
      this.#showError(errorElement, 'Isi nama pengguna dan kata sandi.');
      (username ? this.passwordInput : this.usernameInput).focus();
      return;
    }

    const button = document.getElementById('button-sign-in');
    this.#setLoading(button, true);
    errorElement.classList.add('hidden');
    try {
      await readJson(await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      }));
      this.#enterApplication(button);
    } catch (error) {
      this.#setLoading(button, false);
      this.passwordInput.value = '';
      this.passwordInput.setAttribute('aria-invalid', 'true');
      this.passwordInput.focus();
      this.#showError(errorElement, error.message);
    }
  }

  async #handleSignOut() {
    await fetch('/api/auth/logout', { method: 'POST' }).catch(() => null);
    window.location.replace('/');
  }

  #enterApplication(button) {
    button.querySelector('.btn-label').textContent = 'Berhasil — membuka aplikasi…';
    window.location.assign('/app');
  }

  #setLoading(button, isLoading) {
    button.disabled = isLoading;
    button.classList.toggle('is-loading', isLoading);
  }

  #showError(element, message) {
    element.textContent = message;
    element.classList.remove('hidden', 'shake');
    void element.offsetWidth;
    element.classList.add('shake');
  }

  #togglePasswordVisibility(event) {
    const button = event.currentTarget;
    const reveal = this.passwordInput.type === 'password';
    this.passwordInput.type = reveal ? 'text' : 'password';
    button.setAttribute('aria-pressed', String(reveal));
    button.setAttribute('aria-label', reveal ? 'Sembunyikan kata sandi' : 'Tampilkan kata sandi');
    this.passwordInput.focus();
  }

  #updateCapsLockHint(event) {
    if (typeof event.getModifierState !== 'function') return;
    document.getElementById('caps-hint')
      .classList.toggle('hidden', !event.getModifierState('CapsLock'));
  }
}

/* -------------------------------------------------------------- stats */

class OverviewStats {
  async load() {
    let overview;
    try {
      overview = await readJson(await fetch('/api/public/overview'));
    } catch {
      return;
    }

    document.querySelectorAll('[data-counter]').forEach((element) => {
      this.#countUp(element, Number(overview[element.dataset.counter] ?? 0));
    });
    document.getElementById('stat-updated').textContent = this.#relativeDate(overview.last_indexed_at);
  }

  #countUp(element, target) {
    const format = (value) => Math.round(value).toLocaleString('id-ID');
    if (prefersReducedMotion || target === 0) {
      element.textContent = format(target);
      return;
    }
    const duration = 1400;
    const startedAt = performance.now();
    const step = (now) => {
      const progress = Math.min(1, (now - startedAt) / duration);
      const eased = 1 - Math.pow(1 - progress, 4);
      element.textContent = format(target * eased);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  #relativeDate(isoText) {
    if (!isoText) return 'belum pernah';
    const parsed = new Date(isoText);
    if (Number.isNaN(parsed.getTime())) return '–';
    const days = Math.floor((Date.now() - parsed.getTime()) / 86_400_000);
    if (days <= 0) return 'Hari ini';
    if (days === 1) return 'Kemarin';
    if (days < 7) return `${days} hari lalu`;
    return parsed.toLocaleDateString('id-ID', { day: 'numeric', month: 'short', year: 'numeric' });
  }
}

/* --------------------------------------------------------------- demo */

class AnswerDemo {
  constructor() {
    this.answerElement = document.getElementById('demo-answer-text');
    this.sourcesElement = document.querySelector('.demo-sources');
    this.hasPlayed = false;
  }

  /** Plays once, the first time the demo scrolls into view. */
  observe() {
    if (prefersReducedMotion || !('IntersectionObserver' in window)) return;

    // Split the answer into text and inline elements, so chips appear whole
    // while the words around them are typed character by character.
    this.segments = Array.from(this.answerElement.childNodes).map((node) => node.cloneNode(true));
    this.answerElement.textContent = '';
    this.sourcesElement.classList.add('pending');

    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting) && !this.hasPlayed) {
        this.hasPlayed = true;
        observer.disconnect();
        this.#play();
      }
    }, { threshold: 0.45 });
    observer.observe(this.answerElement.closest('.demo'));
  }

  async #play() {
    this.answerElement.classList.add('is-typing');
    await this.#wait(500);
    for (const segment of this.segments) {
      if (segment.nodeType === Node.TEXT_NODE) {
        const textNode = document.createTextNode('');
        this.answerElement.appendChild(textNode);
        for (const character of segment.nodeValue) {
          textNode.nodeValue += character;
          await this.#wait(character === ' ' ? 14 : 22);
        }
      } else if (segment.classList && segment.classList.contains('chip')) {
        this.answerElement.appendChild(segment);
        await this.#wait(180);
      } else {
        // Inline emphasis: type its text inside the element.
        const shell = segment.cloneNode(false);
        this.answerElement.appendChild(shell);
        for (const character of segment.textContent) {
          shell.textContent += character;
          await this.#wait(26);
        }
      }
    }
    this.answerElement.classList.remove('is-typing');
    await this.#wait(200);
    this.sourcesElement.classList.remove('pending');
  }

  #wait(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }
}

/* ------------------------------------------------------------- scroll */

class ScrollEffects {
  start() {
    const nav = document.getElementById('nav');
    const updateNav = () => nav.classList.toggle('is-scrolled', window.scrollY > 8);
    updateNav();
    window.addEventListener('scroll', updateNav, { passive: true });

    const revealables = document.querySelectorAll('.reveal');
    if (prefersReducedMotion || !('IntersectionObserver' in window)) {
      revealables.forEach((element) => element.classList.add('is-visible'));
      return;
    }

    // Siblings revealed together are staggered so a grid cascades in.
    const groups = new Map();
    revealables.forEach((element) => {
      const siblings = groups.get(element.parentElement) || [];
      element.style.setProperty('--reveal-delay', `${Math.min(siblings.length, 5) * 70}ms`);
      siblings.push(element);
      groups.set(element.parentElement, siblings);
    });

    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
    revealables.forEach((element) => observer.observe(element));
  }
}

/* -------------------------------------------------------------- start */

document.addEventListener('DOMContentLoaded', async function startLandingPage() {
  document.getElementById('footer-year').textContent = String(new Date().getFullYear());

  new ScrollEffects().start();
  new AnswerDemo().observe();
  const signInCard = new SignInCard();
  new OverviewStats().load();

  /* ===== DEMO PAMERAN — HAPUS SELURUH BLOK INI SETELAH PAMERAN =====
     Pasangannya: blok bertanda sama di landing.html dan landing.css. */
  const fillGuestButton = document.getElementById('button-fill-guest');
  if (fillGuestButton) {
    fillGuestButton.addEventListener('click', function fillGuestCredentials() {
      document.getElementById('input-username').value = 'guest';
      document.getElementById('input-password').value = 'i am guest';
      document.getElementById('button-sign-in').focus();
    });
  }
  /* ===== AKHIR DEMO PAMERAN ===== */

  const session = new SessionState();
  const user = await session.load();
  session.applyToPage(user);
  if (user) signInCard.showSignedIn(user);
});
