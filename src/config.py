"""
Central configuration.
Loads env vars and lazily initializes LLM providers (Gemini → Groq fallback).
"""
import os
import sys
import warnings
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Suppress the Google GenAI SDK stderr message about AFC.
# ---------------------------------------------------------------------------
_AFC_PHRASES = (
    "automatic function calling",
    "Direct use of AFC",
    "AFC in Chat.send_message",
)


class _StderrAFCFilter:
    """Proxy for sys.stderr that drops Google GenAI AFC advisory lines."""

    def __init__(self, real: object) -> None:
        self._real = real
        self._pending = ""

    def write(self, text: str) -> int:
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if not any(phrase in line for phrase in _AFC_PHRASES):
                self._real.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._pending and not any(p in self._pending for p in _AFC_PHRASES):
            self._real.write(self._pending)
        self._pending = ""
        self._real.flush()

    def __getattr__(self, name: str):
        return getattr(self._real, name)


if not isinstance(sys.stderr, _StderrAFCFilter):
    sys.stderr = _StderrAFCFilter(sys.stderr)

warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", message=".*AFC.*")

# ── Gemini config ──────────────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Groq config ────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Cached LLM instances ──────────────────────────────────────────
_gemini_llm = None
_groq_llm = None


def gemini_available() -> bool:
    """Check if Gemini is configured (without raising)."""
    return bool(GOOGLE_API_KEY)


def groq_available() -> bool:
    """Check if Groq is configured (without raising)."""
    return bool(GROQ_API_KEY)


def get_llm(temperature: float = 0.2, max_output_tokens: int = 1024):
    """Return a cached ChatGoogleGenerativeAI instance.

    Raises RuntimeError if GOOGLE_API_KEY is not set.
    """
    global _gemini_llm
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. "
            "Copy .env.example to .env and add your key from "
            "https://aistudio.google.com/apikey"
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    if _gemini_llm is None:
        _gemini_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    return _gemini_llm


def get_groq_llm(temperature: float = 0.2, max_output_tokens: int = 1024):
    """Return a cached ChatGroq instance.

    Raises RuntimeError if GROQ_API_KEY is not set.
    """
    global _groq_llm
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set. "
            "Get a free key at https://console.groq.com/keys "
            "and add it to .env"
        )
    from langchain_groq import ChatGroq

    if _groq_llm is None:
        _groq_llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
    return _groq_llm


def get_best_available_llm(temperature: float = 0.2, max_output_tokens: int = 1024):
    """Return the best available LLM, falling back through Gemini → Groq.

    Raises RuntimeError only if neither provider is configured.
    """
    if gemini_available():
        return get_llm(temperature, max_output_tokens)
    if groq_available():
        return get_groq_llm(temperature, max_output_tokens)
    raise RuntimeError(
        "No LLM provider configured. Set either GOOGLE_API_KEY or "
        "GROQ_API_KEY in your .env file."
    )


def llm_provider_name() -> str:
    """Return the name of the currently active (or first available) provider."""
    if gemini_available():
        return "gemini"
    if groq_available():
        return "groq"
    return "none"
