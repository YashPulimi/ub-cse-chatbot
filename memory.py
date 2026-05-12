"""
memory.py — Dialogue State Management for UB CSE Chatbot
=========================================================
Manages short-term conversation memory and optional personalization.

Responsibilities:
  1. Store conversation turns (user + assistant) up to cfg.memory.max_turns
  2. Resolve coreferences: "that course", "his office hours", "the same prof"
  3. Summarize older turns to stay within the LLM context window
  4. Optionally persist important facts per-user (personalization)
  5. Build the conversation history string injected into the generator prompt

Architecture:
  - ConversationMemory: per-session state (turns, entities, summary)
  - PersonalMemory: optional cross-session user profile (name, program, etc.)
  - MemoryManager: facade used by app.py — one instance per Chainlit session

Coreference resolution (lightweight, no NLP):
  Tracks the last-mentioned entity of each type in the conversation:
    last_course  → "CSE 574"  (for "what are its prerequisites?")
    last_faculty → "Rohini Srihari"  (for "what are her office hours?")
    last_program → "MS program"  (for "what are its requirements?")

  When a query contains a pronoun/demonstrative without a clear referent,
  the resolver substitutes the last known entity of the appropriate type.
  Example:
    Turn 1: "Who teaches CSE 574?"    → last_course = "CSE 574"
    Turn 2: "What are its prereqs?"   → resolved to "What are CSE 574's prereqs?"

Summarization:
  Every cfg.memory.summary_every turns, older turns are summarized into a
  one-paragraph string. The summary replaces the raw turns in the prompt
  context while keeping recent turns verbatim.
  This keeps the LLM context window under cfg.llm.context_window tokens.

Personalization (bonus requirement):
  On first interaction, the bot asks if the user wants to personalize.
  If yes, it extracts and stores:
    - Name, program level (MS/PhD/BS), areas of interest, known courses
  Stored in Chainlit session state — not persisted to disk by default
  (no PII stored without explicit user consent).

HOW TO USE:
  from memory import MemoryManager
  mem = MemoryManager(session_id="user_abc")
  mem.add_turn("user", "Who teaches CSE 574?")
  mem.add_turn("assistant", "CSE 574 is taught by Mingchen Gao.")
  resolved = mem.resolve(\"What are his office hours?\")
  # → "What are Mingchen Gao's office hours?"
  history  = mem.build_history_string()
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from config import cfg
from utils import get_logger, extract_course_codes

log = get_logger(__name__)


# ── Turn dataclass ────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role:      str    # "user" | "assistant"
    content:   str
    timestamp: float  = field(default_factory=time.time)
    entities:  dict   = field(default_factory=dict)  # extracted from this turn


# ── Coreference patterns ──────────────────────────────────────────────────────

# Pronouns / demonstratives that likely refer to the last-mentioned entity
_COURSE_REF_RE = re.compile(
    r"\b(it|its|the course|that course|this course|the class|that class"
    r"|the subject|this subject)\b",
    re.I,
)
_FACULTY_REF_RE = re.compile(
    r"\b(he|she|they|his|her|their|the professor|that professor"
    r"|the instructor|that instructor|the faculty|them)\b",
    re.I,
)
_PROGRAM_REF_RE = re.compile(
    r"\b(the program|that program|this program|the degree|that degree"
    r"|the curriculum|it|its)\b",
    re.I,
)

# Entity extractors from text
_FACULTY_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b")


def _extract_entities(text: str) -> dict:
    """
    Extract named entities from a text string for coreference tracking.
    Returns dict with keys: courses, faculty_names
    """
    return {
        "courses":       extract_course_codes(text),
        "faculty_names": _FACULTY_NAME_RE.findall(text),
    }


# ── Conversation memory ───────────────────────────────────────────────────────

class ConversationMemory:
    """
    Per-session conversation state.
    Tracks turns, last-mentioned entities, and optional summary.
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id   = session_id
        self.turns:       list[Turn] = []
        self.summary:     str        = ""   # summarized older turns
        self.summary_turn: int       = 0    # turn index when summary was last made

        # Last-mentioned entities (for coreference resolution)
        self.last_course:  str = ""
        self.last_faculty: str = ""
        self.last_program: str = ""

        # Personalization flag
        self.personalization_asked: bool = False
        self.personalization_on:    bool = False

    # ── Turn management ───────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        """Add a turn and update entity tracking."""
        entities = _extract_entities(content)
        turn     = Turn(role=role, content=content, entities=entities)
        self.turns.append(turn)

        # Update last-mentioned entities
        if entities["courses"]:
            self.last_course = entities["courses"][-1]
        if entities["faculty_names"]:
            # Filter out common false positives
            names = [n for n in entities["faculty_names"]
                     if n.lower() not in {"buffalo", "computer science",
                                          "university", "cse department"}]
            if names:
                self.last_faculty = names[-1]

        # Program detection
        prog_re = re.compile(r"\b(MS|PhD|Bachelor|BS|undergraduate|graduate)\b"
                             r"[\s\w]{0,20}(program|degree|curriculum)?", re.I)
        pm = prog_re.search(content)
        if pm:
            self.last_program = pm.group(0).strip()

        # Trim to max_turns (keep most recent)
        if len(self.turns) > cfg.memory.max_turns:
            excess = len(self.turns) - cfg.memory.max_turns
            self.turns = self.turns[excess:]

        log.debug(
            "Memory turn added: role=%s  last_course=%s  last_faculty=%s",
            role, self.last_course, self.last_faculty,
        )

    # ── Coreference resolution ────────────────────────────────────────────────

    def resolve(self, query: str) -> str:
        """
        Substitute coreferences in query with last-known entities.
        Only substitutes when the query has no explicit entity of that type.

        Examples:
          "What are its prerequisites?" + last_course="CSE 574"
              → "What are CSE 574's prerequisites?"
          "What are her office hours?" + last_faculty="Rohini Srihari"
              → "What are Rohini Srihari's office hours?"
        """
        resolved = query
        query_courses = extract_course_codes(query)
        query_names   = _FACULTY_NAME_RE.findall(query)

        # Course coreference: substitute if no course code in query
        if not query_courses and self.last_course:
            if _COURSE_REF_RE.search(resolved):
                resolved = _COURSE_REF_RE.sub(self.last_course, resolved)
                log.debug("Coref resolved course: %r → %r", query, resolved)

        # Faculty coreference: substitute if no faculty name in query
        if not query_names and self.last_faculty:
            if _FACULTY_REF_RE.search(resolved):
                resolved = _FACULTY_REF_RE.sub(self.last_faculty, resolved)
                log.debug("Coref resolved faculty: %r → %r", query, resolved)

        return resolved

    # ── History string for LLM prompt ────────────────────────────────────────

    def build_history_string(self, max_turns: int | None = None) -> str:
        """
        Build the conversation history string injected into the LLM prompt.
        Uses last 4 turns only to keep context tight for small models.
        """
        n      = min(max_turns or 4, 4)   # cap at 4 turns for 3B model
        lines  = []

        if self.summary:
            lines.append(f"[Earlier: {self.summary}]")

        recent = self.turns[-n:]
        for turn in recent:
            role    = "User" if turn.role == "user" else "Assistant"
            content = turn.content.strip()
            if turn.role == "assistant" and len(content) > 250:
                content = content[:250] + "…"
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    # ── Summarization ─────────────────────────────────────────────────────────

    def maybe_summarize(self) -> bool:
        """
        Summarize older turns if we've hit the summary threshold.
        Returns True if a summary was generated.
        Called automatically by MemoryManager after each assistant turn.
        """
        every = cfg.memory.summary_every
        n     = len(self.turns)

        if n < every or (n - self.summary_turn) < every:
            return False

        # Turns to summarize: all but the most recent (every // 2) turns
        keep_recent  = every // 2
        to_summarize = self.turns[:-keep_recent]
        if not to_summarize:
            return False

        # Build a short extractive summary (no LLM call — keep it fast)
        summary_lines = []
        last_c  = ""
        last_f  = ""
        topics  = []

        for turn in to_summarize:
            e = turn.entities
            if e.get("courses"):
                last_c = e["courses"][-1]
                topics.append(f"courses ({', '.join(e['courses'][:3])})")
            if e.get("faculty_names"):
                last_f = e["faculty_names"][-1]
                topics.append(f"faculty ({e['faculty_names'][0]})")

        if last_c:
            summary_lines.append(f"discussed course {last_c}")
        if last_f:
            summary_lines.append(f"discussed faculty member {last_f}")
        if topics:
            unique = list(dict.fromkeys(topics))[:4]
            summary_lines.append(f"topics: {'; '.join(unique)}")

        self.summary      = "User and assistant " + ", ".join(summary_lines) + "."
        self.summary_turn = n
        # Drop summarized turns, keep recent ones
        self.turns        = self.turns[-keep_recent:]
        log.info("Memory summarized: %s", self.summary)
        return True

    # ── Personalization prompt ────────────────────────────────────────────────

    def should_ask_personalize(self) -> bool:
        """Disabled — personalization prompt interrupts flow."""
        return False

    def personalization_prompt(self) -> str:
        return (
            "Would you like me to personalize my responses? "
            "If you share your program (MS/PhD/BS) and areas of interest, "
            "I can tailor course and faculty suggestions. "
            "Reply **yes** to personalize or **no** to continue."
        )


# ── Personal memory (cross-session user profile) ──────────────────────────────

@dataclass
class PersonalMemory:
    """
    Optional user profile extracted from conversation.
    Stored in Chainlit session state — not persisted to disk.
    """
    name:           str        = ""
    program:        str        = ""   # "MS" | "PhD" | "BS"
    interests:      list[str]  = field(default_factory=list)
    known_courses:  list[str]  = field(default_factory=list)
    advisor:        str        = ""
    raw_statements: list[str]  = field(default_factory=list)

    def extract_from(self, text: str) -> None:
        """
        Extract personal facts from user message.
        Called when personalization is ON.
        """
        # Program level
        prog_m = re.search(r"\b(MS|PhD|Ph\.D|Bachelor|BS|undergrad)\b", text, re.I)
        if prog_m and not self.program:
            self.program = prog_m.group(1).upper().replace("PH.D", "PhD")

        # Name: "I'm [Name]" / "my name is [Name]"
        name_m = re.search(
            r"(?:I(?:'m| am)|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            text, re.I,
        )
        if name_m and not self.name:
            self.name = name_m.group(1).strip()

        # Known courses
        for code in extract_course_codes(text):
            if code not in self.known_courses:
                self.known_courses.append(code)

        # Interests: "interested in X", "working on X", "focusing on X"
        interest_m = re.findall(
            r"(?:interested in|working on|focusing on|studying)\s+([\w\s]{3,40}?)(?:\.|,|$)",
            text, re.I,
        )
        for interest in interest_m:
            clean = interest.strip().lower()
            if clean not in self.interests and len(clean) > 3:
                self.interests.append(clean)

        self.raw_statements.append(text[:200])

    def context_string(self) -> str:
        """
        Build a short context string injected at the top of the LLM prompt
        when personalization is on.
        """
        parts = []
        if self.name:
            parts.append(f"User's name: {self.name}")
        if self.program:
            parts.append(f"Program: {self.program}")
        if self.interests:
            parts.append(f"Interests: {', '.join(self.interests[:4])}")
        if self.known_courses:
            parts.append(f"Courses taken/mentioned: {', '.join(self.known_courses[:6])}")
        if self.advisor:
            parts.append(f"Advisor: {self.advisor}")
        if not parts:
            return ""
        return "[User profile: " + " | ".join(parts) + "]"

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "program":       self.program,
            "interests":     self.interests,
            "known_courses": self.known_courses,
            "advisor":       self.advisor,
        }


# ── Memory manager (facade used by app.py) ───────────────────────────────────

class MemoryManager:
    """
    Top-level memory interface used by app.py.
    One instance per Chainlit session.

    Usage:
        mem = MemoryManager(session_id=cl.user_session.get("id"))
        resolved = mem.process_user_turn(user_message)
        # ... retrieve and generate ...
        mem.add_assistant_turn(assistant_response)
        history  = mem.build_prompt_context()
    """

    def __init__(self, session_id: str = "") -> None:
        self.conversation = ConversationMemory(session_id=session_id)
        self.personal     = PersonalMemory()
        self._pending_personalize = False

    # ── Public API ────────────────────────────────────────────────────────────

    def process_user_turn(self, message: str) -> str:
        """
        Called before retrieval on every user message.
        1. Handles personalization consent response if pending
        2. Extracts personal facts if personalization is ON
        3. Resolves coreferences
        4. Adds turn to memory
        Returns the (possibly resolved) query to pass to the retriever.
        """
        # Handle pending personalization consent
        if self._pending_personalize:
            return self._handle_personalize_response(message)

        # Extract personal facts if opted in
        if self.conversation.personalization_on:
            self.personal.extract_from(message)

        # Resolve coreferences before adding to memory
        resolved = self.conversation.resolve(message)
        self.conversation.add_turn("user", resolved)

        return resolved

    def add_assistant_turn(self, response: str) -> None:
        """Called after generation with the assistant's response."""
        self.conversation.add_turn("assistant", response)
        self.conversation.maybe_summarize()

    def build_prompt_context(self) -> str:
        """
        Build the full context block for the LLM prompt.
        Includes: personal profile (if on) + conversation history.
        """
        parts = []

        # Personal context
        if self.conversation.personalization_on:
            ctx = self.personal.context_string()
            if ctx:
                parts.append(ctx)

        # Conversation history
        history = self.conversation.build_history_string()
        if history:
            parts.append(history)

        return "\n".join(parts)

    def should_ask_personalize(self) -> bool:
        return self.conversation.should_ask_personalize()

    def get_personalize_prompt(self) -> str:
        self._pending_personalize = True
        self.conversation.personalization_asked = True
        return self.conversation.personalization_prompt()

    # ── Personalization consent handler ───────────────────────────────────────

    def _handle_personalize_response(self, message: str) -> str:
        """
        Interprets user's yes/no response to personalization prompt.
        Returns a canned response — does not go to retriever.
        """
        self._pending_personalize = False
        msg_lower = message.strip().lower()

        if any(w in msg_lower for w in ("yes", "sure", "ok", "yeah", "yep", "please")):
            self.conversation.personalization_on = True
            log.info("Personalization enabled for session %s",
                     self.conversation.session_id)
            # Signal to app.py to return this message directly
            return "__PERSONALIZE_YES__"

        # Only treat as "no" if it's clearly a decline
        # If message looks like a real question, don't intercept it
        _decline_words = ("no", "nope", "nah", "skip", "don't", "dont",
                          "not now", "later", "no thanks", "no thank")
        _is_decline = any(w in msg_lower for w in _decline_words)
        _is_question = any(w in msg_lower for w in
                           ("who", "what", "when", "where", "how", "which",
                            "cse", "course", "professor", "faculty", "teach"))

        if _is_question and not _is_decline:
            # Real question slipped through — reset pending and process normally
            self._pending_personalize = False
            self.conversation.personalization_asked = True
            self.conversation.personalization_on = False
            return message  # let it go to the retriever as a normal query

        self.conversation.personalization_on = False
        log.info("Personalization declined for session %s",
                 self.conversation.session_id)
        return "__PERSONALIZE_NO__"

    # ── State inspection ──────────────────────────────────────────────────────

    def state_summary(self) -> dict:
        """Return a summary of current memory state for debugging / UI."""
        return {
            "session_id":        self.conversation.session_id,
            "turns":             len(self.conversation.turns),
            "has_summary":       bool(self.conversation.summary),
            "last_course":       self.conversation.last_course,
            "last_faculty":      self.conversation.last_faculty,
            "last_program":      self.conversation.last_program,
            "personalization":   self.conversation.personalization_on,
            "personal_profile":  self.personal.to_dict(),
        }

    def reset(self) -> None:
        """Clear all memory for this session."""
        self.conversation = ConversationMemory(self.conversation.session_id)
        self.personal     = PersonalMemory()
        self._pending_personalize = False
        log.info("Memory reset for session %s", self.conversation.session_id)


# ── Singleton registry (one MemoryManager per session_id) ────────────────────

_sessions: dict[str, MemoryManager] = {}


def get_memory(session_id: str) -> MemoryManager:
    """
    Return (or create) a MemoryManager for a given session ID.
    Called by app.py with the Chainlit session ID.
    """
    if session_id not in _sessions:
        _sessions[session_id] = MemoryManager(session_id=session_id)
        log.info("New memory session: %s", session_id)
    return _sessions[session_id]


def clear_memory(session_id: str) -> None:
    """Remove a session's memory (called on disconnect)."""
    if session_id in _sessions:
        del _sessions[session_id]
        log.info("Memory cleared for session: %s", session_id)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Interactive demo of memory and coreference resolution."""
    print("\n🧠  Memory Manager demo\n")

    mem = MemoryManager(session_id="demo")

    turns = [
        ("user",      "Who teaches CSE 574?"),
        ("assistant", "CSE 574 is taught by Mingchen Gao (mgao8@buffalo.edu)."),
        ("user",      "What are his office hours?"),
        ("assistant", "Mingchen Gao holds office hours on Tuesdays 2-4pm in Davis 338."),
        ("user",      "What are its prerequisites?"),
        ("assistant", "CSE 574 requires CSE 474 or MTH 309."),
        ("user",      "Tell me about the MS program requirements"),
        ("assistant", "The MS program requires 30 credits including core courses."),
        ("user",      "How many credits does it require?"),
    ]

    for role, content in turns:
        if role == "user":
            resolved = mem.process_user_turn(content)
            print(f"User (original):  {content}")
            if resolved != content:
                print(f"User (resolved):  {resolved}")
        else:
            mem.add_assistant_turn(content)
            print(f"Assistant:        {content[:80]}...")

    print(f"\nState: {json.dumps(mem.state_summary(), indent=2)}")
    print(f"\nPrompt context:\n{mem.build_prompt_context()}")


if __name__ == "__main__":
    import json
    main()