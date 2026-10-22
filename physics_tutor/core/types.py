"""
Core data types shared by every part of the tutor.

The flow of data is:

    QuestionTemplate  --(generator.generate)-->  Question  --(app.py)-->  screen
          |                                         |
     a recipe for a family of                one concrete question:
     questions, with random                  prompt text plus a list of
     integer parameters                      worked-solution Steps

A template never stores a specific question. It holds functions that turn a
dict of integer parameters (e.g. {"a": 3, "b": -2}) into the prompt text and
the worked solution, so each new set of random parameters gives a new question.
"""
from dataclasses import dataclass, field
from typing import Callable, Any, Optional


@dataclass
class Step:
    """A single step in a worked solution, shown as one card in the app.

    Each card is rendered top to bottom as: label, then latex, then note.
    """
    label: str  # Heading for the step. Markdown, so inline maths like "$f(x)$" works.
    latex: str  # The maths for the step, as raw LaTeX (no $ signs). Shown centred via st.latex.
    note: str = ""  # Optional explanation shown under the maths. Markdown, so $...$ works.
    image: Optional[bytes] = None  # Optional PNG (e.g. a sketch from core.plot), shown between latex and note.


@dataclass
class Question:
    """A fully generated question: what the student sees, plus the full solution.

    Built by generator.generate() from a QuestionTemplate and one set of
    random parameters. SymPy does all the maths inside the template's
    solver, so the steps are correct by construction.
    """
    topic: str       # Top-level topic, e.g. "ODEs" (matches the sidebar radio buttons)
    subtopic: str    # e.g. "Separable: dy/dx = f(x)" (matches the sidebar dropdown)
    difficulty: int  # 1-5; stored for future use, not currently displayed
    prompt: str      # The question text shown to the student (Markdown with $...$ / $$...$$ maths)
    params: dict     # The random integers this question was built from, e.g. {"a": 3, "b": -2}
    steps: list[Step]  # Worked solution, revealed one step at a time in the app
    case: str = ""     # Copied from the template's `case` (e.g. "Complex roots"); "" if none
    image: Optional[bytes] = None  # Optional PNG shown under the prompt (from the template's figure_fn)


@dataclass
class QuestionTemplate:
    """
    A recipe for one family of questions, e.g. "find the gradient of
    f = a x^2 y + b yz + c xz^2".

    - `parameters` gives the integer range each random parameter is drawn from.
    - `constraints` rule out bad draws (e.g. a coefficient of 0 that would make
      the question trivial). A draw is accepted only if every constraint returns True.
    - `prompt_fn` turns the parameters into the question text.
    - `solver` turns the same parameters into the list of Steps. It must
      return list[Step].
    """
    id: str          # Unique, stable identifier, e.g. "vc-gradient-polynomial"
    topic: str       # Sidebar topic this template belongs to
    subtopic: str    # Sidebar subtopic; templates sharing a subtopic are picked between at random
    difficulty: int  # 1-5
    prompt_fn: Callable[[dict], str]       # params -> prompt string
    solver: Callable[[dict], list[Step]]   # params -> worked-solution steps
    parameters: dict[str, tuple]           # {"a": (min, max), ...}, both ends inclusive
    constraints: list[Callable] = field(default_factory=list)  # each: params -> bool (True = acceptable)
    max_attempts: int = 50                 # how many random draws to try before giving up on the constraints
    # Optional label for a sub-case within the subtopic, e.g. "Complex roots" for the
    # 2nd-order ODEs. When any template in a subtopic has a case, the app shows a
    # selector on the main page, and only templates matching the chosen case are used.
    case: str = ""
    # Optional text shown on the main page before a case is chosen, explaining
    # what the cases are. "" = the app's default (about types of roots).
    case_intro: str = ""
    # Optional params -> PNG bytes (e.g. from core.plot), shown with the question
    # so the student can see it while working. None = no figure.
    figure_fn: Optional[Callable[[dict], bytes]] = None
