"""
Turns a QuestionTemplate into a concrete Question.

Two stages:
  1. sample_params: draw random integers for each template parameter, retrying
     until every constraint is satisfied.
  2. generate: feed those parameters to the template's solver (SymPy works out
     the steps) and prompt_fn (the question text), then package the result.
"""
from __future__ import annotations

import random
from .types import QuestionTemplate, Question, Step


def sample_params(template: QuestionTemplate, rng: random.Random = None) -> dict | None:
    """
    Sample integer parameters for a template, respecting constraints.
    Returns None if constraints cannot be satisfied within max_attempts.

    Each parameter is drawn uniformly from its inclusive (lo, hi) range. The
    whole set is redrawn whenever any constraint fails, i.e. rejection
    sampling. This is fine because constraints only exclude a small
    fraction of draws (zero coefficients, resonance, etc.).
    """
    if rng is None:
        rng = random.Random()

    for _ in range(template.max_attempts):
        params = {
            k: rng.randint(lo, hi)
            for k, (lo, hi) in template.parameters.items()
        }
        # exclude zero from ranges that straddle it unless explicitly allowed
        # (the templates do this through their own constraints, e.g. p["a"] != 0)
        if all(c(params) for c in template.constraints):
            return params

    return None


def generate(template: QuestionTemplate, seed: int = None) -> Question | None:
    """
    Generate a Question from a template.
    SymPy computation happens inside template.solver — that IS the verification.
    Returns None if param sampling fails.

    The same seed always gives the same question: the seed fixes the random
    parameters, and the solver and prompt are deterministic given the
    parameters.
    """
    # A private RNG seeded per question, so generation never disturbs (or is
    # disturbed by) the global `random` module used elsewhere in the app.
    rng = random.Random(seed)
    params = sample_params(template, rng)
    if params is None:
        return None

    try:
        steps = template.solver(params)
        prompt = template.prompt_fn(params)
        image = template.figure_fn(params) if template.figure_fn else None
    except Exception as e:
        # SymPy failed — treat as unresolvable, don't surface to student.
        # Several solvers also raise deliberately when their own SymPy check
        # fails (e.g. a solution that doesn't satisfy its ODE). That surfaces
        # here too, and the app shows it as "Generation failed".
        raise RuntimeError(f"Solver failed for template '{template.id}' with params {params}: {e}")

    return Question(
        topic=template.topic,
        subtopic=template.subtopic,
        difficulty=template.difficulty,
        prompt=prompt,
        params=params,
        steps=steps,
        case=template.case,
        image=image,
    )
