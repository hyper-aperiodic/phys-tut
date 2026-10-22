# Physics Tutor

SymPy-verified question generator for first-year mathematical physics.

## Topics
- **Vector Calculus** — gradient, divergence, curl, line integrals
- **ODEs** — separable, 2nd-order homogeneous, non-homogeneous, Laplace transforms
- **Fourier Analysis** — square wave series, triangle wave series, Gaussian FT

## Run

```bash
pip install sympy streamlit
streamlit run app.py
```

## Structure

```
app.py                          # Streamlit UI
physics_tutor/
  core/
    types.py                    # Step, Question, QuestionTemplate dataclasses
    generator.py                # Parameter sampler + question generator
  templates/
    vector_calculus.py          # VC templates + solvers
    odes.py                     # ODE templates + solvers
    fourier.py                  # Fourier templates + solvers
    __init__.py                 # Registry (ALL_TEMPLATES, TOPICS)
```

## Adding a Template

1. Write a `solver(params) -> list[Step]` function in the relevant file
2. Define a `QuestionTemplate(...)` with `prompt_fn`, `parameters`, `constraints`, `solver`
3. Add it to the file's `ALL_TEMPLATES` list

