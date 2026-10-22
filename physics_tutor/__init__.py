"""
physics_tutor — SymPy-verified question generator for first-year physics.

Package layout:
    core/       data types (Step, Question, QuestionTemplate), the generator,
                and LaTeX formatting helpers (fmt.py)
    templates/  one module per topic (vector calculus, ODEs, Fourier analysis),
                each defining question templates and an ALL_TEMPLATES list

The Streamlit front end lives outside the package, in ../app.py.
"""
