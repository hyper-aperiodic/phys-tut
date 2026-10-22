"""
Physics Tutor — Streamlit UI
Run with: streamlit run app.py

How the page works
------------------
Streamlit re-runs this whole script from top to bottom every time the user
interacts with anything (clicks a button, changes a dropdown). Anything that
must survive between runs, such as the current question and which steps are
revealed, lives in st.session_state.

Layout:
    Sidebar    topic and subtopic pickers, "Generate new question", and the
               step-reveal controls once a question exists.
    Main area  the question box, then one card per revealed solution step.

The maths all happens in the physics_tutor package. This file only displays
Question objects produced by physics_tutor.core.generate().
"""
import sys, os
# Make the physics_tutor package importable however Streamlit is launched
# (e.g. from a different working directory).
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as components
import random
from physics_tutor.templates import ALL_TEMPLATES, TOPICS
from physics_tutor.core import generate

# ── Page config ───────────────────────────────────────────────────────────────
# Must be the first Streamlit call in the script. "wide" uses the full browser
# width; the sidebar starts open (and on desktop the CSS below stops it being closed).
st.set_page_config(
    page_title="PHYS108 Tutor",
    page_icon="∇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
# Custom CSS injected once per run. What the main rules do:
#   fonts             DM Sans (body), DM Serif Display (titles), DM Mono (labels/buttons)
#   .main-title/.main-subtitle   the heading at the top of the main area
#   .topic-tag, .tag-vc/-ode/-fa coloured pills showing topic/subtopic, one colour per topic
#   ...:has(.question-box-marker) / ...:has(.step-card-marker)
#       Streamlit's bordered containers can't take a CSS class directly. So an
#       empty <span class='...-marker'> goes inside each container, and the
#       :has() selector styles any container holding that marker. This is how
#       the question box and step cards get their own look.
#   stSidebarCollapseButton / stExpandSidebarButton
#       hidden on screens 768px and wider, so the sidebar is always visible
#       there; phones keep the buttons (see the CSS). Streamlit remembers a
#       collapsed sidebar in the browser and ignores initial_sidebar_state on
#       reload, so the [aria-expanded="false"] rule also forces a "collapsed"
#       sidebar back open. These test-ids are Streamlit-version specific
#       (checked against 1.57); if the sidebar becomes collapsible again after
#       an upgrade, check them.
#   .stCustomComponentV1:has(iframe[height="0"])
#       hides the empty gap left by the invisible scroll-to-step script (see the end of the file).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #e8e8e8;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}

.main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #8b8b9a;
    font-size: 1rem;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}

.topic-tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-bottom: 4px;
    font-weight: 500;
    letter-spacing: 0.03em;
}

.tag-vc   { background: #0d3d3a; color: #2dd4bf; border: 1px solid #2dd4bf44; }
.tag-ode  { background: #3d0d2e; color: #f472b6; border: 1px solid #f472b644; }
.tag-fa   { background: #2d2d0d; color: #facc15; border: 1px solid #facc1544; }

[data-testid="stVerticalBlockBorderWrapper"]:has(.question-box-marker) {
    background: #181c27 !important;
    border: 1px solid #2a2d3a !important;
    border-left: 4px solid #4f8ef7 !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.6rem !important;
    margin: 1rem 0 1.5rem 0 !important;
    font-size: 1.05rem !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:has(.step-card-marker) {
    background: #13161f !important;
    border: 1px solid #22263a !important;
    border-radius: 6px !important;
    margin: 0.5rem 0 !important;
    overflow: visible !important;
    transition: border-color 0.2s !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.step-card-marker) > div {
    overflow: visible !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.step-card-marker):hover {
    border-color: #4f8ef7 !important;
}

.step-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}


.param-display {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #6b7280;
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 4px;
    padding: 4px 10px;
    margin-top: 0.5rem;
}

.divider {
    border: none;
    border-top: 1px solid #1e2130;
    margin: 1.5rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1018;
    border-right: 1px solid #1a1e2a;
}

/* Buttons */
.stButton > button {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    border-radius: 6px;
    border: 1px solid #2a2d3a;
    background: #181c27;
    color: #e8e8e8;
    transition: all 0.15s;
}
.stButton > button:hover {
    border-color: #4f8ef7;
    color: #4f8ef7;
    background: #12182f;
}

/* Selectbox, radio */
.stSelectbox label, .stRadio label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #9ca3af;
}

/* Keep the sidebar permanently open: no collapse / expand controls, and undo
   the collapsed styling (zero width, slid off-screen) in case the browser
   remembers it as collapsed from an earlier visit. Desktop only: below 768px
   Streamlit treats the screen as mobile, collapses the sidebar by itself and
   lays it over the page, so phones keep the normal open / close buttons. */
@media (min-width: 768px) {
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] { display: none !important; }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        min-width: 200px !important;
        max-width: 600px !important;
    }
}

/* Hide the wrapper around 0-height scroll-helper iframes */
.stCustomComponentV1:has(iframe[height="0"]) { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
# Values that persist across reruns. Each is set only on the very first run:
#   question      the current Question, or None before the first one is generated
#   show_steps    indices of the revealed solution steps, in reveal order
#   all_revealed  True after "Reveal all steps" (kept for future use)
#   seed          seed of the current question (kept so a question could be reproduced)
#   selection     (set further down) the last (topic, subtopic) seen, for auto-generation
if "question" not in st.session_state:
    st.session_state.question = None
if "show_steps" not in st.session_state:
    st.session_state.show_steps = []
if "all_revealed" not in st.session_state:
    st.session_state.all_revealed = False
if "seed" not in st.session_state:
    st.session_state.seed = random.randint(0, 99999)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # st.markdown("### ∇ PHYS108 Tutor")
    # st.markdown("<div style='color:#6b7280;font-size:0.8rem;margin-bottom:1.2rem'>First-year mathematical physics</div>", unsafe_allow_html=True)
    # st.markdown("---")

    # Topic selector
    # Topics and subtopics come from physics_tutor.templates.TOPICS, so adding a
    # template there makes it appear here automatically.
    topic_choice = st.radio("Topic", list(TOPICS.keys()), label_visibility="collapsed")
    st.markdown("")  # empty markdown = a little vertical spacing

    # The subtopic list depends on the chosen topic. Changing topic rebuilds
    # this dropdown, which resets it to that topic's first subtopic.
    subtopic_map = TOPICS[topic_choice]
    subtopic_choice = st.selectbox("Subtopic", list(subtopic_map.keys()))
    st.markdown("")

    # All templates for this subtopic; each new question picks one at random.
    templates_available = subtopic_map[subtopic_choice]

    eligible = templates_available

    # Case filter (e.g. the three root cases of the 2nd-order ODEs).
    # If the subtopic's templates carry a `case` label, the main page shows a
    # case selector (key "case_choice", drawn further down). This script runs
    # top to bottom, so the selector itself appears after this point. But
    # Streamlit stores a widget's new value in session_state before the rerun
    # starts, so it can be read here. Until a case is chosen, eligible is empty
    # and no question can be generated.
    cases = list(dict.fromkeys(t.case for t in templates_available if t.case))  # unique, in template order
    case_choice = st.session_state.get("case_choice") if cases else None
    if cases:
        eligible = [t for t in templates_available if t.case == case_choice]

    # st.markdown("---")

    # Topic colour
    # (CSS class for the coloured topic/subtopic tags; a new topic needs an entry here and in the CSS)
    # tag_class = {"Vector Calculus": "tag-vc", "ODEs": "tag-ode", "Fourier Analysis": "tag-fa"}[topic_choice]
    # st.markdown(f"<span class='topic-tag {tag_class}'>{topic_choice}</span><span class='topic-tag {tag_class}'>{subtopic_choice}</span>", unsafe_allow_html=True)
    # # st.markdown(f"<div style='color:#6b7280;font-size:0.75rem;margin-top:0.5rem'>{len(eligible)} template(s) available</div>", unsafe_allow_html=True)
    # st.markdown("")

    def _new_question():
        """Generate a fresh question for the current subtopic and reset the revealed steps.

        Picks a random template from the subtopic and a random seed. The
        generator draws the template's parameters from that seed.
        Clears the current question instead if nothing is eligible (a case
        still needs choosing), so a stale question from another subtopic isn't left showing.
        """
        if not eligible:
            st.session_state.question = None
            return
        template = random.choice(eligible)
        seed = random.randint(0, 99999)
        st.session_state.seed = seed
        try:
            q = generate(template, seed=seed)
            st.session_state.question = q
            st.session_state.show_steps = []
            st.session_state.all_revealed = False
        except RuntimeError as e:
            st.error(f"Generation failed: {e}")

    # Auto-generate when the selection changes (but not on first page load)
    # Compare this run's (topic, subtopic, case) with the previous run's. On the
    # very first run there is no stored selection, so nothing is generated and
    # the landing page shows. Changing topic also counts as a change, because
    # the subtopic dropdown resets to the new topic's first entry. Picking a
    # different root case counts too.
    selection = (topic_choice, subtopic_choice, case_choice)
    if "selection" in st.session_state and st.session_state.selection != selection:
        _new_question()
    st.session_state.selection = selection

    # Disabled until a case is chosen, on subtopics that need one.
    if st.button("⟳  Generate new question", use_container_width=True, disabled=not eligible):
        _new_question()


    # ── Step controls ──────────────────────────────────────────────────────────
    # Shown only once a question exists. Each button changes show_steps through
    # an on_click callback. Streamlit runs callbacks *before* the script reruns,
    # so the step counter, the buttons and the main area all see the updated
    # steps in the same run.
    #
    # Don't switch these back to "if st.button(...): ...; st.rerun()".
    # st.rerun() stops the script partway through the sidebar, before the main
    # area draws the case selector. Streamlit then treats the selector as gone
    # and deletes its value, which resets the chosen root case and throws the
    # question away.
    def _reveal_next():
        # Steps are revealed in order, so the next one to show is the count shown so far.
        st.session_state.show_steps.append(len(st.session_state.show_steps))

    def _reveal_all():
        st.session_state.show_steps = list(range(len(st.session_state.question.steps)))
        st.session_state.all_revealed = True

    def _hide_solution():
        st.session_state.show_steps = []
        st.session_state.all_revealed = False

    if st.session_state.question is not None:
        _q = st.session_state.question
        _n_steps = len(_q.steps)
        _n_shown = len(st.session_state.show_steps)

        # st.markdown("---")
        st.markdown("<div style='color:#9ca3af;font-size:0.78rem;font-family:DM Mono,monospace;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem'>Solution</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#6b7280;font-size:0.8rem;margin-bottom:0.6rem'>{_n_shown} / {_n_steps} steps shown</div>", unsafe_allow_html=True)

        if _n_shown < _n_steps:
            st.button("▶  Reveal next step", use_container_width=True, on_click=_reveal_next)

        st.button("⟫  Reveal all steps", use_container_width=True, on_click=_reveal_all)

        if _n_shown > 0:
            st.button("↺  Hide solution", use_container_width=True, on_click=_hide_solution)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>PHYS108 tutor</div>", unsafe_allow_html=True)
# st.markdown("<div class='main-subtitle'>Have a go, then reveal the solution step by step.</div>", unsafe_allow_html=True)

# Case selector, for subtopics split into cases (the 2nd-order ODE root cases).
# Nothing is selected at first, so the student chooses which case to work on.
# Picking a case (or switching to another) generates a question of that case,
# via the selection check in the sidebar code. Clicking the selected case again
# deselects it and returns to the landing message.
if cases:
    st.segmented_control("Which case do you want to work with?", cases, key="case_choice")

if st.session_state.question is None and cases:
    # Landing state for a case-split subtopic: prompt for a case first.
    # The text comes from the subtopic's templates (case_intro) if they set it,
    # otherwise the default, which describes the root cases.
    case_intro = next((t.case_intro for t in templates_available if t.case_intro),
                      "Choose the type of roots you want to work with above. "
                      "You'll get an equation of that type: confirm the case using the discriminant, then find the general solution.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#9ca3af;line-height:1.7'>{case_intro}</div>", unsafe_allow_html=True)

elif st.session_state.question is None:
    # Landing state
    # Shown before any question has been generated. The lorem ipsum is a
    # placeholder for an introduction to the tutor.
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#9ca3af;line-height:1.7'>"
        "This is a practice book for PHYS108 - each topic we cover in the lectures has example questions "
        "for you to try. Select the main topic (calculus, ODEs, Fourier Analysis) and the subtopic, "
        "and a question will be auto-generated. Give it a go before using the 'Reveal' buttons to "
        "check your work.<br><br>"
        "Any issues (the app giving wrong answers, broken links etc.) - email coates92@liv.ac.uk <br>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("")
    st.markdown("<div style='color:#6b7280;text-align:center;margin-top:2rem'>← Select a topic</div>", unsafe_allow_html=True)

else:
    q = st.session_state.question

    # Topic tag + difficulty
    tag_class = {"Vector Calculus": "tag-vc", "ODEs": "tag-ode", "Fourier Analysis": "tag-fa"}[q.topic]
    # diff_pips = "".join(
    #     f"<span class='difficulty-pip' style='background:{'#4f8ef7' if i < q.difficulty else '#22263a'}'></span>"
    #     for i in range(5)
    # )

    st.markdown(
        f"<span class='topic-tag {tag_class}'>{q.topic}</span>"
        f"<span class='topic-tag {tag_class}'>{q.subtopic}</span>"
        # Third tag for the case, if the question has one (e.g. "Complex roots").
        + (f"<span class='topic-tag {tag_class}'>{q.case}</span>" if q.case else ""),
        # f"<span style='margin-left:8px'>{diff_pips}</span>"
        # f"<span style='font-family:DM Mono,monospace;font-size:0.72rem;color:#6b7280;margin-left:6px'>difficulty {q.difficulty}/5</span>",
        unsafe_allow_html=True
    )

    # Question box — use container so Streamlit renders LaTeX in q.prompt
    # The prompt is Markdown: $...$ for inline maths, and $$...$$ on their own
    # lines for centred display maths. The marker span lets the CSS style this
    # container as the question box.
    with st.container(border=True):
        st.markdown("<span class='question-box-marker'></span>**Question**", unsafe_allow_html=True)
        st.markdown(q.prompt)
        # Optional figure for the question (e.g. the field and path of a line integral).
        if q.image:
            st.image(q.image, width=480)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    n_steps = len(q.steps)
    n_shown = len(st.session_state.show_steps)

    st.markdown("")

    if n_shown == 0:
        st.markdown("<div style='color:#6b7280;font-style:italic;font-size:0.9rem'>Work it out on paper first, then use the sidebar controls to reveal steps.</div>", unsafe_allow_html=True)

    # Render steps
    # One bordered card per revealed step, showing three parts top to bottom:
    #   step.label  Markdown heading (may contain $...$ maths)
    #   step.latex  the maths, centred (or plain Markdown text, see below)
    #   step.note   optional Markdown explanation underneath
    # The most recently revealed card gets id="step-last" so the script at the
    # end can scroll to it.
    last_shown = st.session_state.show_steps[-1] if st.session_state.show_steps else None
    for i in st.session_state.show_steps:
        step = q.steps[i]
        is_last = (i == last_shown)
        with st.container(border=True):
            anchor = ' id="step-last"' if is_last else ''
            st.markdown(
                f"<span class='step-card-marker'{anchor}></span>"
                f"<span class='step-label'>Step {i+1}</span>",
                unsafe_allow_html=True
            )
            st.markdown(step.label)  # renders $...$ in labels correctly
            latex_str = step.latex
            # Use markdown only for prose steps (contain em-dash); everything
            # else is valid LaTeX even if it lacks backslashes or superscripts.
            # So an em/en-dash anywhere in a step's latex field switches that
            # step to plain text. Put dashes in labels or notes, not in latex.
            if '—' in latex_str or '–' in latex_str:
                st.markdown(f"<span style='color:#c4c4d0'>{latex_str}</span>", unsafe_allow_html=True)
            else:
                st.latex(latex_str)
            # Optional figure (e.g. a sketch of a periodic function), as PNG bytes.
            if step.image:
                st.image(step.image)
            if step.note:
                st.markdown(step.note)

    # Scroll to the last revealed step
    # components.html runs in a sandboxed iframe, so the script reaches out to
    # the parent page (window.parent) to find the #step-last card and scroll
    # it into view. The iframe has zero height and its wrapper is hidden by the CSS.
    if st.session_state.show_steps:
        components.html(
            "<script>"
            "var el=window.parent.document.getElementById('step-last');"
            "if(el)el.scrollIntoView({behavior:'smooth',block:'start'});"
            "</script>",
            height=0,
        )

    # Completion message
    if n_shown == n_steps:
        st.markdown("")
        st.success("Full solution shown!")
        st.markdown("<div style='color:#6b7280;font-size:0.85rem'>Hit <strong>Generate new question</strong> in the sidebar for another.</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
