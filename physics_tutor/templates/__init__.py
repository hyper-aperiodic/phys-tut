# Collects every question template from the topic modules and builds the
# lookup tables the app uses to populate the sidebar.
#
# To add a new topic: create a module here that defines an ALL_TEMPLATES list,
# import it below, and add it to the ALL_TEMPLATES sum. The order of that sum
# sets the order of topics in the sidebar. The order within each module's list
# sets the order of its subtopics.

from .vector_calculus import ALL_TEMPLATES as VC_TEMPLATES
from .odes import ALL_TEMPLATES as ODE_TEMPLATES
from .fourier import ALL_TEMPLATES as FOURIER_TEMPLATES

# Every template in the app, in display order.
ALL_TEMPLATES = VC_TEMPLATES + ODE_TEMPLATES + FOURIER_TEMPLATES

# Lookup by stable id, e.g. TEMPLATE_INDEX["ode-classify"].
TEMPLATE_INDEX = {t.id: t for t in ALL_TEMPLATES}

# Nested grouping used to build the sidebar:
#   TOPICS[topic][subtopic] -> list of templates
# e.g. TOPICS["Fourier Analysis"]["Fourier Series"] holds both the square-wave and
# triangle-wave templates. The app picks one of them at random per question.
# Plain dicts keep insertion order, so the sidebar follows ALL_TEMPLATES order.
TOPICS = {}
for t in ALL_TEMPLATES:
    TOPICS.setdefault(t.topic, {}).setdefault(t.subtopic, []).append(t)
