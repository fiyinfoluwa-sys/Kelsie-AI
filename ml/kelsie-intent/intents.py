"""Kelsie's routing intents and what each one does downstream.

`route` says where a message goes after classification:
  - "local":     handled by app code, no LLM call (database query, timer, settings change)
  - "llm_small": a short, cheap LLM call (extract a fact or a date, write one line)
  - "llm":       the full LLM with the intent's own system prompt
Messages the classifier is unsure about always fall back to "general" (full LLM).
"""

INTENTS = [
    {"id": "remember",         "route": "llm_small", "desc": "User wants Kelsie to save something for later"},
    {"id": "recall",           "route": "llm",       "desc": "User wants something back from Kelsie's memory"},
    {"id": "deadline_add",     "route": "llm_small", "desc": "User is telling Kelsie about a due date"},
    {"id": "deadline_query",   "route": "local",     "desc": "User asks what is due / coming up"},
    {"id": "focus",            "route": "local",     "desc": "User wants to start, stop or adjust a focus session"},
    {"id": "application",      "route": "llm",       "desc": "Job/internship applications, cover letters, tracking"},
    {"id": "settings_privacy", "route": "local",     "desc": "Change vibe, sites, memory, delete or export data"},
    {"id": "smalltalk",        "route": "local",     "desc": "Greetings, thanks, how-are-you, banter"},
    {"id": "general",          "route": "llm",       "desc": "Anything else: open questions, writing, explaining"},
]

LABELS = [i["id"] for i in INTENTS]
LABEL2ID = {l: n for n, l in enumerate(LABELS)}
ROUTE = {i["id"]: i["route"] for i in INTENTS}
FALLBACK = "general"
