"""Generate a seed dataset for Kelsie's intent router.

This is synthetic seed data so the pipeline works end to end today. Replace or
extend it with real (consented, anonymised) messages from testers as soon as
you have them -- real data is what will make the router good.

Leakage control: each intent's templates are split into train / val / test
groups BEFORE filling slots, so the test set uses sentence patterns the model
never saw in training. Slot values are shared, which is fine (they are the
vocabulary of the domain).

Usage:  python data/generate.py --out data --per-template 12 --seed 7
"""
import argparse
import json
import random
import re
from pathlib import Path

SLOTS = {
    "task": ["econ essay", "problem set 4", "lab report", "history paper", "group presentation",
             "stats assignment", "reading response", "chem quiz", "project proposal", "case study",
             "discussion post", "coding assignment", "final project", "book report", "midterm review sheet",
             "quarterly report", "client deck", "budget sheet", "team retro notes", "sprint demo"],
    "course": ["ECO 102", "MAT 135", "PSY 100", "CSC 148", "HIS 201", "BIO 120", "STA 220", "ENG 140",
               "econ", "calc", "psych", "bio", "chem", "stats", "history"],
    "day": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "tomorrow",
            "tonight", "next week", "the 14th", "nov 20", "end of the month", "this weekend", "march 3rd"],
    "time": ["11:59pm", "9am", "noon", "5pm", "midnight", "8:30", "3pm", "end of day"],
    "thing": ["the article about fiscal stimulus", "that video on neural nets", "my landlord's number",
              "the recipe i looked at", "that internship posting", "the quote from my reading",
              "the meeting notes from monday", "my prof's office hours", "the study guide link",
              "the podcast episode on sleep", "that thread about resumes", "my wifi password hint",
              "the slides from lecture 6", "what maya said about the slides", "the apartment listing"],
    "topic": ["keynesian economics", "photosynthesis", "the french revolution", "recursion", "supply and demand",
              "how mortgages work", "the krebs cycle", "machine learning", "inflation", "the cold war",
              "linear regression", "stoicism", "climate change", "the electoral college", "black holes"],
    "duration": ["25 minutes", "an hour", "45 mins", "2 hours", "30 min", "90 minutes", "20 minutes"],
    "company": ["Maple Bank", "Lumen Health", "Northwind Studio", "a startup", "this company", "the city",
                "a nonprofit", "Shopify", "a consulting firm", "an agency"],
    "role": ["data analyst co-op", "product design intern", "marketing intern", "software engineering intern",
             "research assistant", "summer analyst", "UX intern", "operations coordinator"],
    "site": ["gmail", "linkedin", "canvas", "google docs", "job boards", "every site", "youtube", "reddit"],
    "person": ["maya", "my prof", "my manager", "jordan", "my TA", "my group", "sam", "my mom"],
    "vibe": ["chill", "hype", "straight to the point"],
}

TEMPLATES = {
    "remember": [
        "remember that {thing} is important",
        "kelsie remember this for me: {person} wants the {task} by {day}",
        "save this for later",
        "can you save {thing}",
        "keep this in mind: {person} owns the slides",
        "note that {person} said the {task} is moved",
        "dont let me forget {thing}",
        "make a note that i need to email {person}",
        "store this link for my {course} research",
        "remember i liked {thing}",
        "pls save this page",
        "bookmark this in your memory",
        "hold onto this for my {task}",
        "jot down that office hours are at {time}",
        "add this to my notes",
        "remember my locker code is in my notes app",
        "save the main points from this article",
        "can you remember that {person} prefers {time} meetings",
        "keep track of this for me",
        "this is important, remember it",
        "remember this quote for my {task}",
        "save what {person} just said",
        "note to self: {thing}",
        "file this under {course}",
        "remember that i already read {thing}",
        "log this so i can find it later",
        "remember the name of {thing}",
        "keep this one, i'll need it for {course}",
    ],
    "recall": [
        "what was {thing}",
        "find {thing}",
        "where did i see {thing}",
        "pull up {thing}",
        "what did {person} say about the {task}",
        "remind me what i read about {topic}",
        "what were the key points from {thing}",
        "can you bring back {thing}",
        "i saved something about {topic}, what was it",
        "show me my notes on {topic}",
        "what did i save yesterday",
        "what was that link i saved for {course}",
        "get me the quote i saved for my {task}",
        "do you remember {thing}",
        "what was i working on {day}",
        "search my memory for {topic}",
        "what did i look at about {topic} last week",
        "what were the 3 points from that article",
        "remind me what {person} wanted",
        "whats in my notes about the {task}",
        "find the thing i saved about {company}",
        "what did i read before i started the {task}",
        "what was the name of {thing} again",
        "show me what you remember about {course}",
        "where was that page about {topic}",
        "can you find what i bookmarked on {day}",
    ],
    "deadline_add": [
        "my {task} is due {day}",
        "{task} due {day} at {time}",
        "add a deadline: {task} {day}",
        "i have a {course} quiz {day}",
        "the {task} for {course} is due {day}",
        "remind me {task} is due {day}",
        "put {task} on my deadlines for {day}",
        "{course} midterm is {day}",
        "{person} needs the {task} by {day}",
        "add {task} to my calendar for {day} {time}",
        "track this deadline: {task} on {day}",
        "my application for {company} closes {day}",
        "i need to submit the {task} by {time} {day}",
        "new deadline {task} {day}",
        "set a due date for my {task}, its {day}",
        "{task} is due {day} dont let me miss it",
        "heads up the {task} got moved to {day}",
        "exam for {course} on {day} at {time}",
        "the {task} deadline is {day}",
        "add {course} final to my deadlines, {day}",
        "{task} due end of {day}",
        "save the deadline for my {task}: {day}",
        "i have to hand in the {task} {day}",
        "{course} project due {day}",
    ],
    "deadline_query": [
        "what's due this week",
        "what do i have due {day}",
        "whats coming up",
        "any deadlines {day}",
        "when is my {task} due",
        "what's due next",
        "do i have anything due for {course}",
        "show me my deadlines",
        "what's on my plate this week",
        "when's the {course} midterm",
        "is anything due {day}",
        "how many things are due this week",
        "what assignments are left",
        "list my upcoming deadlines",
        "what should i work on first",
        "whats the next thing due",
        "do i have any exams coming up",
        "when do i have to submit the {task}",
        "what's due before {day}",
        "am i behind on anything",
        "what's my week looking like",
        "check my deadlines for {course}",
        "is the {task} due {day}",
        "what do i need to finish by {day}",
    ],
    "focus": [
        "help me lock in",
        "start a focus session",
        "focus mode for {duration}",
        "i need to focus on my {task}",
        "lock in for {duration}",
        "start a {duration} timer",
        "block distractions for {duration}",
        "pomodoro please",
        "keep me off youtube for {duration}",
        "stop focus mode",
        "end my focus session",
        "pause the timer",
        "i keep getting distracted help",
        "how long is left on my focus timer",
        "let's do a study sprint",
        "focus time, {duration}",
        "i'm procrastinating, make me work",
        "turn on focus mode",
        "add {duration} to my session",
        "i need to grind on the {task} for {duration}",
        "start a deep work block",
        "no more tabs for {duration}",
        "can you keep me on track while i do the {task}",
        "study mode on",
    ],
    "application": [
        "help me apply to {company}",
        "write a cover letter for the {role} role",
        "fill out this application",
        "tailor my resume for {company}",
        "i applied to {company} for {role}",
        "track my application to {company}",
        "what's the status of my {company} application",
        "draft a cover letter for {company}",
        "help me answer this application question",
        "autofill this job form",
        "how many places have i applied to",
        "i got an interview at {company}",
        "mark the {company} application as rejected",
        "write a short answer for why i want to work at {company}",
        "prep me for my {role} interview",
        "save this job posting for {role}",
        "update my {company} application to interview stage",
        "make my cover letter sound less stiff",
        "help me write a follow up email to {company}",
        "can you fill this in from my profile",
        "list the jobs i applied to this month",
        "i want to apply for the {role} position",
        "what should i say in my {company} cover letter",
        "add {company} {role} to my applications",
    ],
    "settings_privacy": [
        "change your vibe to {vibe}",
        "be more {vibe}",
        "stop running on {site}",
        "turn off kelsie on {site}",
        "pause memory",
        "stop remembering things",
        "delete everything you remember",
        "clear my memories",
        "delete my account",
        "download my data",
        "what do you know about me",
        "turn memory back on",
        "add {site} to your sites",
        "change my timezone",
        "change my name to {person}",
        "forget what i saved about {topic}",
        "don't save anything from {site}",
        "what sites are you on",
        "i want to export my notes",
        "sign me out",
        "make your messages shorter",
        "switch to {vibe} mode",
        "stop showing up on {site}",
        "who can see my data",
    ],
    "smalltalk": [
        "hi",
        "hey kelsie",
        "good morning",
        "thanks",
        "thank you so much",
        "you're the best",
        "how are you",
        "lol",
        "ok cool",
        "what's up",
        "gn kelsie",
        "who are you",
        "are you a robot",
        "i love you kelsie",
        "you're kinda funny",
        "bye",
        "haha nice",
        "yo",
        "appreciate it",
        "that was helpful",
        "what can you do",
        "omg thank u",
        "hello there",
        "sup",
    ],
    "general": [
        "explain {topic}",
        "what is {topic}",
        "summarize {topic} in simple terms",
        "help me understand {topic}",
        "write an intro paragraph about {topic}",
        "give me 3 essay ideas about {topic}",
        "is this sentence grammatically correct",
        "translate this to french",
        "how do i cite a website in apa",
        "what's the difference between {topic} and {topic}",
        "make this paragraph sound better",
        "quiz me on {topic}",
        "how do i solve quadratic equations",
        "write a text to {person} saying i'll be late",
        "give me feedback on my thesis",
        "what should i eat for dinner",
        "how do i make my resume one page",
        "brainstorm names for my club",
        "help me reply to this email",
        "why is the sky blue",
        "can you proofread this",
        "what are good study techniques",
        "turn these notes into flashcards on {topic}",
        "how do i ask {person} for an extension",
        "rewrite this to sound more professional",
        "what happened in {topic}",
    ],
}

import sys
sys.path.insert(0, str(Path(__file__).parent))
from templates_more import MORE  # noqa: E402

for _label, _extra in MORE.items():
    TEMPLATES[_label] = TEMPLATES[_label] + _extra

TYPO_RATE = 0.12


def fill(template, rng):
    def sub(m):
        return rng.choice(SLOTS[m.group(1)])
    return re.sub(r"\{(\w+)\}", sub, template)


def typo(word, rng):
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.random()
    if op < 0.4:   # drop a letter
        return word[:i] + word[i + 1:]
    if op < 0.8:   # swap two letters
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]  # double a letter


def noisify(text, rng):
    words = text.split()
    words = [typo(w, rng) if rng.random() < TYPO_RATE else w for w in words]
    text = " ".join(words)
    r = rng.random()
    if r < 0.15:
        text = text.capitalize()
    elif r < 0.2:
        text = text.upper()
    if rng.random() < 0.15:
        text = rng.choice(["kelsie ", "hey kelsie ", "yo ", "ok so "]) + text
    if rng.random() < 0.15:
        text = text + rng.choice([" pls", " please", " rn", " asap", "?", "!", " lol", " thx"])
    return text


def split_templates(templates, rng):
    t = list(templates)
    rng.shuffle(t)
    n = len(t)
    n_test = max(3, round(n * 0.15))
    n_val = max(3, round(n * 0.15))
    return {"test": t[:n_test], "val": t[n_test:n_test + n_val], "train": t[n_test + n_val:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-template", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    for label, templates in TEMPLATES.items():
        groups = split_templates(templates, rng)
        for split, ts in groups.items():
            for t in ts:
                seen = set()
                for _ in range(args.per_template * 3):
                    if len(seen) >= args.per_template:
                        break
                    text = noisify(fill(t, rng), rng)
                    if text in seen:
                        continue
                    seen.add(text)
                    splits[split].append({"text": text, "label": label})

    for split, rows in splits.items():
        rng.shuffle(rows)
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(rows)} examples")


if __name__ == "__main__":
    main()
