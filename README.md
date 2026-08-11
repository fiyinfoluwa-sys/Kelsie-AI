# Kelsie AI 🤖

Kelsie is an ambient AI assistant designed to help you keep track of the things that matter without constantly organizing, saving, or re-explaining them.

The idea is simple: you mention something to Kelsie, she remembers why it mattered, and brings it back when it becomes useful.

---

## Product Strategy

Alongside development, I documented the product thinking behind Kelsie, including the target user, product vision, key problems, feature priorities, and roadmap.

Read the Product Vision and Roadmap [here](https://docs.google.com/document/d/e/2PACX-1vSaO2HSkUQAklphYTHMOJna_qnlpBRKiS-wfG4oDw2w5FTvGa0aznEvuul7MIxVnpyZ90mJCcFFzoOZ/pub) 

---

## What Kelsie does

Kelsie currently runs as a Chrome extension and acts as an AI layer across your browser. While browsing, you can tell Kelsie some things naturally, such as:

- "I want to come back to this."
- "This would be useful for my presentation."
- "I want to compare this before I decide."
- "I should email someone about this."

Instead of making you organize everything into folders, notes, or reminders, Kelsie remembers the useful context: what you saw, why it mattered, and where it came from.

---

### Contextual resurfacing

Kelsie can bring something back when it becomes relevant again. For example, you might say:
> "I want to apply to this PM role, but I need to update my resume first."

Later, when you are working on your resume, Kelsie can surface that role again. The goal is not to constantly interrupt you. Kelsie should only step in when something you previously mentioned is actually useful.

---

### Context-aware reading

Kelsie can also help you understand what you are reading without leaving the page. Its current actions include:

- **Help me understand it** — breaks down the page and works through it with you.
- **Summarize it** — gives you a direct summary of the page.
- **I have questions about this page** — lets you ask questions using the current page as context.
- **Keep this in mind** — saves why the page matters so it can be useful later.

Kelsie works across content such as articles, research, tutorials, explainers, news, and reference material. By default, page context is temporary. Kelsie only keeps it for later when you explicitly ask her to.
