# Kelsie AI 🤖

Kelsie is an ambient AI assistant designed to help you keep track of the things that matter without having to constantly organize, save, or re-explain them.

The idea is as simple as you mentioning something; Kelsie understands why it matters and brings it back when it becomes useful later.

---

## What Kelsie does

Kelsie currently runs as a Chrome extension and acts as an AI layer across the browser.

While browsing, you can tell Kelsie things naturally:

- "I want to come back to this."
- "I need this for my assignment."
- "I want to compare this before I decide."
- "I should email someone about this."
- "This would be useful for my presentation."

Instead of requiring you to manually organize everything into folders, notes, or reminders, Kelsie keeps the relevant context — including what it is, why it mattered, and where it came from.

---

### Contextual resurfacing

Kelsie can bring something back when your current browser context makes it useful.

For example:

You might tell Kelsie:

> "I want to apply to this PM role, but I need to update my resume first."

Later, while working on your resume, Kelsie can surface the role again.

The goal is not to interrupt you whenever two pages look similar. Kelsie is designed to stay quiet unless the previous context is meaningfully useful in the moment.

---

### Context-aware reading

Kelsie can also help while you are reading online.

It currently supports:

- **Help me understand it** — breaks down how the page is structured and works through it with you.
- **Summarize it** — gives you a direct summary.
- **I have questions about this page** — lets you discuss the current page in your normal Kelsie conversation.
- **Keep this in mind** — saves why the page matters so it can be useful later.

Kelsie can adapt to different types of content, including articles, research, tutorials, explainers, news, and reference material.

Page context is temporary unless you explicitly ask Kelsie to keep something.

---

## How Kelsie works

At the centre of Kelsie is a simple loop:

```text
Notice
  ↓
Understand
  ↓
Remember
  ↓
Match
  ↓
Surface
