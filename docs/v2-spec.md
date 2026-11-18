# Kelsie v2 — Product Spec

Nov 17, 2026 · @Fiyin

## Overview

Kelsie v2 is a Chrome extension plus web dashboard that turns any job posting, company page or recruiter profile into a tracked application with a drafted outreach message, in one click.

**Pitch:** Your internship search co-pilot that lives in your browser.

**Target user:** University students and new grads in Canada applying to 20+ internships a term, starting with UW students and NSBE members.

**Problem:** Students juggle postings across company sites, LinkedIn and WaterlooWorks. Key details (deadline, pay, term length, location, language requirements) are buried in long descriptions. Outreach notes have tight limits like LinkedIn's 200 characters. Deadlines get missed because tracking lives in a messy spreadsheet or nowhere.

**What changes from v1:** Kelsie stops being a general ambient assistant. Every feature serves one job: find, understand, apply to and follow up on internships. The intent router stays as the core of the architecture.

## Core features (MVP)

Five features ship in v2. The first two are the demo; the last three are what keep users coming back.

1. **Posting snapshot.** On any job page, the side panel shows a card with: role, company, location, work model (on-site / hybrid / remote), term and length (4, 8 or 12 months), pay range, deadline, language requirement (e.g. "French required" vs "asset"), and student-only or new-grad eligibility.
   - Missing fields show as "Not listed", never guessed.
   - Works on company career sites (Workday, SuccessFactors, Greenhouse, Lever), LinkedIn Jobs, and pasted text as a fallback.
2. **Outreach drafter.** On a LinkedIn profile or after an event, the user picks a context ("met at NSBE x Intact workshop") and a goal (connect, ask for a chat, follow up after applying).
   - Kelsie drafts 3 variants inside a hard character limit: 200 for LinkedIn connection notes, 300 for InMail-style intros, no limit for email.
   - A live counter shows characters left; the user can regenerate one variant or edit inline.
   - Personalizes from a saved profile (school, program, current role, interests), never invents shared experiences.
3. **Company brief.** One click on a company gives a short brief: what it does, size, main Canadian offices, recent news headline, and typical intern pay if the posting lists it. Sources are linked.
4. **Application tracker.** "Save to tracker" stores the snapshot as a row with a status (Saved, Applied, Interview, Offer, Rejected), notes and contacts.
   - Board and table views in the web dashboard.
   - Each contact links to the outreach messages sent to them.
5. **Deadline reminders.** Email reminder 7 days and 2 days before a saved deadline, plus a weekly digest of what's due and who to follow up with.

## User flows

Three flows cover the MVP; each should take under 30 seconds.

**Onboarding (once, \~2 minutes)**

1. Install the extension and sign in with Google (Supabase auth).
2. Fill a short profile: school, program, grad date, current role, target roles, locations, languages.
3. Optionally paste a resume; Kelsie stores a short summary for personalization.

**Flow A: Save a posting**

1. User opens a posting (e.g. Bombardier "Intern, Program Management").
2. Clicks the Kelsie icon; the side panel shows the posting snapshot.
3. User clicks "Save to tracker"; the row appears in the dashboard with status Saved and the deadline set.
4. If a deadline exists, reminders are scheduled automatically.

**Flow B: Message a recruiter**

1. User opens a recruiter's LinkedIn profile.
2. Opens Kelsie, picks the context (event, mutual connection, applied already) and the goal.
3. Kelsie shows 3 drafts under 200 characters with live counts.
4. User copies one into LinkedIn; Kelsie logs the contact and message against the matching application.

**Flow C: Weekly review**

1. User opens the dashboard or the Sunday digest email.
2. Sees deadlines this week, applications with no update in 14 days, and contacts to follow up with.
3. One click opens a follow-up draft for any contact.

## Architecture

Keep the v1 stack (Next.js, Supabase, Groq) and make the intent router the centre: it sends each request to the cheapest path that can handle it.

&#91;embedded content: Kelsie v2 architecture · router picks the cheapest path\]

Only outreach drafts need the large model; saves and most posting fields never touch an LLM.

**Extension:** Chrome Manifest V3 with the Side Panel API. A content script reads the page: first any schema.org `JobPosting` JSON-LD (many career sites include it), then trimmed page text as a fallback.

**Router intents**

| Intent | Trigger | Route | Why |
| --- | --- | --- | --- |
| extract\_posting | Panel opened on a job page | Rules parser, then fast model for missing fields | Most fields follow patterns; cheap and fast |
| draft\_outreach | Panel opened on a profile, or "Draft message" | Large model | User-facing writing needs quality |
| company\_brief | "Brief me" on a company page | Fast model on the page text | Summary task, low stakes |
| track\_action | Save, change status, add note | No LLM: direct database write | Deterministic |
| out\_of\_scope | Anything else | Fixed reply pointing to job-search features | Keeps v2 focused |

**Character limits:** Generate, count characters in code, and if a draft is over the limit, ask the model to shorten it (max 2 retries). Never show a draft over the limit.

**Database tables (Supabase)**

| Table | Key columns |
| --- | --- |
| profiles | user\_id, school, program, grad\_date, current\_role, target\_roles, locations, languages, resume\_summary |
| applications | id, user\_id, company, role, url, location, work\_model, term, pay\_min, pay\_max, deadline, language\_req, status, notes |
| contacts | id, user\_id, application\_id, name, title, linkedin\_url, met\_at |
| messages | id, contact\_id, channel, goal, body, char\_count, created\_at |
| reminders | id, application\_id, send\_at, sent |
| events | id, user\_id, type, route, latency\_ms, tokens, cost\_usd, created\_at |

## Metrics to track

Log these from day one in a Supabase `events` table; they become the numbers in your case study and interviews. Targets are starting goals for the first 6 weeks after launch, not benchmarks.

| Metric | What it measures | How to compute | Target |
| --- | --- | --- | --- |
| Weekly active users | Real usage | Users with 1+ action in 7 days | 30 |
| Postings saved per user | Core value | Saved rows / active users | 5+ |
| Week-2 retention | Habit | Users active in week 2 / week-1 signups | 40% |
| Snapshot accuracy | Extraction quality | Fields correct on a hand-checked set of 50 postings | 90% |
| Draft copy rate | Outreach usefulness | Drafts copied / drafts generated | 50% |
| Router accuracy | Classifier quality | Correct routes on a labelled test set | 95% |
| p50 latency | Speed | Median time from click to card | < 2 s |
| Cost per active user | Efficiency | LLM spend / weekly active users | < $0.10 / week |

## Out of scope for v2

Cutting these keeps v2 shippable in 4 weeks; saying no to them is part of the case study.

- Auto-applying or filling application forms
- Resume and cover letter writing
- Scraping job boards in the background or a job search feed
- Sending LinkedIn messages automatically (user always copies and sends)
- WaterlooWorks integration (log-in walled; revisit later)
- Mobile app, team or club accounts, payments
- The general v1 assistant features (kept in code, hidden from the UI)

## 4-week build plan

Each week ends with something you can use on your own search, so you dogfood from week 2. Shift the start around exams as needed.

&#91;embedded content: Build plan · 4 weeks, one shippable piece per week\]

Week 2 is the riskiest: if posting extraction is unreliable on Workday or SuccessFactors pages, cut LinkedIn support and ship company sites plus pasted text first.

## Launch and portfolio

The goal of launch is 30 real weekly users and a story you can tell in 60 seconds.

**Getting users**

- [ ] Use Kelsie for your own Summer 2027 search first (Bombardier, Pratt & Whitney, Scotiabank, Intact)
- [ ] Post in UW Arts & Business, Systems Design and NSBE group chats with a 30-second demo clip
- [ ] Offer 10 friends a 15-minute onboarding call; collect feedback in a form
- [ ] Share a build-in-public post on LinkedIn at launch and one with results 4 weeks later

**Portfolio pieces**

- [ ] README: problem, GIF demo, live link, architecture paragraph, router section, metrics
- [ ] 60-second Loom: save a posting, draft a 200-character note, show the tracker
- [ ] Case study (1 page): why you narrowed from general assistant to job search, router tradeoffs, metrics, what you cut, what users said
- [ ] Resume line, e.g. "Built and launched Kelsie, an AI internship co-pilot used by X students; trained an intent router that cut LLM cost by Y%"
