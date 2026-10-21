# Kelsie web

The Kelsie profile site, built with Next.js (App Router).

## Pages

- `/` — landing page ("Meet Kelsie")
- `/signup` and `/signin` — account page with a Sign up / Sign in toggle. Sign-in isn't connected yet; the buttons show a preview notice.

## Run it locally

Requires Node.js 20.9 or newer.

```bash
cd web
npm install
npm run dev
```

Then open <http://localhost:3000>. Edits reload automatically.

To check a production build:

```bash
npm run build
npm start
```

## Deploy to Vercel

1. Push the repo to GitHub.
2. In Vercel, choose **Add New → Project** and import the repo.
3. Set **Root Directory** to `web`. Vercel detects Next.js and fills in the build settings.
4. Click **Deploy**.

## Layout

```
app/
  layout.tsx        Fonts (Bricolage Grotesque, Inter) and page metadata
  globals.css       Colour tokens, buttons and base styles
  page.tsx          Landing page
  signup/, signin/  Account pages
components/         Orb, speech bubble, nav, vibe picker, auth page, icons
```
