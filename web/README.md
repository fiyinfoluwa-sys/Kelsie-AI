# Kelsie web

The Kelsie profile site, built with Next.js (App Router) and Supabase Auth.

## Pages

- `/` — landing page
- `/signup`, `/signin` — email + password or Google
- `/forgot-password`, `/reset-password` — password reset by email
- `/onboarding` — five setup steps (signed-in, verified users only)
- `/profile` — your answers, sign out, delete account (signed-in users only)

## Run it locally

Requires Node.js 20.9 or newer and the Supabase setup below.

```bash
cd web
cp .env.example .env.local   # then fill in the two values
npm install
npm run dev
```

Open <http://localhost:3000>.

## How auth is set up

- **The browser never talks to Supabase.** Sign-up, sign-in and every database call run in Server Actions and Route Handlers. Session cookies are `httpOnly` (page JavaScript can't read them), `Secure` in production and `SameSite=Lax`. No Supabase key is sent to the browser.
- **`proxy.ts`** refreshes the session on each request and redirects logged-out visitors away from `/onboarding`, `/profile` and `/reset-password`. Each of those pages checks again on the server (`lib/session.ts`).
- **The database is the final boundary.** Row Level Security limits every user to their own `profiles` row, and CHECK constraints only allow known values. That holds even if someone calls the Supabase API directly with their own session.
- **No service role key.** Account deletion goes through the `delete_my_account()` database function, which can only delete the caller.
- **Redirects** only go to fixed paths on `http://localhost:3000` or `https://kelsie-ai-beta.vercel.app` (`lib/site.ts`). Supabase's Redirect URLs list enforces the same allowlist.

## One-time setup

### 1. Supabase project

1. Create a project at <https://supabase.com/dashboard> (Free plan). Save the database password in a password manager.
2. **SQL Editor → New query**: paste all of [`supabase/migrations/0001_profiles.sql`](supabase/migrations/0001_profiles.sql) and click **Run**. Then do the same with [`0002_profiles_grants.sql`](supabase/migrations/0002_profiles_grants.sql).
3. **Authentication → Sign In / Providers → Email**: keep **Confirm email** on and set **Minimum password length** to `8`.
4. **Authentication → URL Configuration**:
   - Site URL: `https://kelsie-ai-beta.vercel.app`
   - Redirect URLs (add each one exactly, with no wildcards):
     - `http://localhost:3000/auth/callback`
     - `http://localhost:3000/auth/confirm`
     - `https://kelsie-ai-beta.vercel.app/auth/callback`
     - `https://kelsie-ai-beta.vercel.app/auth/confirm`
5. **Authentication → Emails → Templates**:
   - **Confirm signup**, set the body to:
     ```html
     <h2>Confirm your email</h2>
     <p>Tap below to verify your email and finish setting up Kelsie.</p>
     <p><a href="{{ .RedirectTo }}?token_hash={{ .TokenHash }}&type=email">Verify my email</a></p>
     <p>If you didn't sign up for Kelsie, you can ignore this email.</p>
     ```
   - **Reset password**, set the body to:
     ```html
     <h2>Reset your password</h2>
     <p>Tap below to choose a new password for Kelsie. The link works once.</p>
     <p><a href="{{ .RedirectTo }}?token_hash={{ .TokenHash }}&type=recovery">Reset my password</a></p>
     <p>If you didn't ask for this, you can ignore this email. Your password won't change.</p>
     ```
6. **Project Settings → API Keys**: copy the **Project URL** and the **publishable** key (`sb_publishable_…`; on older projects, the `anon` key) into `web/.env.local`. Don't copy the secret / `service_role` key anywhere: the app doesn't use it.

### 2. Google sign-in

1. In <https://console.cloud.google.com>, create a project (or pick one).
2. **APIs & Services → OAuth consent screen**: External, app name "Kelsie", your support email. Only the default `email`, `profile` and `openid` scopes are needed.
3. **APIs & Services → Credentials → Create credentials → OAuth client ID**, type **Web application**:
   - Authorized JavaScript origins: `http://localhost:3000`, `https://kelsie-ai-beta.vercel.app`
   - Authorized redirect URI: `https://<your-project-ref>.supabase.co/auth/v1/callback`. Supabase shows the exact URL under **Authentication → Sign In / Providers → Google**.
4. Paste the client ID and secret into **Supabase → Authentication → Sign In / Providers → Google**, and enable it. The Google secret stays in Supabase; the app never sees it.

### 3. Vercel

**Project → Settings → Environment Variables**: add `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` for Production (and Development if you use `vercel dev`). Mark them **Sensitive**. Then redeploy.

### Before inviting real users

Supabase's built-in email service is meant for testing: it sends very few emails per hour and may only deliver to your project's team members. Set up custom SMTP (for example Resend's free plan) under **Authentication → Emails → SMTP Settings**.

## Deploy

Push to `main`; Vercel builds from the `web` folder.
