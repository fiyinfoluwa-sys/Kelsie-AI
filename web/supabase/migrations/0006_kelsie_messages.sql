-- Kelsie: messages to Kelsie (intent router), focus sessions, router logs
-- and opt-in training examples.
--
-- Run once in SQL Editor after 0001-0005. Safe to re-run.

-- ------------------------------------------------------------
-- "Help improve Kelsie" (off unless the user turns it on)
-- ------------------------------------------------------------

alter table public.profiles add column if not exists help_improve boolean not null default false;

-- ------------------------------------------------------------
-- Focus sessions ("lock in for 25 mins")
-- ------------------------------------------------------------

create table if not exists public.focus_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  started_at timestamptz not null default now(),
  ends_at timestamptz not null,
  ended_at timestamptz,
  check (ends_at > started_at and ends_at <= started_at + interval '8 hours')
);
-- At most one running session per user.
create unique index if not exists focus_sessions_one_active_idx
  on public.focus_sessions (user_id) where ended_at is null;

alter table public.focus_sessions enable row level security;
alter table public.focus_sessions force row level security;
drop policy if exists "Users can read their own focus sessions" on public.focus_sessions;
create policy "Users can read their own focus sessions" on public.focus_sessions for select to authenticated
  using ((select auth.uid()) = user_id);
drop policy if exists "Users can start their own focus sessions" on public.focus_sessions;
create policy "Users can start their own focus sessions" on public.focus_sessions for insert to authenticated
  with check ((select auth.uid()) = user_id);
drop policy if exists "Users can change their own focus sessions" on public.focus_sessions;
create policy "Users can change their own focus sessions" on public.focus_sessions for update to authenticated
  using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
revoke all on table public.focus_sessions from anon;
grant select, insert, update on table public.focus_sessions to authenticated;
revoke delete, truncate, references, trigger on table public.focus_sessions from authenticated;

-- ------------------------------------------------------------
-- Router logs: what the router decided, never what was said.
-- No user ID either, so these are anonymous counts.
-- ------------------------------------------------------------

create table if not exists public.router_events (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  intent text not null check (char_length(intent) <= 32),
  predicted text not null check (char_length(predicted) <= 32),
  route text not null check (route in ('local', 'llm_small', 'llm', 'router_error')),
  handler text not null check (char_length(handler) <= 40),
  confidence real not null check (confidence between 0 and 1),
  fell_back boolean not null,
  -- Whether answering it needed an AI call (to count what the router saves).
  used_ai boolean not null,
  latency_ms real not null check (latency_ms >= 0)
);
alter table public.router_events enable row level security;
alter table public.router_events force row level security;
revoke all on table public.router_events from public, anon, authenticated;

create or replace function public.log_router_event(
  p_intent text, p_predicted text, p_route text, p_handler text,
  p_confidence real, p_fell_back boolean, p_used_ai boolean, p_latency_ms real
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  -- Only signed-in users' messages are logged; the row itself has no user.
  if auth.uid() is null then
    return;
  end if;
  insert into public.router_events (intent, predicted, route, handler, confidence, fell_back, used_ai, latency_ms)
  values (p_intent, p_predicted, p_route, p_handler,
          least(greatest(p_confidence, 0), 1), p_fell_back, coalesce(p_used_ai, true), greatest(p_latency_ms, 0));
end;
$$;
revoke all on function public.log_router_event(text, text, text, text, real, boolean, boolean, real) from public, anon;
grant execute on function public.log_router_event(text, text, text, text, real, boolean, boolean, real) to authenticated;

-- ------------------------------------------------------------
-- Opt-in training examples: message text + predicted intent, only for
-- testers who turned on "Help improve Kelsie". They can see and delete
-- their own; nobody else's.
-- ------------------------------------------------------------

create table if not exists public.router_training_examples (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  created_at timestamptz not null default now(),
  text text not null check (char_length(text) between 1 and 500),
  predicted text not null check (char_length(predicted) <= 32),
  confidence real not null check (confidence between 0 and 1),
  fell_back boolean not null,
  -- For relabelling later (filled in by you, in the dashboard).
  correct_intent text check (correct_intent is null or char_length(correct_intent) <= 32)
);
create index if not exists router_training_examples_user_idx on public.router_training_examples (user_id);

alter table public.router_training_examples enable row level security;
alter table public.router_training_examples force row level security;
drop policy if exists "Users can read their own examples" on public.router_training_examples;
create policy "Users can read their own examples" on public.router_training_examples for select to authenticated
  using ((select auth.uid()) = user_id);
drop policy if exists "Users can delete their own examples" on public.router_training_examples;
create policy "Users can delete their own examples" on public.router_training_examples for delete to authenticated
  using ((select auth.uid()) = user_id);
drop policy if exists "Opted-in users can add their own examples" on public.router_training_examples;
create policy "Opted-in users can add their own examples" on public.router_training_examples for insert to authenticated
  with check (
    (select auth.uid()) = user_id
    and correct_intent is null
    and exists (select 1 from public.profiles p where p.id = (select auth.uid()) and p.help_improve)
  );
revoke all on table public.router_training_examples from anon;
grant select, insert, delete on table public.router_training_examples to authenticated;
revoke update, truncate, references, trigger on table public.router_training_examples from authenticated;

-- ------------------------------------------------------------
-- AI calls from messages: at most 100 per user per day (shared free tier)
-- ------------------------------------------------------------

create table if not exists public.assistant_usage (
  user_id uuid not null references auth.users (id) on delete cascade,
  day date not null,
  calls integer not null default 0 check (calls >= 0),
  primary key (user_id, day)
);
alter table public.assistant_usage enable row level security;
alter table public.assistant_usage force row level security;
revoke all on table public.assistant_usage from public, anon, authenticated;

-- Counts one AI call for the caller; false once today's limit is reached.
-- Users can't read or reset the counter themselves.
create or replace function public.reserve_assistant_call()
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := auth.uid();
  today date := (now() at time zone 'utc')::date;
  used integer;
begin
  if uid is null then
    return false;
  end if;
  insert into public.assistant_usage (user_id, day, calls) values (uid, today, 0)
  on conflict (user_id, day) do nothing;
  update public.assistant_usage set calls = calls + 1
   where user_id = uid and day = today and calls < 100
  returning calls into used;
  delete from public.assistant_usage where user_id = uid and day < today - 7;
  return used is not null;
end;
$$;
revoke all on function public.reserve_assistant_call() from public, anon;
grant execute on function public.reserve_assistant_call() to authenticated;
