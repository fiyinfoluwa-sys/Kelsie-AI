-- Kelsie: profiles table, Row Level Security and account deletion.
--
-- Run once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Safe to re-run: every statement is idempotent.

-- ------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------

-- Helpers live in a schema the Supabase API doesn't expose, so they can't
-- be called from outside the database.
create schema if not exists private;
revoke all on schema private from public, anon;
grant usage on schema private to authenticated;

-- True when an array has no repeated values. Used by CHECK constraints,
-- which can't contain subqueries directly.
create or replace function private.array_is_distinct(values_in text[])
returns boolean
language sql
immutable
set search_path = ''
as $$
  select cardinality(values_in) = cardinality(array(select distinct unnest(values_in)));
$$;

-- ------------------------------------------------------------
-- Table
-- ------------------------------------------------------------

create table if not exists public.profiles (
  -- One row per user, keyed by their auth ID. Deleting the user deletes
  -- the row.
  id uuid primary key references auth.users (id) on delete cascade,

  name text not null default ''
    check (char_length(name) <= 40 and name !~ '[[:cntrl:]]'),

  -- Any of student/working/job_hunting, or "mix" on its own.
  roles text[] not null default '{}'
    check (
      roles <@ array['student', 'working', 'job_hunting', 'mix']
      and private.array_is_distinct(roles)
      and (not ('mix' = any (roles)) or cardinality(roles) = 1)
    ),

  timezone text not null default 'UTC'
    check (
      char_length(timezone) between 1 and 64
      and timezone ~ '^[A-Za-z0-9_+-]+(/[A-Za-z0-9_+-]+)*$'
    ),

  vibe text check (vibe in ('chill', 'hype', 'direct')),

  allowed_sites text[] not null default '{}'
    check (
      allowed_sites <@ array['google_docs', 'canvas', 'gmail', 'linkedin', 'job_boards', 'everything_else']
      and private.array_is_distinct(allowed_sites)
    ),

  onboarding_completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  -- Onboarding can only be marked complete once the required answers exist.
  constraint profiles_complete_requires_answers check (
    onboarding_completed_at is null
    or (cardinality(roles) > 0 and vibe is not null)
  )
);

-- Timestamps are set by the database, whatever the client sends.
create or replace function private.set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if tg_op = 'INSERT' then
    new.created_at := now();
  else
    new.created_at := old.created_at;
  end if;
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at
  before insert or update on public.profiles
  for each row execute function private.set_updated_at();

-- ------------------------------------------------------------
-- Row Level Security: each user can only see and change their own row
-- ------------------------------------------------------------

alter table public.profiles enable row level security;
-- Applies RLS to the table owner too, so nothing bypasses it by accident.
alter table public.profiles force row level security;

drop policy if exists "Users can read their own profile" on public.profiles;
create policy "Users can read their own profile"
  on public.profiles for select
  to authenticated
  using ((select auth.uid()) = id);

drop policy if exists "Users can create their own profile" on public.profiles;
create policy "Users can create their own profile"
  on public.profiles for insert
  to authenticated
  with check ((select auth.uid()) = id);

drop policy if exists "Users can update their own profile" on public.profiles;
create policy "Users can update their own profile"
  on public.profiles for update
  to authenticated
  using ((select auth.uid()) = id)
  with check ((select auth.uid()) = id);

-- No DELETE policy: rows are only removed when the account is deleted
-- (see delete_my_account below).

-- Supabase grants broad table privileges by default. Logged-out visitors
-- (anon) get nothing, and TRUNCATE is revoked because it isn't covered by
-- Row Level Security.
revoke all on table public.profiles from anon;
revoke truncate, delete, references, trigger on table public.profiles from authenticated;

-- ------------------------------------------------------------
-- Account deletion
-- ------------------------------------------------------------

-- Lets a signed-in user delete their own account, and only their own.
-- Runs with the privileges of its owner so it can remove the auth user,
-- which also deletes the profile (on delete cascade). This means the app
-- never needs the service role key.
create or replace function public.delete_my_account()
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := auth.uid();
begin
  if uid is null then
    raise exception 'Not signed in' using errcode = '42501';
  end if;
  delete from auth.users where id = uid;
end;
$$;

revoke all on function public.delete_my_account() from public, anon;
grant execute on function public.delete_my_account() to authenticated;

-- The CHECK constraints run as the signed-in user, so they need to be able
-- to execute the array helper. The trigger function needs no grant.
revoke all on function private.array_is_distinct(text[]) from public, anon;
grant execute on function private.array_is_distinct(text[]) to authenticated;
revoke all on function private.set_updated_at() from public, anon, authenticated;
