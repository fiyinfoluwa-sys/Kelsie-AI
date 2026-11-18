-- Kelsie: memories, deadlines and applications, plus two profile settings.
--
-- Every table is linked to the user, has Row Level Security so users only
-- ever see and change their own rows, and CHECK constraints matching the
-- server-side validation.
--
-- Run once in SQL Editor after 0001 and 0002. Safe to re-run.

-- ------------------------------------------------------------
-- Profile settings
-- ------------------------------------------------------------

-- "Let Kelsie remember things" in Settings and Pause on the timeline.
alter table public.profiles
  add column if not exists memory_enabled boolean not null default true;

-- Set when the Chrome extension links to the account. Only trusted server
-- code may set it (see the trigger below), not the user's own session.
alter table public.profiles
  add column if not exists extension_connected_at timestamptz;

create or replace function private.protect_profile_system_fields()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  -- Requests made with a user's session run as "authenticated".
  if current_user in ('authenticated', 'anon') then
    if tg_op = 'INSERT' then
      new.extension_connected_at := null;
    else
      new.extension_connected_at := old.extension_connected_at;
    end if;
  end if;
  return new;
end;
$$;
revoke all on function private.protect_profile_system_fields() from public, anon, authenticated;

drop trigger if exists profiles_protect_system_fields on public.profiles;
create trigger profiles_protect_system_fields
  before insert or update on public.profiles
  for each row execute function private.protect_profile_system_fields();

-- Timestamps: requests made with a user's session can't choose created_at
-- or updated_at. Trusted SQL (the dashboard, seed scripts) still can.
create or replace function private.set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if current_user in ('authenticated', 'anon') then
    if tg_op = 'INSERT' then
      new.created_at := now();
    else
      new.created_at := old.created_at;
    end if;
    new.updated_at := now();
  elsif tg_op = 'UPDATE' then
    new.updated_at := now();
  end if;
  return new;
end;
$$;
revoke all on function private.set_updated_at() from public, anon, authenticated;

-- ------------------------------------------------------------
-- Memories: short notes Kelsie saves
-- ------------------------------------------------------------

create table if not exists public.memories (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  content text not null check (char_length(content) between 1 and 500),
  source text not null default 'other'
    check (source in ('google_docs', 'canvas', 'gmail', 'linkedin', 'job_boards', 'everything_else', 'other')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists memories_user_created_idx on public.memories (user_id, created_at desc);

-- ------------------------------------------------------------
-- Deadlines: confirmed ones show in "Coming up"; suggested ones wait for
-- the user to say whether they're real.
-- ------------------------------------------------------------

create table if not exists public.deadlines (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  title text not null check (char_length(title) between 1 and 120),
  details text not null default '' check (char_length(details) <= 200),
  -- The due date in the user's own timezone.
  due_date date not null,
  status text not null default 'suggested' check (status in ('suggested', 'confirmed', 'dismissed')),
  source text not null default 'other'
    check (source in ('google_docs', 'canvas', 'gmail', 'linkedin', 'job_boards', 'everything_else', 'other')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists deadlines_user_due_idx on public.deadlines (user_id, due_date);

-- ------------------------------------------------------------
-- Applications: jobs and internships applied to
-- ------------------------------------------------------------

create table if not exists public.applications (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  company text not null check (char_length(company) between 1 and 80),
  role text not null check (char_length(role) between 1 and 120),
  status text not null default 'draft' check (status in ('draft', 'applied', 'interview', 'offer', 'rejected')),
  applied_on date,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists applications_user_idx on public.applications (user_id, created_at desc);

-- ------------------------------------------------------------
-- Same rules on all three tables
-- ------------------------------------------------------------

do $$
declare
  t text;
begin
  foreach t in array array['memories', 'deadlines', 'applications'] loop
    execute format('alter table public.%I enable row level security', t);
    execute format('alter table public.%I force row level security', t);

    execute format('drop policy if exists "Users can read their own rows" on public.%I', t);
    execute format(
      'create policy "Users can read their own rows" on public.%I for select to authenticated
         using ((select auth.uid()) = user_id)', t);

    execute format('drop policy if exists "Users can add their own rows" on public.%I', t);
    execute format(
      'create policy "Users can add their own rows" on public.%I for insert to authenticated
         with check ((select auth.uid()) = user_id)', t);

    execute format('drop policy if exists "Users can change their own rows" on public.%I', t);
    execute format(
      'create policy "Users can change their own rows" on public.%I for update to authenticated
         using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id)', t);

    execute format('drop policy if exists "Users can delete their own rows" on public.%I', t);
    execute format(
      'create policy "Users can delete their own rows" on public.%I for delete to authenticated
         using ((select auth.uid()) = user_id)', t);

    -- Explicit privileges: newer projects grant nothing by default.
    -- (Memories get a stricter insert policy below.)
    execute format('revoke all on table public.%I from anon', t);
    execute format('grant select, insert, update, delete on table public.%I to authenticated', t);
    execute format('revoke truncate, references, trigger on table public.%I from authenticated', t);

    -- Database-managed timestamps (function from 0001).
    execute format('drop trigger if exists %I on public.%I', t || '_set_updated_at', t);
    execute format(
      'create trigger %I before insert or update on public.%I
         for each row execute function private.set_updated_at()', t || '_set_updated_at', t);
  end loop;
end;
$$;

-- While memory is paused ("Let Kelsie remember things" off), no new
-- memories can be saved, whatever tries to save them.
drop policy if exists "Users can add their own rows" on public.memories;
create policy "Users can add their own rows" on public.memories for insert to authenticated
  with check (
    (select auth.uid()) = user_id
    and exists (
      select 1 from public.profiles p
      where p.id = (select auth.uid()) and p.memory_enabled
    )
  );
