-- Kelsie: saving memory notes from the Chrome extension.
--
-- Privacy:
--  - Only the short note is stored, with the page title and which allowed
--    site it came from. Page text is never stored.
--  - Instead of the page's address, a one-way hash of it (page_key) is
--    stored, so revisiting a page updates its note without keeping a list
--    of the addresses you visited. The extension computes the hash; the
--    server never sees the address.
--
-- Run once in SQL Editor after 0001-0004. Safe to re-run.

alter table public.memories add column if not exists page_title text not null default ''
  check (char_length(page_title) <= 200);
alter table public.memories add column if not exists page_key text
  check (page_key is null or page_key ~ '^[0-9a-f]{64}$');

-- One note per page per user.
create unique index if not exists memories_user_page_key_idx
  on public.memories (user_id, page_key) where page_key is not null;

-- ------------------------------------------------------------
-- Usage: max 30 AI notes per user per day (the free AI tier is shared by
-- everyone), and at most one every 15 seconds.
-- ------------------------------------------------------------

create table if not exists public.memory_usage (
  user_id uuid not null references auth.users (id) on delete cascade,
  day date not null,
  notes integer not null default 0 check (notes >= 0),
  last_at timestamptz not null default now(),
  primary key (user_id, day)
);
alter table public.memory_usage enable row level security;
alter table public.memory_usage force row level security;
revoke all on table public.memory_usage from public, anon, authenticated;

-- Called before asking the AI for a note. Checks the extension session,
-- that memory is on, that the site is allowed, and the limits; then
-- counts this attempt. Returns 'ok' or the reason it can't save.
create or replace function public.extension_reserve_memory(p_token_hash text, p_source text)
returns text
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := private.extension_session_user(p_token_hash);
  prof public.profiles%rowtype;
  usage public.memory_usage%rowtype;
  today date := (now() at time zone 'utc')::date;
begin
  if uid is null then
    return 'not_connected';
  end if;
  select * into prof from public.profiles where id = uid;
  if not found or not prof.memory_enabled then
    return 'paused';
  end if;
  if p_source is null or not (p_source = any (prof.allowed_sites)) then
    return 'site_not_allowed';
  end if;

  insert into public.memory_usage (user_id, day, notes, last_at)
  values (uid, today, 0, now() - interval '1 hour')
  on conflict (user_id, day) do nothing;

  select * into usage from public.memory_usage where user_id = uid and day = today for update;
  if usage.notes >= 30 then
    return 'daily_limit';
  end if;
  if usage.last_at > now() - interval '15 seconds' then
    return 'too_soon';
  end if;

  update public.memory_usage set notes = notes + 1, last_at = now()
   where user_id = uid and day = today;
  -- Old usage rows aren't needed.
  delete from public.memory_usage where user_id = uid and day < today - 7;
  return 'ok';
end;
$$;

-- Saves (or, for a page already noted, updates) one note. Re-checks the
-- session, pause and site, since time has passed since the reservation.
-- Returns the note's id, or null if it couldn't be saved.
create or replace function public.extension_save_memory(
  p_token_hash text,
  p_page_key text,
  p_source text,
  p_title text,
  p_note text
)
returns uuid
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := private.extension_session_user(p_token_hash);
  prof public.profiles%rowtype;
  saved uuid;
begin
  if uid is null or p_page_key is null or p_page_key !~ '^[0-9a-f]{64}$' then
    return null;
  end if;
  select * into prof from public.profiles where id = uid;
  if not found or not prof.memory_enabled or not (p_source = any (prof.allowed_sites)) then
    return null;
  end if;

  insert into public.memories (user_id, content, source, page_title, page_key)
  values (uid, p_note, p_source, coalesce(p_title, ''), p_page_key)
  on conflict (user_id, page_key) where page_key is not null
  do update set content = excluded.content,
                page_title = excluded.page_title,
                source = excluded.source,
                created_at = now(),
                updated_at = now()
  returning id into saved;
  return saved;
end;
$$;

-- "Undo" in the orb: deletes one note, only if it belongs to the token's user.
create or replace function public.extension_delete_memory(p_token_hash text, p_memory_id uuid)
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := private.extension_session_user(p_token_hash);
begin
  if uid is null or p_memory_id is null then
    return false;
  end if;
  delete from public.memories where id = p_memory_id and user_id = uid;
  return found;
end;
$$;

do $$
declare
  fn text;
begin
  foreach fn in array array[
    'public.extension_reserve_memory(text, text)',
    'public.extension_save_memory(text, text, text, text, text)',
    'public.extension_delete_memory(text, uuid)'
  ] loop
    execute format('revoke all on function %s from public, authenticated', fn);
    execute format('grant execute on function %s to anon', fn);
  end loop;
end;
$$;
