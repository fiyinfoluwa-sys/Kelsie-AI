-- Kelsie: connecting the Chrome extension to an account.
--
-- How it works:
-- 1. A signed-in user on the website asks for a connection code. The
--    website generates a random code, stores only its SHA-256 hash here,
--    and hands the code to the extension. Codes expire after 5 minutes
--    and work once.
-- 2. The extension trades the code (through the website) for a long random
--    session token. Again, only the token's hash is stored.
-- 3. Every extension request sends the token; these functions look up its
--    hash and only ever act on that token's user.
--
-- Neither table has any policies, and API roles have no privileges on
-- them, so they can only be reached through the functions below.
--
-- Run once in SQL Editor after 0001-0003. Safe to re-run.

create table if not exists public.extension_connect_codes (
  code_hash text primary key check (code_hash ~ '^[0-9a-f]{64}$'),
  user_id uuid not null references auth.users (id) on delete cascade,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null,
  used_at timestamptz
);
create index if not exists extension_connect_codes_user_idx on public.extension_connect_codes (user_id);

create table if not exists public.extension_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  token_hash text not null unique check (token_hash ~ '^[0-9a-f]{64}$'),
  created_at timestamptz not null default now(),
  last_used_at timestamptz not null default now()
);
create index if not exists extension_sessions_user_idx on public.extension_sessions (user_id);

alter table public.extension_connect_codes enable row level security;
alter table public.extension_connect_codes force row level security;
alter table public.extension_sessions enable row level security;
alter table public.extension_sessions force row level security;
revoke all on table public.extension_connect_codes from public, anon, authenticated;
revoke all on table public.extension_sessions from public, anon, authenticated;

-- A session that hasn't been used for 30 days stops working.
create or replace function private.extension_session_user(p_token_hash text)
returns uuid
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid;
begin
  if p_token_hash is null or p_token_hash !~ '^[0-9a-f]{64}$' then
    return null;
  end if;
  update public.extension_sessions
     set last_used_at = now()
   where token_hash = p_token_hash
     and last_used_at > now() - interval '30 days'
  returning user_id into uid;
  return uid;
end;
$$;
revoke all on function private.extension_session_user(text) from public, anon, authenticated;

-- Keeps the profile's "connected" flag (shown on the home page) in step
-- with whether any extension session exists.
create or replace function private.refresh_extension_connected(p_user uuid)
returns void
language sql
security definer
set search_path = ''
as $$
  update public.profiles
     set extension_connected_at = case
       when exists (select 1 from public.extension_sessions s where s.user_id = p_user)
         then coalesce(extension_connected_at, now())
       else null
     end
   where id = p_user;
$$;
revoke all on function private.refresh_extension_connected(uuid) from public, anon, authenticated;

-- ------------------------------------------------------------
-- Called by the signed-in website user
-- ------------------------------------------------------------

-- Stores the hash of a new connection code for the caller.
create or replace function public.create_extension_connect_code(p_code_hash text)
returns boolean
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
  if p_code_hash is null or p_code_hash !~ '^[0-9a-f]{64}$' then
    raise exception 'Bad code' using errcode = '22023';
  end if;
  -- Clean up old codes, and cap how many can be waiting at once.
  delete from public.extension_connect_codes
   where user_id = uid and (expires_at < now() or used_at is not null);
  if (select count(*) from public.extension_connect_codes where user_id = uid) >= 5 then
    raise exception 'Too many codes' using errcode = '54000';
  end if;
  insert into public.extension_connect_codes (code_hash, user_id, expires_at)
  values (p_code_hash, uid, now() + interval '5 minutes');
  return true;
end;
$$;
revoke all on function public.create_extension_connect_code(text) from public, anon;
grant execute on function public.create_extension_connect_code(text) to authenticated;

-- Signing out on the website disconnects every extension for that user.
create or replace function public.revoke_my_extension_sessions()
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := auth.uid();
begin
  if uid is null then
    return;
  end if;
  delete from public.extension_sessions where user_id = uid;
  delete from public.extension_connect_codes where user_id = uid;
  perform private.refresh_extension_connected(uid);
end;
$$;
revoke all on function public.revoke_my_extension_sessions() from public, anon;
grant execute on function public.revoke_my_extension_sessions() to authenticated;

-- ------------------------------------------------------------
-- Called by the website's extension API on the extension's behalf.
-- These run without a user session (role "anon"); possession of a valid
-- code or token is what proves who the caller is. Both are 256-bit random
-- values, so they can't be guessed.
-- ------------------------------------------------------------

-- Trades a connection code for a session. Returns false if the code is
-- unknown, expired or already used. The code is marked used first, in the
-- same statement, so two simultaneous attempts can't both succeed.
create or replace function public.exchange_extension_code(p_code_hash text, p_token_hash text)
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid;
begin
  if p_code_hash is null or p_code_hash !~ '^[0-9a-f]{64}$'
     or p_token_hash is null or p_token_hash !~ '^[0-9a-f]{64}$' then
    return false;
  end if;
  update public.extension_connect_codes
     set used_at = now()
   where code_hash = p_code_hash and used_at is null and expires_at > now()
  returning user_id into uid;
  if uid is null then
    return false;
  end if;
  insert into public.extension_sessions (user_id, token_hash) values (uid, p_token_hash);
  perform private.refresh_extension_connected(uid);
  return true;
end;
$$;

-- What the extension needs to know, and nothing more.
create or replace function public.extension_state(p_token_hash text)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := private.extension_session_user(p_token_hash);
  result jsonb;
begin
  if uid is null then
    return null;
  end if;
  select jsonb_build_object(
           'name', p.name,
           'vibe', p.vibe,
           'allowed_sites', to_jsonb(p.allowed_sites),
           'memory_enabled', p.memory_enabled
         )
    into result
    from public.profiles p
   where p.id = uid;
  return result;
end;
$$;

-- The popup's Pause button (same setting as Pause on the website).
create or replace function public.extension_set_memory(p_token_hash text, p_enabled boolean)
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := private.extension_session_user(p_token_hash);
begin
  if uid is null or p_enabled is null then
    return false;
  end if;
  update public.profiles set memory_enabled = p_enabled where id = uid;
  return true;
end;
$$;

-- The popup's Disconnect button.
create or replace function public.extension_disconnect(p_token_hash text)
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid;
begin
  if p_token_hash is null or p_token_hash !~ '^[0-9a-f]{64}$' then
    return false;
  end if;
  delete from public.extension_sessions where token_hash = p_token_hash returning user_id into uid;
  if uid is not null then
    perform private.refresh_extension_connected(uid);
  end if;
  return uid is not null;
end;
$$;

do $$
declare
  fn text;
begin
  foreach fn in array array[
    'public.exchange_extension_code(text, text)',
    'public.extension_state(text)',
    'public.extension_set_memory(text, boolean)',
    'public.extension_disconnect(text)'
  ] loop
    execute format('revoke all on function %s from public, authenticated', fn);
    execute format('grant execute on function %s to anon', fn);
  end loop;
end;
$$;
