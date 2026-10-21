-- Kelsie: explicit table privileges for profiles.
--
-- Newer Supabase projects don't grant API roles access to tables created in
-- SQL, so grant exactly what the app needs. Row Level Security (from 0001)
-- still limits every signed-in user to their own row.
--
-- Run once in SQL Editor after 0001. Safe to re-run.

grant usage on schema public to authenticated;
grant select, insert, update on table public.profiles to authenticated;

-- Logged-out visitors get nothing, and nobody can delete or truncate rows
-- directly (deletion goes through delete_my_account()).
revoke all on table public.profiles from anon;
revoke truncate, delete, references, trigger on table public.profiles from authenticated;
