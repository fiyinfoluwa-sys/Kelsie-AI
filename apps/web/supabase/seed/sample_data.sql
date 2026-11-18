-- Kelsie: sample memories, deadlines and applications for ONE test account.
--
-- 1. Replace YOUR_EMAIL_HERE below with your test account's email.
--    (Edit it in the SQL Editor only; don't commit your email.)
-- 2. Run in SQL Editor. Re-running replaces the previous sample rows.
--
-- Runs as the database owner, so it can write to your account directly.
-- It refuses to run if the email doesn't match exactly one user.

do $$
declare
  target_email text := 'YOUR_EMAIL_HERE';
  uid uuid;
  tz text;
  today date;
begin
  select id into uid from auth.users where lower(email) = lower(target_email);
  if uid is null then
    raise exception 'No user with email %. Check the address and try again.', target_email;
  end if;

  select coalesce(timezone, 'UTC') into tz from public.profiles where id = uid;
  if tz is null then
    raise exception 'That user has no profile yet. Finish onboarding first.';
  end if;
  today := (now() at time zone tz)::date;

  -- Start fresh so re-running doesn't duplicate.
  delete from public.memories where user_id = uid;
  delete from public.deadlines where user_id = uid;
  delete from public.applications where user_id = uid;

  insert into public.memories (user_id, content, source, created_at) values
    (uid, 'Econ essay thesis: rent control reduces housing supply long-term. Prof wants 3 sources.', 'google_docs', now() - interval '3 hours'),
    (uid, 'Group project meeting moved to Thursday 4pm in the library.', 'gmail', now() - interval '1 day 2 hours'),
    (uid, 'Stats midterm covers chapters 4 to 7, formula sheet allowed.', 'canvas', now() - interval '3 days'),
    (uid, 'Recruiter at Shopify said the next round is a take-home, due within a week.', 'linkedin', now() - interval '6 days');

  insert into public.deadlines (user_id, title, details, due_date, status, source) values
    (uid, 'Econ essay', 'ECON 210 · 2,000 words', today + 1, 'confirmed', 'canvas'),
    (uid, 'Stats midterm', 'STAT 230 · Chapters 4–7', today + 4, 'confirmed', 'canvas'),
    (uid, 'Shopify take-home', 'Frontend challenge', today + 6, 'confirmed', 'gmail'),
    (uid, 'Lab report 3', 'CHEM 120 · spotted in your syllabus PDF', today + 9, 'suggested', 'canvas');

  insert into public.applications (user_id, company, role, status, applied_on) values
    (uid, 'Shopify', 'Frontend Developer Intern', 'interview', today - 12),
    (uid, 'Wealthsimple', 'Product Design Intern', 'applied', today - 5),
    (uid, 'Figma', 'Software Engineering Intern', 'draft', null);

  -- Shows "Riding along on X sites". Remove this line to see the
  -- "Not in Chrome yet" state instead.
  update public.profiles set extension_connected_at = now() where id = uid;

  raise notice 'Sample data added for %', target_email;
end;
$$;
