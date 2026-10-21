import type { AuthError } from "@supabase/supabase-js";

/*
 * Friendly auth error messages.
 *
 * None of them say whether an email is registered: sign-in failures share
 * one message, and sign-up / password reset always look like they worked.
 * That stops someone checking which emails have Kelsie accounts.
 */

export const PASSWORD_MIN = 8;
// Supabase hashes passwords with bcrypt, which only uses the first 72 bytes.
export const PASSWORD_MAX = 72;
const EMAIL_MAX = 254;

export const MESSAGES = {
  badEmail: "That email doesn't look quite right.",
  shortPassword: `Passwords need at least ${PASSWORD_MIN} characters.`,
  longPassword: `Passwords can be up to ${PASSWORD_MAX} characters.`,
  wrongCredentials: "That email and password don't match. Try again or reset your password.",
  unverified: "Verify your email first. The link is in your inbox.",
  tooMany: "Too many tries. Wait a minute and try again.",
  passwordsDiffer: "Those passwords don't match.",
  samePassword: "Pick a password you haven't used before.",
  weakPassword: "That password is too easy to guess. Try a longer one.",
  generic: "Something went wrong. Try again in a moment.",
} as const;

export function normalizeEmail(value: FormDataEntryValue | null): string | null {
  if (typeof value !== "string") return null;
  const email = value.trim().toLowerCase();
  if (email.length > EMAIL_MAX || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return null;
  return email;
}

export function passwordProblem(value: FormDataEntryValue | null): string | null {
  if (typeof value !== "string" || value.length < PASSWORD_MIN) return MESSAGES.shortPassword;
  if (new TextEncoder().encode(value).length > PASSWORD_MAX) return MESSAGES.longPassword;
  return null;
}

export function isRateLimited(error: AuthError): boolean {
  return error.status === 429 || error.code?.startsWith("over_") === true;
}

export function signInMessage(error: AuthError): string {
  if (isRateLimited(error)) return MESSAGES.tooMany;
  if (error.code === "email_not_confirmed") return MESSAGES.unverified;
  // invalid_credentials and anything unexpected get the same wording, so a
  // wrong password and an unknown email look identical.
  return MESSAGES.wrongCredentials;
}
