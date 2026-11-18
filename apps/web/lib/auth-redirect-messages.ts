/*
 * Messages for the ?error= / ?verify= flags that auth redirects add to the
 * sign-in page. Only known flags map to text: nothing from the URL is ever
 * shown on the page directly.
 */
const MESSAGES: Record<string, string> = {
  google: "Google sign-in didn't go through. Try again.",
  link: "That link has expired or was already used. Sign in, or sign up again for a new one.",
  verify: "Verify your email to continue. The link is in your inbox.",
};

export function redirectMessage(params: Record<string, string | string[] | undefined>) {
  if (params.verify) return MESSAGES.verify;
  const key = typeof params.error === "string" ? params.error : undefined;
  return key ? MESSAGES[key] : undefined;
}
