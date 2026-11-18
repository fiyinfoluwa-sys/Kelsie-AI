/*
 * Sends sample messages through POST /api/kelsie/message with the real
 * intent router (ONNX model + tokenizer), and fake Supabase and Groq that
 * record what the route tried to do.
 */
import { createRequire } from "node:module";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fakeSupabase, type Call } from "./fake-supabase";

// router.js / tokenizer.js are CommonJS: load the real files with Node.
const nodeRequire = createRequire(import.meta.url);
vi.mock("@/intent-router/router.js", () => nodeRequire("../intent-router/router.js"));

const USER = { id: "00000000-0000-0000-0000-00000000000a", email_confirmed_at: "2026-01-01T00:00:00Z" };
const PROFILE = {
  name: "Sam",
  roles: ["student"],
  timezone: "America/Toronto",
  vibe: "chill",
  allowed_sites: ["gmail", "canvas"],
  onboarding_completed_at: "2026-01-01T00:00:00Z",
  memory_enabled: true,
  extension_connected_at: null,
  help_improve: false,
};

let fake: ReturnType<typeof fakeSupabase>;
vi.mock("@/lib/supabase/server", () => ({ createClient: async () => fake.client }));

const ai = vi.hoisted(() => ({
  completeJson: vi.fn(async () => ({ ok: true, text: '{"note": "Essay idea saved"}' })),
  completeText: vi.fn(async () => ({ ok: true, text: "Inflation is when prices rise over time." })),
}));
vi.mock("@/lib/ai", () => ai);

const { POST: message } = await import("@/app/api/kelsie/message/route");
const { POST: confirm } = await import("@/app/api/kelsie/confirm/route");

function post(handler: (r: Request) => Promise<Response>, path: string, body: unknown) {
  return handler(
    new Request(`http://localhost${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

const writes = (calls: Call[]) => calls.filter((c) => ["insert", "update", "delete", "upsert"].includes(c.op));

beforeEach(() => {
  fake = fakeSupabase({
    user: USER,
    rows: {
      profiles: [PROFILE],
      deadlines: [{ title: "Econ essay", due_date: "2026-12-31" }],
    },
    rpcResults: { reserve_assistant_call: true },
  });
  ai.completeJson.mockClear();
  ai.completeText.mockClear();
});

describe("POST /api/kelsie/message", () => {
  it("routes 'whats due this week' to the deadline query, with no AI call", async () => {
    const res = await post(message, "/api/kelsie/message", { text: "whats due this week" });
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.handler).toBe("deadline_query");
    expect(ai.completeJson).not.toHaveBeenCalled();
    expect(ai.completeText).not.toHaveBeenCalled();
    expect(fake.calls.some((c) => c.table === "deadlines" && c.op === "select")).toBe(true);
  });

  it("routes 'lock in for 25 mins' to focus and starts a 25-minute session, with no AI call", async () => {
    const before = Date.now();
    const body = await (await post(message, "/api/kelsie/message", { text: "lock in for 25 mins" })).json();
    expect(body.handler).toBe("focus");
    const insert = fake.calls.find((c) => c.table === "focus_sessions" && c.op === "insert");
    const minutes = (Date.parse((insert?.args[0] as { ends_at: string }).ends_at) - before) / 60_000;
    expect(Math.round(minutes)).toBe(25);
    expect(ai.completeJson).not.toHaveBeenCalled();
    expect(ai.completeText).not.toHaveBeenCalled();
  });

  it("routes 'save this for my essay' to remember: one small AI call, then saves the note", async () => {
    const body = await (await post(message, "/api/kelsie/message", { text: "save this for my essay" })).json();
    expect(body.handler).toBe("remember");
    expect(ai.completeJson).toHaveBeenCalledTimes(1);
    expect(ai.completeText).not.toHaveBeenCalled();
    const insert = fake.calls.find((c) => c.table === "memories" && c.op === "insert");
    expect(insert?.args[0]).toMatchObject({ content: "Essay idea saved", user_id: USER.id });
  });

  it("routes 'explain inflation' to the general AI", async () => {
    const body = await (await post(message, "/api/kelsie/message", { text: "explain inflation" })).json();
    expect(body.handler).toBe("general");
    expect(ai.completeText).toHaveBeenCalledTimes(1);
    expect(body.reply).toContain("Inflation");
  });

  it("answers 'delete my account' with a confirm step and deletes nothing", async () => {
    const body = await (await post(message, "/api/kelsie/message", { text: "delete my account" })).json();
    expect(body.handler).toBe("settings");
    expect(body.confirm).toEqual({ action: "delete_account" });
    expect(writes(fake.calls)).toEqual([]);
    expect(fake.calls.some((c) => c.rpc === "delete_my_account")).toBe(false);
    expect(ai.completeJson).not.toHaveBeenCalled();
    expect(ai.completeText).not.toHaveBeenCalled();
  });

  it("asks before clearing memories or removing a site", async () => {
    let body = await (await post(message, "/api/kelsie/message", { text: "wipe all my memories" })).json();
    expect(body.confirm).toEqual({ action: "clear_memories" });
    body = await (await post(message, "/api/kelsie/message", { text: "stop running on gmail" })).json();
    expect(body.confirm).toEqual({ action: "remove_site", site: "gmail" });
    expect(writes(fake.calls)).toEqual([]);
  });

  it("logs the route without the message text", async () => {
    const text = "whats due this week for my secret project";
    await post(message, "/api/kelsie/message", { text });
    const log = fake.calls.find((c) => c.rpc === "log_router_event");
    expect(log).toBeDefined();
    expect(JSON.stringify(log?.args)).not.toContain("secret");
    expect(fake.calls.some((c) => c.table === "router_training_examples")).toBe(false);
  });

  it("saves a training example only when 'Help improve Kelsie' is on", async () => {
    fake = fakeSupabase({
      user: USER,
      rows: { profiles: [{ ...PROFILE, help_improve: true }] },
      rpcResults: { reserve_assistant_call: true },
    });
    await post(message, "/api/kelsie/message", { text: "ty kelsie, email me at sam@example.com" });
    const saved = fake.calls.find((c) => c.table === "router_training_examples" && c.op === "insert");
    expect(saved).toBeDefined();
    expect(JSON.stringify(saved?.args)).not.toContain("sam@example.com");
  });

  it("refuses requests that aren't signed in, too long, or from another site", async () => {
    fake = fakeSupabase({ user: null });
    expect((await post(message, "/api/kelsie/message", { text: "hi" })).status).toBe(401);

    fake = fakeSupabase({ user: USER, rows: { profiles: [PROFILE] } });
    expect((await post(message, "/api/kelsie/message", { text: "x".repeat(1001) })).status).toBe(413);

    const crossSite = await message(
      new Request("http://localhost/api/kelsie/message", {
        method: "POST",
        headers: { "Content-Type": "application/json", Origin: "https://evil.example" },
        body: JSON.stringify({ text: "delete my account" }),
      }),
    );
    expect(crossSite.status).toBe(403);
  });
});

describe("POST /api/kelsie/confirm", () => {
  it("won't delete an account: that stays behind Settings' type-DELETE step", async () => {
    const res = await post(confirm, "/api/kelsie/confirm", { action: "delete_account" });
    expect(res.status).toBe(400);
    expect((await res.json()).error).toBe("use_settings");
    expect(fake.calls.some((c) => c.rpc === "delete_my_account")).toBe(false);
    expect(writes(fake.calls)).toEqual([]);
  });

  it("re-validates a confirmed deadline before saving it", async () => {
    let res = await post(confirm, "/api/kelsie/confirm", { action: "add_deadline", title: "Psych paper", dueDate: "not-a-date" });
    expect(res.status).toBe(400);
    res = await post(confirm, "/api/kelsie/confirm", { action: "add_deadline", title: "", dueDate: "2026-12-01" });
    expect(res.status).toBe(400);
    expect(writes(fake.calls)).toEqual([]);
  });
});
