/*
 * An in-memory stand-in for the Supabase client, recording every call so
 * tests can check exactly what the API tried to read, write or delete.
 */

export type Call = { table?: string; rpc?: string; op: string; args: unknown[] };

type Rows = Record<string, unknown[]>;

export function fakeSupabase(options: { user: { id: string } | null; rows?: Rows; rpcResults?: Record<string, unknown> }) {
  const calls: Call[] = [];

  function builder(table: string) {
    let op = "select";
    let single = false;
    const chain: Record<string, unknown> = {};
    const record = (name: string) =>
      (...args: unknown[]) => {
        if (["select", "insert", "update", "delete", "upsert"].includes(name) && op === "select") {
          op = name === "select" ? "select" : name;
        }
        if (name === "maybeSingle" || name === "single") single = true;
        calls.push({ table, op: name, args });
        return chain;
      };
    for (const name of ["select", "insert", "update", "delete", "upsert", "eq", "neq", "is", "gte", "lte", "order", "limit", "maybeSingle", "single"]) {
      chain[name] = record(name);
    }
    chain.then = (resolve: (value: unknown) => unknown) => {
      const rows = op === "select" ? (options.rows?.[table] ?? []) : [];
      return Promise.resolve(resolve({ data: single ? (rows[0] ?? null) : rows, error: null, count: rows.length }));
    };
    return chain;
  }

  const client = {
    auth: {
      getUser: async () => ({ data: { user: options.user }, error: null }),
    },
    from: (table: string) => builder(table),
    rpc: async (name: string, args?: unknown) => {
      calls.push({ rpc: name, op: "rpc", args: [args] });
      return { data: options.rpcResults?.[name] ?? null, error: null };
    },
  };
  return { client, calls };
}
