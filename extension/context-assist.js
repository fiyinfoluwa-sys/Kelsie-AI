(() => {
  /* =========================================================
     CONFIGURATION
  ========================================================= */

  const ROOT_ID =
    "kelsie-extension-root";

  const CARD_ID =
    "kelsie-context-resurface-card";

  const ITEMS_KEY =
    "kelsie_context_items_v1";

  /*
   * Automatic contextual resurfacing should not begin
   * immediately when a page loads.
   */
  const MIN_VISIBLE_DWELL_MS =
    12000;

  /*
   * If the local match is extremely strong, Kelsie can
   * verify it after the dwell period without requiring
   * another interaction.
   */
  const HIGH_LOCAL_SCORE =
    6.2;

  /*
   * Otherwise we require at least a little real engagement
   * with the page.
   */
  const MIN_INTERACTION_SCORE =
    1;

  /*
   * Only locally plausible items are ever sent to the
   * backend matcher.
   */
  const LOCAL_MATCH_THRESHOLD =
    2.2;

  const MAX_AI_CANDIDATES =
    5;

  const MAX_ACTIVE_ITEMS =
    30;

  const AI_MATCH_THRESHOLD =
    0.88;

  const CAPTURE_THRESHOLD =
    0.82;

  /*
   * Do not repeatedly resurface the same thing.
   */
  const MATCH_COOLDOWN_MS =
    6 * 60 * 60 * 1000;

  /*
   * Lightweight maintenance only.
   *
   * This replaces the old MutationObserver.
   */
  const MAINTENANCE_INTERVAL_MS =
    2500;

  const URL_CHECK_INTERVAL_MS =
    1500;

  /* =========================================================
     GUARDS
  ========================================================= */

  if (
    window.top !==
    window.self
  ) {
    return;
  }

  if (
    window.__KELSIE_CONTEXT_ASSIST__
  ) {
    return;
  }

  window.__KELSIE_CONTEXT_ASSIST__ =
    true;

  const PARENT_SOURCE =
    "kelsie-context-assist";

  const HOOK_SOURCE =
    "kelsie-keeping-hook";

  /* =========================================================
     STATE
  ========================================================= */

  const pendingRpc =
    new Map();

  const dismissedOnPage =
    new Set();

  let observedUrl =
    window.location.href;

  let currentMatch =
    null;

  let currentMode =
    "idle";

  let matchTimer =
    null;

  let matchRequestInFlight =
    false;

  let matchEvaluatedForUrl =
    false;

  let firstVisibleAt =
    document.visibilityState ===
      "visible"
      ? Date.now()
      : null;

  let interactionScore =
    0;

  let startScrollY =
    window.scrollY;

  let readingSuppressedForPage =
    false;

  let maintenanceTimer =
    null;

  let urlTimer =
    null;

  /* =========================================================
     GENERIC HELPERS
  ========================================================= */

  const clean = (
    value,
    limit = 1000
  ) =>
    String(
      value || ""
    )
      .replace(
        /\s+/g,
        " "
      )
      .trim()
      .slice(
        0,
        limit
      );

  const clamp = (
    value,
    min,
    max
  ) =>
    Math.min(
      Math.max(
        value,
        min
      ),
      max
    );

  function newId(
    prefix = "ctx"
  ) {
    if (
      window.crypto &&
      typeof window.crypto
        .randomUUID ===
        "function"
    ) {
      return (
        `${prefix}-`
        + window.crypto
          .randomUUID()
      );
    }

    return (
      `${prefix}-`
      + `${Date.now()}-`
      + Math.random()
        .toString(16)
        .slice(2)
    );
  }

  /* =========================================================
     LOCAL SEMANTIC FILTER
     ---------------------------------------------------------
     This is deliberately cheap.

     It does NOT call the AI.

     Its only job is to answer:
     "Is there even enough overlap to justify asking
     the backend to think about this?"
  ========================================================= */

  const STOP_WORDS =
    new Set([
      "a",
      "about",
      "after",
      "again",
      "all",
      "also",
      "am",
      "an",
      "and",
      "are",
      "as",
      "at",
      "be",
      "been",
      "before",
      "being",
      "but",
      "by",
      "can",
      "could",
      "did",
      "do",
      "does",
      "doing",
      "for",
      "from",
      "get",
      "got",
      "had",
      "has",
      "have",
      "he",
      "her",
      "here",
      "him",
      "his",
      "how",
      "i",
      "if",
      "in",
      "into",
      "is",
      "it",
      "its",
      "just",
      "me",
      "more",
      "my",
      "need",
      "of",
      "on",
      "or",
      "our",
      "out",
      "should",
      "so",
      "some",
      "that",
      "the",
      "their",
      "them",
      "then",
      "there",
      "they",
      "this",
      "to",
      "up",
      "us",
      "want",
      "was",
      "we",
      "were",
      "what",
      "when",
      "where",
      "which",
      "who",
      "why",
      "will",
      "with",
      "would",
      "you",
      "your",
    ]);

  function normalizeToken(
    token
  ) {
    let value =
      String(
        token || ""
      )
        .toLowerCase()
        .replace(
          /[^a-z0-9']/g,
          ""
        );

    /*
     * Very small amount of stemming.
     *
     * We are not trying to do NLP here.
     * We simply want:
     *
     * editing -> edit
     * applications -> application
     * compares -> compare
     */
    if (
      value.length > 6 &&
      value.endsWith(
        "ing"
      )
    ) {
      value =
        value.slice(
          0,
          -3
        );
    } else if (
      value.length > 5 &&
      value.endsWith(
        "ed"
      )
    ) {
      value =
        value.slice(
          0,
          -2
        );
    } else if (
      value.length > 5 &&
      value.endsWith(
        "s"
      ) &&
      !value.endsWith(
        "ss"
      )
    ) {
      value =
        value.slice(
          0,
          -1
        );
    }

    return value;
  }

  function tokens(
    value
  ) {
    const result =
      new Set();

    String(
      value || ""
    )
      .toLowerCase()
      .match(
        /[a-z0-9']+/g
      )
      ?.forEach(
        (
          raw
        ) => {
          const token =
            normalizeToken(
              raw
            );

          if (
            token.length >=
              3 &&
            !STOP_WORDS.has(
              token
            )
          ) {
            result.add(
              token
            );
          }
        }
      );

    return result;
  }

  function overlapCount(
    first,
    second
  ) {
    if (
      !first.size ||
      !second.size
    ) {
      return 0;
    }

    let count =
      0;

    first.forEach(
      (
        value
      ) => {
        if (
          second.has(
            value
          )
        ) {
          count +=
            1;
        }
      }
    );

    return count;
  }

  function itemLocalScore(
    item,
    pageTokens
  ) {
    const thingTokens =
      tokens(
        item.thing
      );

    const intentTokens =
      tokens(
        item.intent
      );

    const reasonTokens =
      tokens(
        item.reason
      );

    const entityTokens =
      tokens(
        (
          item.entities ||
          []
        ).join(
          " "
        )
      );

    const futureTokens =
      tokens(
        (
          item.futureRelevance ||
          []
        ).join(
          " "
        )
      );

    const sourceTokens =
      tokens(
        item.sourceTitle
      );

    const thingOverlap =
      overlapCount(
        thingTokens,
        pageTokens
      );

    const intentOverlap =
      overlapCount(
        intentTokens,
        pageTokens
      );

    const reasonOverlap =
      overlapCount(
        reasonTokens,
        pageTokens
      );

    const entityOverlap =
      overlapCount(
        entityTokens,
        pageTokens
      );

    const futureOverlap =
      overlapCount(
        futureTokens,
        pageTokens
      );

    const sourceOverlap =
      overlapCount(
        sourceTokens,
        pageTokens
      );

    /*
     * Future relevance and entities are intentionally
     * weighted most heavily.
     *
     * Example:
     *
     * item future relevance:
     * "editing a resume"
     *
     * current page:
     * "Fiyin Resume - Google Docs"
     *
     * That should make it through the local filter.
     */
    return (
      futureOverlap *
        2.25
      +
      entityOverlap *
        1.75
      +
      intentOverlap *
        1.4
      +
      thingOverlap *
        1.2
      +
      reasonOverlap *
        0.8
      +
      sourceOverlap *
        0.45
    );
  }

  /* =========================================================
     WAIT FOR EXISTING EXTENSION SHELL
  ========================================================= */

  function waitForShell() {
    const host =
      document.getElementById(
        ROOT_ID
      );

    const shadow =
      host?.shadowRoot;

    const launcher =
      shadow
        ?.getElementById(
          "kelsie-launcher"
        );

    const panelHost =
      shadow
        ?.getElementById(
          "kelsie-panel-host"
        );

    const reminderPopup =
      shadow
        ?.getElementById(
          "kelsie-reminder-popup"
        );

    const widgetFrame =
      shadow
        ?.getElementById(
          "kelsie-widget-frame"
        );

    if (
      !host ||
      !shadow ||
      !launcher ||
      !panelHost ||
      !widgetFrame
    ) {
      window.setTimeout(
        waitForShell,
        150
      );

      return;
    }

    start({
      host,
      shadow,
      launcher,
      panelHost,
      reminderPopup,
      widgetFrame,
    });
  }

  /* =========================================================
     MAIN
  ========================================================= */

  function start({
    host,
    shadow,
    launcher,
    panelHost,
    reminderPopup,
    widgetFrame,
  }) {
    if (
      shadow.getElementById(
        CARD_ID
      )
    ) {
      return;
    }

    /* =======================================================
       UI STYLE
    ======================================================= */

    const style =
      document.createElement(
        "style"
      );

    style.textContent = `
      #${CARD_ID} {
        position: fixed;
        z-index: 2147483647;

        width:
          min(
            308px,
            calc(
              100vw - 20px
            )
          );

        box-sizing:
          border-box;

        overflow:
          hidden;

        border:
          1px solid
          rgba(
            49,
            55,
            64,
            0.085
          );

        border-radius:
          20px;

        background:
          linear-gradient(
            145deg,
            rgba(
              255,
              255,
              255,
              0.982
            ),
            rgba(
              244,
              246,
              248,
              0.958
            )
          );

        box-shadow:
          0 18px 46px
            rgba(
              28,
              33,
              42,
              0.13
            ),
          0 4px 13px
            rgba(
              28,
              33,
              42,
              0.055
            ),
          inset
            0 1px 0
            rgba(
              255,
              255,
              255,
              0.94
            );

        backdrop-filter:
          blur(22px)
          saturate(1.02);

        -webkit-backdrop-filter:
          blur(22px)
          saturate(1.02);

        pointer-events:
          auto;

        font-family:
          Inter,
          -apple-system,
          BlinkMacSystemFont,
          "Segoe UI",
          Helvetica,
          Arial,
          sans-serif;

        animation:
          kelsieContextIn
          180ms
          ease-out;
      }

      #${CARD_ID}[hidden] {
        display:
          none !important;
      }

      .kc-nudge,
      .kc-capture {
        padding:
          13px
          13px
          12px;
      }

      .kc-topline {
        display:
          flex;

        align-items:
          flex-start;

        justify-content:
          space-between;

        gap:
          10px;
      }

      .kc-identity {
        display:
          flex;

        align-items:
          center;

        gap:
          8px;

        min-width:
          0;
      }

      .kc-orb {
        position:
          relative;

        width:
          23px;

        height:
          23px;

        flex:
          0 0 auto;

        overflow:
          hidden;

        border:
          1px solid
          rgba(
            240,
            242,
            247,
            0.88
          );

        border-radius:
          50%;

        background:
          radial-gradient(
            circle at
              25% 19%,
            rgba(
              255,
              255,
              255,
              0.98
            )
              0 6%,
            rgba(
              240,
              242,
              247,
              0.72
            )
              7% 14%,
            transparent
              27%
          ),
          conic-gradient(
            from 218deg
              at 52% 50%,
            #b8becb 0%,
            #9ca3b1 18%,
            #808899 38%,
            #727a8b 58%,
            #949baa 78%,
            #b1b7c4 100%
          );

        box-shadow:
          0 3px 8px
            rgba(
              70,
              76,
              91,
              0.16
            ),
          0 0 9px
            rgba(
              151,
              160,
              184,
              0.22
            ),
          inset
            2px 2px 4px
            rgba(
              255,
              255,
              255,
              0.65
            ),
          inset
            -3px -4px 5px
            rgba(
              72,
              79,
              95,
              0.18
            );
      }

      .kc-eyebrow {
        color:
          #858c95;

        font-size:
          8px;

        line-height:
          1.2;

        font-weight:
          620;
      }

      .kc-close {
        width:
          26px;

        height:
          26px;

        min-width:
          26px;

        display:
          grid;

        place-items:
          center;

        padding:
          0;

        border:
          0;

        border-radius:
          50%;

        background:
          transparent;

        color:
          #9aa0a7;

        cursor:
          pointer;

        font-family:
          inherit;

        font-size:
          17px;

        font-weight:
          300;

        line-height:
          1;
      }

      .kc-close:hover {
        color:
          #5f666f;

        background:
          rgba(
            54,
            61,
            70,
            0.045
          );
      }

      .kc-message {
        margin-top:
          10px;

        color:
          #343a44;

        font-size:
          10.5px;

        line-height:
          1.5;

        font-weight:
          590;

        letter-spacing:
          -0.006em;
      }

      .kc-source {
        margin-top:
          6px;

        color:
          #9aa0a7;

        font-size:
          7.8px;

        line-height:
          1.3;
      }

      .kc-actions {
        display:
          flex;

        align-items:
          center;

        gap:
          5px;

        margin-top:
          10px;
      }

      .kc-action {
        min-height:
          27px;

        padding:
          0 8px;

        border:
          0;

        border-radius:
          8px;

        background:
          transparent;

        color:
          #7c838d;

        cursor:
          pointer;

        font-family:
          inherit;

        font-size:
          8px;

        font-weight:
          620;
      }

      .kc-action:hover {
        color:
          #4f5760;

        background:
          rgba(
            60,
            67,
            77,
            0.045
          );
      }

      .kc-action.done {
        color:
          #66766f;
      }

      .kc-capture-title {
        margin-top:
          9px;

        color:
          #353b44;

        font-size:
          11px;

        line-height:
          1.35;

        font-weight:
          640;
      }

      .kc-capture-copy {
        margin-top:
          4px;

        color:
          #858c95;

        font-size:
          8.5px;

        line-height:
          1.4;
      }

      .kc-capture-form {
        display:
          flex;

        gap:
          6px;

        margin-top:
          10px;
      }

      .kc-capture-input {
        min-width:
          0;

        flex:
          1;

        min-height:
          34px;

        padding:
          7px 10px;

        border:
          1px solid
          rgba(
            59,
            66,
            76,
            0.105
          );

        border-radius:
          11px;

        background:
          rgba(
            250,
            251,
            252,
            0.86
          );

        color:
          #333942;

        outline:
          none;

        font:
          inherit;

        font-size:
          9px;
      }

      .kc-capture-input:focus {
        border-color:
          rgba(
            128,
            136,
            153,
            0.32
          );

        box-shadow:
          0 0 0 2px
          rgba(
            128,
            136,
            153,
            0.075
          );
      }

      .kc-save {
        min-height:
          34px;

        padding:
          0 10px;

        border:
          1px solid
          rgba(
            50,
            56,
            66,
            0.08
          );

        border-radius:
          10px;

        background:
          rgba(
            239,
            241,
            244,
            0.94
          );

        color:
          #59616b;

        cursor:
          pointer;

        font:
          inherit;

        font-size:
          8px;

        font-weight:
          650;
      }

      .kc-save:disabled {
        opacity:
          0.45;

        cursor:
          default;
      }

      .kc-capture-status {
        margin-top:
          8px;

        color:
          #777f88;

        font-size:
          8px;

        line-height:
          1.35;
      }

      @keyframes
      kelsieContextIn {
        from {
          opacity:
            0;

          transform:
            translateY(
              4px
            )
            scale(
              0.99
            );
        }

        to {
          opacity:
            1;

          transform:
            translateY(
              0
            )
            scale(
              1
            );
        }
      }

      @media (
        prefers-reduced-motion:
          reduce
      ) {
        #${CARD_ID} {
          animation:
            none !important;
        }
      }
    `;

    shadow.appendChild(
      style
    );

    /* =======================================================
       UI MARKUP
    ======================================================= */

    const card =
      document.createElement(
        "section"
      );

    card.id =
      CARD_ID;

    card.hidden =
      true;

    card.setAttribute(
      "aria-live",
      "polite"
    );

    card.innerHTML = `
      <div
        class="kc-nudge"
        id="kc-nudge"
        hidden
      >
        <div class="kc-topline">
          <div class="kc-identity">
            <span
              class="kc-orb"
              aria-hidden="true"
            ></span>

            <span
              class="kc-eyebrow"
            >
              You mentioned this earlier
            </span>
          </div>

          <button
            class="kc-close"
            id="kc-dismiss"
            type="button"
            aria-label="Dismiss this suggestion"
          >
            ×
          </button>
        </div>

        <div
          class="kc-message"
          id="kc-message"
        ></div>

        <div
          class="kc-source"
          id="kc-source"
        ></div>

        <div
          class="kc-actions"
        >
          <button
            class="kc-action"
            id="kc-open-source"
            type="button"
          >
            Open source
          </button>

          <button
            class="kc-action done"
            id="kc-done"
            type="button"
          >
            Done
          </button>
        </div>
      </div>

      <div
        class="kc-capture"
        id="kc-capture"
        hidden
      >
        <div class="kc-topline">
          <div class="kc-identity">
            <span
              class="kc-orb"
              aria-hidden="true"
            ></span>

            <span
              class="kc-eyebrow"
            >
              Keep this in mind
            </span>
          </div>

          <button
            class="kc-close"
            id="kc-capture-close"
            type="button"
            aria-label="Cancel"
          >
            ×
          </button>
        </div>

        <div
          class="kc-capture-title"
        >
          What should Kelsie keep this
          in mind for?
        </div>

        <div
          class="kc-capture-copy"
        >
          Give it just enough context
          to know when this may become
          useful later.
        </div>

        <div
          class="kc-capture-form"
        >
          <input
            class="kc-capture-input"
            id="kc-capture-input"
            type="text"
            maxlength="300"
            autocomplete="off"
            placeholder="e.g. my ECON assignment"
          />

          <button
            class="kc-save"
            id="kc-capture-save"
            type="button"
          >
            Save
          </button>
        </div>

        <div
          class="kc-capture-status"
          id="kc-capture-status"
          hidden
        ></div>
      </div>
    `;

    shadow.appendChild(
      card
    );

    const nudgeView =
      card.querySelector(
        "#kc-nudge"
      );

    const captureView =
      card.querySelector(
        "#kc-capture"
      );

    const messageElement =
      card.querySelector(
        "#kc-message"
      );

    const sourceElement =
      card.querySelector(
        "#kc-source"
      );

    const openSourceButton =
      card.querySelector(
        "#kc-open-source"
      );

    const captureInput =
      card.querySelector(
        "#kc-capture-input"
      );

    const captureSave =
      card.querySelector(
        "#kc-capture-save"
      );

    const captureStatus =
      card.querySelector(
        "#kc-capture-status"
      );

    /* =======================================================
       IFRAME / BACKEND BRIDGE
    ======================================================= */

    function postToHook(
      type,
      payload = {}
    ) {
      if (
        !widgetFrame
          .contentWindow
      ) {
        return;
      }

      widgetFrame
        .contentWindow
        .postMessage(
          {
            source:
              PARENT_SOURCE,

            type,

            ...payload,
          },
          "*"
        );
    }

    function rpcToHook(
      type,
      payload,
      timeoutMs =
        45000
    ) {
      const requestId =
        newId(
          "rpc"
        );

      return new Promise(
        (
          resolve,
          reject
        ) => {
          const timeout =
            window.setTimeout(
              () => {
                pendingRpc.delete(
                  requestId
                );

                reject(
                  new Error(
                    "Kelsie context request timed out."
                  )
                );
              },
              timeoutMs
            );

          pendingRpc.set(
            requestId,
            {
              resolve,
              reject,
              timeout,
            }
          );

          postToHook(
            type,
            {
              request_id:
                requestId,

              payload,
            }
          );
        }
      );
    }

    /* =======================================================
       STORAGE
    ======================================================= */

    async function loadItems() {
      try {
        const result =
          await chrome
            .storage
            .local
            .get({
              [ITEMS_KEY]:
                [],
            });

        const rawItems =
          Array.isArray(
            result[
              ITEMS_KEY
            ]
          )
            ? result[
                ITEMS_KEY
              ]
            : [];

        return rawItems
          .map(
            normalizeStoredItem
          )
          .filter(
            Boolean
          );
      } catch (error) {
        console.error(
          "[Kelsie Context] Could not load items:",
          error
        );

        return [];
      }
    }

    async function saveItems(
      items
    ) {
      try {
        await chrome
          .storage
          .local
          .set({
            [ITEMS_KEY]:
              items,
          });
      } catch (error) {
        console.error(
          "[Kelsie Context] Could not save items:",
          error
        );
      }
    }

    function normalizeStoredItem(
      raw
    ) {
      if (
        !raw ||
        typeof raw !==
          "object"
      ) {
        return null;
      }

      const id =
        clean(
          raw.id,
          140
        );

      const thing =
        clean(
          raw.thing,
          220
        );

      if (
        !id ||
        !thing
      ) {
        return null;
      }

      return {
        id,

        thing,

        reason:
          clean(
            raw.reason,
            420
          ),

        intent:
          clean(
            raw.intent,
            320
          ),

        kind:
          clean(
            raw.kind,
            40
          ) ||
          "saved_context",

        entities:
          Array.isArray(
            raw.entities
          )
            ? raw.entities
                .map(
                  (
                    item
                  ) =>
                    clean(
                      item,
                      120
                    )
                )
                .filter(
                  Boolean
                )
                .slice(
                  0,
                  6
                )
            : [],

        futureRelevance:
          Array.isArray(
            raw.futureRelevance
          )
            ? raw
                .futureRelevance
                .map(
                  (
                    item
                  ) =>
                    clean(
                      item,
                      180
                    )
                )
                .filter(
                  Boolean
                )
                .slice(
                  0,
                  6
                )
            : [],

        sourceTitle:
          clean(
            raw.sourceTitle,
            300
          ),

        sourceUrl:
          clean(
            raw.sourceUrl,
            1200
          ),

        sourceDomain:
          clean(
            raw.sourceDomain,
            180
          ),

        sourceExcerpt:
          clean(
            raw.sourceExcerpt,
            800
          ),

        confidence:
          Number(
            raw.confidence ||
            0
          ),

        status:
          [
            "active",
            "completed",
            "forgotten",
          ].includes(
            raw.status
          )
            ? raw.status
            : "active",

        createdAt:
          clean(
            raw.createdAt,
            80
          ),

        updatedAt:
          clean(
            raw.updatedAt,
            80
          ),

        completedAt:
          clean(
            raw.completedAt,
            80
          ),

        lastSurfacedAt:
          clean(
            raw.lastSurfacedAt,
            80
          ),

        surfaceCount:
          Number(
            raw.surfaceCount ||
            0
          ),
      };
    }

    function activeItems(
      items
    ) {
      return items
        .filter(
          (
            item
          ) =>
            item.status ===
            "active"
        )
        .sort(
          (
            a,
            b
          ) =>
            String(
              b.updatedAt ||
              b.createdAt
            ).localeCompare(
              String(
                a.updatedAt ||
                a.createdAt
              )
            )
        );
    }

    /* =======================================================
       PAGE PRIVACY / CONTEXT
    ======================================================= */

    function safeDomain() {
      return String(
        window.location
          .hostname ||
        ""
      )
        .replace(
          /^www\./,
          ""
        )
        .toLowerCase();
    }

    function pageIsSensitive() {
      /*
       * Never automatically analyse pages containing
       * credential/payment fields.
       */
      if (
        document.querySelector(
          [
            'input[type="password"]',
            'input[autocomplete="cc-number"]',
            'input[autocomplete="cc-csc"]',
            'input[autocomplete="current-password"]',
            'input[autocomplete="new-password"]',
          ].join(
            ", "
          )
        )
      ) {
        return true;
      }

      const locationText =
        (
          window.location
            .pathname +
          " " +
          document.title
        )
          .toLowerCase();

      return /\b(?:checkout|payment|billing|banking|password)\b/.test(
        locationText
      );
    }

    function structuralSignals() {
      const signals =
        [];

      const active =
        document.activeElement;

      if (
        active &&
        (
          active.matches?.(
            "textarea"
          ) ||
          active.getAttribute?.(
            "contenteditable"
          ) ===
            "true"
        )
      ) {
        signals.push(
          "active writing surface"
        );
      }

      if (
        document.querySelector(
          'textarea, [contenteditable="true"]'
        )
      ) {
        signals.push(
          "writing or editing surface"
        );
      }

      if (
        document.querySelector(
          [
            'input[type="email"]',
            'input[autocomplete="email"]',
          ].join(
            ", "
          )
        )
      ) {
        signals.push(
          "communication field"
        );
      }

      return signals;
    }

    /*
     * AUTOMATIC MATCHING:
     *
     * Only lightweight metadata.
     *
     * No page paragraphs.
     * No large DOM scan.
     */
    function lightweightPageContext() {
      const description =
        document
          .querySelector(
            [
              'meta[name="description"]',
              'meta[property="og:description"]',
            ].join(
              ", "
            )
          )
          ?.getAttribute(
            "content"
          ) ||
        "";

      const heading =
        document
          .querySelector(
            "h1"
          )
          ?.textContent ||
        document
          .querySelector(
            "[role='heading']"
          )
          ?.textContent ||
        "";

      const secondary =
        Array.from(
          document
            .querySelectorAll(
              "h2, h3"
            )
        )
          .slice(
            0,
            4
          )
          .map(
            (
              node
            ) =>
              clean(
                node.textContent,
                110
              )
          )
          .filter(
            Boolean
          )
          .join(
            " · "
          );

      return {
        title:
          clean(
            document.title,
            300
          ),

        url:
          window.location.href,

        domain:
          safeDomain(),

        description:
          clean(
            description,
            600
          ),

        heading:
          clean(
            heading,
            300
          ),

        text:
          clean(
            [
              secondary,
              ...structuralSignals(),
            ]
              .filter(
                Boolean
              )
              .join(
                " "
              ),
            650
          ),
      };
    }

    /*
     * EXPLICIT CAPTURE:
     *
     * We may read a small amount of actual page text because
     * the user explicitly said the page matters.
     */
    function capturePageContext() {
      const base =
        lightweightPageContext();

      if (
        pageIsSensitive()
      ) {
        return base;
      }

      const root =
        document.querySelector(
          [
            "article",
            "main",
            "[role='main']",
          ].join(
            ", "
          )
        ) ||
        document.body;

      if (!root) {
        return base;
      }

      const excerpt =
        Array.from(
          root.querySelectorAll(
            "p, li"
          )
        )
          .map(
            (
              node
            ) =>
              clean(
                node.textContent,
                340
              )
          )
          .filter(
            (
              text
            ) =>
              text.length >=
              45
          )
          .slice(
            0,
            5
          )
          .join(
            " "
          );

      return {
        ...base,

        text:
          clean(
            [
              base.text,
              excerpt,
            ]
              .filter(
                Boolean
              )
              .join(
                " "
              ),
            1800
          ),
      };
    }

    /* =======================================================
       USER-INITIATED CAPTURE
    ======================================================= */

    function looksLikeCaptureCandidate(
      message
    ) {
      const text =
        clean(
          message,
          1800
        )
          .toLowerCase();

      if (
        !text ||
        text.length <
          4
      ) {
        return false;
      }

      if (
        /^(?:hi|hello|hey|thanks|thank you|okay|ok|lol|haha|bye)\b/.test(
          text
        )
      ) {
        return false;
      }

      if (
        /\b(?:password|passcode|api key|secret key|credit card|debit card|bank account|passport number|social insurance number|social security number)\b/.test(
          text
        )
      ) {
        return false;
      }

      /*
       * For browser-context capture, there should normally
       * be some relationship to the current page.
       *
       * This keeps "I need to study for ECON" from creating
       * another browser item just because Kelsie is open.
       */
      const pageReference =
        /\b(?:this|that|it|here|page|article|role|job|product|item|recipe|course|hotel|source|statistic|link|site|post|paper|guide)\b/.test(
          text
        );

      if (
        !pageReference
      ) {
        return false;
      }

      const futureIntent =
        /\b(?:remember|save|keep|come back|return to|want to|need to|have to|should|plan to|planning to|considering|thinking about|useful for|good for|compare|apply|email|message|call|ask|book|buy|order|try|read later|watch later|use later|send|cancel|follow up)\b/.test(
          text
        );

      if (
        !futureIntent
      ) {
        return false;
      }

      if (
        text.endsWith(
          "?"
        ) &&
        !/\b(?:remember|save|keep|come back|follow up)\b/.test(
          text
        )
      ) {
        return false;
      }

      return true;
    }

    function itemFromInterpretation(
      result,
      page
    ) {
      const now =
        new Date()
          .toISOString();

      return {
        id:
          newId(
            "item"
          ),

        thing:
          clean(
            result.thing,
            220
          ) ||
          clean(
            page.title,
            220
          ) ||
          "Saved context",

        reason:
          clean(
            result.reason,
            420
          ) ||
          (
            "You asked Kelsie "
            + "to keep this "
            + "available for later."
          ),

        intent:
          clean(
            result.intent,
            320
          ),

        kind:
          clean(
            result.kind,
            40
          ) ||
          "saved_context",

        entities:
          Array.isArray(
            result.entities
          )
            ? result.entities
                .map(
                  (
                    item
                  ) =>
                    clean(
                      item,
                      120
                    )
                )
                .filter(
                  Boolean
                )
                .slice(
                  0,
                  6
                )
            : [],

        futureRelevance:
          Array.isArray(
            result
              .future_relevance
          )
            ? result
                .future_relevance
                .map(
                  (
                    item
                  ) =>
                    clean(
                      item,
                      180
                    )
                )
                .filter(
                  Boolean
                )
                .slice(
                  0,
                  6
                )
            : [],

        sourceTitle:
          clean(
            page.title,
            300
          ),

        sourceUrl:
          clean(
            page.url,
            1200
          ),

        sourceDomain:
          clean(
            page.domain,
            180
          ),

        sourceExcerpt:
          clean(
            page.text,
            800
          ),

        confidence:
          Number(
            result.confidence ||
            0
          ),

        status:
          "active",

        createdAt:
          now,

        updatedAt:
          now,

        completedAt:
          "",

        lastSurfacedAt:
          "",

        surfaceCount:
          0,
      };
    }

    async function upsertCapturedItem(
      item
    ) {
      const items =
        await loadItems();

      const existingIndex =
        items.findIndex(
          (
            candidate
          ) =>
            candidate.status ===
              "active" &&
            candidate.sourceUrl &&
            candidate.sourceUrl ===
              item.sourceUrl &&
            candidate
              .thing
              .toLowerCase() ===
              item
                .thing
                .toLowerCase()
        );

      if (
        existingIndex >=
        0
      ) {
        items[
          existingIndex
        ] = {
          ...items[
            existingIndex
          ],

          ...item,

          id:
            items[
              existingIndex
            ].id,

          createdAt:
            items[
              existingIndex
            ].createdAt ||
            item.createdAt,

          updatedAt:
            new Date()
              .toISOString(),
        };
      } else {
        items.push(
          item
        );
      }

      await saveItems(
        items
      );

      const activeCount =
        activeItems(
          items
        ).length;

      postToHook(
        "KELSIE_CONTEXT_CAPTURED",
        {
          item,

          active_count:
            activeCount,
        }
      );

      /*
       * A newly added item may be useful on a future page,
       * but we do NOT trigger a match on this same page.
       */
      return item;
    }

    async function captureFromMessage(
      message
    ) {
      const page =
        capturePageContext();

      try {
        const result =
          await rpcToHook(
            "KELSIE_CONTEXT_INTERPRET_REQUEST",
            {
              message,
              page,
            }
          );

        if (
          !result
            ?.should_capture ||
          Number(
            result.confidence ||
            0
          ) <
            CAPTURE_THRESHOLD
        ) {
          return null;
        }

        const item =
          itemFromInterpretation(
            result,
            page
          );

        return (
          upsertCapturedItem(
            item
          )
        );
      } catch (error) {
        console.error(
          "[Kelsie Context] Capture failed:",
          error
        );

        return null;
      }
    }

    /* =======================================================
       EXISTING READING UI INTEGRATION
    ======================================================= */

    function readingCard() {
      return (
        shadow.getElementById(
          "kelsie-reading-assist-card"
        ) ||
        shadow.getElementById(
          "kelsie-page-card"
        ) ||
        shadow.querySelector(
          "[data-kelsie-reading-card]"
        )
      );
    }

    function setReadingSuppressed(
      suppressed
    ) {
      const reading =
        readingCard();

      if (!reading) {
        return;
      }

      if (
        suppressed ||
        readingSuppressedForPage
      ) {
        reading.style
          .setProperty(
            "display",
            "none",
            "important"
          );
      } else {
        reading.style
          .removeProperty(
            "display"
          );
      }
    }

    function injectReadingKeepOption() {
      const reading =
        readingCard();

      if (!reading) {
        return false;
      }

      const options =
        reading.querySelector(
          ".kr-options"
        );

      if (
        !options ||
        options.querySelector(
          "#kr-keep-in-mind"
        )
      ) {
        return false;
      }

      const button =
        document.createElement(
          "button"
        );

      button.className =
        "kr-option";

      button.id =
        "kr-keep-in-mind";

      button.type =
        "button";

      const title =
        document.createElement(
          "span"
        );

      title.className =
        "kr-option-title";

      title.textContent =
        "Keep this in mind";

      const copy =
        document.createElement(
          "span"
        );

      copy.className =
        "kr-option-copy";

      copy.textContent =
        (
          "Save why this matters "
          + "so Kelsie can bring "
          + "it back later."
        );

      button.append(
        title,
        copy
      );

      button.addEventListener(
        "click",
        () => {
          currentMode =
            "capture";

          currentMatch =
            null;

          captureInput.value =
            "";

          captureStatus.hidden =
            true;

          captureStatus.textContent =
            "";

          render();

          window.setTimeout(
            () => {
              captureInput.focus();
            },
            80
          );
        }
      );

      options.appendChild(
        button
      );

      return true;
    }

    /* =======================================================
       PRIORITY
    ======================================================= */

    function panelIsOpen() {
      return (
        !panelHost.hidden
      );
    }

    function reminderIsVisible() {
      return Boolean(
        reminderPopup &&
        !reminderPopup.hidden
      );
    }

    /* =======================================================
       CARD POSITIONING
    ======================================================= */

    function positionCard() {
      if (
        card.hidden
      ) {
        return;
      }

      const launcherRect =
        launcher
          .getBoundingClientRect();

      const cardRect =
        card
          .getBoundingClientRect();

      const margin =
        8;

      const gap =
        10;

      const edge =
        host.getAttribute(
          "data-edge"
        ) ||
        "right";

      let left =
        launcherRect.left;

      let top =
        launcherRect.top;

      if (
        edge ===
        "right"
      ) {
        left =
          launcherRect.left -
          cardRect.width -
          gap;

        top =
          launcherRect.top +
          launcherRect.height /
            2 -
          cardRect.height /
            2;
      } else if (
        edge ===
        "left"
      ) {
        left =
          launcherRect.right +
          gap;

        top =
          launcherRect.top +
          launcherRect.height /
            2 -
          cardRect.height /
            2;
      } else if (
        edge ===
        "top"
      ) {
        left =
          launcherRect.left +
          launcherRect.width /
            2 -
          cardRect.width /
            2;

        top =
          launcherRect.bottom +
          gap;
      } else {
        left =
          launcherRect.left +
          launcherRect.width /
            2 -
          cardRect.width /
            2;

        top =
          launcherRect.top -
          cardRect.height -
          gap;
      }

      card.style.left =
        `${clamp(
          left,
          margin,
          Math.max(
            margin,
            window.innerWidth -
              cardRect.width -
              margin
          )
        )}px`;

      card.style.top =
        `${clamp(
          top,
          margin,
          Math.max(
            margin,
            window.innerHeight -
              cardRect.height -
              margin
          )
        )}px`;
    }

    function render() {
      nudgeView.hidden =
        currentMode !==
        "nudge";

      captureView.hidden =
        currentMode !==
        "capture";

      /*
       * Priority:
       *
       * reminder
       *     ↓
       * user-opened Kelsie
       *     ↓
       * contextual resurfacing
       *     ↓
       * reading suggestion
       */
      const shouldShow =
        [
          "nudge",
          "capture",
        ].includes(
          currentMode
        ) &&
        !panelIsOpen() &&
        !reminderIsVisible() &&
        !host.hasAttribute(
          "data-dragging"
        );

      card.hidden =
        !shouldShow;

      setReadingSuppressed(
        shouldShow
      );

      if (
        shouldShow
      ) {
        requestAnimationFrame(
          positionCard
        );
      }
    }

    /* =======================================================
       ITEM STATE
    ======================================================= */

    function clearNudge() {
      currentMatch =
        null;

      currentMode =
        "idle";

      render();
    }

    function dismissCurrentNudge() {
      if (
        currentMatch
          ?.item
          ?.id
      ) {
        dismissedOnPage.add(
          (
            currentMatch
              .item
              .id
            + "::"
            + window.location.href
          )
        );
      }

      clearNudge();
    }

    async function completeItem(
      itemId
    ) {
      const items =
        await loadItems();

      const index =
        items.findIndex(
          (
            item
          ) =>
            item.id ===
            itemId
        );

      if (
        index <
        0
      ) {
        return;
      }

      items[
        index
      ] = {
        ...items[
          index
        ],

        status:
          "completed",

        completedAt:
          new Date()
            .toISOString(),

        updatedAt:
          new Date()
            .toISOString(),
      };

      await saveItems(
        items
      );

      postToHook(
        "KELSIE_CONTEXT_ITEMS_CHANGED",
        {
          active_count:
            activeItems(
              items
            ).length,
        }
      );

      clearNudge();
    }

    async function forgetItem(
      itemId
    ) {
      const items =
        await loadItems();

      const next =
        items.filter(
          (
            item
          ) =>
            item.id !==
            itemId
        );

      await saveItems(
        next
      );

      postToHook(
        "KELSIE_CONTEXT_ITEMS_CHANGED",
        {
          active_count:
            activeItems(
              next
            ).length,
        }
      );

      if (
        currentMatch
          ?.item
          ?.id ===
        itemId
      ) {
        clearNudge();
      }
    }

    function formatSource(
      item
    ) {
      return (
        clean(
          item.sourceDomain,
          120
        ) ||
        clean(
          item.sourceTitle,
          180
        )
      );
    }

    function showMatch(
      item,
      result
    ) {
      currentMatch = {
        item,
        result,
      };

      currentMode =
        "nudge";

      messageElement
        .textContent =
        clean(
          result.surface_message,
          360
        ) ||
        clean(
          item.reason,
          360
        ) ||
        (
          "Something you asked "
          + "Kelsie to keep in mind "
          + "may be useful here."
        );

      sourceElement
        .textContent =
        formatSource(
          item
        );

      sourceElement.hidden =
        !sourceElement
          .textContent;

      openSourceButton.hidden =
        !item.sourceUrl;

      render();
    }

    async function markSurfaced(
      itemId
    ) {
      const items =
        await loadItems();

      const index =
        items.findIndex(
          (
            item
          ) =>
            item.id ===
            itemId
        );

      if (
        index <
        0
      ) {
        return;
      }

      items[
        index
      ] = {
        ...items[
          index
        ],

        lastSurfacedAt:
          new Date()
            .toISOString(),

        surfaceCount:
          Number(
            items[
              index
            ].surfaceCount ||
            0
          ) +
          1,
      };

      await saveItems(
        items
      );
    }

    /* =======================================================
       LOCAL CANDIDATE SELECTION
    ======================================================= */

    function itemEligibleForMatch(
      item
    ) {
      if (
        item.status !==
        "active"
      ) {
        return false;
      }

      /*
       * Do not resurface the item on the exact page where
       * the user originally saved it.
       */
      if (
        item.sourceUrl &&
        item.sourceUrl ===
          window.location.href
      ) {
        return false;
      }

      if (
        dismissedOnPage.has(
          (
            item.id +
            "::" +
            window.location.href
          )
        )
      ) {
        return false;
      }

      if (
        item.lastSurfacedAt
      ) {
        const last =
          new Date(
            item.lastSurfacedAt
          )
            .getTime();

        if (
          Number.isFinite(
            last
          ) &&
          Date.now() -
            last <
            MATCH_COOLDOWN_MS
        ) {
          return false;
        }
      }

      return true;
    }

    function localCandidates(
      items,
      page
    ) {
      const pageTokens =
        tokens(
          [
            page.title,
            page.domain,
            page.description,
            page.heading,
            page.text,
          ]
            .filter(
              Boolean
            )
            .join(
              " "
            )
        );

      return items
        .filter(
          itemEligibleForMatch
        )
        .map(
          (
            item
          ) => ({
            item,

            score:
              itemLocalScore(
                item,
                pageTokens
              ),
          })
        )
        .filter(
          (
            candidate
          ) =>
            candidate.score >=
            LOCAL_MATCH_THRESHOLD
        )
        .sort(
          (
            a,
            b
          ) =>
            b.score -
            a.score
        )
        .slice(
          0,
          MAX_AI_CANDIDATES
        );
    }

    /* =======================================================
       PERFORMANCE-SAFE MATCHING
    ======================================================= */

    function visibleDwellMs() {
      if (
        firstVisibleAt ===
        null
      ) {
        return 0;
      }

      return (
        Date.now() -
        firstVisibleAt
      );
    }

    function scheduleMatchCheck(
      delay =
        600
    ) {
      if (
        matchEvaluatedForUrl ||
        matchRequestInFlight
      ) {
        return;
      }

      window.clearTimeout(
        matchTimer
      );

      matchTimer =
        window.setTimeout(
          () => {
            tryAutomaticMatch();
          },
          delay
        );
    }

    async function tryAutomaticMatch() {
      /*
       * ONE:
       * Never work in a background tab.
       */
      if (
        document.visibilityState !==
        "visible"
      ) {
        return;
      }

      /*
       * TWO:
       * Never automatically process sensitive pages.
       */
      if (
        pageIsSensitive()
      ) {
        matchEvaluatedForUrl =
          true;

        return;
      }

      /*
       * THREE:
       * Do not compete with existing Kelsie attention.
       */
      if (
        panelIsOpen() ||
        reminderIsVisible() ||
        currentMode !==
          "idle"
      ) {
        return;
      }

      /*
       * FOUR:
       * Require real dwell.
       */
      const dwell =
        visibleDwellMs();

      if (
        dwell <
        MIN_VISIBLE_DWELL_MS
      ) {
        scheduleMatchCheck(
          MIN_VISIBLE_DWELL_MS -
          dwell +
          250
        );

        return;
      }

      /*
       * FIVE:
       * Check whether there is even anything to match.
       *
       * Still no AI call.
       */
      const stored =
        await loadItems();

      const active =
        activeItems(
          stored
        )
          .slice(
            0,
            MAX_ACTIVE_ITEMS
          );

      if (
        active.length ===
        0
      ) {
        matchEvaluatedForUrl =
          true;

        return;
      }

      /*
       * SIX:
       * Lightweight page metadata only.
       */
      const page =
        lightweightPageContext();

      /*
       * SEVEN:
       * Local filtering.
       *
       * If no item is plausibly connected, STOP.
       *
       * No fetch.
       * No LLM.
       * No backend request.
       */
      const candidates =
        localCandidates(
          active,
          page
        );

      if (
        candidates.length ===
        0
      ) {
        matchEvaluatedForUrl =
          true;

        console.debug(
          "[Kelsie Context] "
          + "No local relevance candidate. "
          + "No AI call."
        );

        return;
      }

      const bestLocalScore =
        candidates[
          0
        ].score;

      /*
       * EIGHT:
       * A normal plausible match still requires some
       * evidence that the user is actually engaged
       * with this page.
       */
      if (
        interactionScore <
          MIN_INTERACTION_SCORE &&
        bestLocalScore <
          HIGH_LOCAL_SCORE
      ) {
        console.debug(
          "[Kelsie Context] "
          + "Local candidate exists, "
          + "waiting for user engagement."
        );

        return;
      }

      /*
       * From this point onward we allow exactly ONE automatic
       * backend match call for this URL.
       */
      matchEvaluatedForUrl =
        true;

      matchRequestInFlight =
        true;

      const compactItems =
        candidates.map(
          (
            {
              item,
              score,
            }
          ) => ({
            id:
              item.id,

            thing:
              item.thing,

            reason:
              item.reason,

            intent:
              item.intent,

            kind:
              item.kind,

            source_title:
              item.sourceTitle,

            source_url:
              item.sourceUrl,

            source_domain:
              item.sourceDomain,

            entities:
              item.entities,

            future_relevance:
              item.futureRelevance,

            created_at:
              item.createdAt,

            last_surfaced_at:
              item.lastSurfacedAt,

            /*
             * Not used by the current backend schema.
             * Intentionally not transmitted.
             *
             * score remains local only.
             */
            _local_score:
              score,
          })
        )
        .map(
          (
            item
          ) => {
            const {
              _local_score,
              ...safeItem
            } =
              item;

            return safeItem;
          }
        );

      try {
        console.debug(
          "[Kelsie Context] "
          + `AI verification for ${
            compactItems.length
          } local candidate(s).`
        );

        const result =
          await rpcToHook(
            "KELSIE_CONTEXT_MATCH_REQUEST",
            {
              page,

              items:
                compactItems,
            }
          );

        if (
          !result
            ?.should_surface ||
          Number(
            result.confidence ||
            0
          ) <
            AI_MATCH_THRESHOLD ||
          !result.item_id
        ) {
          return;
        }

        const selected =
          candidates.find(
            (
              candidate
            ) =>
              candidate.item.id ===
              result.item_id
          );

        if (
          !selected
        ) {
          return;
        }

        showMatch(
          selected.item,
          result
        );

        await markSurfaced(
          selected.item.id
        );

        console.log(
          "[Kelsie Context] "
          + "Useful contextual match surfaced.",
          {
            item:
              selected.item
                .thing,

            confidence:
              result.confidence,

            localScore:
              selected.score,
          }
        );
      } catch (error) {
        console.error(
          "[Kelsie Context] "
          + "Context match failed:",
          error
        );
      } finally {
        matchRequestInFlight =
          false;
      }
    }

    /* =======================================================
       ACTIVITY SIGNALS
    ======================================================= */

    function registerActivity(
      amount = 1
    ) {
      if (
        document.visibilityState !==
        "visible"
      ) {
        return;
      }

      interactionScore =
        Math.min(
          10,
          interactionScore +
          amount
        );

      /*
       * If dwell is already satisfied, user activity is
       * a good time to perform the cheap local check.
       */
      if (
        visibleDwellMs() >=
        MIN_VISIBLE_DWELL_MS
      ) {
        scheduleMatchCheck(
          450
        );
      }
    }

    function handleScroll() {
      const distance =
        Math.abs(
          window.scrollY -
          startScrollY
        );

      if (
        distance >=
        220
      ) {
        startScrollY =
          window.scrollY;

        registerActivity(
          1
        );
      }
    }

    document.addEventListener(
      "scroll",
      handleScroll,
      {
        passive:
          true,
      }
    );

    document.addEventListener(
      "pointerdown",
      () => {
        registerActivity(
          1
        );
      },
      {
        passive:
          true,
        capture:
          true,
      }
    );

    document.addEventListener(
      "keydown",
      () => {
        registerActivity(
          1
        );
      },
      {
        capture:
          true,
      }
    );

    document.addEventListener(
      "visibilitychange",
      () => {
        if (
          document.visibilityState ===
          "visible"
        ) {
          firstVisibleAt =
            Date.now();

          startScrollY =
            window.scrollY;

          scheduleMatchCheck(
            MIN_VISIBLE_DWELL_MS
          );
        } else {
          firstVisibleAt =
            null;

          window.clearTimeout(
            matchTimer
          );
        }
      }
    );

    /* =======================================================
       EXPLICIT "KEEP THIS PAGE" FLOW
    ======================================================= */

    async function saveCapturePrompt() {
      const reason =
        clean(
          captureInput.value,
          300
        );

      if (!reason) {
        captureInput.focus();

        return;
      }

      captureSave.disabled =
        true;

      captureStatus.hidden =
        false;

      captureStatus.textContent =
        "Saving…";

      const page =
        capturePageContext();

      const message =
        (
          "Keep this page "
          + "in mind for "
          + `${reason}.`
        );

      try {
        const result =
          await rpcToHook(
            "KELSIE_CONTEXT_INTERPRET_REQUEST",
            {
              message,
              page,
            }
          );

        let item =
          null;

        if (
          result
            ?.should_capture &&
          Number(
            result.confidence ||
            0
          ) >=
            CAPTURE_THRESHOLD
        ) {
          item =
            itemFromInterpretation(
              result,
              page
            );
        } else {
          /*
           * The user explicitly clicked Save,
           * so their instruction wins even if
           * the parser is conservative.
           */
          const now =
            new Date()
              .toISOString();

          item = {
            id:
              newId(
                "item"
              ),

            thing:
              clean(
                page.title,
                220
              ) ||
              "This page",

            reason:
              (
                "You wanted to "
                + "keep this in mind "
                + `for ${reason}.`
              ),

            intent:
              (
                "Use this for "
                + reason
              ),

            kind:
              "saved_context",

            entities:
              [],

            futureRelevance:
              [
                reason,
              ],

            sourceTitle:
              clean(
                page.title,
                300
              ),

            sourceUrl:
              clean(
                page.url,
                1200
              ),

            sourceDomain:
              clean(
                page.domain,
                180
              ),

            sourceExcerpt:
              clean(
                page.text,
                800
              ),

            confidence:
              1,

            status:
              "active",

            createdAt:
              now,

            updatedAt:
              now,

            completedAt:
              "",

            lastSurfacedAt:
              "",

            surfaceCount:
              0,
          };
        }

        await upsertCapturedItem(
          item
        );

        captureStatus
          .textContent =
          "Saved.";

        readingSuppressedForPage =
          true;

        window.setTimeout(
          () => {
            currentMode =
              "idle";

            render();
          },
          650
        );
      } catch (error) {
        captureStatus
          .textContent =
          error instanceof
            Error
            ? error.message
            : (
                "Kelsie could not "
                + "save that right now."
              );
      } finally {
        captureSave.disabled =
          false;
      }
    }

    /* =======================================================
       WIDGET "KEEPING IN MIND" LIST
    ======================================================= */

    function respondWithItems() {
      loadItems()
        .then(
          (
            items
          ) => {
            const active =
              activeItems(
                items
              );

            postToHook(
              "KELSIE_CONTEXT_ITEMS_RESPONSE",
              {
                items:
                  active,

                active_count:
                  active.length,
              }
            );
          }
        );
    }

    async function handleItemAction(
      data
    ) {
      const itemId =
        clean(
          data.item_id,
          140
        );

      const action =
        clean(
          data.action,
          40
        );

      if (!itemId) {
        return;
      }

      const items =
        await loadItems();

      const item =
        items.find(
          (
            candidate
          ) =>
            candidate.id ===
            itemId
        );

      if (!item) {
        respondWithItems();

        return;
      }

      if (
        action ===
        "complete"
      ) {
        await completeItem(
          itemId
        );

        respondWithItems();

        return;
      }

      if (
        action ===
        "forget"
      ) {
        await forgetItem(
          itemId
        );

        respondWithItems();
      }
    }

    /* =======================================================
       IFRAME MESSAGES
    ======================================================= */

    window.addEventListener(
      "message",
      (
        event
      ) => {
        if (
          event.source !==
          widgetFrame
            .contentWindow
        ) {
          return;
        }

        const data =
          event.data;

        if (
          !data ||
          data.source !==
            HOOK_SOURCE
        ) {
          return;
        }

        if (
          data.type ===
          "KELSIE_CONTEXT_RPC_RESPONSE"
        ) {
          const pending =
            pendingRpc.get(
              data.request_id
            );

          if (!pending) {
            return;
          }

          window.clearTimeout(
            pending.timeout
          );

          pendingRpc.delete(
            data.request_id
          );

          if (
            data.ok
          ) {
            pending.resolve(
              data.data
            );
          } else {
            pending.reject(
              new Error(
                clean(
                  data.error,
                  500
                ) ||
                (
                  "Kelsie context "
                  + "request failed."
                )
              )
            );
          }

          return;
        }

        if (
          data.type ===
          "KELSIE_CONTEXT_CAPTURE_CANDIDATE"
        ) {
          const message =
            clean(
              data.message,
              1800
            );

          if (
            looksLikeCaptureCandidate(
              message
            )
          ) {
            /*
             * User initiated.
             *
             * This may call the interpretation endpoint,
             * but it is NOT a background page-load request.
             */
            captureFromMessage(
              message
            );
          }

          return;
        }

        if (
          data.type ===
          "KELSIE_CONTEXT_ITEMS_REQUEST"
        ) {
          respondWithItems();

          return;
        }

        if (
          data.type ===
          "KELSIE_CONTEXT_ITEM_ACTION"
        ) {
          handleItemAction(
            data
          );
        }
      }
    );

    /* =======================================================
       CARD EVENTS
    ======================================================= */

    card
      .querySelector(
        "#kc-dismiss"
      )
      .addEventListener(
        "click",
        dismissCurrentNudge
      );

    card
      .querySelector(
        "#kc-done"
      )
      .addEventListener(
        "click",
        () => {
          if (
            currentMatch
              ?.item
              ?.id
          ) {
            completeItem(
              currentMatch
                .item
                .id
            );
          }
        }
      );

    openSourceButton
      .addEventListener(
        "click",
        () => {
          const sourceUrl =
            currentMatch
              ?.item
              ?.sourceUrl;

          if (
            sourceUrl
          ) {
            window.open(
              sourceUrl,
              "_blank",
              "noopener,noreferrer"
            );
          }
        }
      );

    card
      .querySelector(
        "#kc-capture-close"
      )
      .addEventListener(
        "click",
        () => {
          currentMode =
            "idle";

          render();
        }
      );

    captureSave
      .addEventListener(
        "click",
        saveCapturePrompt
      );

    captureInput
      .addEventListener(
        "keydown",
        (
          event
        ) => {
          if (
            event.key ===
            "Enter"
          ) {
            event.preventDefault();

            saveCapturePrompt();
          }
        }
      );

    /* =======================================================
       NO MUTATION OBSERVER
       -------------------------------------------------------
       This is the important performance fix.

       Old:
         DOM mutation
             ↓
         observer
             ↓
         render()
             ↓
         DOM mutation
             ↓
         observer
             ↓
         ...

       New:
         cheap timer every 2.5 sec
         checking only Kelsie's own Shadow DOM.
    ======================================================= */

    maintenanceTimer =
      window.setInterval(
        () => {
          injectReadingKeepOption();

          /*
           * Keeps priority state correct if the user
           * opens/closes the existing Kelsie panel or
           * a reminder appears.
           */
          render();
        },
        MAINTENANCE_INTERVAL_MS
      );

    /* =======================================================
       SPA / URL CHANGES
    ======================================================= */

    function resetForNewUrl() {
      currentMatch =
        null;

      currentMode =
        "idle";

      matchRequestInFlight =
        false;

      matchEvaluatedForUrl =
        false;

      readingSuppressedForPage =
        false;

      interactionScore =
        0;

      startScrollY =
        window.scrollY;

      firstVisibleAt =
        document.visibilityState ===
          "visible"
          ? Date.now()
          : null;

      dismissedOnPage.clear();

      window.clearTimeout(
        matchTimer
      );

      render();

      if (
        document.visibilityState ===
        "visible"
      ) {
        scheduleMatchCheck(
          MIN_VISIBLE_DWELL_MS
        );
      }
    }

    urlTimer =
      window.setInterval(
        () => {
          if (
            window.location.href ===
            observedUrl
          ) {
            return;
          }

          observedUrl =
            window.location.href;

          resetForNewUrl();
        },
        URL_CHECK_INTERVAL_MS
      );

    /* =======================================================
       WINDOW EVENTS
    ======================================================= */

    window.addEventListener(
      "resize",
      () => {
        if (
          !card.hidden
        ) {
          requestAnimationFrame(
            positionCard
          );
        }
      }
    );

    window.addEventListener(
      "pagehide",
      () => {
        window.clearTimeout(
          matchTimer
        );

        window.clearInterval(
          maintenanceTimer
        );

        window.clearInterval(
          urlTimer
        );

        pendingRpc.forEach(
          (
            pending
          ) => {
            window.clearTimeout(
              pending.timeout
            );
          }
        );

        pendingRpc.clear();
      },
      {
        once:
          true,
      }
    );

    /* =======================================================
       INITIALIZATION
    ======================================================= */

    /*
     * Update the "Keeping in mind" badge.
     *
     * Local storage only.
     * No backend call.
     */
    respondWithItems();

    injectReadingKeepOption();

    /*
     * This timer does NOT automatically mean an AI request.
     *
     * At 12 seconds:
     *
     * 1. visible tab?
     * 2. non-sensitive?
     * 3. active items?
     * 4. local lexical relationship?
     * 5. actual engagement?
     *
     * Only then may Kelsie call /api/context/match.
     */
    if (
      document.visibilityState ===
      "visible"
    ) {
      scheduleMatchCheck(
        MIN_VISIBLE_DWELL_MS
      );
    }

    console.log(
      "[Kelsie Context] "
      + "Performance-safe contextual "
      + "resurfacing initialized."
    );
  }

  waitForShell();
})();