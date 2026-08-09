(() => {
  const ROOT_ID =
    "kelsie-extension-root";

  const OLD_PAGE_CARD_ID =
    "kelsie-page-card";

  const CARD_ID =
    "kelsie-reading-assist-card";

  const READING_INITIAL_DELAY =
    2500;

  const READING_RETRY_INTERVAL =
    2500;

  const READING_MAX_DETECTION_WINDOW =
    18000;

  const READING_MIN_DWELL_MS =
    7000;

  const READING_SCROLL_THRESHOLD =
    260;

  const PAGE_URL_CHECK_INTERVAL =
    1200;

  const MAX_EXTRACTED_PAGE_CHARS =
    18000;

  const MAX_READING_COACH_TURNS =
    4;

  const READING_SITE_STATS_KEY =
    "kelsieReadingSiteStatsV2";

  const READING_SITE_DISMISSAL_LIMIT =
    3;

  const READING_SITE_SUPPRESSION_MS =
    7 * 24 * 60 * 60 * 1000;

  if (
    window.top !==
    window.self
  ) {
    return;
  }

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

  const normalizeWhitespace = (
    value
  ) =>
    String(
      value || ""
    )
      .replace(
        /\s+/g,
        " "
      )
      .trim();

  const safeText = (
    value,
    maximumLength = 1000
  ) =>
    normalizeWhitespace(
      value
    ).slice(
      0,
      maximumLength
    );

  function isElementVisible(
    element
  ) {
    if (
      !(
        element instanceof
        Element
      )
    ) {
      return false;
    }

    const style =
      window.getComputedStyle(
        element
      );

    if (
      style.display ===
        "none" ||
      style.visibility ===
        "hidden" ||
      Number(
        style.opacity
      ) === 0
    ) {
      return false;
    }

    const rect =
      element
        .getBoundingClientRect();

    return (
      rect.width > 0 &&
      rect.height > 0
    );
  }

  function waitForExtensionShell() {
    const host =
      document.getElementById(
        ROOT_ID
      );

    const shadow =
      host?.shadowRoot;

    const launcher =
      shadow?.getElementById(
        "kelsie-launcher"
      );

    const panelHost =
      shadow?.getElementById(
        "kelsie-panel-host"
      );

    const reminderPopup =
      shadow?.getElementById(
        "kelsie-reminder-popup"
      );

    const widgetFrame =
      shadow?.getElementById(
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
        waitForExtensionShell,
        120
      );

      return;
    }

    startReadingAssist({
      host,
      shadow,
      launcher,
      panelHost,
      reminderPopup,
      widgetFrame,
    });
  }

  function startReadingAssist({
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

    /*
     * Keep the older Level 1 page card from appearing.
     * Nothing else from content.js is touched.
     */
    const compatibilityStyle =
      document.createElement(
        "style"
      );

    compatibilityStyle.textContent = `
      #${OLD_PAGE_CARD_ID} {
        display: none !important;
      }
    `;

    shadow.appendChild(
      compatibilityStyle
    );

    const style =
      document.createElement(
        "style"
      );

    style.textContent = `
      #${CARD_ID} {
        position: fixed;
        z-index: 2147483647;
        width: min(
          316px,
          calc(100vw - 20px)
        );
        overflow: hidden;
        box-sizing: border-box;

        border:
          1px solid
          rgba(
            48,
            54,
            64,
            0.085
          );

        border-radius: 20px;

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

        pointer-events: auto;

        font-family:
          Inter,
          -apple-system,
          BlinkMacSystemFont,
          "Segoe UI",
          Helvetica,
          Arial,
          sans-serif;

        animation:
          kelsieReadingIn
          180ms
          ease-out;
      }

      #${CARD_ID}[hidden] {
        display:
          none !important;
      }

      .kr-suggestion {
        min-height: 62px;

        display: flex;
        align-items: center;

        gap: 7px;

        padding:
          7px
          8px
          7px
          10px;
      }

      .kr-main {
        flex: 1;
        min-width: 0;
        min-height: 48px;

        display: flex;
        align-items: center;

        gap: 9px;

        padding: 5px 6px;

        border: 0;
        border-radius: 13px;

        background:
          transparent;

        color: #303641;

        text-align: left;
        cursor: pointer;

        font: inherit;
      }

      .kr-main:hover {
        background:
          rgba(
            55,
            61,
            74,
            0.032
          );
      }

      .kr-orb {
        position: relative;

        width: 23px;
        height: 23px;

        flex: 0 0 auto;

        overflow: hidden;
        isolation: isolate;

        border:
          1px solid
          rgba(
            240,
            242,
            247,
            0.88
          );

        border-radius: 50%;

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

      .kr-copy,
      .kr-heading-copy {
        min-width: 0;

        display: grid;

        gap: 3px;
      }

      .kr-suggestion-title {
        color: #2f3540;

        font-size: 11px;
        line-height: 1.2;
        font-weight: 640;

        letter-spacing:
          -0.01em;
      }

      .kr-note,
      .kr-trust {
        color: #9298a1;

        font-size: 8px;
        line-height: 1.3;
        font-weight: 500;
      }

      /*
       * Visually light close button.
       * The hit area stays usable while
       * the icon no longer dominates.
       */
      .kr-close {
        width: 26px;
        height: 26px;
        min-width: 26px;

        display: grid;
        place-items: center;

        padding: 0;

        border: 0;
        border-radius: 50%;

        background:
          transparent;

        color: #9ba0a7;

        cursor: pointer;

        font-family: inherit;
        font-size: 17px;
        font-weight: 300;
        line-height: 1;

        transition:
          color
            140ms ease,
          background
            140ms ease;
      }

      .kr-close:hover {
        color: #5f666f;

        background:
          rgba(
            54,
            61,
            70,
            0.045
          );
      }

      .kr-menu,
      .kr-detail {
        padding:
          15px
          15px
          14px;
      }

      .kr-header {
        display: flex;
        align-items: center;
        justify-content:
          space-between;

        gap: 12px;

        margin-bottom: 14px;
      }

      .kr-identity {
        display: flex;
        align-items: center;

        gap: 8px;

        min-width: 0;
      }

      .kr-title {
        color: #303641;

        font-size: 11px;
        line-height: 1.2;
        font-weight: 655;

        letter-spacing:
          -0.01em;
      }

      .kr-menu-question,
      .kr-section-title {
        margin: 0;

        color: #3b414b;

        font-size: 12px;
        line-height: 1.3;
        font-weight: 620;

        letter-spacing:
          -0.012em;
      }

      .kr-menu-question {
        margin-bottom: 8px;
      }

      /*
       * No arrows.
       * No individual bordered cards.
       * These are simple rows separated
       * by very faint rules.
       */
      .kr-options {
        display: grid;
      }

      .kr-option {
        width: 100%;

        display: grid;

        gap: 3px;

        padding:
          11px 4px;

        border: 0;

        border-bottom:
          1px solid
          rgba(
            54,
            61,
            70,
            0.065
          );

        border-radius: 0;

        background:
          transparent;

        color: #343a44;

        text-align: left;
        cursor: pointer;

        font-family: inherit;

        transition:
          padding-left
            150ms ease,
          background
            150ms ease;
      }

      .kr-option:first-child {
        border-top:
          1px solid
          rgba(
            54,
            61,
            70,
            0.065
          );
      }

      .kr-option:hover {
        padding-left: 7px;

        background:
          rgba(
            57,
            64,
            74,
            0.022
          );
      }

      .kr-option-title {
        color: #383e48;

        font-size: 10px;
        line-height: 1.25;
        font-weight: 630;
      }

      .kr-option-copy {
        color: #9298a1;

        font-size: 8px;
        line-height: 1.35;
        font-weight: 500;
      }

      .kr-loading {
        min-height: 92px;

        display: flex;
        align-items: center;

        gap: 9px;

        color: #747b85;

        font-size: 9px;
        line-height: 1.4;
      }

      .kr-loading-dot {
        width: 6px;
        height: 6px;

        flex: 0 0 auto;

        border-radius: 50%;

        background: #808899;

        animation:
          kelsieReadingPulse
          1.35s
          ease-in-out
          infinite;
      }

      .kr-result {
        display: grid;

        gap: 13px;
      }

      .kr-summary {
        color: #49505a;

        font-size: 9.5px;
        line-height: 1.55;
        font-weight: 500;
      }

      .kr-structure {
        display: grid;

        gap: 10px;
      }

      .kr-structure-item {
        display: grid;

        gap: 3px;
      }

      /*
       * Sentence case, quieter than
       * the actual thinking prompt.
       */
      .kr-structure-label {
        color: #737a84;

        font-size: 8px;
        line-height: 1.15;
        font-weight: 650;
      }

      .kr-structure-content {
        color: #4a515b;

        font-size: 9px;
        line-height: 1.48;
        font-weight: 500;
      }

      /*
       * The question is intentionally
       * the strongest part of the
       * understand surface.
       */
      .kr-question-box {
        margin-top: 1px;

        padding:
          12px
          12px
          11px;

        border:
          1px solid
          rgba(
            71,
            78,
            89,
            0.07
          );

        border-radius: 15px;

        background:
          rgba(
            255,
            255,
            255,
            0.48
          );

        box-shadow:
          inset
          0 1px 0
          rgba(
            255,
            255,
            255,
            0.72
          );
      }

      .kr-question {
        color: #343a44;

        font-size: 10.5px;
        line-height: 1.5;
        font-weight: 610;

        letter-spacing:
          -0.008em;
      }

      .kr-input-row {
        margin-top: 9px;

        display: flex;
        align-items: center;

        gap: 6px;
      }

      .kr-input {
        min-width: 0;
        flex: 1;

        min-height: 33px;

        padding:
          7px
          10px;

        border:
          1px solid
          rgba(
            59,
            66,
            76,
            0.105
          );

        border-radius: 11px;

        background:
          rgba(
            250,
            251,
            252,
            0.86
          );

        color: #333942;

        outline: none;

        font-family: inherit;
        font-size: 9px;
        line-height: 1.3;

        box-shadow:
          inset
          0 1px 0
          rgba(
            255,
            255,
            255,
            0.82
          );
      }

      .kr-input:focus {
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
            ),
          inset
            0 1px 0
            rgba(
              255,
              255,
              255,
              0.82
            );
      }

      .kr-send {
        min-height: 32px;

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

        border-radius: 10px;

        background:
          rgba(
            241,
            243,
            246,
            0.9
          );

        color: #5d6570;

        font-family: inherit;
        font-size: 8.5px;
        line-height: 1;
        font-weight: 630;

        cursor: pointer;
      }

      .kr-send:hover:not(
        :disabled
      ) {
        background:
          rgba(
            235,
            238,
            242,
            0.96
          );

        color: #424952;
      }

      .kr-send:disabled {
        opacity: 0.45;

        cursor: default;
      }

      .kr-coach-response {
        margin-top: 9px;

        color: #69717b;

        font-size: 8.8px;
        line-height: 1.45;
      }

      .kr-error {
        min-height: 60px;

        color: #626a74;

        font-size: 9.5px;
        line-height: 1.5;
      }

      .kr-footer {
        margin-top: 2px;

        padding-top: 10px;

        border-top:
          1px solid
          rgba(
            54,
            61,
            70,
            0.055
          );

        display: flex;
        align-items: center;
        justify-content:
          space-between;

        gap: 8px;
      }

      /*
       * Text navigation instead of
       * chevrons / heavy buttons.
       */
      .kr-text-action {
        min-height: 27px;

        padding:
          0 4px;

        border: 0;
        border-radius: 8px;

        background:
          transparent;

        color: #858b94;

        font-family: inherit;
        font-size: 8px;
        line-height: 1;
        font-weight: 590;

        cursor: pointer;
      }

      .kr-text-action:hover {
        color: #565e68;

        background:
          rgba(
            60,
            67,
            77,
            0.035
          );
      }

      .kr-text-action.primary {
        color: #626a75;

        font-weight: 630;
      }

      @keyframes
      kelsieReadingIn {
        from {
          opacity: 0;

          transform:
            translateY(4px)
            scale(0.99);
        }

        to {
          opacity: 1;

          transform:
            translateY(0)
            scale(1);
        }
      }

      @keyframes
      kelsieReadingPulse {
        0%,
        100% {
          opacity: 0.45;

          transform:
            scale(0.9);
        }

        50% {
          opacity: 1;

          transform:
            scale(1.06);
        }
      }

      @media (
        prefers-reduced-motion:
          reduce
      ) {
        #${CARD_ID},
        .kr-loading-dot,
        .kr-option {
          animation:
            none !important;

          transition:
            none !important;
        }
      }
    `;

    shadow.appendChild(
      style
    );

    const card =
      document.createElement(
        "section"
      );

    card.id =
      CARD_ID;

    card.setAttribute(
      "aria-live",
      "polite"
    );

    card.hidden =
      true;

    card.innerHTML = `
      <div
        class="kr-suggestion"
        id="kr-suggestion-view"
      >
        <button
          class="kr-main"
          id="kr-open-menu"
          type="button"
        >
          <span
            class="kr-orb"
            aria-hidden="true"
          ></span>

          <span
            class="kr-copy"
          >
            <span
              class="kr-suggestion-title"
            >
              Help with this page?
            </span>

            <span
              class="kr-note"
            >
              Nothing from the page is sent until you choose.
            </span>
          </span>
        </button>

        <button
          class="kr-close"
          id="kr-dismiss"
          type="button"
          aria-label="Dismiss page help"
          title="Not on this page"
        >
          ×
        </button>
      </div>

      <div
        class="kr-menu"
        id="kr-menu-view"
        hidden
      >
        <div
          class="kr-header"
        >
          <div
            class="kr-identity"
          >
            <span
              class="kr-orb"
              aria-hidden="true"
            ></span>

            <span
              class="kr-heading-copy"
            >
              <span
                class="kr-title"
              >
                Kelsie
              </span>

              <span
                class="kr-trust"
              >
                Choose how much help you want
              </span>
            </span>
          </div>

          <button
            class="kr-close"
            id="kr-menu-close"
            type="button"
            aria-label="Close page help"
          >
            ×
          </button>
        </div>

        <div
          class="kr-menu-question"
        >
          What would help?
        </div>

        <div
          class="kr-options"
        >
          <button
            class="kr-option"
            id="kr-understand"
            type="button"
          >
            <span
              class="kr-option-title"
            >
              Help me understand it
            </span>

            <span
              class="kr-option-copy"
            >
              Work through what matters, one question at a time.
            </span>
          </button>

          <button
            class="kr-option"
            id="kr-summarize"
            type="button"
          >
            <span
              class="kr-option-title"
            >
              Summarize it
            </span>

            <span
              class="kr-option-copy"
            >
              Give me the useful version for this kind of page.
            </span>
          </button>

          <button
            class="kr-option"
            id="kr-questions"
            type="button"
          >
            <span
              class="kr-option-title"
            >
              I have questions about this page
            </span>

            <span
              class="kr-option-copy"
            >
              Bring the page into my normal Kelsie conversation.
            </span>
          </button>
        </div>
      </div>

      <div
        class="kr-detail"
        id="kr-detail-view"
        hidden
      >
        <div
          class="kr-header"
        >
          <div
            class="kr-identity"
          >
            <span
              class="kr-orb"
              aria-hidden="true"
            ></span>

            <span
              class="kr-heading-copy"
            >
              <span
                class="kr-title"
                id="kr-detail-title"
              >
                Kelsie
              </span>

              <span
                class="kr-trust"
                id="kr-trust-line"
              >
                This page only · not saved to memory
              </span>
            </span>
          </div>

          <button
            class="kr-close"
            id="kr-detail-close"
            type="button"
            aria-label="Close reading help"
          >
            ×
          </button>
        </div>

        <div
          class="kr-loading"
          id="kr-loading"
          hidden
        >
          <span
            class="kr-loading-dot"
            aria-hidden="true"
          ></span>

          <span
            id="kr-loading-copy"
          >
            Reading this page only…
          </span>
        </div>

        <div
          class="kr-result"
          id="kr-result"
          hidden
        >
          <div
            class="kr-section-title"
            id="kr-section-title"
          ></div>

          <div
            class="kr-summary"
            id="kr-summary"
            hidden
          ></div>

          <div
            class="kr-structure"
            id="kr-structure"
          ></div>

          <div
            class="kr-question-box"
            id="kr-question-box"
            hidden
          >
            <div
              class="kr-question"
              id="kr-question"
            ></div>

            <div
              class="kr-input-row"
            >
              <input
                class="kr-input"
                id="kr-response-input"
                type="text"
                maxlength="700"
                autocomplete="off"
                placeholder="Your take…"
              />

              <button
                class="kr-send"
                id="kr-response-send"
                type="button"
              >
                Send
              </button>
            </div>

            <div
              class="kr-coach-response"
              id="kr-coach-response"
              hidden
            ></div>
          </div>

          <div
            class="kr-footer"
          >
            <button
              class="kr-text-action"
              id="kr-back"
              type="button"
            >
              Other options
            </button>

            <button
              class="kr-text-action primary"
              id="kr-continue"
              type="button"
              hidden
            >
              Continue with Kelsie
            </button>
          </div>
        </div>

        <div
          id="kr-error-view"
          hidden
        >
          <div
            class="kr-error"
            id="kr-error"
          ></div>

          <div
            class="kr-footer"
          >
            <button
              class="kr-text-action"
              id="kr-error-back"
              type="button"
            >
              Other options
            </button>

            <button
              class="kr-text-action primary"
              id="kr-retry"
              type="button"
            >
              Try again
            </button>
          </div>
        </div>
      </div>
    `;

    shadow.appendChild(
      card
    );

    const $ = (
      selector
    ) =>
      card.querySelector(
        selector
      );

    const suggestionView =
      $(
        "#kr-suggestion-view"
      );

    const menuView =
      $(
        "#kr-menu-view"
      );

    const detailView =
      $(
        "#kr-detail-view"
      );

    const loading =
      $(
        "#kr-loading"
      );

    const loadingCopy =
      $(
        "#kr-loading-copy"
      );

    const resultView =
      $(
        "#kr-result"
      );

    const detailTitle =
      $(
        "#kr-detail-title"
      );

    const trustLine =
      $(
        "#kr-trust-line"
      );

    const sectionTitle =
      $(
        "#kr-section-title"
      );

    const summaryElement =
      $(
        "#kr-summary"
      );

    const structureElement =
      $(
        "#kr-structure"
      );

    const questionBox =
      $(
        "#kr-question-box"
      );

    const questionElement =
      $(
        "#kr-question"
      );

    const responseInput =
      $(
        "#kr-response-input"
      );

    const responseSend =
      $(
        "#kr-response-send"
      );

    const coachResponse =
      $(
        "#kr-coach-response"
      );

    const continueButton =
      $(
        "#kr-continue"
      );

    const errorView =
      $(
        "#kr-error-view"
      );

    const errorElement =
      $(
        "#kr-error"
      );

    let state =
      "inactive";

    let candidate =
      null;

    let pageContent =
      null;

    let resultData =
      null;

    let errorMessage =
      "";

    let requestMode =
      null;

    let detectionTimer =
      null;

    let detectionStartedAt =
      0;

    let pageEnteredAt =
      Date.now();

    let scrollDistance =
      0;

    let lastScrollY =
      window.scrollY;

    let coachTurns =
      [];

    let currentQuestion =
      "";

    let observedUrl =
      window.location.href;

    function log(
      message,
      extra = null
    ) {
      if (extra) {
        console.log(
          `[Kelsie Reading] ${message}`,
          extra
        );

        return;
      }

      console.log(
        `[Kelsie Reading] ${message}`
      );
    }

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

    function currentEdge() {
      const edge =
        host.getAttribute(
          "data-edge"
        );

      return [
        "top",
        "right",
        "bottom",
        "left",
      ].includes(
        edge
      )
        ? edge
        : "right";
    }

    function positionCard() {
      if (card.hidden) {
        return;
      }

      const launcherRect =
        launcher
          .getBoundingClientRect();

      const cardRect =
        card
          .getBoundingClientRect();

      const margin = 8;
      const gap = 10;

      const edge =
        currentEdge();

      let left =
        launcherRect.left;

      let top =
        launcherRect.top;

      if (
        edge === "right"
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
        edge === "left"
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
        edge === "top"
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

    function isUnsupportedPage() {
      const protocol =
        window.location.protocol;

      if (
        protocol !== "http:" &&
        protocol !== "https:"
      ) {
        return true;
      }

      if (
        String(
          document.contentType ||
          ""
        ).toLowerCase() ===
        "application/pdf"
      ) {
        return true;
      }

      return (
        /\.pdf(?:$|[?#])/i.test(
          window.location.href
        )
      );
    }

    function pageLooksLikeUtility() {
      const descriptor = [
        document.title,
        document.body?.id,
        document.body
          ?.className,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      return [
        "search results",
        "shopping cart",
        "checkout",
        "sign in",
        "log in",
        "dashboard",
        "inbox",
      ].some(
        (term) =>
          descriptor.includes(
            term
          )
      );
    }

    function substantialParagraphs(
      root
    ) {
      if (
        !(
          root instanceof
          Element
        )
      ) {
        return [];
      }

      return Array.from(
        root.querySelectorAll(
          "p, blockquote"
        )
      ).filter(
        (element) => {
          if (
            element.closest(
              "nav, footer, aside, form, [role='navigation'], [role='dialog']"
            )
          ) {
            return false;
          }

          if (
            !isElementVisible(
              element
            )
          ) {
            return false;
          }

          return (
            normalizeWhitespace(
              element.textContent
            ).length >= 75
          );
        }
      );
    }

    function rootMetrics(
      root,
      kind
    ) {
      const paragraphs =
        substantialParagraphs(
          root
        );

      const totalChars =
        paragraphs.reduce(
          (
            sum,
            paragraph
          ) =>
            sum +
            normalizeWhitespace(
              paragraph.textContent
            ).length,
          0
        );

      const heading =
        root.querySelector(
          "h1"
        ) ||
        document.querySelector(
          "h1"
        );

      const linkTextChars =
        Array.from(
          root.querySelectorAll(
            "a"
          )
        ).reduce(
          (
            sum,
            link
          ) =>
            sum +
            normalizeWhitespace(
              link.textContent
            ).length,
          0
        );

      const linkRatio =
        totalChars > 0
          ? linkTextChars /
            totalChars
          : 1;

      const formCount =
        root
          .querySelectorAll(
            "form"
          )
          .length;

      let score = 0;

      if (
        kind === "article" ||
        kind ===
          "article-body"
      ) {
        score += 4;
      } else if (
        kind === "main"
      ) {
        score += 2;
      }

      if (heading) {
        score += 1;
      }

      if (
        paragraphs.length >= 3
      ) {
        score += 2;
      }

      if (
        paragraphs.length >= 6
      ) {
        score += 1;
      }

      if (
        totalChars >= 900
      ) {
        score += 2;
      }

      if (
        totalChars >= 1800
      ) {
        score += 1;
      }

      if (
        linkRatio < 0.24
      ) {
        score += 1;
      }

      if (
        linkRatio > 0.48
      ) {
        score -= 3;
      }

      if (
        formCount > 4
      ) {
        score -= 2;
      }

      return {
        root,
        kind,

        paragraphCount:
          paragraphs.length,

        totalChars,
        linkRatio,
        score,
      };
    }

    function hasArticleMetadata() {
      const ogType =
        document
          .querySelector(
            'meta[property="og:type"]'
          )
          ?.getAttribute(
            "content"
          ) || "";

      if (
        /article/i.test(
          ogType
        )
      ) {
        return true;
      }

      if (
        document.querySelector(
          'meta[property="article:published_time"], meta[name="article:section"]'
        )
      ) {
        return true;
      }

      return Array.from(
        document.querySelectorAll(
          'script[type="application/ld+json"]'
        )
      ).some(
        (script) =>
          /"@type"\s*:\s*"(?:Article|NewsArticle|BlogPosting|ScholarlyArticle|HowTo)"/i.test(
            script.textContent ||
            ""
          )
      );
    }

    function detectCandidate() {
      if (
        isUnsupportedPage() ||
        pageLooksLikeUtility()
      ) {
        return null;
      }

      const candidates =
        [];

      const seen =
        new Set();

      function addRoots(
        selector,
        kind
      ) {
        document
          .querySelectorAll(
            selector
          )
          .forEach(
            (root) => {
              if (
                seen.has(root) ||
                !isElementVisible(
                  root
                )
              ) {
                return;
              }

              seen.add(
                root
              );

              candidates.push(
                rootMetrics(
                  root,
                  kind
                )
              );
            }
          );
      }

      addRoots(
        "article",
        "article"
      );

      addRoots(
        '[itemprop="articleBody"]',
        "article-body"
      );

      addRoots(
        "main, [role='main']",
        "main"
      );

      if (
        candidates.length ===
          0 &&
        document.body
      ) {
        candidates.push(
          rootMetrics(
            document.body,
            "body"
          )
        );
      }

      const metadataBoost =
        hasArticleMetadata()
          ? 2
          : 0;

      candidates.forEach(
        (item) => {
          item.score +=
            metadataBoost;
        }
      );

      candidates.sort(
        (a, b) =>
          b.score -
          a.score
      );

      const best =
        candidates[0];

      if (
        !best ||
        best.score < 6 ||
        best.paragraphCount <
          3 ||
        best.totalChars < 850
      ) {
        return null;
      }

      return best;
    }

    function engagementReady(
      detected
    ) {
      const dwell =
        Date.now() -
        pageEnteredAt;

      const highConfidence =
        detected.score >= 9;

      return (
        dwell >=
          READING_MIN_DWELL_MS &&
        (
          scrollDistance >=
            READING_SCROLL_THRESHOLD ||
          highConfidence
        )
      );
    }

    function siteKey() {
      return String(
        window.location
          .hostname ||
        "unknown"
      ).toLowerCase();
    }

    async function getSiteStats() {
      try {
        const result =
          await chrome
            .storage
            .local
            .get({
              [READING_SITE_STATS_KEY]:
                {},
            });

        const allStats =
          result[
            READING_SITE_STATS_KEY
          ] || {};

        return {
          allStats,

          siteStats:
            allStats[
              siteKey()
            ] || {
              dismissals: 0,
              opens: 0,

              lastDismissedAt:
                0,
            },
        };
      } catch (error) {
        console.error(
          "[Kelsie Reading] Could not read site stats:",
          error
        );

        return {
          allStats: {},

          siteStats: {
            dismissals: 0,
            opens: 0,

            lastDismissedAt:
              0,
          },
        };
      }
    }

    async function recordSiteEvent(
      eventName
    ) {
      const {
        allStats,
        siteStats,
      } =
        await getSiteStats();

      const next = {
        ...siteStats,
      };

      if (
        eventName ===
        "dismissed"
      ) {
        next.dismissals =
          Number(
            next.dismissals ||
            0
          ) + 1;

        next.lastDismissedAt =
          Date.now();
      }

      if (
        eventName ===
        "opened"
      ) {
        next.opens =
          Number(
            next.opens ||
            0
          ) + 1;
      }

      allStats[
        siteKey()
      ] = next;

      try {
        await chrome
          .storage
          .local
          .set({
            [READING_SITE_STATS_KEY]:
              allStats,
          });
      } catch (error) {
        console.error(
          "[Kelsie Reading] Could not save site stats:",
          error
        );
      }
    }

    async function siteIsSuppressed() {
      const {
        siteStats,
      } =
        await getSiteStats();

      const dismissals =
        Number(
          siteStats
            .dismissals ||
          0
        );

      const opens =
        Number(
          siteStats.opens ||
          0
        );

      const lastDismissedAt =
        Number(
          siteStats
            .lastDismissedAt ||
          0
        );

      if (
        dismissals <
          READING_SITE_DISMISSAL_LIMIT ||
        opens > 0
      ) {
        return false;
      }

      return (
        Date.now() -
          lastDismissedAt <
        READING_SITE_SUPPRESSION_MS
      );
    }

    function stopDetection() {
      if (!detectionTimer) {
        return;
      }

      clearTimeout(
        detectionTimer
      );

      detectionTimer =
        null;
    }

    function scheduleDetection(
      delay =
        READING_INITIAL_DELAY
    ) {
      stopDetection();

      if (
        state !== "inactive"
      ) {
        return;
      }

      if (
        !detectionStartedAt
      ) {
        detectionStartedAt =
          Date.now();
      }

      detectionTimer =
        window.setTimeout(
          async () => {
            detectionTimer =
              null;

            if (
              state !==
              "inactive"
            ) {
              return;
            }

            if (
              document
                .visibilityState !==
              "visible"
            ) {
              scheduleDetection(
                READING_RETRY_INTERVAL
              );

              return;
            }

            const detected =
              detectCandidate();

            if (detected) {
              log(
                "candidate found",
                {
                  kind:
                    detected.kind,

                  score:
                    detected.score,

                  paragraphs:
                    detected
                      .paragraphCount,

                  chars:
                    detected
                      .totalChars,

                  scroll:
                    Math.round(
                      scrollDistance
                    ),

                  dwell_ms:
                    Date.now() -
                    pageEnteredAt,
                }
              );

              if (
                engagementReady(
                  detected
                )
              ) {
                if (
                  await siteIsSuppressed()
                ) {
                  log(
                    "site suppressed after repeated dismissals"
                  );

                  return;
                }

                candidate =
                  detected;

                state =
                  "suggested";

                render();

                log(
                  "showing Help with this page?"
                );

                return;
              }
            } else {
              log(
                "no eligible reading candidate yet"
              );
            }

            if (
              Date.now() -
                detectionStartedAt <
              READING_MAX_DETECTION_WINDOW
            ) {
              scheduleDetection(
                READING_RETRY_INTERVAL
              );
            } else {
              log(
                "detection window ended without a popup"
              );
            }
          },
          delay
        );
    }

    function extractPage() {
      const root =
        candidate?.root;

      if (
        !(
          root instanceof
          Element
        )
      ) {
        return null;
      }

      const nodes =
        Array.from(
          root.querySelectorAll(
            "h1, h2, h3, p, blockquote, li"
          )
        );

      const lines =
        [];

      const seen =
        new Set();

      let totalLength =
        0;

      for (
        const node
        of nodes
      ) {
        if (
          node.closest(
            "nav, footer, aside, form, [role='navigation'], [role='dialog']"
          )
        ) {
          continue;
        }

        if (
          !isElementVisible(
            node
          )
        ) {
          continue;
        }

        let text =
          normalizeWhitespace(
            node.textContent
          );

        if (!text) {
          continue;
        }

        if (
          node.matches(
            "p, blockquote"
          ) &&
          text.length < 35
        ) {
          continue;
        }

        if (
          node.matches("li") &&
          text.length < 20
        ) {
          continue;
        }

        text =
          text.slice(
            0,
            1000
          );

        const signature =
          text.toLowerCase();

        if (
          seen.has(
            signature
          )
        ) {
          continue;
        }

        seen.add(
          signature
        );

        const remaining =
          MAX_EXTRACTED_PAGE_CHARS -
          totalLength;

        if (
          remaining <= 0
        ) {
          break;
        }

        if (
          text.length >
          remaining
        ) {
          if (
            remaining >= 100
          ) {
            lines.push(
              text.slice(
                0,
                remaining
              )
            );
          }

          break;
        }

        lines.push(
          text
        );

        totalLength +=
          text.length +
          2;
      }

      const combined =
        lines
          .join("\n\n")
          .trim();

      if (
        combined.length <
        500
      ) {
        return null;
      }

      const heading =
        safeText(
          root
            .querySelector(
              "h1"
            )
            ?.textContent ||
            document
              .querySelector(
                "h1"
              )
              ?.textContent ||
            document.title ||
            "",
          300
        );

      return {
        title: heading,

        url:
          window.location.href,

        text:
          combined,
      };
    }

    function ensurePageContent() {
      if (!pageContent) {
        pageContent =
          extractPage();
      }

      return pageContent;
    }

    function sendBackgroundRequest(
      type,
      payload
    ) {
      return new Promise(
        (
          resolve,
          reject
        ) => {
          chrome.runtime.sendMessage(
            {
              type,
              payload,
            },
            (response) => {
              const runtimeError =
                chrome.runtime
                  .lastError;

              if (
                runtimeError
              ) {
                reject(
                  new Error(
                    runtimeError
                      .message
                  )
                );

                return;
              }

              if (
                !response ||
                response.ok !==
                  true
              ) {
                reject(
                  new Error(
                    response
                      ?.error ||
                    "Kelsie could not complete that request."
                  )
                );

                return;
              }

              resolve(
                response.data
              );
            }
          );
        }
      );
    }

    function beginRequest(
      mode
    ) {
      const page =
        ensurePageContent();

      requestMode =
        mode;

      if (!page) {
        state =
          "error";

        errorMessage =
          "I couldn't find enough readable content on this page.";

        render();

        return null;
      }

      state =
        "loading";

      resultData =
        null;

      errorMessage =
        "";

      coachResponse.hidden =
        true;

      render();

      return page;
    }

    async function requestAnalysis() {
      const page =
        beginRequest(
          "understand"
        );

      if (!page) {
        return;
      }

      try {
        resultData =
          await sendBackgroundRequest(
            "KELSIE_READING_ANALYZE",
            page
          );

        coachTurns =
          [];

        currentQuestion =
          safeText(
            resultData
              ?.first_question,
            700
          );

        /*
         * Help-me-understand is not
         * allowed to silently become
         * a structure dump.
         */
        if (
          !currentQuestion
        ) {
          throw new Error(
            "Kelsie didn't create a useful reading question. Please try again."
          );
        }

        state =
          "understand-result";
      } catch (error) {
        console.error(
          "[Kelsie] Reading analysis failed:",
          error
        );

        state =
          "error";

        errorMessage =
          error instanceof
          Error
            ? error.message
            : "Kelsie could not unpack this page right now.";
      }

      render();
    }

    async function requestSummary() {
      const page =
        beginRequest(
          "summary"
        );

      if (!page) {
        return;
      }

      try {
        resultData =
          await sendBackgroundRequest(
            "KELSIE_READING_SUMMARIZE",
            page
          );

        state =
          "summary-result";
      } catch (error) {
        console.error(
          "[Kelsie] Reading summary failed:",
          error
        );

        state =
          "error";

        errorMessage =
          error instanceof
          Error
            ? error.message
            : "Kelsie could not summarize this page right now.";
      }

      render();
    }

    function widgetOrigin() {
      try {
        return new URL(
          widgetFrame.src ||
            "http://127.0.0.1:8000"
        ).origin;
      } catch (_error) {
        return (
          "http://127.0.0.1:8000"
        );
      }
    }

    /*
     * Instead of maintaining a second mini-chat,
     * the page can become optional temporary
     * context for Kelsie's existing conversation.
     */
    function activatePageContext(
      mode,
      introMessage = ""
    ) {
      const page =
        ensurePageContent();

      if (
        !page ||
        !widgetFrame
          .contentWindow
      ) {
        state =
          "error";

        errorMessage =
          "I couldn't bring this page into Kelsie right now.";

        render();

        return;
      }

      widgetFrame
        .contentWindow
        .postMessage(
          {
            source:
              "kelsie-reading-assist",

            type:
              "KELSIE_PAGE_CONTEXT_ACTIVATE",

            intro_message:
              introMessage,

            context: {
              ...page,

              content_type:
                safeText(
                  resultData
                    ?.content_type ||
                    "other",
                  40
                ),

              structure:
                Array.isArray(
                  resultData
                    ?.structure
                )
                  ? resultData
                      .structure
                  : [],

              mode,

              active_question:
                mode ===
                  "scaffold"
                  ? currentQuestion
                  : "",

              page_turns:
                coachTurns.flatMap(
                  (turn) => [
                    {
                      role:
                        "user",

                      content:
                        turn.answer ||
                        "",
                    },
                    {
                      role:
                        "assistant",

                      content:
                        [
                          turn.response,
                          turn.next_question,
                        ]
                          .filter(
                            Boolean
                          )
                          .join(
                            " "
                          ),
                    },
                  ]
                ),
            },
          },
          widgetOrigin()
        );

      state =
        "context-open";

      render();

      if (
        !panelIsOpen()
      ) {
        launcher.click();
      }
    }

    /*
     * "I have questions..." now moves into
     * normal Kelsie instead of creating a
     * separate question field.
     */
    async function openQuestionsInKelsie() {
      const page =
        ensurePageContent();

      requestMode =
        "questions";

      if (!page) {
        state =
          "error";

        errorMessage =
          "I couldn't find enough readable content on this page.";

        render();

        return;
      }

      activatePageContext(
        "questions",
        "I've got the page. What do you want to know?"
      );
    }

    async function sendCoachResponse() {
      const userAnswer =
        safeText(
          responseInput.value,
          700
        );

      if (!userAnswer) {
        responseInput.focus();

        return;
      }

      if (
        !resultData ||
        !pageContent ||
        !currentQuestion
      ) {
        return;
      }

      responseSend.disabled =
        true;

      coachResponse.hidden =
        true;

      try {
        const data =
          await sendBackgroundRequest(
            "KELSIE_READING_COACH",
            {
              ...pageContent,

              content_type:
                resultData
                  .content_type ||
                "other",

              structure:
                Array.isArray(
                  resultData
                    .structure
                )
                  ? resultData
                      .structure
                  : [],

              prior_question:
                currentQuestion,

              user_answer:
                userAnswer,

              turns:
                coachTurns.slice(
                  -MAX_READING_COACH_TURNS
                ),
            }
          );

        const responseText =
          safeText(
            data?.response,
            900
          );

        const nextQuestion =
          safeText(
            data
              ?.next_question,
            700
          );

        coachTurns.push({
          question:
            currentQuestion,

          answer:
            userAnswer,

          response:
            responseText,

          next_question:
            nextQuestion,
        });

        if (
          responseText
        ) {
          coachResponse
            .textContent =
            responseText;

          coachResponse
            .hidden =
            false;
        }

        if (
          nextQuestion
        ) {
          currentQuestion =
            nextQuestion;

          questionElement
            .textContent =
            nextQuestion;
        }

        responseInput.value =
          "";
      } catch (error) {
        coachResponse
          .textContent =
          error instanceof
          Error
            ? error.message
            : "Kelsie could not continue that thought right now.";

        coachResponse.hidden =
          false;
      } finally {
        responseSend.disabled =
          false;

        requestAnimationFrame(
          positionCard
        );
      }
    }

    function structureHeading(
      contentType
    ) {
      const headings = {
        argument:
          "How the argument works",

        news:
          "What this report is telling you",

        research:
          "How the study works",

        how_to:
          "How this page works",

        explainer:
          "How the idea works",

        reference:
          "What this page gives you",

        other:
          "What matters on this page",
      };

      return (
        headings[
          contentType
        ] ||
        headings.other
      );
    }

    function renderStructure(
      items
    ) {
      structureElement
        .replaceChildren();

      if (
        !Array.isArray(
          items
        )
      ) {
        return;
      }

      items
        .slice(
          0,
          5
        )
        .forEach(
          (item) => {
            const label =
              safeText(
                item?.label,
                80
              );

            const content =
              safeText(
                item?.content,
                520
              );

            if (
              !label ||
              !content
            ) {
              return;
            }

            const wrapper =
              document.createElement(
                "div"
              );

            wrapper.className =
              "kr-structure-item";

            const labelElement =
              document.createElement(
                "div"
              );

            labelElement.className =
              "kr-structure-label";

            labelElement.textContent =
              label;

            const contentElement =
              document.createElement(
                "div"
              );

            contentElement.className =
              "kr-structure-content";

            contentElement.textContent =
              content;

            wrapper.append(
              labelElement,
              contentElement
            );

            structureElement
              .appendChild(
                wrapper
              );
          }
        );
    }

    function renderContents() {
      suggestionView.hidden =
        state !==
        "suggested";

      menuView.hidden =
        state !==
        "menu";

      detailView.hidden =
        ![
          "loading",
          "understand-result",
          "summary-result",
          "error",
        ].includes(
          state
        );

      loading.hidden =
        state !==
        "loading";

      resultView.hidden =
        ![
          "understand-result",
          "summary-result",
        ].includes(
          state
        );

      errorView.hidden =
        state !==
        "error";

      if (
        state ===
        "loading"
      ) {
        detailTitle.textContent =
          "Kelsie";

        trustLine.textContent =
          "This page only · not saved to memory";

        loadingCopy.textContent =
          requestMode ===
          "summary"
            ? "Summarizing this page only…"
            : "Reading this page only…";
      }

      if (
        state ===
          "understand-result" &&
        resultData
      ) {
        detailTitle.textContent =
          "Understand this page";

        trustLine.textContent =
          "This page only · not saved to memory";

        sectionTitle.textContent =
          structureHeading(
            resultData
              .content_type
          );

        summaryElement.hidden =
          true;

        summaryElement
          .textContent =
          "";

        renderStructure(
          resultData
            .structure
        );

        questionElement
          .textContent =
          currentQuestion;

        questionBox.hidden =
          !currentQuestion;

        continueButton.hidden =
          false;
      }

      if (
        state ===
          "summary-result" &&
        resultData
      ) {
        detailTitle.textContent =
          "Kelsie summary";

        trustLine.textContent =
          "This page only · not saved to memory";

        sectionTitle.textContent =
          "The useful version";

        const summary =
          safeText(
            resultData.summary,
            1500
          );

        summaryElement
          .textContent =
          summary;

        summaryElement.hidden =
          !summary;

        renderStructure(
          resultData
            .sections
        );

        questionBox.hidden =
          true;

        continueButton.hidden =
          true;
      }

      if (
        state === "error"
      ) {
        detailTitle.textContent =
          "Kelsie";

        trustLine.textContent =
          "Nothing from this page was saved";

        errorElement.textContent =
          errorMessage ||
          "Kelsie could not help with this page right now.";
      }
    }

    function render() {
      renderContents();

      const visibleState =
        [
          "suggested",
          "menu",
          "loading",
          "understand-result",
          "summary-result",
          "error",
        ].includes(
          state
        );

      /*
       * Existing due reminders continue to outrank
       * the reading surface.
       */
      const shouldShow =
        visibleState &&
        !panelIsOpen() &&
        !reminderIsVisible() &&
        !host.hasAttribute(
          "data-dragging"
        );

      card.hidden =
        !shouldShow;

      if (shouldShow) {
        requestAnimationFrame(
          positionCard
        );
      }
    }

    function dismissForPage() {
      stopDetection();

      recordSiteEvent(
        "dismissed"
      );

      state =
        "dismissed";

      candidate =
        null;

      pageContent =
        null;

      resultData =
        null;

      errorMessage =
        "";

      coachTurns =
        [];

      currentQuestion =
        "";

      render();
    }

    function showMenu() {
      state =
        "menu";

      resultData =
        null;

      errorMessage =
        "";

      requestMode =
        null;

      coachResponse.hidden =
        true;

      render();
    }

    function resetForNavigation() {
      stopDetection();

      state =
        "inactive";

      candidate =
        null;

      pageContent =
        null;

      resultData =
        null;

      errorMessage =
        "";

      requestMode =
        null;

      detectionStartedAt =
        0;

      pageEnteredAt =
        Date.now();

      scrollDistance =
        0;

      lastScrollY =
        window.scrollY;

      coachTurns =
        [];

      currentQuestion =
        "";

      try {
        widgetFrame
          .contentWindow
          ?.postMessage(
            {
              source:
                "kelsie-reading-assist",

              type:
                "KELSIE_PAGE_CONTEXT_CLEAR",
            },
            widgetOrigin()
          );
      } catch (_error) {
        // Best effort.
      }

      render();

      scheduleDetection();
    }

    $(
      "#kr-open-menu"
    ).addEventListener(
      "click",
      () => {
        recordSiteEvent(
          "opened"
        );

        showMenu();
      }
    );

    $(
      "#kr-dismiss"
    ).addEventListener(
      "click",
      dismissForPage
    );

    $(
      "#kr-menu-close"
    ).addEventListener(
      "click",
      dismissForPage
    );

    $(
      "#kr-detail-close"
    ).addEventListener(
      "click",
      dismissForPage
    );

    $(
      "#kr-understand"
    ).addEventListener(
      "click",
      requestAnalysis
    );

    $(
      "#kr-summarize"
    ).addEventListener(
      "click",
      requestSummary
    );

    $(
      "#kr-questions"
    ).addEventListener(
      "click",
      openQuestionsInKelsie
    );

    $(
      "#kr-back"
    ).addEventListener(
      "click",
      showMenu
    );

    $(
      "#kr-error-back"
    ).addEventListener(
      "click",
      showMenu
    );

    $(
      "#kr-retry"
    ).addEventListener(
      "click",
      () => {
        if (
          requestMode ===
          "summary"
        ) {
          requestSummary();
        } else {
          requestAnalysis();
        }
      }
    );

    $(
      "#kr-continue"
    ).addEventListener(
      "click",
      () => {
        const intro =
          currentQuestion
            ? `Let's keep working through it. ${currentQuestion}`
            : "Let's keep working through this page.";

        activatePageContext(
          "scaffold",
          intro
        );
      }
    );

    responseSend
      .addEventListener(
        "click",
        sendCoachResponse
      );

    responseInput
      .addEventListener(
        "keydown",
        (event) => {
          if (
            event.key !==
            "Enter"
          ) {
            return;
          }

          event.preventDefault();

          sendCoachResponse();
        }
      );

    window.addEventListener(
      "resize",
      () => {
        requestAnimationFrame(
          positionCard
        );
      }
    );

    window.addEventListener(
      "scroll",
      () => {
        if (
          state !==
          "inactive"
        ) {
          return;
        }

        const currentY =
          window.scrollY;

        scrollDistance +=
          Math.abs(
            currentY -
            lastScrollY
          );

        lastScrollY =
          currentY;
      },
      {
        passive: true,
      }
    );

    document.addEventListener(
      "visibilitychange",
      () => {
        if (
          document
            .visibilityState ===
            "visible" &&
          state ===
            "inactive"
        ) {
          scheduleDetection(
            500
          );
        }
      }
    );

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

        resetForNavigation();
      },
      PAGE_URL_CHECK_INTERVAL
    );

    /*
     * The iframe can tell us when the user manually removes
     * page context from normal Kelsie.
     */
    window.addEventListener(
      "message",
      (event) => {
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
            "kelsie-page-context-hook"
        ) {
          return;
        }

        if (
          data.type ===
            "KELSIE_PAGE_CONTEXT_CLEARED" &&
          state ===
            "context-open"
        ) {
          state =
            "dismissed";

          render();
        }
      }
    );

    /*
     * Re-render if Kelsie opens/closes, a reminder changes,
     * or the launcher moves. This keeps priority behavior
     * consistent with the existing extension.
     */
    const observer =
      new MutationObserver(
        render
      );

    observer.observe(
      panelHost,
      {
        attributes: true,

        attributeFilter: [
          "hidden",
          "class",
        ],
      }
    );

    if (reminderPopup) {
      observer.observe(
        reminderPopup,
        {
          attributes: true,

          attributeFilter: [
            "hidden",
            "aria-hidden",
            "class",
          ],
        }
      );
    }

    observer.observe(
      host,
      {
        attributes: true,

        attributeFilter: [
          "data-edge",
          "data-dragging",
        ],
      }
    );

    scheduleDetection();

    console.log(
      "[Kelsie Reading] Reading assistance v2 initialized."
    );
  }

  waitForExtensionShell();
})();