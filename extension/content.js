(() => {
    const ROOT_ID = "kelsie-extension-root";

    const PLACEMENT_KEY = "kelsieLauncherPlacement";
    const MOVE_HINT_KEY = "kelsieMoveHintSeen";

    const WIDGET_URL =
        "http://127.0.0.1:8000/static/widget.html?kelsie_extension_embed=1";

    const WIDGET_ORIGIN =
        "http://127.0.0.1:8000";

    const VALID_EDGES = [
        "top",
        "right",
        "bottom",
        "left",
    ];

    const DEFAULT_PLACEMENT = {
        edge: "right",
        ratio: 0.72,
    };

    const DRAG_THRESHOLD = 5;
    const SNAP_PREVIEW_DISTANCE = 72;
    const MOVE_HINT_DURATION = 5000;

    /*
     * Give the user time to actually land on the page before
     * Kelsie offers help.
     */
    const PAGE_SUGGESTION_DELAY = 7000;

    const PAGE_URL_CHECK_INTERVAL = 1200;
    const MAX_EXTRACTED_PAGE_CHARS = 18000;

    if (
        document.getElementById(
            ROOT_ID
        )
    ) {
        return;
    }

    if (window.top !== window.self) {
        return;
    }

    function clamp(
        value,
        min,
        max
    ) {
        return Math.min(
            Math.max(
                value,
                min
            ),
            max
        );
    }

    function normalizeWhitespace(
        value
    ) {
        return String(
            value || ""
        )
            .replace(/\s+/g, " ")
            .trim();
    }

    function normalizePlacement(
        value
    ) {
        if (
            !value ||
            !VALID_EDGES.includes(
                value.edge
            )
        ) {
            return {
                ...DEFAULT_PLACEMENT,
            };
        }

        return {
            edge: value.edge,

            ratio:
                typeof value.ratio ===
                "number"
                    ? clamp(
                        value.ratio,
                        0,
                        1
                    )
                    : DEFAULT_PLACEMENT
                        .ratio,
        };
    }

    function isElementVisible(
        element
    ) {
        if (!(element instanceof Element)) {
            return false;
        }

        const style =
            window.getComputedStyle(
                element
            );

        if (
            style.display === "none" ||
            style.visibility === "hidden" ||
            Number(style.opacity) === 0
        ) {
            return false;
        }

        const rect =
            element.getBoundingClientRect();

        return (
            rect.width > 0 &&
            rect.height > 0
        );
    }

    function safeText(
        value,
        maximumLength = 1000
    ) {
        return normalizeWhitespace(
            value
        ).slice(
            0,
            maximumLength
        );
    }

    // =========================================================
    // EXTENSION HOST
    // =========================================================

    const host =
        document.createElement(
            "div"
        );

    host.id = ROOT_ID;

    const shadow =
        host.attachShadow({
            mode: "open",
        });

    const stylesheet =
        document.createElement(
            "link"
        );

    stylesheet.rel =
        "stylesheet";

    stylesheet.href =
        chrome.runtime.getURL(
            "content.css"
        );

    shadow.appendChild(
        stylesheet
    );

    const ambientStyle =
        document.createElement(
            "style"
        );

    ambientStyle.textContent = `
        .kelsie-launcher-due-dot {
            position: absolute;
            top: 5px;
            right: 8px;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #808899;
            border: 1.5px solid rgba(255, 255, 255, 0.98);
            box-shadow: 0 1px 5px rgba(35, 39, 48, 0.24);
            opacity: 0;
            transform: scale(0.65);
            pointer-events: none;
            transition:
                opacity 150ms ease,
                transform 150ms ease;
        }

        .kelsie-launcher-due-dot.visible {
            opacity: 1;
            transform: scale(1);
        }

        .kelsie-reminder-popup,
        .kelsie-page-card {
            position: fixed;
            z-index: 2147483647;
            border: 1px solid rgba(42, 47, 58, 0.10);
            background:
                linear-gradient(
                    145deg,
                    rgba(255, 255, 255, 0.975),
                    rgba(244, 246, 249, 0.955)
                );
            box-shadow:
                0 14px 38px rgba(28, 32, 41, 0.14),
                0 4px 12px rgba(28, 32, 41, 0.07),
                inset 0 1px 0 rgba(255, 255, 255, 0.94);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            pointer-events: auto;
            animation:
                kelsie-ambient-surface-in
                180ms
                ease-out;
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Helvetica,
                Arial,
                sans-serif;
            box-sizing: border-box;
        }

        .kelsie-reminder-popup[hidden],
        .kelsie-page-card[hidden],
        .kelsie-page-view[hidden] {
            display: none !important;
        }

        .kelsie-reminder-popup {
            width: min(
                270px,
                calc(100vw - 20px)
            );
            min-height: 68px;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 8px 8px 10px;
            border-radius: 18px;
        }

        .kelsie-reminder-main {
            flex: 1;
            min-width: 0;
            min-height: 50px;
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 5px 6px;
            border: 0;
            border-radius: 12px;
            background: transparent;
            color: #303641;
            text-align: left;
            cursor: pointer;
            font: inherit;
        }

        .kelsie-reminder-main:hover {
            background:
                rgba(
                    55,
                    61,
                    74,
                    0.035
                );
        }

        .kelsie-ambient-orb {
            width: 24px;
            height: 24px;
            flex: 0 0 auto;
            border-radius: 50%;
            background:
                radial-gradient(
                    circle at 30% 23%,
                    rgba(255, 255, 255, 0.76),
                    rgba(255, 255, 255, 0) 42%
                ),
                linear-gradient(
                    145deg,
                    #a1a8b5,
                    #808899 55%,
                    #686f7e
                );
            border:
                1px solid
                rgba(255, 255, 255, 0.58);
            box-shadow:
                inset 0 1px 1px
                    rgba(255, 255, 255, 0.72),
                0 3px 8px
                    rgba(34, 39, 49, 0.15);
        }

        .kelsie-reminder-copy {
            min-width: 0;
            display: grid;
            gap: 3px;
        }

        .kelsie-ambient-kicker {
            color: #858c99;
            font-size: 8px;
            line-height: 1;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .kelsie-reminder-title {
            overflow: hidden;
            color: #303641;
            font-size: 11px;
            line-height: 1.35;
            font-weight: 630;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .kelsie-reminder-actions {
            flex: 0 0 auto;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .kelsie-icon-button,
        .kelsie-arrow-button {
            display: grid;
            place-items: center;
            margin: 0;
            padding: 0;
            border:
                1px solid
                rgba(50, 56, 68, 0.09);
            border-radius: 50%;
            background:
                rgba(
                    255,
                    255,
                    255,
                    0.76
                );
            color: #6c7481;
            cursor: pointer;
            font: inherit;
            transition:
                transform 130ms ease,
                background 130ms ease,
                color 130ms ease,
                opacity 130ms ease;
        }

        .kelsie-icon-button {
            width: 32px;
            height: 32px;
        }

        .kelsie-arrow-button {
            width: 30px;
            height: 30px;
        }

        .kelsie-icon-button svg {
            width: 16px;
            height: 16px;
            fill: none;
            stroke: currentColor;
            stroke-width: 1.8;
            stroke-linecap: round;
            stroke-linejoin: round;
        }

        /*
         * Explicit arrow characters rather than hidden/implicit SVG
         * treatment. These are the arrows the user sees.
         */
        .kelsie-arrow-glyph {
            display: block;
            font-size: 16px;
            line-height: 1;
            font-weight: 650;
            transform:
                translateY(-0.5px);
        }

        .kelsie-icon-button:hover:not(:disabled),
        .kelsie-arrow-button:hover:not(:disabled) {
            transform:
                translateY(-1px);
            background:
                rgba(
                    240,
                    243,
                    247,
                    0.98
                );
            color: #343a45;
        }

        .kelsie-icon-button:disabled,
        .kelsie-arrow-button:disabled {
            opacity: 0.42;
            cursor: default;
        }

        .kelsie-icon-button:focus-visible,
        .kelsie-arrow-button:focus-visible,
        .kelsie-reminder-main:focus-visible,
        .kelsie-page-choice:focus-visible,
        .kelsie-page-primary:focus-visible {
            outline:
                2px solid
                rgba(
                    128,
                    136,
                    153,
                    0.50
                );
            outline-offset: 2px;
        }

        /* =====================================================
           PAGE HELP
        ===================================================== */

        .kelsie-page-card {
            width:
                min(
                    318px,
                    calc(100vw - 20px)
                );
            overflow: hidden;
            border-radius: 19px;
        }

        .kelsie-page-offer {
            min-height: 68px;
            display: flex;
            align-items: center;
            gap: 9px;
            padding:
                8px
                8px
                8px
                11px;
        }

        .kelsie-page-offer-copy {
            flex: 1;
            min-width: 0;
            color: #303641;
            font-size: 11px;
            line-height: 1.3;
            font-weight: 650;
        }

        .kelsie-page-offer-actions {
            flex: 0 0 auto;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .kelsie-page-choice-view,
        .kelsie-page-detail {
            padding:
                13px
                14px
                14px;
        }

        .kelsie-page-choice-header,
        .kelsie-page-detail-header {
            display: flex;
            align-items: center;
            justify-content:
                space-between;
            gap: 10px;
            margin-bottom: 10px;
        }

        .kelsie-page-heading {
            min-width: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .kelsie-page-heading-copy {
            min-width: 0;
            display: grid;
            gap: 2px;
        }

        .kelsie-page-heading-title {
            color: #313742;
            font-size: 11px;
            line-height: 1.2;
            font-weight: 680;
        }

        .kelsie-page-heading-note {
            color: #8a919b;
            font-size: 8px;
            line-height: 1.25;
            font-weight: 540;
        }

        .kelsie-page-choice-list {
            display: grid;
            gap: 5px;
        }

        .kelsie-page-choice {
            width: 100%;
            min-height: 42px;
            display: flex;
            align-items: center;
            justify-content:
                space-between;
            gap: 12px;
            padding:
                0
                8px
                0
                11px;
            border:
                1px solid
                rgba(
                    49,
                    55,
                    67,
                    0.075
                );
            border-radius: 13px;
            background:
                rgba(
                    255,
                    255,
                    255,
                    0.57
                );
            color: #444b57;
            text-align: left;
            cursor: pointer;
            font-family: inherit;
            font-size: 10px;
            line-height: 1.25;
            font-weight: 620;
            transition:
                background 130ms ease,
                border-color 130ms ease,
                transform 130ms ease;
        }

        .kelsie-page-choice:hover {
            transform:
                translateY(-1px);
            border-color:
                rgba(
                    49,
                    55,
                    67,
                    0.11
                );
            background:
                rgba(
                    248,
                    249,
                    251,
                    0.96
                );
        }

        .kelsie-page-choice-arrow {
            width: 27px;
            height: 27px;
            flex: 0 0 auto;
            display: grid;
            place-items: center;
            border-radius: 50%;
            background:
                rgba(
                    239,
                    242,
                    246,
                    0.94
                );
            color: #737b87;
        }

        .kelsie-page-loading-row {
            min-height: 74px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: #666e7b;
            font-size: 10px;
            line-height: 1.4;
        }

        .kelsie-page-loading-dot {
            width: 8px;
            height: 8px;
            flex: 0 0 auto;
            border-radius: 50%;
            background: #808899;
            animation:
                kelsie-page-loading-breathe
                1.35s
                ease-in-out
                infinite;
        }

        .kelsie-page-answer {
            color: #3a404b;
            font-size: 10px;
            line-height: 1.55;
            font-weight: 500;
        }

        .kelsie-page-points-label {
            margin-top: 12px;
            color: #858c98;
            font-size: 8px;
            line-height: 1;
            font-weight: 700;
            letter-spacing: 0.065em;
            text-transform: uppercase;
        }

        .kelsie-page-points {
            margin:
                8px
                0
                0;
            padding: 0;
            display: grid;
            gap: 7px;
            list-style: none;
        }

        .kelsie-page-point {
            position: relative;
            padding-left: 13px;
            color: #4c535f;
            font-size: 9px;
            line-height: 1.48;
        }

        .kelsie-page-point::before {
            content: "";
            position: absolute;
            left: 1px;
            top: 0.55em;
            width: 4px;
            height: 4px;
            border-radius: 50%;
            background: #8b929e;
        }

        .kelsie-page-error {
            padding:
                3px
                0
                2px;
            color: #5e6672;
            font-size: 10px;
            line-height: 1.5;
        }

        .kelsie-page-actions {
            margin-top: 13px;
            display: flex;
            align-items: center;
            justify-content:
                space-between;
            gap: 7px;
        }

        .kelsie-page-primary {
            min-height: 32px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
            padding:
                0
                11px;
            border:
                1px solid
                rgba(
                    49,
                    55,
                    67,
                    0.10
                );
            border-radius: 11px;
            background:
                rgba(
                    245,
                    247,
                    249,
                    0.96
                );
            color: #4e5561;
            font-size: 9px;
            line-height: 1;
            font-weight: 650;
            cursor: pointer;
            font-family: inherit;
        }

        .kelsie-page-primary:hover {
            background:
                rgba(
                    239,
                    242,
                    246,
                    0.98
                );
        }

        .kelsie-page-inline-arrow {
            display: grid;
            place-items: center;
        }

        @keyframes kelsie-ambient-surface-in {
            from {
                opacity: 0;
                transform:
                    translateY(5px)
                    scale(0.988);
            }

            to {
                opacity: 1;
                transform:
                    translateY(0)
                    scale(1);
            }
        }

        @keyframes kelsie-page-loading-breathe {
            0%,
            100% {
                opacity: 0.55;
                transform:
                    scale(0.88);
            }

            50% {
                opacity: 1;
                transform:
                    scale(1.08);
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .kelsie-launcher-due-dot,
            .kelsie-reminder-popup,
            .kelsie-page-card,
            .kelsie-icon-button,
            .kelsie-arrow-button,
            .kelsie-page-choice,
            .kelsie-page-loading-dot {
                animation:
                    none !important;
                transition:
                    none !important;
            }
        }
    `;

    shadow.appendChild(
        ambientStyle
    );

    // =========================================================
    // UI
    // =========================================================

    const shell =
        document.createElement(
            "div"
        );

    shell.className =
        "kelsie-shell";

    shell.innerHTML = `
        <section
            class="kelsie-panel-host"
            id="kelsie-panel-host"
            aria-label="Kelsie assistant"
            hidden
        >
            <div
                class="kelsie-frame-loading"
                id="kelsie-frame-loading"
                role="status"
                aria-live="polite"
            >
                <span
                    class="kelsie-loading-orb"
                    aria-hidden="true"
                ></span>

                <span>
                    Opening Kelsie…
                </span>
            </div>

            <iframe
                class="kelsie-widget-frame"
                id="kelsie-widget-frame"
                title="Kelsie assistant"
                referrerpolicy="no-referrer"
            ></iframe>
        </section>

        <section
            class="kelsie-reminder-popup"
            id="kelsie-reminder-popup"
            role="status"
            aria-live="assertive"
            hidden
        >
            <button
                class="kelsie-reminder-main"
                id="kelsie-reminder-main"
                type="button"
                aria-label="Open Kelsie"
            >
                <span
                    class="kelsie-ambient-orb"
                    aria-hidden="true"
                ></span>

                <span
                    class="kelsie-reminder-copy"
                >
                    <span
                        class="kelsie-ambient-kicker"
                    >
                        Kelsie reminder
                    </span>

                    <span
                        class="kelsie-reminder-title"
                        id="kelsie-reminder-title"
                    ></span>
                </span>
            </button>

            <div
                class="kelsie-reminder-actions"
            >
                <button
                    class="kelsie-icon-button"
                    id="kelsie-reminder-done"
                    type="button"
                    aria-label="Mark reminder done"
                    title="Done"
                >
                    <svg
                        viewBox="0 0 24 24"
                        aria-hidden="true"
                    >
                        <path
                            d="M5 12.5 9.4 17 19 7.5"
                        ></path>
                    </svg>
                </button>

                <button
                    class="kelsie-icon-button"
                    id="kelsie-reminder-dismiss"
                    type="button"
                    aria-label="Dismiss reminder for 15 minutes"
                    title="Dismiss for 15 minutes"
                >
                    <svg
                        viewBox="0 0 24 24"
                        aria-hidden="true"
                    >
                        <path
                            d="M7 7 17 17M17 7 7 17"
                        ></path>
                    </svg>
                </button>
            </div>
        </section>

        <section
            class="kelsie-page-card"
            id="kelsie-page-card"
            aria-live="polite"
            hidden
        >
            <!-- First stage -->
            <div
                class="kelsie-page-view kelsie-page-offer"
                id="kelsie-page-offer-view"
            >
                <span
                    class="kelsie-ambient-orb"
                    aria-hidden="true"
                ></span>

                <span
                    class="kelsie-page-offer-copy"
                >
                    Help with this page?
                </span>

                <span
                    class="kelsie-page-offer-actions"
                >
                    <button
                        class="kelsie-arrow-button"
                        id="kelsie-page-offer-next"
                        type="button"
                        aria-label="See ways Kelsie can help"
                        title="Continue"
                    >
                        <span
                            class="kelsie-arrow-glyph"
                            aria-hidden="true"
                        >
                            →
                        </span>
                    </button>

                    <button
                        class="kelsie-icon-button"
                        id="kelsie-page-dismiss"
                        type="button"
                        aria-label="Dismiss page help"
                        title="Not on this page"
                    >
                        <svg
                            viewBox="0 0 24 24"
                            aria-hidden="true"
                        >
                            <path
                                d="M7 7 17 17M17 7 7 17"
                            ></path>
                        </svg>
                    </button>
                </span>
            </div>

            <!-- Second stage -->
            <div
                class="kelsie-page-view kelsie-page-choice-view"
                id="kelsie-page-choice-view"
                hidden
            >
                <div
                    class="kelsie-page-choice-header"
                >
                    <div
                        class="kelsie-page-heading"
                    >
                        <span
                            class="kelsie-ambient-orb"
                            aria-hidden="true"
                        ></span>

                        <span
                            class="kelsie-page-heading-copy"
                        >
                            <span
                                class="kelsie-page-heading-title"
                            >
                                What would help?
                            </span>
                        </span>
                    </div>

                    <button
                        class="kelsie-icon-button"
                        id="kelsie-page-choice-close"
                        type="button"
                        aria-label="Close page help"
                        title="Close"
                    >
                        <svg
                            viewBox="0 0 24 24"
                            aria-hidden="true"
                        >
                            <path
                                d="M7 7 17 17M17 7 7 17"
                            ></path>
                        </svg>
                    </button>
                </div>

                <div
                    class="kelsie-page-choice-list"
                >
                    <button
                        class="kelsie-page-choice"
                        id="kelsie-page-summarize"
                        type="button"
                    >
                        <span>
                            Summarize
                        </span>

                        <span
                            class="kelsie-page-choice-arrow"
                            aria-hidden="true"
                        >
                            <span
                                class="kelsie-arrow-glyph"
                            >
                                →
                            </span>
                        </span>
                    </button>

                    <button
                        class="kelsie-page-choice"
                        id="kelsie-page-explain"
                        type="button"
                    >
                        <span>
                            Help me understand
                        </span>

                        <span
                            class="kelsie-page-choice-arrow"
                            aria-hidden="true"
                        >
                            <span
                                class="kelsie-arrow-glyph"
                            >
                                →
                            </span>
                        </span>
                    </button>

                    <button
                        class="kelsie-page-choice"
                        id="kelsie-page-questions"
                        type="button"
                    >
                        <span>
                            I have questions
                        </span>

                        <span
                            class="kelsie-page-choice-arrow"
                            aria-hidden="true"
                        >
                            <span
                                class="kelsie-arrow-glyph"
                            >
                                →
                            </span>
                        </span>
                    </button>
                </div>
            </div>

            <!-- Result/loading stage -->
            <div
                class="kelsie-page-view kelsie-page-detail"
                id="kelsie-page-detail-view"
                hidden
            >
                <div
                    class="kelsie-page-detail-header"
                >
                    <div
                        class="kelsie-page-heading"
                    >
                        <span
                            class="kelsie-ambient-orb"
                            aria-hidden="true"
                        ></span>

                        <span
                            class="kelsie-page-heading-copy"
                        >
                            <span
                                class="kelsie-page-heading-title"
                                id="kelsie-page-detail-title"
                            >
                                Kelsie
                            </span>

                            <span
                                class="kelsie-page-heading-note"
                                id="kelsie-page-detail-note"
                            ></span>
                        </span>
                    </div>

                    <button
                        class="kelsie-icon-button"
                        id="kelsie-page-detail-close"
                        type="button"
                        aria-label="Close page help"
                        title="Close"
                    >
                        <svg
                            viewBox="0 0 24 24"
                            aria-hidden="true"
                        >
                            <path
                                d="M7 7 17 17M17 7 7 17"
                            ></path>
                        </svg>
                    </button>
                </div>

                <div
                    class="kelsie-page-loading-row"
                    id="kelsie-page-loading"
                    hidden
                >
                    <span
                        class="kelsie-page-loading-dot"
                        aria-hidden="true"
                    ></span>

                    <span>
                        Reading this page only…
                    </span>
                </div>

                <div
                    id="kelsie-page-result"
                    hidden
                >
                    <div
                        class="kelsie-page-answer"
                        id="kelsie-page-answer"
                    ></div>

                    <div
                        class="kelsie-page-points-label"
                        id="kelsie-page-points-label"
                    >
                        Key points
                    </div>

                    <ul
                        class="kelsie-page-points"
                        id="kelsie-page-points"
                    ></ul>

                    <div
                        class="kelsie-page-actions"
                    >
                        <span></span>

                        <button
                            class="kelsie-page-primary"
                            id="kelsie-page-ask"
                            type="button"
                        >
                            <span>
                                Ask about this
                            </span>

                            <span
                                class="kelsie-page-inline-arrow"
                                aria-hidden="true"
                            >
                                <span
                                    class="kelsie-arrow-glyph"
                                >
                                    →
                                </span>
                            </span>
                        </button>
                    </div>
                </div>

                <div
                    id="kelsie-page-error-view"
                    hidden
                >
                    <div
                        class="kelsie-page-error"
                        id="kelsie-page-error"
                    ></div>

                    <div
                        class="kelsie-page-actions"
                    >
                        <button
                            class="kelsie-page-primary"
                            id="kelsie-page-back"
                            type="button"
                        >
                            Back
                        </button>

                        <button
                            class="kelsie-page-primary"
                            id="kelsie-page-retry"
                            type="button"
                        >
                            Try again
                        </button>
                    </div>
                </div>
            </div>
        </section>

        <button
            class="kelsie-launcher"
            id="kelsie-launcher"
            type="button"
            aria-label="Open Kelsie"
            aria-expanded="false"
            title="Open Kelsie"
        >
            <span
                class="kelsie-launcher-orb"
                aria-hidden="true"
            ></span>

            <span
                class="kelsie-launcher-label"
            >
                Kelsie
            </span>

            <span
                class="kelsie-launcher-due-dot"
                id="kelsie-launcher-due-dot"
                aria-hidden="true"
            ></span>
        </button>

        <div
            class="kelsie-move-hint"
            id="kelsie-move-hint"
            role="status"
            hidden
        >
            Drag me to any edge
        </div>
    `;

    shadow.appendChild(
        shell
    );

    const mountTarget =
        document.body ||
        document.documentElement;

    mountTarget.appendChild(
        host
    );

    // =========================================================
    // ELEMENTS
    // =========================================================

    const panelHost =
        shadow.getElementById(
            "kelsie-panel-host"
        );

    const launcher =
        shadow.getElementById(
            "kelsie-launcher"
        );

    const moveHint =
        shadow.getElementById(
            "kelsie-move-hint"
        );

    const widgetFrame =
        shadow.getElementById(
            "kelsie-widget-frame"
        );

    const frameLoading =
        shadow.getElementById(
            "kelsie-frame-loading"
        );

    const reminderPopup =
        shadow.getElementById(
            "kelsie-reminder-popup"
        );

    const reminderMain =
        shadow.getElementById(
            "kelsie-reminder-main"
        );

    const reminderTitle =
        shadow.getElementById(
            "kelsie-reminder-title"
        );

    const reminderDone =
        shadow.getElementById(
            "kelsie-reminder-done"
        );

    const reminderDismiss =
        shadow.getElementById(
            "kelsie-reminder-dismiss"
        );

    const launcherDueDot =
        shadow.getElementById(
            "kelsie-launcher-due-dot"
        );

    const pageCard =
        shadow.getElementById(
            "kelsie-page-card"
        );

    const pageOfferView =
        shadow.getElementById(
            "kelsie-page-offer-view"
        );

    const pageOfferNext =
        shadow.getElementById(
            "kelsie-page-offer-next"
        );

    const pageDismissButton =
        shadow.getElementById(
            "kelsie-page-dismiss"
        );

    const pageChoiceView =
        shadow.getElementById(
            "kelsie-page-choice-view"
        );

    const pageChoiceClose =
        shadow.getElementById(
            "kelsie-page-choice-close"
        );

    const pageSummarizeButton =
        shadow.getElementById(
            "kelsie-page-summarize"
        );

    const pageExplainButton =
        shadow.getElementById(
            "kelsie-page-explain"
        );

    const pageQuestionsButton =
        shadow.getElementById(
            "kelsie-page-questions"
        );

    const pageDetailView =
        shadow.getElementById(
            "kelsie-page-detail-view"
        );

    const pageDetailClose =
        shadow.getElementById(
            "kelsie-page-detail-close"
        );

    const pageDetailTitle =
        shadow.getElementById(
            "kelsie-page-detail-title"
        );

    const pageDetailNote =
        shadow.getElementById(
            "kelsie-page-detail-note"
        );

    const pageLoading =
        shadow.getElementById(
            "kelsie-page-loading"
        );

    const pageResult =
        shadow.getElementById(
            "kelsie-page-result"
        );

    const pageAnswer =
        shadow.getElementById(
            "kelsie-page-answer"
        );

    const pagePointsLabel =
        shadow.getElementById(
            "kelsie-page-points-label"
        );

    const pagePoints =
        shadow.getElementById(
            "kelsie-page-points"
        );

    const pageAskButton =
        shadow.getElementById(
            "kelsie-page-ask"
        );

    const pageErrorView =
        shadow.getElementById(
            "kelsie-page-error-view"
        );

    const pageError =
        shadow.getElementById(
            "kelsie-page-error"
        );

    const pageBack =
        shadow.getElementById(
            "kelsie-page-back"
        );

    const pageRetry =
        shadow.getElementById(
            "kelsie-page-retry"
        );

    // =========================================================
    // STATE
    // =========================================================

    let panelOpen = false;

    let placement = {
        ...DEFAULT_PLACEMENT,
    };

    let dragState = null;
    let suppressClickAfterDrag = false;

    let moveHintTimer = null;

    let frameStarted = false;
    let frameReady = false;

    let currentReminder = null;
    let reminderActionBusy = false;

    let pageState = "inactive";
    let pageCandidate = null;
    let pageAssistData = null;
    let pageAssistError = "";
    let pageAssistAction = "summarize";
    let pageSuggestionTimer = null;
    let observedPageUrl =
        window.location.href;

    let pendingPageContext = null;

    // =========================================================
    // EMBEDDED KELSIE
    // =========================================================

    function startWidgetFrame() {
        if (frameStarted) {
            return;
        }

        frameStarted = true;
        frameReady = false;

        panelHost.classList.remove(
            "frame-ready"
        );

        frameLoading.hidden =
            false;

        /*
         * This starts Kelsie's hidden runtime immediately.
         * The panel remains visually collapsed.
         *
         * Keeping the iframe alive is what allows existing
         * reminder polling to continue while Kelsie is minimized.
         */
        widgetFrame.src =
            WIDGET_URL;
    }

    function sendWidgetVisibility(
        visible
    ) {
        if (
            !frameStarted ||
            !widgetFrame.contentWindow
        ) {
            return;
        }

        widgetFrame
            .contentWindow
            .postMessage(
                {
                    source:
                        "kelsie-extension-shell",

                    type:
                        "KELSIE_EXTENSION_VISIBILITY",

                    visible:
                        Boolean(
                            visible
                        ),
                },
                WIDGET_ORIGIN
            );
    }

    function sendPageContextToWidget(
        context
    ) {
        pendingPageContext =
            context;

        if (
            !frameReady ||
            !context ||
            !widgetFrame.contentWindow
        ) {
            return;
        }

        widgetFrame
            .contentWindow
            .postMessage(
                {
                    source:
                        "kelsie-extension-shell",

                    type:
                        "KELSIE_EXTENSION_PAGE_CONTEXT",

                    context,
                },
                WIDGET_ORIGIN
            );
    }

    function sendReminderAction(
        action
    ) {
        if (
            !currentReminder ||
            reminderActionBusy ||
            !widgetFrame.contentWindow
        ) {
            return;
        }

        reminderActionBusy =
            true;

        renderAmbientSurfaces();

        widgetFrame
            .contentWindow
            .postMessage(
                {
                    source:
                        "kelsie-extension-shell",

                    type:
                        "KELSIE_EXTENSION_REMINDER_ACTION",

                    action,

                    reminderId:
                        Number(
                            currentReminder.id
                        ),
                },
                WIDGET_ORIGIN
            );

        window.setTimeout(
            () => {
                if (
                    currentReminder
                ) {
                    reminderActionBusy =
                        false;

                    renderAmbientSurfaces();
                }
            },
            3500
        );
    }

    window.addEventListener(
        "message",
        (event) => {
            if (
                event.source !==
                widgetFrame.contentWindow
            ) {
                return;
            }

            if (
                event.origin !==
                WIDGET_ORIGIN
            ) {
                return;
            }

            const data =
                event.data;

            if (
                !data ||
                data.source !==
                    "kelsie-widget-bridge"
            ) {
                return;
            }

            if (
                data.type ===
                "KELSIE_EXTENSION_READY"
            ) {
                frameReady =
                    true;

                panelHost
                    .classList
                    .add(
                        "frame-ready"
                    );

                frameLoading.hidden =
                    true;

                sendWidgetVisibility(
                    panelOpen
                );

                if (
                    pendingPageContext
                ) {
                    sendPageContextToWidget(
                        pendingPageContext
                    );
                }

                if (
                    panelOpen
                ) {
                    requestAnimationFrame(
                        positionPanel
                    );
                }

                return;
            }

            if (
                data.type ===
                "KELSIE_EXTENSION_CLOSE"
            ) {
                setPanelOpen(
                    false
                );

                return;
            }

            if (
                data.type ===
                "KELSIE_EXTENSION_REMINDER_DUE"
            ) {
                const reminderId =
                    Number(
                        data.reminderId
                    );

                if (
                    !Number.isFinite(
                        reminderId
                    ) ||
                    reminderId <= 0
                ) {
                    return;
                }

                const changedReminder =
                    !currentReminder ||
                    Number(
                        currentReminder.id
                    ) !== reminderId;

                currentReminder = {
                    id:
                        reminderId,

                    title:
                        safeText(
                            data.title ||
                            "Reminder",
                            180
                        ),

                    displayText:
                        safeText(
                            data.displayText ||
                            data.title ||
                            "Reminder",
                            220
                        ),
                };

                if (
                    changedReminder
                ) {
                    reminderActionBusy =
                        false;
                }

                renderAmbientSurfaces();

                return;
            }

            if (
                data.type ===
                "KELSIE_EXTENSION_REMINDER_CLEAR"
            ) {
                currentReminder =
                    null;

                reminderActionBusy =
                    false;

                renderAmbientSurfaces();
            }
        }
    );

    // =========================================================
    // MOVE HINT
    // =========================================================

    function hideMoveHint() {
        if (
            moveHintTimer
        ) {
            clearTimeout(
                moveHintTimer
            );

            moveHintTimer =
                null;
        }

        moveHint.hidden =
            true;
    }

    function positionMoveHint() {
        if (
            moveHint.hidden
        ) {
            return;
        }

        const launcherRect =
            launcher
                .getBoundingClientRect();

        const hintRect =
            moveHint
                .getBoundingClientRect();

        const margin = 8;
        const gap = 9;

        let left =
            launcherRect.left;

        let top =
            launcherRect.top;

        if (
            placement.edge ===
            "right"
        ) {
            left =
                launcherRect.left -
                hintRect.width -
                gap;

            top =
                launcherRect.top +
                launcherRect.height /
                    2 -
                hintRect.height /
                    2;
        }

        if (
            placement.edge ===
            "left"
        ) {
            left =
                launcherRect.right +
                gap;

            top =
                launcherRect.top +
                launcherRect.height /
                    2 -
                hintRect.height /
                    2;
        }

        if (
            placement.edge ===
            "top"
        ) {
            left =
                launcherRect.left +
                launcherRect.width /
                    2 -
                hintRect.width /
                    2;

            top =
                launcherRect.bottom +
                gap;
        }

        if (
            placement.edge ===
            "bottom"
        ) {
            left =
                launcherRect.left +
                launcherRect.width /
                    2 -
                hintRect.width /
                    2;

            top =
                launcherRect.top -
                hintRect.height -
                gap;
        }

        moveHint.style.left =
            `${clamp(
                left,
                margin,
                Math.max(
                    margin,
                    window.innerWidth -
                        hintRect.width -
                        margin
                )
            )}px`;

        moveHint.style.top =
            `${clamp(
                top,
                margin,
                Math.max(
                    margin,
                    window.innerHeight -
                        hintRect.height -
                        margin
                )
            )}px`;
    }

    async function markMoveHintSeen() {
        try {
            await chrome
                .storage
                .local
                .set({
                    [MOVE_HINT_KEY]:
                        true,
                });
        } catch (error) {
            console.error(
                "[Kelsie] Could not save move-hint state:",
                error
            );
        }
    }

    async function showMoveHintOnce() {
        try {
            const result =
                await chrome
                    .storage
                    .local
                    .get({
                        [MOVE_HINT_KEY]:
                            false,
                    });

            if (
                result[
                    MOVE_HINT_KEY
                ] ||
                panelOpen
            ) {
                return;
            }

            moveHint.hidden =
                false;

            requestAnimationFrame(
                positionMoveHint
            );

            await markMoveHintSeen();

            moveHintTimer =
                window.setTimeout(
                    hideMoveHint,
                    MOVE_HINT_DURATION
                );
        } catch (error) {
            console.error(
                "[Kelsie] Could not load move-hint state:",
                error
            );
        }
    }

    // =========================================================
    // PLACEMENT
    // =========================================================

    function clearHostPosition() {
        host.style.left = "";
        host.style.right = "";
        host.style.top = "";
        host.style.bottom = "";
    }

    function getHostSize() {
        return {
            width:
                host.offsetWidth ||
                100,

            height:
                host.offsetHeight ||
                38,
        };
    }

    function applyPlacement() {
        placement =
            normalizePlacement(
                placement
            );

        host.setAttribute(
            "data-edge",
            placement.edge
        );

        host.removeAttribute(
            "data-preview-edge"
        );

        clearHostPosition();

        const {
            width,
            height,
        } =
            getHostSize();

        if (
            placement.edge ===
                "left" ||
            placement.edge ===
                "right"
        ) {
            const available =
                Math.max(
                    0,

                    window.innerHeight -
                        height
                );

            const top =
                available *
                placement.ratio;

            host.style.top =
                `${top}px`;

            if (
                placement.edge ===
                "left"
            ) {
                host.style.left =
                    "0px";
            } else {
                host.style.right =
                    "0px";
            }
        } else {
            const available =
                Math.max(
                    0,

                    window.innerWidth -
                        width
                );

            const left =
                available *
                placement.ratio;

            host.style.left =
                `${left}px`;

            if (
                placement.edge ===
                "top"
            ) {
                host.style.top =
                    "0px";
            } else {
                host.style.bottom =
                    "0px";
            }
        }

        if (
            panelOpen
        ) {
            requestAnimationFrame(
                positionPanel
            );
        }

        requestAnimationFrame(
            () => {
                positionMoveHint();
                positionAmbientSurfaces();
            }
        );
    }

    async function savePlacement() {
        try {
            await chrome
                .storage
                .local
                .set({
                    [PLACEMENT_KEY]:
                        placement,
                });
        } catch (error) {
            console.error(
                "[Kelsie] Could not save launcher placement:",
                error
            );
        }
    }

    async function loadPlacement() {
        try {
            const result =
                await chrome
                    .storage
                    .local
                    .get({
                        [PLACEMENT_KEY]:
                            DEFAULT_PLACEMENT,
                    });

            placement =
                normalizePlacement(
                    result[
                        PLACEMENT_KEY
                    ]
                );
        } catch (error) {
            console.error(
                "[Kelsie] Could not load launcher placement:",
                error
            );

            placement = {
                ...DEFAULT_PLACEMENT,
            };
        }

        applyPlacement();

        requestAnimationFrame(
            applyPlacement
        );
    }

    function getEdgeDistances() {
        const rect =
            host
                .getBoundingClientRect();

        return {
            left:
                Math.max(
                    0,
                    rect.left
                ),

            right:
                Math.max(
                    0,

                    window.innerWidth -
                        rect.right
                ),

            top:
                Math.max(
                    0,
                    rect.top
                ),

            bottom:
                Math.max(
                    0,

                    window.innerHeight -
                        rect.bottom
                ),
        };
    }

    function getNearestEdgeInfo() {
        const distances =
            getEdgeDistances();

        const sorted =
            Object
                .entries(
                    distances
                )
                .sort(
                    (a, b) =>
                        a[1] -
                        b[1]
                );

        return {
            edge:
                sorted[0][0],

            distance:
                sorted[0][1],
        };
    }

    function updateSnapPreview() {
        const {
            edge,
            distance,
        } =
            getNearestEdgeInfo();

        if (
            distance <=
            SNAP_PREVIEW_DISTANCE
        ) {
            host.setAttribute(
                "data-preview-edge",
                edge
            );
        } else {
            host.removeAttribute(
                "data-preview-edge"
            );
        }
    }

    function placementFromCurrentPosition(
        edge
    ) {
        const rect =
            host
                .getBoundingClientRect();

        if (
            edge === "left" ||
            edge === "right"
        ) {
            const available =
                Math.max(
                    1,

                    window.innerHeight -
                        rect.height
                );

            return {
                edge,

                ratio:
                    clamp(
                        rect.top /
                            available,

                        0,
                        1
                    ),
            };
        }

        const available =
            Math.max(
                1,

                window.innerWidth -
                    rect.width
            );

        return {
            edge,

            ratio:
                clamp(
                    rect.left /
                        available,

                    0,
                    1
                ),
        };
    }

    // =========================================================
    // DRAGGING
    // =========================================================

    launcher.addEventListener(
        "pointerdown",
        (event) => {
            if (
                event.button !== 0 &&
                event.pointerType ===
                    "mouse"
            ) {
                return;
            }

            const rect =
                host
                    .getBoundingClientRect();

            dragState = {
                pointerId:
                    event.pointerId,

                startX:
                    event.clientX,

                startY:
                    event.clientY,

                grabX:
                    event.clientX -
                    rect.left,

                grabY:
                    event.clientY -
                    rect.top,

                dragging:
                    false,
            };

            launcher.setPointerCapture(
                event.pointerId
            );
        }
    );

    launcher.addEventListener(
        "pointermove",
        (event) => {
            if (
                !dragState ||
                dragState.pointerId !==
                    event.pointerId
            ) {
                return;
            }

            const movementX =
                event.clientX -
                dragState.startX;

            const movementY =
                event.clientY -
                dragState.startY;

            const distance =
                Math.hypot(
                    movementX,
                    movementY
                );

            if (
                !dragState.dragging &&
                distance <
                    DRAG_THRESHOLD
            ) {
                return;
            }

            if (
                !dragState.dragging
            ) {
                dragState.dragging =
                    true;

                host.setAttribute(
                    "data-dragging",
                    "true"
                );

                hideMoveHint();

                markMoveHintSeen();

                renderAmbientSurfaces();
            }

            event.preventDefault();

            const {
                width,
                height,
            } =
                getHostSize();

            const left =
                clamp(
                    event.clientX -
                        dragState.grabX,

                    0,

                    window.innerWidth -
                        width
                );

            const top =
                clamp(
                    event.clientY -
                        dragState.grabY,

                    0,

                    window.innerHeight -
                        height
                );

            clearHostPosition();

            host.style.left =
                `${left}px`;

            host.style.top =
                `${top}px`;

            updateSnapPreview();
        }
    );

    async function finishDrag(
        event
    ) {
        if (
            !dragState ||
            dragState.pointerId !==
                event.pointerId
        ) {
            return;
        }

        const wasDragging =
            dragState.dragging;

        dragState =
            null;

        if (
            launcher.hasPointerCapture(
                event.pointerId
            )
        ) {
            launcher.releasePointerCapture(
                event.pointerId
            );
        }

        host.removeAttribute(
            "data-dragging"
        );

        if (
            !wasDragging
        ) {
            host.removeAttribute(
                "data-preview-edge"
            );

            return;
        }

        suppressClickAfterDrag =
            true;

        const {
            edge: nearestEdge,
        } =
            getNearestEdgeInfo();

        placement =
            placementFromCurrentPosition(
                nearestEdge
            );

        applyPlacement();

        await savePlacement();

        renderAmbientSurfaces();

        setTimeout(
            () => {
                suppressClickAfterDrag =
                    false;
            },
            120
        );
    }

    launcher.addEventListener(
        "pointerup",
        finishDrag
    );

    launcher.addEventListener(
        "pointercancel",
        finishDrag
    );

    // =========================================================
    // POSITIONING
    // =========================================================

    function positionPanel() {
        if (
            !panelOpen
        ) {
            return;
        }

        const launcherRect =
            launcher
                .getBoundingClientRect();

        const panelRect =
            panelHost
                .getBoundingClientRect();

        const viewportWidth =
            window.innerWidth;

        const viewportHeight =
            window.innerHeight;

        const margin = 8;
        const gap = 8;

        let left =
            margin;

        let top =
            margin;

        if (
            placement.edge ===
            "right"
        ) {
            left =
                viewportWidth -
                panelRect.width -
                margin;

            top =
                launcherRect.top +
                launcherRect.height /
                    2 -
                panelRect.height /
                    2;
        }

        if (
            placement.edge ===
            "left"
        ) {
            left =
                margin;

            top =
                launcherRect.top +
                launcherRect.height /
                    2 -
                panelRect.height /
                    2;
        }

        if (
            placement.edge ===
            "top"
        ) {
            left =
                launcherRect.left +
                launcherRect.width /
                    2 -
                panelRect.width /
                    2;

            top =
                launcherRect.bottom +
                gap;
        }

        if (
            placement.edge ===
            "bottom"
        ) {
            left =
                launcherRect.left +
                launcherRect.width /
                    2 -
                panelRect.width /
                    2;

            top =
                launcherRect.top -
                panelRect.height -
                gap;
        }

        left =
            clamp(
                left,
                margin,

                Math.max(
                    margin,

                    viewportWidth -
                        panelRect.width -
                        margin
                )
            );

        top =
            clamp(
                top,
                margin,

                Math.max(
                    margin,

                    viewportHeight -
                        panelRect.height -
                        margin
                )
            );

        panelHost.style.left =
            `${left}px`;

        panelHost.style.top =
            `${top}px`;
    }

    function positionFloatingSurface(
        surface
    ) {
        if (
            !surface ||
            surface.hidden
        ) {
            return;
        }

        const launcherRect =
            launcher
                .getBoundingClientRect();

        const surfaceRect =
            surface
                .getBoundingClientRect();

        const margin = 8;
        const gap = 10;

        let left =
            launcherRect.left;

        let top =
            launcherRect.top;

        if (
            placement.edge ===
            "right"
        ) {
            left =
                launcherRect.left -
                surfaceRect.width -
                gap;

            top =
                launcherRect.top +
                launcherRect.height /
                    2 -
                surfaceRect.height /
                    2;
        }

        if (
            placement.edge ===
            "left"
        ) {
            left =
                launcherRect.right +
                gap;

            top =
                launcherRect.top +
                launcherRect.height /
                    2 -
                surfaceRect.height /
                    2;
        }

        if (
            placement.edge ===
            "top"
        ) {
            left =
                launcherRect.left +
                launcherRect.width /
                    2 -
                surfaceRect.width /
                    2;

            top =
                launcherRect.bottom +
                gap;
        }

        if (
            placement.edge ===
            "bottom"
        ) {
            left =
                launcherRect.left +
                launcherRect.width /
                    2 -
                surfaceRect.width /
                    2;

            top =
                launcherRect.top -
                surfaceRect.height -
                gap;
        }

        surface.style.left =
            `${clamp(
                left,
                margin,

                Math.max(
                    margin,

                    window.innerWidth -
                        surfaceRect.width -
                        margin
                )
            )}px`;

        surface.style.top =
            `${clamp(
                top,
                margin,

                Math.max(
                    margin,

                    window.innerHeight -
                        surfaceRect.height -
                        margin
                )
            )}px`;
    }

    function positionAmbientSurfaces() {
        positionFloatingSurface(
            reminderPopup
        );

        positionFloatingSurface(
            pageCard
        );
    }

    // =========================================================
    // REMINDER AMBIENT SURFACE
    // =========================================================

    function renderReminder() {
        const hasDueReminder =
            Boolean(
                currentReminder
            );

        launcherDueDot
            .classList
            .toggle(
                "visible",
                hasDueReminder
            );

        const shouldShow =
            hasDueReminder &&
            !panelOpen &&
            !host.hasAttribute(
                "data-dragging"
            );

        reminderPopup.hidden =
            !shouldShow;

        reminderDone.disabled =
            reminderActionBusy;

        reminderDismiss.disabled =
            reminderActionBusy;

        if (
            !shouldShow
        ) {
            return;
        }

        reminderTitle.textContent =
            currentReminder
                .displayText ||
            currentReminder
                .title ||
            "Reminder";

        requestAnimationFrame(
            () => {
                positionFloatingSurface(
                    reminderPopup
                );
            }
        );
    }

    // =========================================================
    // LEVEL 1 — READING SURFACE DETECTION
    // =========================================================

    function isUnsupportedPageForHelp() {
        const protocol =
            window.location.protocol;

        if (
            protocol !== "http:" &&
            protocol !== "https:"
        ) {
            return true;
        }

        /*
         * PDF support stays separate for now rather than pretending
         * Chrome's built-in PDF viewer behaves like a normal DOM page.
         */
        if (
            String(
                document.contentType ||
                ""
            ).toLowerCase() ===
            "application/pdf"
        ) {
            return true;
        }

        if (
            /\.pdf(?:$|[?#])/i.test(
                window.location.href
            )
        ) {
            return true;
        }

        /*
         * Conservative privacy boundary for pages containing
         * password fields.
         */
        if (
            document.querySelector(
                'input[type="password"]'
            )
        ) {
            return true;
        }

        return false;
    }

    function paragraphMetrics(
        root
    ) {
        if (
            !(root instanceof Element)
        ) {
            return {
                paragraphs: [],
                totalChars: 0,
                averageChars: 0,
            };
        }

        const paragraphs =
            Array
                .from(
                    root.querySelectorAll(
                        "p"
                    )
                )
                .filter(
                    (paragraph) => {
                        if (
                            paragraph.closest(
                                "nav, footer, aside, form, " +
                                "[role='navigation']"
                            )
                        ) {
                            return false;
                        }

                        if (
                            !isElementVisible(
                                paragraph
                            )
                        ) {
                            return false;
                        }

                        return (
                            normalizeWhitespace(
                                paragraph.textContent
                            ).length >=
                            100
                        );
                    }
                );

        const lengths =
            paragraphs.map(
                (paragraph) =>
                    normalizeWhitespace(
                        paragraph.textContent
                    ).length
            );

        const totalChars =
            lengths.reduce(
                (
                    sum,
                    length
                ) =>
                    sum +
                    length,
                0
            );

        return {
            paragraphs,

            totalChars,

            averageChars:
                paragraphs.length
                    ? totalChars /
                        paragraphs.length
                    : 0,
        };
    }

    function readingCandidateScore(
        root,
        kind
    ) {
        const metrics =
            paragraphMetrics(
                root
            );

        const interactiveCount =
            root.querySelectorAll(
                "textarea, " +
                "[contenteditable='true'], " +
                "form"
            ).length;

        const hasHeading =
            Boolean(
                root.querySelector(
                    "h1, h2"
                ) ||
                document.querySelector(
                    "h1"
                )
            );

        const strongReadingRoot =
            [
                "article",
                "article-body",
                "markdown",
                "documentation",
            ].includes(
                kind
            );

        const minimumParagraphs =
            strongReadingRoot
                ? 4
                : 6;

        const minimumChars =
            strongReadingRoot
                ? 1500
                : 2200;

        if (
            !hasHeading ||
            metrics.paragraphs.length <
                minimumParagraphs ||
            metrics.totalChars <
                minimumChars ||
            metrics.averageChars <
                125 ||
            interactiveCount >
                7
        ) {
            return null;
        }

        return {
            root,
            kind,
            ...metrics,

            score:
                metrics.totalChars +
                metrics.paragraphs.length *
                    (
                        strongReadingRoot
                            ? 280
                            : 220
                    ),
        };
    }

    function detectReadingSurfaceCandidate() {
        if (
            isUnsupportedPageForHelp()
        ) {
            return null;
        }

        const candidates = [];
        const seenRoots =
            new Set();

        const selectorGroups = [
            [
                "article",
                "article",
            ],

            [
                "[itemprop='articleBody']",
                "article-body",
            ],

            [
                ".article-body, " +
                ".post-content, " +
                ".entry-content",
                "article-body",
            ],

            [
                ".markdown-body, .prose",
                "markdown",
            ],

            [
                ".documentation, " +
                ".docs-content, " +
                ".theme-doc-markdown",
                "documentation",
            ],

            [
                "main, [role='main']",
                "main",
            ],
        ];

        selectorGroups.forEach(
            (
                [
                    selector,
                    kind,
                ]
            ) => {
                document
                    .querySelectorAll(
                        selector
                    )
                    .forEach(
                        (root) => {
                            if (
                                seenRoots.has(
                                    root
                                )
                            ) {
                                return;
                            }

                            seenRoots.add(
                                root
                            );

                            const candidate =
                                readingCandidateScore(
                                    root,
                                    kind
                                );

                            if (
                                candidate
                            ) {
                                candidates.push(
                                    candidate
                                );
                            }
                        }
                    );
            }
        );

        candidates.sort(
            (a, b) =>
                b.score -
                a.score
        );

        return (
            candidates[0] ||
            null
        );
    }

    // =========================================================
    // PAGE EXTRACTION — ONLY AFTER USER ACTION
    // =========================================================

    function extractReadingSurfaceText() {
        const root =
            pageCandidate?.root;

        if (
            !(root instanceof Element)
        ) {
            return null;
        }

        const nodes =
            Array.from(
                root.querySelectorAll(
                    "h1, h2, h3, " +
                    "p, blockquote, li"
                )
            );

        const lines = [];
        const seen =
            new Set();

        let totalLength = 0;

        for (
            const node
            of nodes
        ) {
            if (
                node.closest(
                    "nav, footer, aside, form, " +
                    "[role='navigation']"
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

            if (
                !text
            ) {
                continue;
            }

            if (
                node.matches(
                    "p, blockquote"
                ) &&
                text.length <
                    40
            ) {
                continue;
            }

            if (
                node.matches(
                    "li"
                ) &&
                text.length <
                    24
            ) {
                continue;
            }

            text =
                text.slice(
                    0,
                    900
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

            const nextLength =
                totalLength +
                text.length +
                2;

            if (
                nextLength >
                MAX_EXTRACTED_PAGE_CHARS
            ) {
                const remaining =
                    MAX_EXTRACTED_PAGE_CHARS -
                    totalLength;

                if (
                    remaining >
                    100
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

            totalLength =
                nextLength;
        }

        const combined =
            lines
                .join(
                    "\n\n"
                )
                .trim();

        if (
            combined.length <
            600
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
            title:
                heading,

            text:
                combined,
        };
    }

    // =========================================================
    // PAGE HELP OFFER
    // =========================================================

    function schedulePageSuggestion() {
        if (
            pageSuggestionTimer
        ) {
            clearTimeout(
                pageSuggestionTimer
            );

            pageSuggestionTimer =
                null;
        }

        if (
            pageState !==
            "inactive"
        ) {
            return;
        }

        /*
         * Wait seven seconds before even deciding whether to show
         * the ambient suggestion.
         */
        pageSuggestionTimer =
            window.setTimeout(
                () => {
                    pageSuggestionTimer =
                        null;

                    if (
                        pageState !==
                        "inactive"
                    ) {
                        return;
                    }

                    if (
                        document.visibilityState !==
                        "visible"
                    ) {
                        schedulePageSuggestion();
                        return;
                    }

                    const candidate =
                        detectReadingSurfaceCandidate();

                    if (
                        !candidate
                    ) {
                        return;
                    }

                    /*
                     * This stores the DOM root only.
                     *
                     * It does NOT extract the page body or send
                     * page text to Kelsie's backend.
                     */
                    pageCandidate =
                        candidate;

                    pageState =
                        "offered";

                    renderAmbientSurfaces();
                },
                PAGE_SUGGESTION_DELAY
            );
    }

    function resetPageSuggestionForNavigation() {
        if (
            pageSuggestionTimer
        ) {
            clearTimeout(
                pageSuggestionTimer
            );

            pageSuggestionTimer =
                null;
        }

        pageState =
            "inactive";

        pageCandidate =
            null;

        pageAssistData =
            null;

        pageAssistError =
            "";

        pageAssistAction =
            "summarize";

        pendingPageContext =
            null;

        renderAmbientSurfaces();

        schedulePageSuggestion();
    }

    function dismissPageHelp() {
        /*
         * Dismissal is only for the current page.
         */
        pageState =
            "dismissed";

        pageAssistData =
            null;

        pageAssistError =
            "";

        renderAmbientSurfaces();
    }

    // =========================================================
    // BACKGROUND MESSAGING
    // =========================================================

    function sendPageAssistMessage(
        payload
    ) {
        return new Promise(
            (
                resolve,
                reject
            ) => {
                chrome.runtime.sendMessage(
                    {
                        type:
                            "KELSIE_PAGE_ASSIST",

                        payload,
                    },
                    (result) => {
                        const runtimeError =
                            chrome.runtime
                                .lastError;

                        if (
                            runtimeError
                        ) {
                            console.error(
                                "[Kelsie] Extension background did not respond:",
                                runtimeError.message
                            );

                            reject(
                                new Error(
                                    "Kelsie's page helper did not respond. Reload the Kelsie extension and refresh this page."
                                )
                            );

                            return;
                        }

                        resolve(
                            result
                        );
                    }
                );
            }
        );
    }

    // =========================================================
    // SUMMARIZE / EXPLAIN
    // =========================================================

    async function requestPageAssist(
        action
    ) {
        if (
            pageState ===
            "loading"
        ) {
            return;
        }

        /*
         * This is the moment page text is actually extracted.
         *
         * The user has explicitly selected Summarize or
         * Help me understand.
         */
        const extracted =
            extractReadingSurfaceText();

        if (
            !extracted
        ) {
            pageAssistAction =
                action;

            pageState =
                "error";

            pageAssistError =
                "I couldn’t find enough readable text on this page.";

            renderAmbientSurfaces();

            return;
        }

        pageAssistAction =
            action;

        pageState =
            "loading";

        pageAssistData =
            null;

        pageAssistError =
            "";

        renderAmbientSurfaces();

        try {
            const response =
                await sendPageAssistMessage(
                    {
                        action,
                        ...extracted,
                    }
                );

            if (
                !response ||
                response.ok !==
                    true ||
                !response.data
            ) {
                throw new Error(
                    response?.error ||
                    "Kelsie could not help with this page right now."
                );
            }

            const answer =
                safeText(
                    response
                        .data
                        .answer ||
                    response
                        .data
                        .summary,
                    1600
                );

            const keyPoints =
                Array.isArray(
                    response
                        .data
                        .key_points
                )
                    ? response
                        .data
                        .key_points
                        .map(
                            (point) =>
                                safeText(
                                    point,
                                    360
                                )
                        )
                        .filter(
                            Boolean
                        )
                        .slice(
                            0,
                            3
                        )
                    : [];

            if (
                !answer
            ) {
                throw new Error(
                    "Kelsie returned an incomplete response."
                );
            }

            pageAssistData = {
                action,
                answer,
                keyPoints,
            };

            pageState =
                "result";
        } catch (error) {
            console.error(
                "[Kelsie] Page help failed:",
                error
            );

            pageAssistError =
                error instanceof Error &&
                error.message
                    ? error.message
                    : "Kelsie could not help with this page right now.";

            pageState =
                "error";
        }

        renderAmbientSurfaces();
    }

    // =========================================================
    // QUESTIONS
    // =========================================================

    function openPageQuestions() {
        const extracted =
            extractReadingSurfaceText();

        if (
            !extracted
        ) {
            pageAssistAction =
                "question";

            pageState =
                "error";

            pageAssistError =
                "I couldn’t find enough readable text on this page.";

            renderAmbientSurfaces();

            return;
        }

        /*
         * The page context is attached temporarily to the existing
         * Kelsie widget. widget-bridge.js handles questions while
         * this temporary context is active.
         */
        pageState =
            "dismissed";

        pageAssistData =
            null;

        pageAssistError =
            "";

        sendPageContextToWidget({
            title:
                extracted.title,

            text:
                extracted.text,
        });

        setPanelOpen(
            true
        );
    }

    // =========================================================
    // PAGE HELP RENDERING
    // =========================================================

    function renderPageCardContents() {
        pageOfferView.hidden =
            pageState !==
            "offered";

        pageChoiceView.hidden =
            pageState !==
            "choices";

        pageDetailView.hidden =
            ![
                "loading",
                "result",
                "error",
            ].includes(
                pageState
            );

        pageLoading.hidden =
            pageState !==
            "loading";

        pageResult.hidden =
            pageState !==
            "result";

        pageErrorView.hidden =
            pageState !==
            "error";

        if (
            pageState ===
            "loading"
        ) {
            pageDetailTitle.textContent =
                "Kelsie";

            pageDetailNote.textContent =
                "";
        }

        if (
            pageState ===
                "result" &&
            pageAssistData
        ) {
            const isExplain =
                pageAssistData
                    .action ===
                "explain";

            pageDetailTitle.textContent =
                isExplain
                    ? "Kelsie"
                    : "Kelsie summary";

            pageDetailNote.textContent =
                "Not saved to memory";

            pageAnswer.textContent =
                pageAssistData
                    .answer;

            pagePointsLabel.textContent =
                isExplain
                    ? "What matters"
                    : "Key points";

            pagePoints
                .replaceChildren();

            pageAssistData
                .keyPoints
                .forEach(
                    (point) => {
                        const item =
                            document
                                .createElement(
                                    "li"
                                );

                        item.className =
                            "kelsie-page-point";

                        item.textContent =
                            point;

                        pagePoints
                            .appendChild(
                                item
                            );
                    }
                );

            const hasPoints =
                pageAssistData
                    .keyPoints
                    .length >
                0;

            pagePointsLabel.hidden =
                !hasPoints;

            pagePoints.hidden =
                !hasPoints;
        }

        if (
            pageState ===
            "error"
        ) {
            pageDetailTitle.textContent =
                "Kelsie";

            pageDetailNote.textContent =
                "";

            pageError.textContent =
                pageAssistError ||
                "I couldn’t help with this page right now.";
        }
    }

    function renderPageCard() {
        renderPageCardContents();

        const pageHasVisibleState =
            [
                "offered",
                "choices",
                "loading",
                "result",
                "error",
            ].includes(
                pageState
            );

        /*
         * Reminders remain the highest priority ambient surface.
         */
        const shouldShow =
            pageHasVisibleState &&
            !currentReminder &&
            !panelOpen &&
            !host.hasAttribute(
                "data-dragging"
            );

        pageCard.hidden =
            !shouldShow;

        if (
            !shouldShow
        ) {
            return;
        }

        requestAnimationFrame(
            () => {
                positionFloatingSurface(
                    pageCard
                );
            }
        );
    }

    function renderAmbientSurfaces() {
        renderReminder();
        renderPageCard();
    }

    // =========================================================
    // PANEL STATE
    // =========================================================

    function render() {
        panelHost.hidden =
            !panelOpen;

        launcher.setAttribute(
            "aria-expanded",
            String(
                panelOpen
            )
        );

        launcher.setAttribute(
            "aria-label",
            panelOpen
                ? "Kelsie is open"
                : "Open Kelsie"
        );

        launcher.setAttribute(
            "title",
            panelOpen
                ? "Kelsie is open"
                : "Open Kelsie"
        );

        shell.classList.toggle(
            "panel-open",
            panelOpen
        );

        if (
            panelOpen
        ) {
            hideMoveHint();

            requestAnimationFrame(
                positionPanel
            );

            if (
                frameReady
            ) {
                sendWidgetVisibility(
                    true
                );
            }
        } else if (
            frameStarted
        ) {
            sendWidgetVisibility(
                false
            );
        }

        renderAmbientSurfaces();
    }

    function setPanelOpen(
        nextOpen
    ) {
        panelOpen =
            Boolean(
                nextOpen
            );

        if (
            !panelOpen
        ) {
            pendingPageContext =
                null;
        }

        render();
    }

    // =========================================================
    // INTERACTIONS
    // =========================================================

    launcher.addEventListener(
        "click",
        () => {
            if (
                suppressClickAfterDrag ||
                panelOpen
            ) {
                return;
            }

            setPanelOpen(
                true
            );
        }
    );

    reminderMain.addEventListener(
        "click",
        () => {
            setPanelOpen(
                true
            );
        }
    );

    reminderDone.addEventListener(
        "click",
        () => {
            sendReminderAction(
                "done"
            );
        }
    );

    reminderDismiss.addEventListener(
        "click",
        () => {
            sendReminderAction(
                "dismiss"
            );
        }
    );

    /*
     * First arrow:
     * Help with this page? →
     */
    pageOfferNext.addEventListener(
        "click",
        () => {
            if (
                pageState !==
                "offered"
            ) {
                return;
            }

            pageState =
                "choices";

            renderAmbientSurfaces();
        }
    );

    pageDismissButton.addEventListener(
        "click",
        dismissPageHelp
    );

    pageChoiceClose.addEventListener(
        "click",
        dismissPageHelp
    );

    /*
     * Summarize →
     */
    pageSummarizeButton.addEventListener(
        "click",
        () => {
            requestPageAssist(
                "summarize"
            );
        }
    );

    /*
     * Help me understand →
     */
    pageExplainButton.addEventListener(
        "click",
        () => {
            requestPageAssist(
                "explain"
            );
        }
    );

    /*
     * I have questions →
     */
    pageQuestionsButton.addEventListener(
        "click",
        openPageQuestions
    );

    pageDetailClose.addEventListener(
        "click",
        dismissPageHelp
    );

    /*
     * Result screen:
     * Ask about this →
     */
    pageAskButton.addEventListener(
        "click",
        openPageQuestions
    );

    pageBack.addEventListener(
        "click",
        () => {
            pageState =
                "choices";

            pageAssistError =
                "";

            renderAmbientSurfaces();
        }
    );

    pageRetry.addEventListener(
        "click",
        () => {
            requestPageAssist(
                pageAssistAction ===
                    "explain"
                    ? "explain"
                    : "summarize"
            );
        }
    );

    shadow.addEventListener(
        "keydown",
        (event) => {
            if (
                event.key ===
                    "Escape" &&
                panelOpen
            ) {
                setPanelOpen(
                    false
                );

                launcher.focus();
            }
        }
    );

    window.addEventListener(
        "resize",
        () => {
            applyPlacement();

            if (
                panelOpen
            ) {
                requestAnimationFrame(
                    positionPanel
                );
            }

            requestAnimationFrame(
                positionAmbientSurfaces
            );
        }
    );

    document.addEventListener(
        "visibilitychange",
        () => {
            if (
                document.visibilityState ===
                    "visible" &&
                pageState ===
                    "inactive"
            ) {
                schedulePageSuggestion();
            }
        }
    );

    /*
     * Handles sites that change routes without a full reload.
     */
    window.setInterval(
        () => {
            if (
                window.location.href ===
                observedPageUrl
            ) {
                return;
            }

            observedPageUrl =
                window.location.href;

            resetPageSuggestionForNavigation();
        },
        PAGE_URL_CHECK_INTERVAL
    );

    // =========================================================
    // START
    // =========================================================

    panelOpen =
        false;

    render();

    /*
     * Start Kelsie immediately in the background so reminder
     * polling still works while the visible panel is minimized.
     */
    startWidgetFrame();

    loadPlacement()
        .then(
            () => {
                window.setTimeout(
                    showMoveHintOnce,
                    550
                );
            }
        );

    /*
     * Page help does NOT appear immediately.
     *
     * Kelsie waits seven seconds, checks whether this actually
     * looks like a useful reading surface, and only then offers
     * "Help with this page?".
     *
     * The page body still isn't extracted until the user selects
     * one of the actual help actions.
     */
    schedulePageSuggestion();

    console.log(
        "[Kelsie] Extension mounted with ambient reminders and Level 1 page help."
    );
})();