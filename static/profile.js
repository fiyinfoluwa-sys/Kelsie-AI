import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const INTRO_MODEL_URL = new URL(
    "./kelsie-avatar.glb",
    import.meta.url
).href;

const PROFILE_API = "/api/profile";

const INTRO_PREVIEW_ONLY =
    new URLSearchParams(window.location.search).get(
        "kelsie_intro_preview"
    ) === "1";

export const TIMEZONE_OPTIONS = Object.freeze([
    {
        group: "North America",
        value: "America/St_Johns",
        label: "Newfoundland Time (NST/NDT) — St. John's",
    },
    {
        group: "North America",
        value: "America/Halifax",
        label: "Atlantic Time (AST/ADT) — Halifax",
    },
    {
        group: "North America",
        value: "America/Toronto",
        label: "Eastern Time (EST/EDT) — Toronto",
    },
    {
        group: "North America",
        value: "America/New_York",
        label: "Eastern Time (EST/EDT) — New York",
    },
    {
        group: "North America",
        value: "America/Chicago",
        label: "Central Time (CST/CDT) — Chicago",
    },
    {
        group: "North America",
        value: "America/Winnipeg",
        label: "Central Time (CST/CDT) — Winnipeg",
    },
    {
        group: "North America",
        value: "America/Denver",
        label: "Mountain Time (MST/MDT) — Denver",
    },
    {
        group: "North America",
        value: "America/Edmonton",
        label: "Mountain Time (MST/MDT) — Edmonton",
    },
    {
        group: "North America",
        value: "America/Phoenix",
        label: "Mountain Standard Time (MST) — Phoenix",
    },
    {
        group: "North America",
        value: "America/Los_Angeles",
        label: "Pacific Time (PST/PDT) — Los Angeles",
    },
    {
        group: "North America",
        value: "America/Vancouver",
        label: "Pacific Time (PST/PDT) — Vancouver",
    },
    {
        group: "North America",
        value: "America/Anchorage",
        label: "Alaska Time (AKST/AKDT) — Anchorage",
    },
    {
        group: "North America",
        value: "Pacific/Honolulu",
        label: "Hawaii Time (HST) — Honolulu",
    },
    {
        group: "North America",
        value: "America/Mexico_City",
        label: "Central Mexico Time — Mexico City",
    },

    {
        group: "UTC & United Kingdom",
        value: "UTC",
        label: "Coordinated Universal Time (UTC)",
    },
    {
        group: "UTC & United Kingdom",
        value: "Europe/London",
        label: "United Kingdom (GMT/BST) — London",
    },
    {
        group: "UTC & United Kingdom",
        value: "Europe/Dublin",
        label: "Ireland (GMT/IST) — Dublin",
    },

    {
        group: "Europe",
        value: "Europe/Lisbon",
        label: "Western European Time — Lisbon",
    },
    {
        group: "Europe",
        value: "Europe/Paris",
        label: "Central European Time (CET/CEST) — Paris",
    },
    {
        group: "Europe",
        value: "Europe/Berlin",
        label: "Central European Time (CET/CEST) — Berlin",
    },
    {
        group: "Europe",
        value: "Europe/Rome",
        label: "Central European Time (CET/CEST) — Rome",
    },
    {
        group: "Europe",
        value: "Europe/Athens",
        label: "Eastern European Time (EET/EEST) — Athens",
    },
    {
        group: "Europe",
        value: "Europe/Helsinki",
        label: "Eastern European Time (EET/EEST) — Helsinki",
    },

    {
        group: "Africa",
        value: "Africa/Accra",
        label: "Greenwich Mean Time (GMT) — Accra",
    },
    {
        group: "Africa",
        value: "Africa/Lagos",
        label: "West Africa Time (WAT, GMT+1) — Lagos",
    },
    {
        group: "Africa",
        value: "Africa/Johannesburg",
        label: "South Africa Standard Time (SAST) — Johannesburg",
    },
    {
        group: "Africa",
        value: "Africa/Cairo",
        label: "Eastern European Time — Cairo",
    },
    {
        group: "Africa",
        value: "Africa/Nairobi",
        label: "East Africa Time (EAT, GMT+3) — Nairobi",
    },

    {
        group: "Middle East & Asia",
        value: "Asia/Dubai",
        label: "Gulf Standard Time (GST, GMT+4) — Dubai",
    },
    {
        group: "Middle East & Asia",
        value: "Asia/Karachi",
        label: "Pakistan Standard Time (PKT, GMT+5) — Karachi",
    },
    {
        group: "Middle East & Asia",
        value: "Asia/Kolkata",
        label: "India Standard Time (IST, GMT+5:30) — India",
    },
    {
        group: "Middle East & Asia",
        value: "Asia/Dhaka",
        label: "Bangladesh Standard Time (GMT+6) — Dhaka",
    },
    {
        group: "Middle East & Asia",
        value: "Asia/Bangkok",
        label: "Indochina Time (ICT, GMT+7) — Bangkok",
    },
    {
        group: "Middle East & Asia",
        value: "Asia/Singapore",
        label: "Singapore Time (SGT, GMT+8) — Singapore",
    },
    {
        group: "Middle East & Asia",
        value: "Asia/Shanghai",
        label: "China Standard Time (CST, GMT+8) — Shanghai",
    },
    {
        group: "Middle East & Asia",
        value: "Asia/Hong_Kong",
        label: "Hong Kong Time (HKT, GMT+8) — Hong Kong",
    },
    {
        group: "Middle East & Asia",
        value: "Asia/Tokyo",
        label: "Japan Standard Time (JST, GMT+9) — Tokyo",
    },
    {
        group: "Middle East & Asia",
        value: "Asia/Seoul",
        label: "Korea Standard Time (KST, GMT+9) — Seoul",
    },

    {
        group: "Australia & Pacific",
        value: "Australia/Perth",
        label: "Australian Western Time (AWST) — Perth",
    },
    {
        group: "Australia & Pacific",
        value: "Australia/Adelaide",
        label: "Australian Central Time (ACST/ACDT) — Adelaide",
    },
    {
        group: "Australia & Pacific",
        value: "Australia/Sydney",
        label: "Australian Eastern Time (AEST/AEDT) — Sydney",
    },
    {
        group: "Australia & Pacific",
        value: "Pacific/Auckland",
        label: "New Zealand Time (NZST/NZDT) — Auckland",
    },

    {
        group: "South America",
        value: "America/Sao_Paulo",
        label: "Brasília Time (BRT) — São Paulo",
    },
    {
        group: "South America",
        value: "America/Buenos_Aires",
        label: "Argentina Time (ART) — Buenos Aires",
    },
]);

const DEVICE_TIMEZONE =
    Intl.DateTimeFormat().resolvedOptions().timeZone ||
    "America/Toronto";

const DEVICE_REDUCES_MOTION =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    ).matches;

const DEFAULT_PROFILE = Object.freeze({
    timezone: DEVICE_TIMEZONE,

    proactivity: "balanced",
    proactivity_level: "balanced",

    initial_context: "",

    quiet_hours_enabled: true,
    quiet_hours_start: "23:00",
    quiet_hours_end: "08:00",

    accessibility_large_text: false,
    accessibility_high_contrast: false,
    accessibility_reduce_motion:
        DEVICE_REDUCES_MOTION,
    accessibility_simplified_language:
        false,

    memory_enabled: true,
    adaptive_tone: true,

    mode: "both",
});

/* =========================================================
   USER ID
========================================================= */

export function getOrCreateUserId() {
    const storageKey =
        "kelsie_user_id";

    let userId =
        window.localStorage.getItem(
            storageKey
        );

    if (userId) {
        return userId;
    }

    if (
        window.crypto &&
        typeof window.crypto.randomUUID ===
            "function"
    ) {
        userId =
            window.crypto.randomUUID();
    } else {
        userId =
            `kelsie-${Date.now()}-${Math.random()
                .toString(16)
                .slice(2)}`;
    }

    window.localStorage.setItem(
        storageKey,
        userId
    );

    return userId;
}

/* =========================================================
   API
========================================================= */

async function requestJson(
    url,
    options = {}
) {
    const response =
        await fetch(url, {
            ...options,

            headers: {
                "Content-Type":
                    "application/json",

                ...(options.headers ||
                    {}),
            },
        });

    const data =
        await response
            .json()
            .catch(() => ({}));

    if (!response.ok) {
        throw new Error(
            data.detail ||
                "The request could not be completed."
        );
    }

    return data;
}

function unwrapProfile(result) {
    if (!result) {
        return null;
    }

    if (
        typeof result ===
            "object" &&
        result.profile &&
        typeof result.profile ===
            "object"
    ) {
        return result.profile;
    }

    return result;
}

async function fetchProfile(
    userId
) {
    return unwrapProfile(
        await requestJson(
            `${PROFILE_API}/${encodeURIComponent(
                userId
            )}`
        )
    );
}

async function createProfile(
    profile
) {
    return unwrapProfile(
        await requestJson(
            PROFILE_API,
            {
                method: "POST",

                body:
                    JSON.stringify(
                        profile
                    ),
            }
        )
    );
}

async function seedInitialContext(
    userId,
    text
) {
    const cleaned =
        String(text || "").trim();

    if (!cleaned) {
        return null;
    }

    try {
        return await requestJson(
            `/api/memory/${encodeURIComponent(
                userId
            )}/seed`,
            {
                method: "POST",

                body:
                    JSON.stringify({
                        text:
                            cleaned,
                    }),
            }
        );
    } catch (error) {
        console.warn(
            "Kelsie saved the profile, but could not seed memory yet:",
            error
        );

        return null;
    }
}

/* =========================================================
   TIMEZONES
========================================================= */

export function timezoneLabel(
    value
) {
    const match =
        TIMEZONE_OPTIONS.find(
            (option) =>
                option.value ===
                value
        );

    return match
        ? match.label
        : String(value || "");
}

export function populateTimezoneSelect(
    select,
    selectedValue
) {
    if (!select) {
        return;
    }

    const value =
        String(
            selectedValue ||
                DEVICE_TIMEZONE ||
                "America/Toronto"
        );

    select.innerHTML = "";

    const knownValues =
        new Set(
            TIMEZONE_OPTIONS.map(
                (option) =>
                    option.value
            )
        );

    if (
        value &&
        !knownValues.has(value)
    ) {
        const detectedGroup =
            document.createElement(
                "optgroup"
            );

        detectedGroup.label =
            "Detected timezone";

        const detectedOption =
            document.createElement(
                "option"
            );

        detectedOption.value =
            value;

        detectedOption.textContent =
            value;

        detectedGroup.appendChild(
            detectedOption
        );

        select.appendChild(
            detectedGroup
        );
    }

    const groups =
        new Map();

    for (
        const option
        of TIMEZONE_OPTIONS
    ) {
        if (
            !groups.has(
                option.group
            )
        ) {
            const group =
                document.createElement(
                    "optgroup"
                );

            group.label =
                option.group;

            groups.set(
                option.group,
                group
            );

            select.appendChild(
                group
            );
        }

        const element =
            document.createElement(
                "option"
            );

        element.value =
            option.value;

        element.textContent =
            option.label;

        groups
            .get(option.group)
            .appendChild(
                element
            );
    }

    select.value =
        value;

    if (!select.value) {
        select.value =
            "America/Toronto";
    }
}

/* =========================================================
   ONBOARDING STYLES
========================================================= */

function injectOnboardingStyles() {
    if (
        document.getElementById(
            "kelsie-onboarding-styles"
        )
    ) {
        return;
    }

    const style =
        document.createElement(
            "style"
        );

    style.id =
        "kelsie-onboarding-styles";

    style.textContent = `
        #kelsie-onboarding {
            position: absolute;
            inset: 0;
            z-index: 80;

            display: none;

            align-items: stretch;
            justify-content: stretch;

            overflow: hidden;

            border-radius: inherit;

            background:
                linear-gradient(
                    145deg,
                    rgba(
                        255,
                        255,
                        255,
                        0.985
                    ),
                    rgba(
                        243,
                        245,
                        248,
                        0.97
                    )
                );

            backdrop-filter:
                blur(24px);

            -webkit-backdrop-filter:
                blur(24px);
        }

        #kelsie-onboarding.visible {
            display: flex;
        }

        .onboarding-shell {
            width: 100%;
            min-height: 100%;

            display: flex;
            flex-direction: column;

            padding:
                17px
                17px
                15px;

            color: #252a35;
        }

        /* =================================================
           INTRO
        ================================================= */

        .onboarding-intro {
            align-items: center;
            justify-content: center;

            text-align: center;

            padding:
                4px
                8px
                0;
        }

        .onboarding-intro-character {
            position: relative;

            width: 176px;
            height: 178px;

            margin:
                -2px
                auto
                7px;
        }

        #kelsie-intro-avatar {
            position: absolute;

            inset: 0;

            width: 100%;
            height: 100%;

            display: block;

            border: 0;

            outline: 0;

            background:
                transparent;

            pointer-events:
                none;
        }

        .onboarding-intro-shadow {
            position: absolute;

            left: 50%;
            bottom: 17px;

            width: 76px;
            height: 11px;

            transform:
                translateX(-50%);

            border-radius: 50%;

            background:
                rgba(
                    70,
                    76,
                    91,
                    0.11
                );

            filter:
                blur(6px);

            pointer-events:
                none;
        }

        .onboarding-intro
        .onboarding-title {
            font-size: 25px;
        }

        .onboarding-intro-copy {
            max-width: 272px;

            margin:
                9px
                auto
                19px;

            color: #707887;

            font-size: 11px;

            line-height: 1.58;
        }

        .onboarding-intro
        .onboarding-button {
            min-width: 116px;
        }

        /* =================================================
           PROGRESS
        ================================================= */

        .onboarding-progress-row {
            display: flex;

            align-items: center;

            justify-content:
                space-between;

            gap: 10px;

            margin-bottom:
                13px;
        }

        .onboarding-progress-copy {
            color: #848b99;

            font-size: 9px;

            font-weight: 650;

            letter-spacing:
                0.06em;

            text-transform:
                uppercase;
        }

        .onboarding-progress-track {
            flex: 1;

            height: 3px;

            overflow: hidden;

            border-radius: 99px;

            background:
                rgba(
                    128,
                    136,
                    153,
                    0.13
                );
        }

        .onboarding-progress-fill {
            width: 16.6667%;

            height: 100%;

            border-radius:
                inherit;

            background:
                #808899;

            transition:
                width
                0.24s
                ease;
        }

        /* =================================================
           STEPS
        ================================================= */

        .onboarding-step {
            display: none;

            flex: 1;

            min-height: 0;

            overflow-y: auto;

            padding:
                1px
                1px
                0;

            scrollbar-width:
                thin;

            scrollbar-color:
                rgba(
                    128,
                    136,
                    153,
                    0.24
                )
                transparent;
        }

        .onboarding-step.active {
            display: flex;

            flex-direction:
                column;
        }

        .onboarding-eyebrow {
            margin-bottom: 7px;

            color: #808899;

            font-size: 9px;

            font-weight: 700;

            letter-spacing:
                0.08em;

            text-transform:
                uppercase;
        }

        .onboarding-title {
            margin: 0;

            color: #252a35;

            font-size: 22px;

            line-height: 1.14;

            letter-spacing:
                -0.035em;
        }

        .onboarding-description {
            margin:
                8px
                0
                17px;

            color: #707887;

            font-size: 11px;

            line-height: 1.5;
        }

        /* =================================================
           FIELDS
        ================================================= */

        .onboarding-field {
            display: grid;

            gap: 7px;
        }

        .onboarding-field label,
        .onboarding-field-label {
            color: #505866;

            font-size: 10px;

            font-weight: 650;
        }

        .onboarding-input,
        .onboarding-select,
        .onboarding-time,
        .onboarding-textarea {
            width: 100%;

            border:
                1px
                solid
                rgba(
                    54,
                    61,
                    75,
                    0.10
                );

            border-radius:
                14px;

            outline: none;

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.78
                );

            color: #272d39;

            box-shadow:
                inset
                0
                1px
                0
                rgba(
                    255,
                    255,
                    255,
                    0.82
                );

            font: inherit;

            font-size:
                11px;

            transition:
                border-color
                0.2s
                ease,

                box-shadow
                0.2s
                ease;
        }

        .onboarding-input,
        .onboarding-select,
        .onboarding-time {
            height: 42px;

            padding:
                0
                12px;
        }

        .onboarding-textarea {
            min-height: 105px;

            resize: vertical;

            padding:
                11px
                12px;

            line-height: 1.5;
        }

        .onboarding-input:focus,
        .onboarding-select:focus,
        .onboarding-time:focus,
        .onboarding-textarea:focus {
            border-color:
                rgba(
                    128,
                    136,
                    153,
                    0.55
                );

            box-shadow:
                0
                0
                0
                4px
                rgba(
                    128,
                    136,
                    153,
                    0.10
                );
        }

        /* =================================================
           TIMEZONE
        ================================================= */

        .onboarding-detected {
            display: flex;

            align-items: center;

            gap: 6px;

            margin:
                0
                0
                9px;

            color: #697180;

            font-size: 9.5px;
        }

        .onboarding-detected-dot {
            width: 6px;
            height: 6px;

            border-radius: 50%;

            background:
                #808899;
        }

        /* =================================================
           CHOICES
        ================================================= */

        .onboarding-choice-list {
            display: grid;

            gap: 8px;
        }

        .onboarding-choice {
            width: 100%;

            display: grid;

            gap: 4px;

            padding:
                11px
                12px;

            text-align: left;

            border:
                1px
                solid
                rgba(
                    54,
                    61,
                    75,
                    0.09
                );

            border-radius: 15px;

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.68
                );

            color: #2e3440;

            cursor: pointer;

            transition:
                transform
                0.18s
                ease,

                border-color
                0.18s
                ease,

                background
                0.18s
                ease,

                box-shadow
                0.18s
                ease;
        }

        .onboarding-choice:hover {
            transform:
                translateY(-1px);

            border-color:
                rgba(
                    128,
                    136,
                    153,
                    0.34
                );
        }

        .onboarding-choice.selected {
            border-color:
                rgba(
                    128,
                    136,
                    153,
                    0.64
                );

            background:
                rgba(
                    246,
                    247,
                    249,
                    0.96
                );

            box-shadow:
                0
                7px
                20px
                rgba(
                    62,
                    68,
                    82,
                    0.07
                );
        }

        .onboarding-choice-title {
            display: flex;

            align-items: center;

            gap: 6px;

            font-size: 11px;

            font-weight: 680;
        }

        .onboarding-recommended {
            padding:
                2px
                6px;

            border-radius: 99px;

            background:
                rgba(
                    128,
                    136,
                    153,
                    0.12
                );

            color: #727a89;

            font-size: 7.5px;

            font-weight: 700;

            letter-spacing:
                0.04em;

            text-transform:
                uppercase;
        }

        .onboarding-choice-description {
            color: #717988;

            font-size: 9.5px;

            line-height: 1.42;
        }

        .onboarding-choice-best {
            color: #8a919e;

            font-size: 8.7px;

            line-height: 1.35;
        }

        /* =================================================
           CONTEXT CHIPS
        ================================================= */

        .onboarding-helper-chips {
            display: flex;

            flex-wrap: wrap;

            gap: 6px;

            margin:
                0
                0
                9px;
        }

        .onboarding-helper-chip {
            padding:
                6px
                8px;

            border:
                1px
                solid
                rgba(
                    61,
                    68,
                    82,
                    0.08
                );

            border-radius:
                99px;

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.68
                );

            color: #68717f;

            font-size: 8.5px;

            cursor: pointer;
        }

        .onboarding-helper-chip:hover {
            border-color:
                rgba(
                    128,
                    136,
                    153,
                    0.34
                );

            color: #343b48;
        }

        .onboarding-note {
            margin-top: 8px;

            color: #8a919e;

            font-size: 8.8px;

            line-height: 1.4;
        }

        /* =================================================
           TOGGLES
        ================================================= */

        .onboarding-toggle-card {
            display: flex;

            align-items:
                flex-start;

            justify-content:
                space-between;

            gap: 12px;

            padding:
                11px
                12px;

            border:
                1px
                solid
                rgba(
                    54,
                    61,
                    75,
                    0.09
                );

            border-radius:
                15px;

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.68
                );
        }

        .onboarding-toggle-copy {
            display: grid;

            gap: 3px;
        }

        .onboarding-toggle-title {
            color: #343b47;

            font-size: 10.5px;

            font-weight: 670;
        }

        .onboarding-toggle-description {
            color: #7b8391;

            font-size: 9px;

            line-height: 1.38;
        }

        .onboarding-checkbox {
            width: 18px;
            height: 18px;

            margin: 0;

            accent-color:
                #808899;

            cursor: pointer;
        }

        .onboarding-time-grid {
            display: grid;

            grid-template-columns:
                1fr
                1fr;

            gap: 9px;

            margin-top: 10px;
        }

        /* =================================================
           ACCESSIBILITY
        ================================================= */

        .onboarding-access-list {
            display: grid;

            gap: 7px;
        }

        .onboarding-access-option {
            display: flex;

            align-items:
                flex-start;

            gap: 9px;

            padding:
                9px
                10px;

            border:
                1px
                solid
                rgba(
                    54,
                    61,
                    75,
                    0.075
                );

            border-radius:
                13px;

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.62
                );
        }

        .onboarding-access-copy {
            display: grid;

            gap: 2px;
        }

        .onboarding-access-copy strong {
            color: #3c4350;

            font-size: 10px;

            font-weight: 660;
        }

        .onboarding-access-copy span {
            color: #7e8694;

            font-size: 8.8px;

            line-height: 1.35;
        }

        .onboarding-device-note {
            color:
                #808899
                !important;

            font-weight: 650;
        }

        /* =================================================
           BUTTONS
        ================================================= */

        .onboarding-actions {
            display: flex;

            align-items: center;

            justify-content:
                space-between;

            gap: 8px;

            margin-top: auto;

            padding-top: 14px;
        }

        .onboarding-actions-right {
            display: flex;

            align-items: center;

            gap: 7px;

            margin-left: auto;
        }

        .onboarding-button {
            min-height: 38px;

            padding:
                0
                14px;

            border: 0;

            border-radius:
                20px;

            font-size: 10px;

            font-weight: 680;

            cursor: pointer;

            transition:
                transform
                0.18s
                ease,

                opacity
                0.18s
                ease;
        }

        .onboarding-button:hover:not(
            :disabled
        ) {
            transform:
                translateY(-1px);
        }

        .onboarding-button.primary {
            color: white;

            background:
                linear-gradient(
                    145deg,
                    #808899,
                    #5f6677
                );

            box-shadow:
                0
                7px
                18px
                rgba(
                    75,
                    82,
                    98,
                    0.18
                );
        }

        .onboarding-button.secondary {
            color: #68717f;

            background:
                rgba(
                    54,
                    61,
                    75,
                    0.055
                );
        }

        .onboarding-button.text {
            min-height: 34px;

            padding:
                0
                8px;

            color: #828a98;

            background:
                transparent;
        }

        .onboarding-button:disabled {
            opacity: 0.45;

            cursor: default;

            transform: none;
        }

        .onboarding-error {
            min-height: 15px;

            margin-top: 8px;

            color: #bd5966;

            font-size: 8.8px;

            line-height: 1.35;
        }

        /* =================================================
           COMPLETE
        ================================================= */

        .onboarding-complete {
            align-items: center;

            justify-content:
                center;

            text-align:
                center;
        }

        .onboarding-complete
        .mini-orb {
            width: 44px;
            height: 44px;

            margin-bottom:
                14px;
        }

        .onboarding-complete
        .onboarding-description {
            max-width: 245px;
        }

        /* =================================================
           ACCESSIBILITY CLASSES
        ================================================= */

        body.kelsie-large-text
        .onboarding-title {
            font-size: 25px;
        }

        body.kelsie-large-text
        .onboarding-description,

        body.kelsie-large-text
        .onboarding-choice-copy,

        body.kelsie-large-text
        .onboarding-choice-best,

        body.kelsie-large-text
        .onboarding-note,

        body.kelsie-large-text
        .onboarding-field label,

        body.kelsie-large-text
        .onboarding-input,

        body.kelsie-large-text
        .onboarding-select,

        body.kelsie-large-text
        .onboarding-textarea,

        body.kelsie-large-text
        .onboarding-button {
            font-size: 13px;
        }

        body.kelsie-high-contrast
        .onboarding-shell {
            color: #11151d;
        }

        body.kelsie-high-contrast
        .onboarding-title,

        body.kelsie-high-contrast
        .onboarding-choice-title,

        body.kelsie-high-contrast
        .onboarding-toggle-title {
            color: #11151d;
        }

        body.kelsie-high-contrast
        .onboarding-description,

        body.kelsie-high-contrast
        .onboarding-note,

        body.kelsie-high-contrast
        .onboarding-choice-best {
            color: #3f4856;
        }

        body.kelsie-high-contrast
        .onboarding-choice,

        body.kelsie-high-contrast
        .onboarding-toggle-card,

        body.kelsie-high-contrast
        .onboarding-input,

        body.kelsie-high-contrast
        .onboarding-select,

        body.kelsie-high-contrast
        .onboarding-textarea {
            border-color:
                rgba(
                    17,
                    21,
                    29,
                    0.28
                );
        }

        body.kelsie-reduce-motion
        .onboarding-progress-fill,

        body.kelsie-reduce-motion
        .onboarding-choice,

        body.kelsie-reduce-motion
        .onboarding-button {
            animation:
                none
                !important;

            transition:
                none
                !important;
        }

        @media (
            prefers-reduced-motion:
                reduce
        ) {
            .onboarding-progress-fill,
            .onboarding-choice,
            .onboarding-button {
                transition:
                    none
                    !important;
            }
        }
    `;

    document.head.appendChild(
        style
    );
}

/* =========================================================
   ONBOARDING HTML
========================================================= */

function createOnboarding(
    chatCard
) {
    const onboarding =
        document.createElement(
            "div"
        );

    onboarding.id =
        "kelsie-onboarding";

    onboarding.setAttribute(
        "aria-hidden",
        "true"
    );

    onboarding.innerHTML = `
        <div class="onboarding-shell">

            <!-- ============================================
                 INTRO — NOT COUNTED AS A PROFILE STEP
            ============================================= -->

            <section
                class="onboarding-step onboarding-intro active"
                data-profile-step="0"
            >
                <div
                    class="onboarding-intro-character"
                    aria-hidden="true"
                >
                    <div
                        class="onboarding-intro-shadow"
                    ></div>

                    <canvas
                        id="kelsie-intro-avatar"
                    ></canvas>
                </div>

                <div class="onboarding-eyebrow">
                    Your ambient AI
                </div>

                <h2 class="onboarding-title">
                    Meet Kelsie.
                </h2>

                <p class="onboarding-intro-copy">
                    A personal AI that remembers useful
                    context, keeps track of things you
                    mention, and brings them back when
                    they matter.
                </p>

                <button
                    id="onboarding-intro-start"
                    class="onboarding-button primary"
                    type="button"
                >
                    Get started
                </button>
            </section>

            <!-- ============================================
                 PROFILE PROGRESS
            ============================================= -->

            <div
                class="onboarding-progress-row"
                data-onboarding-progress-row
            >
                <span
                    class="onboarding-progress-copy"
                    id="onboarding-progress-copy"
                >
                    1 of 6
                </span>

                <div
                    class="onboarding-progress-track"
                    aria-hidden="true"
                >
                    <div
                        class="onboarding-progress-fill"
                        id="onboarding-progress-fill"
                    ></div>
                </div>
            </div>

            <!-- ============================================
                 STEP 1 — NAME
            ============================================= -->

            <section
                class="onboarding-step"
                data-profile-step="1"
            >
                <div class="onboarding-eyebrow">
                    Start with you
                </div>

                <h2 class="onboarding-title">
                    What should Kelsie call you?
                </h2>

                <p class="onboarding-description">
                    This is how Kelsie will know you.
                    She does not need to use your name
                    in every conversation.
                </p>

                <div class="onboarding-field">
                    <label
                        for="profile-onboarding-name"
                    >
                        Your name
                    </label>

                    <input
                        id="profile-onboarding-name"
                        class="onboarding-input"
                        type="text"
                        maxlength="80"
                        autocomplete="name"
                        placeholder="Enter your name"
                    />
                </div>

                <div
                    id="onboarding-name-error"
                    class="onboarding-error"
                    role="status"
                ></div>

                <div class="onboarding-actions">
                    <span></span>

                    <button
                        id="onboarding-name-next"
                        class="onboarding-button primary"
                        type="button"
                    >
                        Continue
                    </button>
                </div>
            </section>

            <!-- ============================================
                 STEP 2 — TIMEZONE
            ============================================= -->

            <section
                class="onboarding-step"
                data-profile-step="2"
            >
                <div class="onboarding-eyebrow">
                    Your time
                </div>

                <h2 class="onboarding-title">
                    What timezone are you in?
                </h2>

                <p class="onboarding-description">
                    Kelsie uses this to understand things
                    like “tomorrow,” “tonight,” and
                    “in two hours,” and to make reminders
                    happen when you expect them.
                </p>

                <div class="onboarding-detected">
                    <span
                        class="onboarding-detected-dot"
                    ></span>

                    <span
                        id="onboarding-detected-timezone"
                    ></span>
                </div>

                <div class="onboarding-field">
                    <label
                        for="profile-onboarding-timezone"
                    >
                        Timezone
                    </label>

                    <select
                        id="profile-onboarding-timezone"
                        class="onboarding-select"
                    ></select>
                </div>

                <p class="onboarding-note">
                    Familiar labels such as EST/EDT,
                    PST/PDT, GMT, WAT, IST, JST and
                    others are shown, while Kelsie saves
                    the underlying timezone so
                    daylight-saving changes stay accurate.
                </p>

                <div class="onboarding-actions">
                    <button
                        class="onboarding-button secondary"
                        data-back-to="1"
                        type="button"
                    >
                        Back
                    </button>

                    <button
                        id="onboarding-timezone-next"
                        class="onboarding-button primary"
                        type="button"
                    >
                        Continue
                    </button>
                </div>
            </section>

            <!-- ============================================
                 STEP 3 — PROACTIVITY
            ============================================= -->

            <section
                class="onboarding-step"
                data-profile-step="3"
            >
                <div class="onboarding-eyebrow">
                    Kelsie's involvement
                </div>

                <h2 class="onboarding-title">
                    How involved should Kelsie be?
                </h2>

                <p class="onboarding-description">
                    Kelsie can simply respond when you
                    need her, or occasionally take the
                    initiative when something seems worth
                    bringing back up. You can change this
                    anytime.
                </p>

                <div
                    class="onboarding-choice-list"
                    role="radiogroup"
                    aria-label="Kelsie involvement"
                >
                    <button
                        class="onboarding-choice"
                        type="button"
                        data-proactivity="low"
                        role="radio"
                        aria-checked="false"
                    >
                        <span
                            class="onboarding-choice-title"
                        >
                            Only when I ask
                        </span>

                        <span
                            class="onboarding-choice-description"
                        >
                            Kelsie stays mostly in the
                            background. She remembers
                            useful context, but generally
                            waits for you to start.
                        </span>

                        <span
                            class="onboarding-choice-best"
                        >
                            Best if you want a quiet
                            assistant that does not
                            interrupt.
                        </span>
                    </button>

                    <button
                        class="onboarding-choice selected"
                        type="button"
                        data-proactivity="balanced"
                        role="radio"
                        aria-checked="true"
                    >
                        <span
                            class="onboarding-choice-title"
                        >
                            Balanced

                            <span
                                class="onboarding-recommended"
                            >
                                Recommended
                            </span>
                        </span>

                        <span
                            class="onboarding-choice-description"
                        >
                            Kelsie can follow up or bring
                            something back when it seems
                            genuinely useful, without
                            constantly checking in.
                        </span>

                        <span
                            class="onboarding-choice-best"
                        >
                            Best if you want help when it
                            matters without too much
                            interruption.
                        </span>
                    </button>

                    <button
                        class="onboarding-choice"
                        type="button"
                        data-proactivity="high"
                        role="radio"
                        aria-checked="false"
                    >
                        <span
                            class="onboarding-choice-title"
                        >
                            More proactive
                        </span>

                        <span
                            class="onboarding-choice-description"
                        >
                            Kelsie can check in about
                            ongoing situations, unfinished
                            things, reminders, and things
                            you have said matter to you
                            more often.
                        </span>

                        <span
                            class="onboarding-choice-best"
                        >
                            Best if you want Kelsie to
                            take more initiative.
                        </span>
                    </button>
                </div>

                <div class="onboarding-actions">
                    <button
                        class="onboarding-button secondary"
                        data-back-to="2"
                        type="button"
                    >
                        Back
                    </button>

                    <button
                        id="onboarding-proactivity-next"
                        class="onboarding-button primary"
                        type="button"
                    >
                        Continue
                    </button>
                </div>
            </section>

            <!-- ============================================
                 STEP 4 — CONTEXT
            ============================================= -->

            <section
                class="onboarding-step"
                data-profile-step="4"
            >
                <div class="onboarding-eyebrow">
                    A little context
                </div>

                <h2 class="onboarding-title">
                    Give Kelsie a little context
                </h2>

                <p class="onboarding-description">
                    Share anything that would make it
                    easier to understand your life right
                    now: something you are working toward,
                    an ongoing project, people you mention
                    often, or how you like to be helped.
                </p>

                <div
                    class="onboarding-helper-chips"
                    aria-label="Context examples"
                >
                    <button
                        class="onboarding-helper-chip"
                        type="button"
                        data-context-example="I'm currently working toward "
                    >
                        Something I'm working toward
                    </button>

                    <button
                        class="onboarding-helper-chip"
                        type="button"
                        data-context-example="I'm working on a project called "
                    >
                        A project I'm working on
                    </button>

                    <button
                        class="onboarding-helper-chip"
                        type="button"
                        data-context-example="Someone I mention often is "
                    >
                        People important to me
                    </button>

                    <button
                        class="onboarding-helper-chip"
                        type="button"
                        data-context-example="I usually like Kelsie to "
                    >
                        How I like to be helped
                    </button>
                </div>

                <div class="onboarding-field">
                    <label
                        for="profile-onboarding-context"
                    >
                        Anything you want Kelsie to know
                    </label>

                    <textarea
                        id="profile-onboarding-context"
                        class="onboarding-textarea"
                        maxlength="1800"
                        placeholder="For example: I'm applying for internships, building a side project, and I usually prefer direct answers."
                    ></textarea>
                </div>

                <p class="onboarding-note">
                    Optional. You do not need to explain
                    everything now. Kelsie will continue
                    learning naturally as you talk.
                </p>

                <div class="onboarding-actions">
                    <button
                        class="onboarding-button secondary"
                        data-back-to="3"
                        type="button"
                    >
                        Back
                    </button>

                    <div
                        class="onboarding-actions-right"
                    >
                        <button
                            id="onboarding-context-skip"
                            class="onboarding-button text"
                            type="button"
                        >
                            Skip
                        </button>

                        <button
                            id="onboarding-context-next"
                            class="onboarding-button primary"
                            type="button"
                        >
                            Continue
                        </button>
                    </div>
                </div>
            </section>

            <!-- ============================================
                 STEP 5 — QUIET HOURS
            ============================================= -->

            <section
                class="onboarding-step"
                data-profile-step="5"
            >
                <div class="onboarding-eyebrow">
                    Your boundaries
                </div>

                <h2 class="onboarding-title">
                    When should Kelsie leave you alone?
                </h2>

                <p class="onboarding-description">
                    Set a time when Kelsie should not
                    proactively check in or get your
                    attention. You can still open Kelsie
                    and talk normally whenever you want.
                </p>

                <label
                    class="onboarding-toggle-card"
                    for="profile-quiet-enabled"
                >
                    <span
                        class="onboarding-toggle-copy"
                    >
                        <span
                            class="onboarding-toggle-title"
                        >
                            Use quiet hours
                        </span>

                        <span
                            class="onboarding-toggle-description"
                        >
                            Pause Kelsie-initiated
                            check-ins during this window.
                        </span>
                    </span>

                    <input
                        id="profile-quiet-enabled"
                        class="onboarding-checkbox"
                        type="checkbox"
                        checked
                    />
                </label>

                <div
                    class="onboarding-time-grid"
                    id="profile-quiet-time-grid"
                >
                    <div
                        class="onboarding-field"
                    >
                        <label
                            for="profile-quiet-start"
                        >
                            From
                        </label>

                        <input
                            id="profile-quiet-start"
                            class="onboarding-time"
                            type="time"
                            value="23:00"
                        />
                    </div>

                    <div
                        class="onboarding-field"
                    >
                        <label
                            for="profile-quiet-end"
                        >
                            Until
                        </label>

                        <input
                            id="profile-quiet-end"
                            class="onboarding-time"
                            type="time"
                            value="08:00"
                        />
                    </div>
                </div>

                <p class="onboarding-note">
                    Quiet hours are for Kelsie-initiated
                    behaviour. A reminder you explicitly
                    scheduled can still appear when it
                    is due.
                </p>

                <div class="onboarding-actions">
                    <button
                        class="onboarding-button secondary"
                        data-back-to="4"
                        type="button"
                    >
                        Back
                    </button>

                    <button
                        id="onboarding-quiet-next"
                        class="onboarding-button primary"
                        type="button"
                    >
                        Continue
                    </button>
                </div>
            </section>

            <!-- ============================================
                 STEP 6 — ACCESSIBILITY
            ============================================= -->

            <section
                class="onboarding-step"
                data-profile-step="6"
            >
                <div class="onboarding-eyebrow">
                    Accessibility
                </div>

                <h2 class="onboarding-title">
                    Make Kelsie easier for you to use
                </h2>

                <p class="onboarding-description">
                    Optional. Choose any adjustments
                    that make the experience more
                    comfortable. Your device settings
                    are respected automatically where
                    possible.
                </p>

                <div
                    class="onboarding-access-list"
                >
                    <label
                        class="onboarding-access-option"
                    >
                        <input
                            id="profile-access-large-text"
                            class="onboarding-checkbox"
                            type="checkbox"
                        />

                        <span
                            class="onboarding-access-copy"
                        >
                            <strong>
                                Larger text
                            </strong>

                            <span>
                                Increase text size
                                throughout the Kelsie
                                interface.
                            </span>
                        </span>
                    </label>

                    <label
                        class="onboarding-access-option"
                    >
                        <input
                            id="profile-access-high-contrast"
                            class="onboarding-checkbox"
                            type="checkbox"
                        />

                        <span
                            class="onboarding-access-copy"
                        >
                            <strong>
                                Higher contrast
                            </strong>

                            <span>
                                Strengthen text, borders
                                and interface contrast.
                            </span>
                        </span>
                    </label>

                    <label
                        class="onboarding-access-option"
                    >
                        <input
                            id="profile-access-reduce-motion"
                            class="onboarding-checkbox"
                            type="checkbox"
                        />

                        <span
                            class="onboarding-access-copy"
                        >
                            <strong>
                                Reduce motion
                            </strong>

                            <span>
                                Reduce orb, character and
                                interface animation.
                            </span>

                            <span
                                id="profile-motion-device-note"
                                class="onboarding-device-note"
                                hidden
                            >
                                Using your device's
                                reduced-motion preference.
                            </span>
                        </span>
                    </label>

                    <label
                        class="onboarding-access-option"
                    >
                        <input
                            id="profile-access-simple-language"
                            class="onboarding-checkbox"
                            type="checkbox"
                        />

                        <span
                            class="onboarding-access-copy"
                        >
                            <strong>
                                Simpler responses
                            </strong>

                            <span>
                                Use clearer sentence
                                structure, less jargon
                                and less dense wording
                                without talking down to
                                you.
                            </span>
                        </span>
                    </label>
                </div>

                <div
                    id="onboarding-finish-error"
                    class="onboarding-error"
                    role="status"
                ></div>

                <div class="onboarding-actions">
                    <button
                        class="onboarding-button secondary"
                        data-back-to="5"
                        type="button"
                    >
                        Back
                    </button>

                    <div
                        class="onboarding-actions-right"
                    >
                        <button
                            id="onboarding-access-skip"
                            class="onboarding-button text"
                            type="button"
                        >
                            Skip
                        </button>

                        <button
                            id="onboarding-finish"
                            class="onboarding-button primary"
                            type="button"
                        >
                            Continue
                        </button>
                    </div>
                </div>
            </section>

            <!-- ============================================
                 COMPLETE
            ============================================= -->

            <section
                class="onboarding-step onboarding-complete"
                data-profile-step="7"
            >
                <div
                    class="mini-orb"
                    aria-hidden="true"
                ></div>

                <h2 class="onboarding-title">
                    That's enough to start.
                </h2>

                <p class="onboarding-description">
                    Kelsie will learn the rest
                    naturally as you talk.
                </p>

                <button
                    id="onboarding-start"
                    class="onboarding-button primary"
                    type="button"
                >
                    Start talking to Kelsie
                </button>
            </section>

        </div>
    `;

    chatCard.appendChild(
        onboarding
    );

    return onboarding;
}

/* =========================================================
   INTRO STICKMAN / CHARACTER

   This is intentionally only used for the onboarding
   introduction. It is not part of the permanent chat UI.
========================================================= */

function createIntroAvatar(
    onboarding
) {
    const canvas =
        onboarding.querySelector(
            "#kelsie-intro-avatar"
        );

    if (!canvas) {
        return {
            setVisible() {},
        };
    }

    const stage =
        canvas.closest(
            ".onboarding-intro-character"
        );

    const scene =
        new THREE.Scene();

    scene.background =
        null;

    const cameraHeight =
        2.65;

    const camera =
        new THREE.OrthographicCamera(
            -1,
            1,
            1,
            -1,
            0.1,
            100
        );

    camera.position.set(
        0,
        1,
        5
    );

    camera.lookAt(
        0,
        1,
        0
    );

    const renderer =
        new THREE.WebGLRenderer({
            canvas,

            antialias: true,

            alpha: true,

            premultipliedAlpha:
                true,
        });

    renderer.setClearColor(
        0x000000,
        0
    );

    renderer.setPixelRatio(
        Math.min(
            window.devicePixelRatio ||
                1,
            2
        )
    );

    renderer.outputColorSpace =
        THREE.SRGBColorSpace;

    renderer.toneMapping =
        THREE.NoToneMapping;

    const clock =
        new THREE.Clock();

    const characterGroup =
        new THREE.Group();

    scene.add(
        characterGroup
    );

    let mixer = null;

    let activeAction =
        null;

    let activeGesture =
        null;

    let modelReady =
        false;

    let visible =
        false;

    let wavePlayedForVisit =
        false;

    const actions =
        new Map();

    function reducesMotion() {
        return (
            DEVICE_REDUCES_MOTION ||
            document.body.classList.contains(
                "kelsie-reduce-motion"
            )
        );
    }

    function getAction(name) {
        return (
            actions.get(
                String(name)
                    .toLowerCase()
            ) ||
            null
        );
    }

    function resize() {
        if (!stage) {
            return;
        }

        const width =
            Math.max(
                stage.clientWidth,
                1
            );

        const height =
            Math.max(
                stage.clientHeight,
                1
            );

        const aspect =
            width / height;

        const halfHeight =
            cameraHeight / 2;

        const halfWidth =
            halfHeight *
            aspect;

        camera.left =
            -halfWidth;

        camera.right =
            halfWidth;

        camera.top =
            halfHeight;

        camera.bottom =
            -halfHeight;

        camera.updateProjectionMatrix();

        renderer.setSize(
            width,
            height,
            false
        );
    }

    function fitModel(
        object
    ) {
        object.updateMatrixWorld(
            true
        );

        let bounds =
            new THREE.Box3()
                .setFromObject(
                    object
                );

        const initialSize =
            bounds.getSize(
                new THREE.Vector3()
            );

        if (
            initialSize.y <= 0
        ) {
            return;
        }

        object.scale.setScalar(
            (
                cameraHeight *
                0.86
            ) /
                initialSize.y
        );

        object.updateMatrixWorld(
            true
        );

        bounds =
            new THREE.Box3()
                .setFromObject(
                    object
                );

        const center =
            bounds.getCenter(
                new THREE.Vector3()
            );

        object.position.x -=
            center.x;

        object.position.y -=
            bounds.min.y;

        object.position.z -=
            center.z;

        object.updateMatrixWorld(
            true
        );
    }

    /* -----------------------------------------------------
       Idle
    ----------------------------------------------------- */

    function playIdle() {
        if (
            !visible ||
            !mixer ||
            reducesMotion()
        ) {
            return false;
        }

        const idle =
            getAction("idle");

        if (
            !idle ||
            activeGesture
        ) {
            return false;
        }

        if (
            activeAction ===
                idle &&
            idle.isRunning()
        ) {
            return true;
        }

        if (
            activeAction &&
            activeAction !== idle
        ) {
            activeAction.fadeOut(
                0.12
            );
        }

        idle.reset();

        idle.enabled =
            true;

        idle.clampWhenFinished =
            false;

        idle.setLoop(
            THREE.LoopRepeat,
            Infinity
        );

        idle.fadeIn(
            0.12
        );

        idle.play();

        activeAction =
            idle;

        return true;
    }

    /* -----------------------------------------------------
       One small welcome wave
    ----------------------------------------------------- */

    function wave() {
        if (
            !visible ||
            !mixer ||
            reducesMotion()
        ) {
            return false;
        }

        const action =
            getAction("wave");

        if (!action) {
            return playIdle();
        }

        if (
            activeGesture
        ) {
            activeGesture.stop();
        }

        if (
            activeAction &&
            activeAction !== action
        ) {
            activeAction.fadeOut(
                0.1
            );
        }

        activeGesture =
            action;

        action.reset();

        action.enabled =
            true;

        action.clampWhenFinished =
            true;

        action.setLoop(
            THREE.LoopOnce,
            1
        );

        action.fadeIn(
            0.1
        );

        action.play();

        activeAction =
            action;

        return true;
    }

    /* -----------------------------------------------------
       Render
    ----------------------------------------------------- */

    function renderLoop() {
        window.requestAnimationFrame(
            renderLoop
        );

        const delta =
            Math.min(
                clock.getDelta(),
                0.05
            );

        if (!visible) {
            return;
        }

        if (
            mixer &&
            !reducesMotion()
        ) {
            mixer.update(
                delta
            );
        }

        renderer.render(
            scene,
            camera
        );
    }

    /* -----------------------------------------------------
       Load GLB
    ----------------------------------------------------- */

    const loader =
        new GLTFLoader();

    loader.load(
        INTRO_MODEL_URL,

        (gltf) => {
            const model =
                gltf.scene;

            model.traverse(
                (child) => {
                    if (
                        !child.isMesh
                    ) {
                        return;
                    }

                    child.frustumCulled =
                        false;

                    child.castShadow =
                        false;

                    child.receiveShadow =
                        false;

                    const materials =
                        Array.isArray(
                            child.material
                        )
                            ? child.material
                            : [
                                  child.material,
                              ];

                    materials.forEach(
                        (material) => {
                            if (
                                !material
                            ) {
                                return;
                            }

                            material.toneMapped =
                                false;

                            material.needsUpdate =
                                true;
                        }
                    );
                }
            );

            fitModel(
                model
            );

            characterGroup.add(
                model
            );

            if (
                gltf.animations
                    ?.length
            ) {
                mixer =
                    new THREE.AnimationMixer(
                        model
                    );

                gltf.animations.forEach(
                    (clip) => {
                        actions.set(
                            clip.name.toLowerCase(),

                            mixer.clipAction(
                                clip
                            )
                        );
                    }
                );

                mixer.addEventListener(
                    "finished",
                    (event) => {
                        if (
                            event.action !==
                            activeGesture
                        ) {
                            return;
                        }

                        activeGesture =
                            null;

                        activeAction =
                            null;

                        playIdle();
                    }
                );
            }

            modelReady =
                true;

            resize();

            if (visible) {
                if (
                    !wavePlayedForVisit &&
                    !reducesMotion()
                ) {
                    wavePlayedForVisit =
                        true;

                    window.setTimeout(
                        wave,
                        180
                    );
                } else {
                    playIdle();
                }
            }
        },

        undefined,

        (error) => {
            console.error(
                "Could not load Kelsie intro character:",
                error
            );
        }
    );

    /* -----------------------------------------------------
       Resize observer
    ----------------------------------------------------- */

    if (
        typeof ResizeObserver ===
            "function" &&
        stage
    ) {
        const observer =
            new ResizeObserver(
                resize
            );

        observer.observe(
            stage
        );
    } else {
        window.addEventListener(
            "resize",
            resize
        );
    }

    resize();

    renderLoop();

    return {
        setVisible(
            nextVisible
        ) {
            visible =
                Boolean(
                    nextVisible
                );

            clock.getDelta();

            if (!visible) {
                wavePlayedForVisit =
                    false;

                if (mixer) {
                    mixer.stopAllAction();
                }

                activeAction =
                    null;

                activeGesture =
                    null;

                return;
            }

            resize();

            if (!modelReady) {
                return;
            }

            if (
                !wavePlayedForVisit &&
                !reducesMotion()
            ) {
                wavePlayedForVisit =
                    true;

                window.setTimeout(
                    wave,
                    180
                );
            } else {
                playIdle();
            }
        },
    };
}

/* =========================================================
   ACCESSIBILITY PREVIEW
========================================================= */

function applyAccessibilityPreview({
    largeText = false,
    highContrast = false,
    reduceMotion = false,
} = {}) {
    if (
        typeof document ===
        "undefined"
    ) {
        return;
    }

    document.body.classList.toggle(
        "kelsie-large-text",
        Boolean(largeText)
    );

    document.body.classList.toggle(
        "kelsie-high-contrast",
        Boolean(highContrast)
    );

    document.body.classList.toggle(
        "kelsie-reduce-motion",

        DEVICE_REDUCES_MOTION ||
            Boolean(
                reduceMotion
            )
    );
}

/* =========================================================
   PROFILE UI
========================================================= */

export async function initProfileUI({
    userId,
    orb,
    chatForm,
    chatInput,
    closeButton,
    onProfileReady,
}) {
    injectOnboardingStyles();

    const chatCard =
        document.getElementById(
            "chat-card"
        ) ||
        document.querySelector(
            ".chat-card"
        );

    if (!chatCard) {
        throw new Error(
            "Kelsie chat card was not found."
        );
    }

    /* -----------------------------------------------------
       Create onboarding
    ----------------------------------------------------- */

    const onboarding =
        createOnboarding(
            chatCard
        );

    let introAvatar =
        null;

    function setIntroAvatarVisible(
        visible
    ) {
        if (
            visible &&
            !introAvatar
        ) {
            introAvatar =
                createIntroAvatar(
                    onboarding
                );
        }

        introAvatar?.setVisible(
            Boolean(
                visible
            )
        );
    }

    const sendButton =
        chatForm.querySelector(
            'button[type="submit"]'
        );

    const draft = {
        name: "",

        ...DEFAULT_PROFILE,
    };

    let profile =
        null;

    let currentStep =
        0;

    let profileSaved =
        false;

    /* -----------------------------------------------------
       Elements
    ----------------------------------------------------- */

    const progressRow =
        onboarding.querySelector(
            "[data-onboarding-progress-row]"
        );

    const progressCopy =
        document.getElementById(
            "onboarding-progress-copy"
        );

    const progressFill =
        document.getElementById(
            "onboarding-progress-fill"
        );

    const nameInput =
        document.getElementById(
            "profile-onboarding-name"
        );

    const timezoneSelect =
        document.getElementById(
            "profile-onboarding-timezone"
        );

    const contextInput =
        document.getElementById(
            "profile-onboarding-context"
        );

    const quietEnabled =
        document.getElementById(
            "profile-quiet-enabled"
        );

    const quietStart =
        document.getElementById(
            "profile-quiet-start"
        );

    const quietEnd =
        document.getElementById(
            "profile-quiet-end"
        );

    const quietGrid =
        document.getElementById(
            "profile-quiet-time-grid"
        );

    const largeText =
        document.getElementById(
            "profile-access-large-text"
        );

    const highContrast =
        document.getElementById(
            "profile-access-high-contrast"
        );

    const reduceMotion =
        document.getElementById(
            "profile-access-reduce-motion"
        );

    const simpleLanguage =
        document.getElementById(
            "profile-access-simple-language"
        );

    const motionDeviceNote =
        document.getElementById(
            "profile-motion-device-note"
        );

    /* -----------------------------------------------------
       Initial defaults
    ----------------------------------------------------- */

    populateTimezoneSelect(
        timezoneSelect,
        draft.timezone
    );

    document.getElementById(
        "onboarding-detected-timezone"
    ).textContent =
        `Detected: ${
            timezoneLabel(
                DEVICE_TIMEZONE
            ) ||
            DEVICE_TIMEZONE
        }`;

    if (
        DEVICE_REDUCES_MOTION
    ) {
        reduceMotion.checked =
            true;

        reduceMotion.disabled =
            true;

        motionDeviceNote.hidden =
            false;
    }

    /* -----------------------------------------------------
       Accessibility preview
    ----------------------------------------------------- */

    function previewAccessibilityChoices() {
        applyAccessibilityPreview({
            largeText:
                largeText.checked,

            highContrast:
                highContrast.checked,

            reduceMotion:
                DEVICE_REDUCES_MOTION ||
                reduceMotion.checked,
        });
    }

    [
        largeText,
        highContrast,
        reduceMotion,
    ].forEach(
        (control) => {
            control.addEventListener(
                "change",
                previewAccessibilityChoices
            );
        }
    );

    /* -----------------------------------------------------
       Chat enabled / disabled
    ----------------------------------------------------- */

    function setChatEnabled(
        enabled
    ) {
        chatInput.disabled =
            !enabled;

        if (sendButton) {
            sendButton.disabled =
                !enabled ||
                !chatInput.value.trim();
        }
    }

    /* -----------------------------------------------------
       Quiet hours state
    ----------------------------------------------------- */

    function updateQuietState() {
        const enabled =
            quietEnabled.checked;

        quietGrid.style.opacity =
            enabled
                ? "1"
                : "0.48";

        quietStart.disabled =
            !enabled;

        quietEnd.disabled =
            !enabled;
    }

    /* -----------------------------------------------------
       Onboarding navigation
    ----------------------------------------------------- */

    function showStep(step) {
        currentStep =
            step;

        onboarding
            .querySelectorAll(
                "[data-profile-step]"
            )
            .forEach(
                (element) => {
                    element
                        .classList
                        .toggle(
                            "active",

                            Number(
                                element
                                    .dataset
                                    .profileStep
                            ) === step
                        );
                }
            );

        const isIntro =
            step === 0;

        const isComplete =
            step === 7;

        /*
         * Intro and final screen do not count as
         * profile-question steps.
         */

        progressRow.style.display =
            isIntro ||
            isComplete
                ? "none"
                : "flex";

        /*
         * The GLB stickman only exists while the intro
         * screen is active.
         */

        setIntroAvatarVisible(
            isIntro
        );

        if (
            !isIntro &&
            !isComplete
        ) {
            progressCopy.textContent =
                `${step} of 6`;

            progressFill.style.width =
                `${
                    (step / 6) *
                    100
                }%`;
        }

        window.setTimeout(
            () => {
                if (
                    step === 1
                ) {
                    nameInput.focus();
                } else if (
                    step === 2
                ) {
                    timezoneSelect.focus();
                }
            },
            40
        );
    }

    function showOnboarding() {
        onboarding.classList.add(
            "visible"
        );

        onboarding.setAttribute(
            "aria-hidden",
            "false"
        );

        setChatEnabled(
            false
        );

        /*
         * Start with Meet Kelsie,
         * not Question 1.
         */

        showStep(0);
    }

    function hideOnboarding() {
        setIntroAvatarVisible(
            false
        );

        onboarding.classList.remove(
            "visible"
        );

        onboarding.setAttribute(
            "aria-hidden",
            "true"
        );

        setChatEnabled(
            true
        );
    }

    /* -----------------------------------------------------
       Proactivity
    ----------------------------------------------------- */

    function selectProactivity(
        value
    ) {
        draft.proactivity =
            value;

        draft.proactivity_level =
            value;

        onboarding
            .querySelectorAll(
                "[data-proactivity]"
            )
            .forEach(
                (button) => {
                    const selected =
                        button
                            .dataset
                            .proactivity ===
                        value;

                    button
                        .classList
                        .toggle(
                            "selected",
                            selected
                        );

                    button.setAttribute(
                        "aria-checked",

                        selected
                            ? "true"
                            : "false"
                    );
                }
            );
    }

    /* -----------------------------------------------------
       Save profile
    ----------------------------------------------------- */

    async function saveProfile() {
        const finishButton =
            document.getElementById(
                "onboarding-finish"
            );

        const errorElement =
            document.getElementById(
                "onboarding-finish-error"
            );

        draft.timezone =
            timezoneSelect.value ||
            DEVICE_TIMEZONE;

        draft.initial_context =
            contextInput
                .value
                .trim();

        draft.quiet_hours_enabled =
            quietEnabled.checked;

        draft.quiet_hours_start =
            quietEnabled.checked
                ? quietStart.value ||
                  "23:00"
                : null;

        draft.quiet_hours_end =
            quietEnabled.checked
                ? quietEnd.value ||
                  "08:00"
                : null;

        draft.accessibility_large_text =
            largeText.checked;

        draft.accessibility_high_contrast =
            highContrast.checked;

        draft.accessibility_reduce_motion =
            DEVICE_REDUCES_MOTION ||
            reduceMotion.checked;

        draft.accessibility_simplified_language =
            simpleLanguage.checked;

        previewAccessibilityChoices();

        finishButton.disabled =
            true;

        errorElement.textContent =
            "";

        try {
            profile =
                await createProfile({
                    user_id:
                        userId,

                    id:
                        userId,

                    name:
                        draft.name,

                    mode:
                        "both",

                    timezone:
                        draft.timezone,

                    proactivity:
                        draft.proactivity,

                    proactivity_level:
                        draft.proactivity_level,

                    initial_context:
                        draft.initial_context,

                    quiet_hours_enabled:
                        draft.quiet_hours_enabled,

                    quiet_hours_start:
                        draft.quiet_hours_start,

                    quiet_hours_end:
                        draft.quiet_hours_end,

                    accessibility_large_text:
                        draft.accessibility_large_text,

                    accessibility_high_contrast:
                        draft.accessibility_high_contrast,

                    accessibility_reduce_motion:
                        draft.accessibility_reduce_motion,

                    accessibility_simplified_language:
                        draft.accessibility_simplified_language,

                    memory_enabled:
                        true,

                    adaptive_tone:
                        true,
                });

            profileSaved =
                true;

            if (
                draft.initial_context
            ) {
                await seedInitialContext(
                    userId,
                    draft.initial_context
                );
            }

            onProfileReady?.(
                profile
            );

            showStep(7);
        } catch (error) {
            errorElement.textContent =
                error.message;
        } finally {
            finishButton.disabled =
                false;
        }
    }

    /* =====================================================
       INTRO
    ===================================================== */

    document
        .getElementById(
            "onboarding-intro-start"
        )
        .addEventListener(
            "click",
            () => {
                /*
                 * Preview mode lets an existing user see
                 * the intro without overwriting their
                 * profile.
                 */

                if (
                    INTRO_PREVIEW_ONLY &&
                    profile
                ) {
                    hideOnboarding();

                    window.setTimeout(
                        () =>
                            chatInput.focus(),
                        60
                    );

                    return;
                }

                showStep(1);
            }
        );

    /* =====================================================
       STEP 1 — NAME
    ===================================================== */

    document
        .getElementById(
            "onboarding-name-next"
        )
        .addEventListener(
            "click",
            () => {
                const errorElement =
                    document.getElementById(
                        "onboarding-name-error"
                    );

                const name =
                    nameInput
                        .value
                        .trim();

                if (!name) {
                    errorElement.textContent =
                        "Enter the name Kelsie should use.";

                    nameInput.focus();

                    return;
                }

                draft.name =
                    name;

                errorElement.textContent =
                    "";

                showStep(2);
            }
        );

    /* =====================================================
       STEP 2 — TIMEZONE
    ===================================================== */

    document
        .getElementById(
            "onboarding-timezone-next"
        )
        .addEventListener(
            "click",
            () => {
                draft.timezone =
                    timezoneSelect.value ||
                    DEVICE_TIMEZONE;

                showStep(3);
            }
        );

    /* =====================================================
       STEP 3 — PROACTIVITY
    ===================================================== */

    onboarding
        .querySelectorAll(
            "[data-proactivity]"
        )
        .forEach(
            (button) => {
                button.addEventListener(
                    "click",
                    () => {
                        selectProactivity(
                            button.dataset
                                .proactivity
                        );
                    }
                );
            }
        );

    document
        .getElementById(
            "onboarding-proactivity-next"
        )
        .addEventListener(
            "click",
            () => {
                showStep(4);
            }
        );

    /* =====================================================
       STEP 4 — INITIAL CONTEXT
    ===================================================== */

    onboarding
        .querySelectorAll(
            "[data-context-example]"
        )
        .forEach(
            (button) => {
                button.addEventListener(
                    "click",
                    () => {
                        if (
                            !contextInput
                                .value
                                .trim()
                        ) {
                            contextInput.value =
                                button
                                    .dataset
                                    .contextExample ||
                                "";
                        }

                        contextInput.focus();

                        contextInput
                            .setSelectionRange(
                                contextInput
                                    .value
                                    .length,

                                contextInput
                                    .value
                                    .length
                            );
                    }
                );
            }
        );

    document
        .getElementById(
            "onboarding-context-next"
        )
        .addEventListener(
            "click",
            () => {
                draft.initial_context =
                    contextInput
                        .value
                        .trim();

                showStep(5);
            }
        );

    document
        .getElementById(
            "onboarding-context-skip"
        )
        .addEventListener(
            "click",
            () => {
                contextInput.value =
                    "";

                draft.initial_context =
                    "";

                showStep(5);
            }
        );

    /* =====================================================
       STEP 5 — QUIET HOURS
    ===================================================== */

    quietEnabled.addEventListener(
        "change",
        updateQuietState
    );

    updateQuietState();

    document
        .getElementById(
            "onboarding-quiet-next"
        )
        .addEventListener(
            "click",
            () => {
                draft.quiet_hours_enabled =
                    quietEnabled.checked;

                draft.quiet_hours_start =
                    quietStart.value ||
                    "23:00";

                draft.quiet_hours_end =
                    quietEnd.value ||
                    "08:00";

                showStep(6);
            }
        );

    /* =====================================================
       STEP 6 — ACCESSIBILITY
    ===================================================== */

    document
        .getElementById(
            "onboarding-access-skip"
        )
        .addEventListener(
            "click",
            () => {
                largeText.checked =
                    false;

                highContrast.checked =
                    false;

                simpleLanguage.checked =
                    false;

                if (
                    !DEVICE_REDUCES_MOTION
                ) {
                    reduceMotion.checked =
                        false;
                }

                previewAccessibilityChoices();

                saveProfile();
            }
        );

    document
        .getElementById(
            "onboarding-finish"
        )
        .addEventListener(
            "click",
            saveProfile
        );

    /* =====================================================
       BACK BUTTONS
    ===================================================== */

    onboarding
        .querySelectorAll(
            "[data-back-to]"
        )
        .forEach(
            (button) => {
                button.addEventListener(
                    "click",
                    () => {
                        showStep(
                            Number(
                                button
                                    .dataset
                                    .backTo
                            )
                        );
                    }
                );
            }
        );

    /* =====================================================
       FINAL SCREEN
    ===================================================== */

    document
        .getElementById(
            "onboarding-start"
        )
        .addEventListener(
            "click",
            () => {
                if (
                    !profileSaved ||
                    !profile
                ) {
                    return;
                }

                hideOnboarding();

                window.setTimeout(
                    () =>
                        chatInput.focus(),
                    60
                );
            }
        );

    /* =====================================================
       ORB
    ===================================================== */

    orb.addEventListener(
        "click",
        () => {
            if (
                !profile &&
                currentStep === 1
            ) {
                window.setTimeout(
                    () =>
                        nameInput.focus(),
                    350
                );
            }
        }
    );

    /*
     * Chat stays disabled while profile state
     * is being resolved.
     */

    setChatEnabled(
        false
    );

    /* =====================================================
       LOAD EXISTING PROFILE
    ===================================================== */

    try {
        profile =
            await fetchProfile(
                userId
            );

        if (profile) {
            applyAccessibilityPreview({
                largeText:
                    profile
                        .accessibility_large_text ===
                    true,

                highContrast:
                    profile
                        .accessibility_high_contrast ===
                    true,

                reduceMotion:
                    profile
                        .accessibility_reduce_motion ===
                    true,
            });

            onProfileReady?.(
                profile
            );

            /*
             * Existing users normally skip onboarding.
             *
             * ?kelsie_intro_preview=1 intentionally
             * shows only the intro so it can be tested
             * safely.
             */

            if (
                INTRO_PREVIEW_ONLY
            ) {
                showOnboarding();
            } else {
                hideOnboarding();
            }
        } else {
            /*
             * Brand-new user:
             *
             * Meet Kelsie
             *      ↓
             * six profile questions
             */

            showOnboarding();
        }
    } catch (error) {
        console.error(
            "Could not load Kelsie profile:",
            error
        );

        showOnboarding();

        document.getElementById(
            "onboarding-name-error"
        ).textContent =
            "Kelsie could not load your profile. Check that the backend is running.";
    }

    /* =====================================================
       PUBLIC PROFILE STATE
    ===================================================== */

    window.KelsieProfile = {
        get userId() {
            return userId;
        },

        get profile() {
            return profile;
        },
    };

    return {
        profile,
    };
}