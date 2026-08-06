const PROFILE_API = "/api/profile";

const DEFAULT_PROFILE = Object.freeze({
    timezone:
        Intl.DateTimeFormat()
            .resolvedOptions()
            .timeZone || "America/Toronto",

    daily_overview_time: "08:00",
    quiet_start: "23:00",
    quiet_end: "08:00",
    proactivity_level: "balanced",
    memory_enabled: true,
    adaptive_tone: true,
});


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


async function requestJson(
    url,
    options = {}
) {
    const response = await fetch(
        url,
        {
            ...options,

            headers: {
                "Content-Type":
                    "application/json",

                ...(options.headers || {}),
            },
        }
    );

    const data =
        await response.json().catch(
            () => ({})
        );

    if (!response.ok) {
        throw new Error(
            data.detail ||
            "The request could not be completed."
        );
    }

    return data;
}


async function fetchProfile(
    userId
) {
    const result = await requestJson(
        `${PROFILE_API}/${encodeURIComponent(
            userId
        )}`
    );

    if (
        result &&
        typeof result === "object" &&
        "profile" in result
    ) {
        return result;
    }

    return {
        exists: Boolean(result),
        profile: result || null,
    };
}


async function createProfile(
    profile
) {
    const result = await requestJson(
        PROFILE_API,
        {
            method: "POST",
            body: JSON.stringify(profile),
        }
    );

    return (
        result &&
        typeof result === "object" &&
        "profile" in result
    )
        ? result
        : {
            exists: true,
            profile: result,
        };
}


async function updateProfile(
    userId,
    updates
) {
    const result = await requestJson(
        `${PROFILE_API}/${encodeURIComponent(
            userId
        )}`,
        {
            method: "PATCH",
            body: JSON.stringify(updates),
        }
    );

    return (
        result &&
        typeof result === "object" &&
        "profile" in result
    )
        ? result
        : {
            exists: true,
            profile: result,
        };
}


function injectProfileStyles() {
    if (
        document.getElementById(
            "kelsie-profile-styles"
        )
    ) {
        return;
    }

    const style =
        document.createElement("style");

    style.id =
        "kelsie-profile-styles";

    style.textContent = `
        .profile-header-actions {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        #profile-settings-button {
            width: 31px;
            height: 31px;

            display: grid;
            place-items: center;

            padding: 0;

            border: 1px solid
                rgba(21, 29, 43, 0.04);

            border-radius: 50%;

            background:
                rgba(24, 32, 46, 0.045);

            color: #778091;

            cursor: pointer;

            transition:
                background 0.2s ease,
                color 0.2s ease,
                transform 0.2s ease;
        }

        #profile-settings-button:hover {
            color: #161b26;

            background:
                rgba(24, 32, 46, 0.085);

            transform: scale(1.05);
        }

        #profile-settings-button svg {
            width: 15px;
            height: 15px;
        }

        .profile-overlay {
            position: absolute;
            inset: 0;
            z-index: 30;

            display: none;
            align-items: center;
            justify-content: center;

            padding: 16px;

            border-radius: inherit;

            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.97),
                    rgba(240,245,250,0.95)
                );

            backdrop-filter:
                blur(24px);

            -webkit-backdrop-filter:
                blur(24px);
        }

        .profile-overlay.visible {
            display: flex;
        }

        .profile-card {
            width: 100%;
            max-height: 100%;

            overflow-y: auto;

            padding: 5px 3px;

            color: #161b26;
        }

        .profile-eyebrow {
            margin-bottom: 7px;

            color: #7f8998;

            font-size: 10px;
            font-weight: 650;

            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .profile-title {
            margin: 0 0 7px;

            font-size: 22px;
            line-height: 1.15;
            letter-spacing: -0.035em;
        }

        .profile-description {
            margin: 0 0 18px;

            color: #687284;

            font-size: 12px;
            line-height: 1.5;
        }

        .profile-field {
            display: flex;
            flex-direction: column;
            gap: 6px;

            margin-bottom: 13px;
        }

        .profile-field label {
            color: #566173;

            font-size: 10px;
            font-weight: 650;
        }

        .profile-field input,
        .profile-field select {
            width: 100%;
            height: 40px;

            padding: 0 12px;

            border: 1px solid
                rgba(21, 29, 43, 0.09);

            border-radius: 13px;
            outline: none;

            background:
                rgba(255,255,255,0.82);

            color: #18202d;

            font-size: 12px;

            transition:
                border-color 0.2s ease,
                box-shadow 0.2s ease;
        }

        .profile-toggle-row {
            grid-column: 1 / -1;

            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;

            min-height: 52px;
            padding: 10px 12px;

            border: 1px solid
                rgba(21,29,43,0.07);

            border-radius: 14px;

            background:
                rgba(255,255,255,0.62);
        }

        .profile-toggle-copy {
            display: grid;
            gap: 3px;
        }

        .profile-toggle-copy strong {
            color: #273142;

            font-size: 11px;
            font-weight: 650;
        }

        .profile-toggle-copy span {
            color: #7a8494;

            font-size: 9px;
            line-height: 1.35;
        }

        .profile-toggle-row
        input[type="checkbox"] {
            width: 36px;
            height: 20px;

            margin: 0;

            accent-color: #808899;
            cursor: pointer;
        }

        .profile-field input:focus,
        .profile-field select:focus {
            border-color:
                rgba(112,198,235,0.55);

            box-shadow:
                0 0 0 4px
                rgba(112,198,235,0.12);
        }

        .profile-mode-grid {
            display: grid;
            grid-template-columns:
                repeat(3, 1fr);

            gap: 8px;
            margin-bottom: 17px;
        }

        .profile-mode-button {
            min-height: 62px;

            padding: 8px;

            border: 1px solid
                rgba(21,29,43,0.08);

            border-radius: 15px;

            background:
                rgba(255,255,255,0.72);

            color: #4e596b;

            cursor: pointer;

            font-size: 11px;
            font-weight: 600;

            transition:
                border-color 0.2s ease,
                background 0.2s ease,
                transform 0.2s ease;
        }

        .profile-mode-button:hover {
            transform: translateY(-1px);

            border-color:
                rgba(112,198,235,0.4);
        }

        .profile-mode-button.selected {
            border-color:
                rgba(112,198,235,0.65);

            background:
                linear-gradient(
                    145deg,
                    rgba(236,249,255,0.95),
                    rgba(245,242,255,0.92)
                );

            color: #18202d;

            box-shadow:
                0 7px 20px
                rgba(37,58,78,0.07);
        }

        .profile-summary {
            display: grid;
            gap: 8px;

            margin-bottom: 17px;
            padding: 12px;

            border: 1px solid
                rgba(21,29,43,0.06);

            border-radius: 15px;

            background:
                rgba(255,255,255,0.62);
        }

        .profile-summary-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;

            color: #687284;
            font-size: 11px;
        }

        .profile-summary-row strong {
            color: #273142;
            font-weight: 650;
        }

        .profile-actions {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 8px;
        }

        .profile-button {
            height: 39px;

            padding: 0 15px;

            border: 0;
            border-radius: 20px;

            cursor: pointer;

            font-size: 11px;
            font-weight: 650;

            transition:
                transform 0.2s ease,
                opacity 0.2s ease;
        }

        .profile-button:hover {
            transform: translateY(-1px);
        }

        .profile-button.primary {
            color: white;

            background:
                linear-gradient(
                    145deg,
                    #273141,
                    #121823
                );

            box-shadow:
                0 8px 19px
                rgba(13,19,30,0.18);
        }

        .profile-button.secondary {
            color: #5d6879;

            background:
                rgba(20,30,45,0.055);
        }

        .profile-button:disabled {
            opacity: 0.4;
            cursor: default;
            transform: none;
        }

        .profile-error {
            min-height: 16px;
            margin: 8px 0 0;

            color: #cf5363;

            font-size: 10px;
            line-height: 1.4;
        }

        .profile-step {
            display: none;
        }

        .profile-step.active {
            display: block;
        }

        .profile-settings-grid {
            display: grid;
            grid-template-columns:
                1fr 1fr;

            gap: 0 10px;
        }

        .profile-settings-grid
        .profile-field.full {
            grid-column: 1 / -1;
        }
    `;

    document.head.appendChild(style);
}


function createSettingsButton(
    closeButton
) {
    const existing =
        document.getElementById(
            "profile-settings-button"
        );

    if (existing) {
        return existing;
    }

    const headerActions =
        document.createElement("div");

    headerActions.className =
        "profile-header-actions";

    const settingsButton =
        document.createElement("button");

    settingsButton.id =
        "profile-settings-button";

    settingsButton.type =
        "button";

    settingsButton.setAttribute(
        "aria-label",
        "Open profile settings"
    );

    settingsButton.innerHTML = `
        <svg
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
        >
            <path
                d="M12 15.25A3.25 3.25 0 1 0 12 8.75a3.25 3.25 0 0 0 0 6.5Z"
                stroke="currentColor"
                stroke-width="1.6"
            />

            <path
                d="M19.1 13.8a7.6 7.6 0 0 0 0-3.6l2-1.55-2-3.45-2.47 1a7.5 7.5 0 0 0-3.12-1.8L13.16 2h-4l-.35 2.4a7.5 7.5 0 0 0-3.12 1.8l-2.47-1-2 3.45 2 1.55a7.6 7.6 0 0 0 0 3.6l-2 1.55 2 3.45 2.47-1a7.5 7.5 0 0 0 3.12 1.8l.35 2.4h4l.35-2.4a7.5 7.5 0 0 0 3.12-1.8l2.47 1 2-3.45-2-1.55Z"
                stroke="currentColor"
                stroke-width="1.3"
                stroke-linejoin="round"
            />
        </svg>
    `;

    const parent =
        closeButton.parentElement;

    parent.insertBefore(
        headerActions,
        closeButton
    );

    headerActions.appendChild(
        settingsButton
    );

    headerActions.appendChild(
        closeButton
    );

    return settingsButton;
}


function createProfileOverlays(
    chatCard
) {
    const onboarding =
        document.createElement("div");

    onboarding.id =
        "kelsie-onboarding";

    onboarding.className =
        "profile-overlay";

    onboarding.innerHTML = `
        <div class="profile-card">
            <div
                class="profile-step active"
                data-profile-step="1"
            >
                <div class="profile-eyebrow">
                    Meet Kelsie
                </div>

                <h2 class="profile-title">
                    Let’s make this yours.
                </h2>

                <p class="profile-description">
                    First, what should Kelsie call you?
                </p>

                <div class="profile-field">
                    <label for="profile-onboarding-name">
                        Your name
                    </label>

                    <input
                        id="profile-onboarding-name"
                        type="text"
                        maxlength="60"
                        autocomplete="name"
                        placeholder="Enter your name"
                    />
                </div>

                <div class="profile-actions">
                    <button
                        id="profile-name-next"
                        class="profile-button primary"
                        type="button"
                    >
                        Continue
                    </button>
                </div>

                <div
                    id="profile-onboarding-error"
                    class="profile-error"
                ></div>
            </div>

            <div
                class="profile-step"
                data-profile-step="2"
            >
                <div class="profile-eyebrow">
                    Your everyday rhythm
                </div>

                <h2 class="profile-title">
                    What best describes you?
                </h2>

                <p class="profile-description">
                    This changes what Kelsie prioritizes
                    when helping you.
                </p>

                <div class="profile-mode-grid">
                    <button
                        class="profile-mode-button"
                        type="button"
                        data-profile-mode="student"
                    >
                        Student
                    </button>

                    <button
                        class="profile-mode-button"
                        type="button"
                        data-profile-mode="professional"
                    >
                        Professional
                    </button>

                    <button
                        class="profile-mode-button"
                        type="button"
                        data-profile-mode="both"
                    >
                        Both
                    </button>
                </div>

                <div class="profile-actions">
                    <button
                        class="profile-button secondary"
                        id="profile-mode-back"
                        type="button"
                    >
                        Back
                    </button>

                    <button
                        class="profile-button primary"
                        id="profile-mode-next"
                        type="button"
                        disabled
                    >
                        Continue
                    </button>
                </div>
            </div>

            <div
                class="profile-step"
                data-profile-step="3"
            >
                <div class="profile-eyebrow">
                    Final details
                </div>

                <h2 class="profile-title">
                    You’re ready.
                </h2>

                <p class="profile-description">
                    These defaults can be changed anytime
                    from Kelsie’s settings.
                </p>

                <div class="profile-field">
                    <label for="profile-onboarding-timezone">
                        Timezone
                    </label>

                    <input
                        id="profile-onboarding-timezone"
                        type="text"
                    />
                </div>

                <div class="profile-summary">
                    <div class="profile-summary-row">
                        <span>Daily overview</span>
                        <strong>8:00 AM</strong>
                    </div>

                    <div class="profile-summary-row">
                        <span>Quiet hours</span>
                        <strong>11:00 PM–8:00 AM</strong>
                    </div>

                    <div class="profile-summary-row">
                        <span>Check-ins</span>
                        <strong>Balanced</strong>
                    </div>
                </div>

                <div class="profile-actions">
                    <button
                        class="profile-button secondary"
                        id="profile-timezone-back"
                        type="button"
                    >
                        Back
                    </button>

                    <button
                        class="profile-button primary"
                        id="profile-finish"
                        type="button"
                    >
                        Start with Kelsie
                    </button>
                </div>

                <div
                    id="profile-finish-error"
                    class="profile-error"
                ></div>
            </div>
        </div>
    `;

    const settings =
        document.createElement("div");

    settings.id =
        "kelsie-profile-settings";

    settings.className =
        "profile-overlay";

    settings.innerHTML = `
        <div class="profile-card">
            <div class="profile-eyebrow">
                Kelsie settings
            </div>

            <h2 class="profile-title">
                Your profile
            </h2>

            <p class="profile-description">
                Control how Kelsie addresses you
                and when she should check in.
            </p>

            <div class="profile-settings-grid">
                <div class="profile-field full">
                    <label for="settings-name">
                        Name
                    </label>

                    <input
                        id="settings-name"
                        type="text"
                        maxlength="60"
                    />
                </div>

                <div class="profile-field full">
                    <label for="settings-mode">
                        Mode
                    </label>

                    <select id="settings-mode">
                        <option value="student">
                            Student
                        </option>

                        <option value="professional">
                            Professional
                        </option>

                        <option value="both">
                            Both
                        </option>
                    </select>
                </div>

                <div class="profile-field full">
                    <label for="settings-timezone">
                        Timezone
                    </label>

                    <input
                        id="settings-timezone"
                        type="text"
                    />
                </div>

                <div class="profile-field">
                    <label for="settings-overview">
                        Daily overview
                    </label>

                    <input
                        id="settings-overview"
                        type="time"
                    />
                </div>

                <div class="profile-field">
                    <label for="settings-proactivity">
                        Check-ins
                    </label>

                    <select id="settings-proactivity">
                        <option value="necessary">
                            Only when necessary
                        </option>

                        <option value="balanced">
                            Balanced
                        </option>

                        <option value="proactive">
                            More proactive
                        </option>
                    </select>
                </div>

                <div class="profile-field">
                    <label for="settings-quiet-start">
                        Quiet hours start
                    </label>

                    <input
                        id="settings-quiet-start"
                        type="time"
                    />
                </div>

                <div class="profile-field">
                    <label for="settings-quiet-end">
                        Quiet hours end
                    </label>

                    <input
                        id="settings-quiet-end"
                        type="time"
                    />
                </div>

                <label
                    class="profile-toggle-row"
                    for="settings-memory-enabled"
                >
                    <span class="profile-toggle-copy">
                        <strong>
                            Long-term memory
                        </strong>

                        <span>
                            Remember useful context
                            across conversations.
                        </span>
                    </span>

                    <input
                        id="settings-memory-enabled"
                        type="checkbox"
                        checked
                    />
                </label>

                <label
                    class="profile-toggle-row"
                    for="settings-adaptive-tone"
                >
                    <span class="profile-toggle-copy">
                        <strong>
                            Adaptive conversation style
                        </strong>

                        <span>
                            Match your directness, tone
                            and usual reply length.
                        </span>
                    </span>

                    <input
                        id="settings-adaptive-tone"
                        type="checkbox"
                        checked
                    />
                </label>
            </div>

            <div class="profile-actions">
                <button
                    id="settings-cancel"
                    class="profile-button secondary"
                    type="button"
                >
                    Cancel
                </button>

                <button
                    id="settings-save"
                    class="profile-button primary"
                    type="button"
                >
                    Save changes
                </button>
            </div>

            <div
                id="settings-error"
                class="profile-error"
            ></div>
        </div>
    `;

    chatCard.appendChild(onboarding);
    chatCard.appendChild(settings);

    return {
        onboarding,
        settings,
    };
}


export async function initProfileUI({
    userId,
    orb,
    chatForm,
    chatInput,
    closeButton,
    onProfileReady,
}) {
    injectProfileStyles();

    const chatCard =
        document.querySelector(
            ".chat-card"
        );

    if (!chatCard) {
        throw new Error(
            "Kelsie chat card was not found."
        );
    }

    const settingsButton =
        createSettingsButton(
            closeButton
        );

    const {
        onboarding,
        settings,
    } = createProfileOverlays(
        chatCard
    );

    const sendButton =
        chatForm.querySelector(
            'button[type="submit"]'
        );

    const draft = {
        name: "",
        mode: "",
        ...DEFAULT_PROFILE,
    };

    let profile = null;
    let currentStep = 1;

    function setChatEnabled(
        enabled
    ) {
        chatInput.disabled = !enabled;

        if (sendButton) {
            sendButton.disabled =
                !enabled ||
                !chatInput.value.trim();
        }
    }

    function showOnboardingStep(
        step
    ) {
        currentStep = step;

        onboarding
            .querySelectorAll(
                "[data-profile-step]"
            )
            .forEach((element) => {
                element.classList.toggle(
                    "active",
                    Number(
                        element.dataset
                            .profileStep
                    ) === step
                );
            });
    }

    function showOnboarding() {
        onboarding.classList.add(
            "visible"
        );

        settings.classList.remove(
            "visible"
        );

        setChatEnabled(false);
    }

    function hideOnboarding() {
        onboarding.classList.remove(
            "visible"
        );

        setChatEnabled(true);
    }

    function populateSettings(
        currentProfile
    ) {
        document.getElementById(
            "settings-name"
        ).value =
            currentProfile.name || "";

        document.getElementById(
            "settings-mode"
        ).value =
            currentProfile.mode ||
            "student";

        document.getElementById(
            "settings-timezone"
        ).value =
            currentProfile.timezone ||
            DEFAULT_PROFILE.timezone;

        document.getElementById(
            "settings-overview"
        ).value =
            currentProfile
                .daily_overview_time ||
            "08:00";

        document.getElementById(
            "settings-quiet-start"
        ).value =
            currentProfile.quiet_start ||
            "23:00";

        document.getElementById(
            "settings-quiet-end"
        ).value =
            currentProfile.quiet_end ||
            "08:00";

        document.getElementById(
            "settings-proactivity"
        ).value =
            currentProfile
                .proactivity_level ||
            "balanced";

        document.getElementById(
            "settings-memory-enabled"
        ).checked =
            currentProfile
                .memory_enabled !== false;

        document.getElementById(
            "settings-adaptive-tone"
        ).checked =
            currentProfile
                .adaptive_tone !== false;
    }

    function openSettings() {
        if (!profile) {
            return;
        }

        populateSettings(profile);

        onboarding.classList.remove(
            "visible"
        );

        settings.classList.add(
            "visible"
        );

        setChatEnabled(false);
    }

    function closeSettings() {
        settings.classList.remove(
            "visible"
        );

        setChatEnabled(true);

        window.setTimeout(() => {
            chatInput.focus();
        }, 50);
    }

    document.getElementById(
        "profile-name-next"
    ).addEventListener(
        "click",
        () => {
            const nameInput =
                document.getElementById(
                    "profile-onboarding-name"
                );

            const errorElement =
                document.getElementById(
                    "profile-onboarding-error"
                );

            const name =
                nameInput.value.trim();

            if (!name) {
                errorElement.textContent =
                    "Enter the name Kelsie should use.";

                return;
            }

            errorElement.textContent = "";
            draft.name = name;

            showOnboardingStep(2);
        }
    );

    document
        .querySelectorAll(
            "[data-profile-mode]"
        )
        .forEach((button) => {
            button.addEventListener(
                "click",
                () => {
                    draft.mode =
                        button.dataset
                            .profileMode;

                    document
                        .querySelectorAll(
                            "[data-profile-mode]"
                        )
                        .forEach(
                            (
                                modeButton
                            ) => {
                                modeButton
                                    .classList
                                    .toggle(
                                        "selected",
                                        modeButton ===
                                            button
                                    );
                            }
                        );

                    document.getElementById(
                        "profile-mode-next"
                    ).disabled = false;
                }
            );
        });

    document.getElementById(
        "profile-mode-back"
    ).addEventListener(
        "click",
        () => {
            showOnboardingStep(1);
        }
    );

    document.getElementById(
        "profile-mode-next"
    ).addEventListener(
        "click",
        () => {
            if (!draft.mode) {
                return;
            }

            document.getElementById(
                "profile-onboarding-timezone"
            ).value =
                draft.timezone;

            showOnboardingStep(3);
        }
    );

    document.getElementById(
        "profile-timezone-back"
    ).addEventListener(
        "click",
        () => {
            showOnboardingStep(2);
        }
    );

    document.getElementById(
        "profile-finish"
    ).addEventListener(
        "click",
        async () => {
            const finishButton =
                document.getElementById(
                    "profile-finish"
                );

            const errorElement =
                document.getElementById(
                    "profile-finish-error"
                );

            draft.timezone =
                document.getElementById(
                    "profile-onboarding-timezone"
                ).value.trim() ||
                DEFAULT_PROFILE.timezone;

            finishButton.disabled = true;
            errorElement.textContent = "";

            try {
                const result =
                    await createProfile({
                        id: userId,
                        name: draft.name,
                        mode: draft.mode,

                        timezone:
                            draft.timezone,

                        daily_overview_time:
                            draft
                                .daily_overview_time,

                        quiet_start:
                            draft.quiet_start,

                        quiet_end:
                            draft.quiet_end,

                        proactivity_level:
                            draft
                                .proactivity_level,

                        memory_enabled:
                            draft.memory_enabled,

                        adaptive_tone:
                            draft.adaptive_tone,
                    });

                profile =
                    result.profile;

                hideOnboarding();

                onProfileReady?.(
                    profile
                );

            } catch (error) {
                errorElement.textContent =
                    error.message;

            } finally {
                finishButton.disabled =
                    false;
            }
        }
    );

    settingsButton.addEventListener(
        "click",
        openSettings
    );

    document.getElementById(
        "settings-cancel"
    ).addEventListener(
        "click",
        closeSettings
    );

    document.getElementById(
        "settings-save"
    ).addEventListener(
        "click",
        async () => {
            const saveButton =
                document.getElementById(
                    "settings-save"
                );

            const errorElement =
                document.getElementById(
                    "settings-error"
                );

            const updates = {
                name:
                    document.getElementById(
                        "settings-name"
                    ).value.trim(),

                mode:
                    document.getElementById(
                        "settings-mode"
                    ).value,

                timezone:
                    document.getElementById(
                        "settings-timezone"
                    ).value.trim(),

                daily_overview_time:
                    document.getElementById(
                        "settings-overview"
                    ).value,

                quiet_start:
                    document.getElementById(
                        "settings-quiet-start"
                    ).value,

                quiet_end:
                    document.getElementById(
                        "settings-quiet-end"
                    ).value,

                proactivity_level:
                    document.getElementById(
                        "settings-proactivity"
                    ).value,

                memory_enabled:
                    document.getElementById(
                        "settings-memory-enabled"
                    ).checked,

                adaptive_tone:
                    document.getElementById(
                        "settings-adaptive-tone"
                    ).checked,
            };

            if (!updates.name) {
                errorElement.textContent =
                    "Name cannot be empty.";

                return;
            }

            if (!updates.timezone) {
                errorElement.textContent =
                    "Timezone cannot be empty.";

                return;
            }

            saveButton.disabled = true;
            errorElement.textContent = "";

            try {
                const result =
                    await updateProfile(
                        userId,
                        updates
                    );

                profile =
                    result.profile;

                onProfileReady?.(
                    profile
                );

                closeSettings();

            } catch (error) {
                errorElement.textContent =
                    error.message;

            } finally {
                saveButton.disabled =
                    false;
            }
        }
    );

    orb.addEventListener(
        "click",
        () => {
            if (!profile) {
                window.setTimeout(() => {
                    const nameInput =
                        document.getElementById(
                            "profile-onboarding-name"
                        );

                    if (
                        currentStep === 1
                    ) {
                        nameInput.focus();
                    }
                }, 380);
            }
        }
    );

    setChatEnabled(false);

    try {
        const result =
            await fetchProfile(
                userId
            );

        if (
            result.exists &&
            result.profile
        ) {
            profile =
                result.profile;

            hideOnboarding();

            onProfileReady?.(
                profile
            );

        } else {
            showOnboarding();
        }

    } catch (error) {
        showOnboarding();

        document.getElementById(
            "profile-onboarding-error"
        ).textContent =
            "Kelsie could not load your profile. Check that the backend is running.";
    }

    window.KelsieProfile = {
        get userId() {
            return userId;
        },

        get profile() {
            return profile;
        },

        openSettings,
    };

    return {
        profile,
        openSettings,
    };
}