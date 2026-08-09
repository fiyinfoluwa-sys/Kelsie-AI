const KELSIE_API_BASE = "http://127.0.0.1:8000";
const KELSIE_API_FALLBACK = "http://localhost:8000";
const PAGE_ASSIST_TIMEOUT_MS = 45000;

function normalizePageAssistMessage(message) {
    if (!message || typeof message !== "object") {
        return null;
    }

    const rawPayload =
        message.payload && typeof message.payload === "object"
            ? message.payload
            : {};

    if (message.type === "KELSIE_PAGE_ASSIST") {
        return {
            action:
                typeof rawPayload.action === "string"
                    ? rawPayload.action
                    : "summarize",
            title:
                typeof rawPayload.title === "string"
                    ? rawPayload.title
                    : "",
            text:
                typeof rawPayload.text === "string"
                    ? rawPayload.text
                    : "",
            question:
                typeof rawPayload.question === "string"
                    ? rawPayload.question
                    : "",
        };
    }

    // Compatibility with the first Level 1 build.
    if (message.type === "KELSIE_SUMMARIZE_PAGE") {
        return {
            action: "summarize",
            title:
                typeof rawPayload.title === "string"
                    ? rawPayload.title
                    : "",
            text:
                typeof rawPayload.text === "string"
                    ? rawPayload.text
                    : "",
            question: "",
        };
    }

    // Compatibility with the temporary reading-assist prototype.
    if (message.type === "KELSIE_READING_SUMMARIZE") {
        return {
            action: "summarize",
            title:
                typeof rawPayload.title === "string"
                    ? rawPayload.title
                    : "",
            text:
                typeof rawPayload.text === "string"
                    ? rawPayload.text
                    : "",
            question: "",
        };
    }

    if (message.type === "KELSIE_READING_ANALYZE") {
        return {
            action: "explain",
            title:
                typeof rawPayload.title === "string"
                    ? rawPayload.title
                    : "",
            text:
                typeof rawPayload.text === "string"
                    ? rawPayload.text
                    : "",
            question: "",
        };
    }

    if (message.type === "KELSIE_READING_ASK") {
        return {
            action: "question",
            title:
                typeof rawPayload.title === "string"
                    ? rawPayload.title
                    : "",
            text:
                typeof rawPayload.text === "string"
                    ? rawPayload.text
                    : "",
            question:
                typeof rawPayload.question === "string"
                    ? rawPayload.question
                    : "",
        };
    }

    return null;
}

async function postPageAssist(
    baseUrl,
    payload
) {
    const controller =
        new AbortController();

    const timeoutId =
        setTimeout(
            () => {
                controller.abort();
            },
            PAGE_ASSIST_TIMEOUT_MS
        );

    try {
        const response =
            await fetch(
                `${baseUrl}/api/page/assist`,
                {
                    method: "POST",
                    headers: {
                        Accept: "application/json",
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(
                        payload
                    ),
                    signal:
                        controller.signal,
                }
            );

        let data = null;

        try {
            data =
                await response.json();
        } catch (_error) {
            data = null;
        }

        if (!response.ok) {
            const detail =
                typeof data?.detail ===
                "string"
                    ? data.detail
                    : `Kelsie could not help with this page (${response.status}).`;

            throw new Error(
                detail
            );
        }

        return data;
    } finally {
        clearTimeout(
            timeoutId
        );
    }
}

async function pageAssist(
    payload
) {
    let firstError = null;

    try {
        return await postPageAssist(
            KELSIE_API_BASE,
            payload
        );
    } catch (error) {
        firstError = error;
    }

    try {
        return await postPageAssist(
            KELSIE_API_FALLBACK,
            payload
        );
    } catch (fallbackError) {
        throw (
            fallbackError ||
            firstError
        );
    }
}

chrome.runtime.onMessage.addListener(
    (
        message,
        _sender,
        sendResponse
    ) => {
        const payload =
            normalizePageAssistMessage(
                message
            );

        if (!payload) {
            return false;
        }

        if (!payload.text.trim()) {
            sendResponse({
                ok: false,
                error:
                    "Kelsie could not read enough of this page to help yet.",
            });

            return false;
        }

        pageAssist(payload)
            .then(
                (data) => {
                    sendResponse({
                        ok: true,
                        data,
                    });
                }
            )
            .catch(
                (error) => {
                    console.error(
                        "[Kelsie] Page help failed:",
                        error
                    );

                    sendResponse({
                        ok: false,
                        error:
                            error instanceof Error &&
                            error.message
                                ? error.message
                                : "Kelsie could not help with this page right now.",
                    });
                }
            );

        // Keep the MV3 message channel open until the async fetch finishes.
        return true;
    }
);

console.log(
    "[Kelsie] Background service worker ready for page assistance."
);