(() => {
    const params =
        new URLSearchParams(
            window.location.search
        );

    if (
        params.get(
            "kelsie_extension_embed"
        ) !== "1"
    ) {
        return;
    }

    const PARENT_MESSAGE_SOURCE =
        "kelsie-extension-shell";

    const BRIDGE_MESSAGE_SOURCE =
        "kelsie-widget-bridge";

    let extensionVisible = false;
    let suppressParentClose = false;
    let lastReminderSignature = null;

    let pageContext = null;
    let pageQuestionBusy = false;
    let originalChatPlaceholder = "Message Kelsie";

    function postToParent(
        type,
        extra = {}
    ) {
        window.parent.postMessage(
            {
                source:
                    BRIDGE_MESSAGE_SOURCE,

                type,

                ...extra,
            },
            "*"
        );
    }

    function cleanReminderTitle(
        displayText
    ) {
        return String(
            displayText ||
            ""
        )
            .replace(
                /^Hey\s*[—-]\s*/i,
                ""
            )
            .replace(
                /\s+now\.\s*$/i,
                ""
            )
            .trim();
    }

    function waitForWidget() {
        const shell =
            document.getElementById(
                "kelsie-shell"
            );

        const orb =
            document.getElementById(
                "kelsie-orb"
            );

        const closeButton =
            document.getElementById(
                "close-widget"
            );

        const reminderPopover =
            document.getElementById(
                "orb-reminder-popover"
            );

        const reminderTitle =
            document.getElementById(
                "orb-reminder-title"
            );

        const reminderDone =
            document.getElementById(
                "orb-reminder-done"
            );

        const reminderDismiss =
            document.getElementById(
                "orb-reminder-close"
            );

        const chatForm =
            document.getElementById(
                "chat-form"
            );

        const chatInput =
            document.getElementById(
                "chat-input"
            );

        const sendButton =
            document.getElementById(
                "send-message"
            );

        const messageList =
            document.getElementById(
                "message-list"
            );

        const typingRow =
            document.getElementById(
                "typing-row"
            );

        if (
            !shell ||
            !orb ||
            !closeButton ||
            !reminderPopover ||
            !reminderTitle ||
            !reminderDone ||
            !reminderDismiss ||
            !chatForm ||
            !chatInput ||
            !sendButton ||
            !messageList ||
            !typingRow
        ) {
            window.setTimeout(
                waitForWidget,
                100
            );

            return;
        }

        originalChatPlaceholder =
            chatInput.getAttribute(
                "placeholder"
            ) ||
            "Message Kelsie";

        function scrollMessagesToBottom() {
            window.requestAnimationFrame(
                () => {
                    messageList.scrollTop =
                        messageList.scrollHeight;
                }
            );
        }

        function appendBridgeMessage(
            role,
            text
        ) {
            const row =
                document.createElement(
                    "div"
                );

            row.className =
                `message-row ${role}`;

            row.dataset.kelsiePageMessage =
                "true";

            const bubble =
                document.createElement(
                    "div"
                );

            bubble.className =
                "message-bubble";

            bubble.textContent =
                String(text || "");

            row.appendChild(bubble);

            messageList.insertBefore(
                row,
                typingRow
            );

            scrollMessagesToBottom();
        }

        function showBridgeTyping(
            visible
        ) {
            typingRow.classList.toggle(
                "visible",
                Boolean(visible)
            );

            scrollMessagesToBottom();
        }

        function removeTemporaryPageMessages() {
            messageList
                .querySelectorAll(
                    '[data-kelsie-page-message="true"]'
                )
                .forEach(
                    (element) => {
                        element.remove();
                    }
                );
        }

        function clearPageContext() {
            pageContext = null;
            pageQuestionBusy = false;

            chatInput.placeholder =
                originalChatPlaceholder;

            chatInput.removeAttribute(
                "data-kelsie-page-context"
            );

            showBridgeTyping(false);

            removeTemporaryPageMessages();
        }

        function activatePageContext(
            context
        ) {
            if (
                !context ||
                typeof context.text !== "string" ||
                !context.text.trim()
            ) {
                return;
            }

            pageContext = {
                title:
                    String(
                        context.title ||
                        ""
                    ).trim(),

                text:
                    String(
                        context.text ||
                        ""
                    ).trim(),
            };

            pageQuestionBusy = false;

            chatInput.placeholder =
                "Ask about this page";

            chatInput.setAttribute(
                "data-kelsie-page-context",
                "true"
            );

            ensureOpen();

            window.setTimeout(
                () => {
                    if (
                        !chatInput.disabled
                    ) {
                        chatInput.focus();
                    }
                },
                120
            );
        }

        function ensureOpen() {
            if (
                !extensionVisible ||
                shell.classList.contains(
                    "open"
                )
            ) {
                return;
            }

            orb.click();
        }

        function ensureClosed() {
            if (
                extensionVisible ||
                !shell.classList.contains(
                    "open"
                )
            ) {
                return;
            }

            suppressParentClose = true;

            closeButton.click();

            window.setTimeout(
                () => {
                    suppressParentClose =
                        false;
                },
                60
            );
        }

        function publishReminderState() {
            const reminderId =
                Number(
                    reminderPopover
                        .dataset
                        .reminderId
                );

            const displayText =
                String(
                    reminderTitle
                        .textContent ||
                    ""
                ).trim();

            const popoverVisible =
                reminderPopover
                    .getAttribute(
                        "aria-hidden"
                    ) !== "true";

            const hasReminder =
                popoverVisible &&
                Number.isFinite(
                    reminderId
                ) &&
                reminderId > 0 &&
                Boolean(
                    displayText
                );

            if (!hasReminder) {
                if (
                    lastReminderSignature !==
                    null
                ) {
                    lastReminderSignature =
                        null;

                    postToParent(
                        "KELSIE_EXTENSION_REMINDER_CLEAR"
                    );
                }

                return;
            }

            const signature =
                `${reminderId}:${displayText}`;

            if (
                signature ===
                lastReminderSignature
            ) {
                return;
            }

            lastReminderSignature =
                signature;

            postToParent(
                "KELSIE_EXTENSION_REMINDER_DUE",
                {
                    reminderId,

                    title:
                        cleanReminderTitle(
                            displayText
                        ),

                    displayText,
                }
            );
        }

        async function askPageQuestion(
            question
        ) {
            if (
                !pageContext ||
                pageQuestionBusy
            ) {
                return;
            }

            const cleanQuestion =
                String(
                    question ||
                    ""
                ).trim();

            if (!cleanQuestion) {
                return;
            }

            pageQuestionBusy = true;

            appendBridgeMessage(
                "user",
                cleanQuestion
            );

            chatInput.value = "";
            chatInput.disabled = true;
            sendButton.disabled = true;

            showBridgeTyping(true);

            try {
                const response =
                    await new Promise(
                        (
                            resolve,
                            reject
                        ) => {
                            chrome.runtime.sendMessage(
                                {
                                    type:
                                        "KELSIE_PAGE_ASSIST",

                                    payload: {
                                        action:
                                            "question",

                                        title:
                                            pageContext.title,

                                        text:
                                            pageContext.text,

                                        question:
                                            cleanQuestion,
                                    },
                                },
                                (result) => {
                                    const runtimeError =
                                        chrome.runtime.lastError;

                                    if (runtimeError) {
                                        reject(
                                            new Error(
                                                runtimeError.message
                                            )
                                        );
                                        return;
                                    }

                                    resolve(result);
                                }
                            );
                        }
                    );

                if (
                    !response ||
                    response.ok !== true ||
                    !response.data
                ) {
                    throw new Error(
                        response?.error ||
                        "Kelsie could not answer that from this page right now."
                    );
                }

                const answer =
                    String(
                        response.data.answer ||
                        ""
                    ).trim();

                if (!answer) {
                    throw new Error(
                        "Kelsie returned an empty answer."
                    );
                }

                appendBridgeMessage(
                    "assistant",
                    answer
                );
            } catch (error) {
                console.error(
                    "[Kelsie] Page question failed:",
                    error
                );

                appendBridgeMessage(
                    "assistant",
                    error instanceof Error &&
                    error.message
                        ? error.message
                        : "I couldn’t answer that from this page right now."
                );
            } finally {
                pageQuestionBusy = false;
                showBridgeTyping(false);

                chatInput.disabled = false;
                sendButton.disabled = true;

                chatInput.focus();
            }
        }

        closeButton.addEventListener(
            "click",
            () => {
                clearPageContext();

                if (
                    suppressParentClose
                ) {
                    return;
                }

                postToParent(
                    "KELSIE_EXTENSION_CLOSE"
                );
            }
        );

        chatForm.addEventListener(
            "submit",
            (event) => {
                if (!pageContext) {
                    return;
                }

                event.preventDefault();
                event.stopImmediatePropagation();

                askPageQuestion(
                    chatInput.value
                );
            },
            true
        );

        const shellObserver =
            new MutationObserver(
                () => {
                    if (
                        extensionVisible &&
                        !shell
                            .classList
                            .contains(
                                "open"
                            )
                    ) {
                        window.setTimeout(
                            ensureOpen,
                            90
                        );
                    }
                }
            );

        shellObserver.observe(
            shell,
            {
                attributes: true,
                attributeFilter: [
                    "class",
                ],
            }
        );

        const reminderObserver =
            new MutationObserver(
                publishReminderState
            );

        reminderObserver.observe(
            reminderPopover,
            {
                attributes: true,

                attributeFilter: [
                    "aria-hidden",
                    "data-reminder-id",
                ],

                childList: true,
                subtree: true,
            }
        );

        reminderObserver.observe(
            reminderTitle,
            {
                childList: true,
                characterData: true,
                subtree: true,
            }
        );

        window.addEventListener(
            "message",
            (event) => {
                if (
                    event.source !==
                    window.parent
                ) {
                    return;
                }

                const data =
                    event.data;

                if (
                    !data ||
                    data.source !==
                        PARENT_MESSAGE_SOURCE
                ) {
                    return;
                }

                if (
                    data.type ===
                    "KELSIE_EXTENSION_VISIBILITY"
                ) {
                    extensionVisible =
                        Boolean(
                            data.visible
                        );

                    if (
                        extensionVisible
                    ) {
                        ensureOpen();

                        lastReminderSignature =
                            null;

                        postToParent(
                            "KELSIE_EXTENSION_REMINDER_CLEAR"
                        );

                        return;
                    }

                    clearPageContext();
                    ensureClosed();

                    window.setTimeout(
                        publishReminderState,
                        80
                    );

                    return;
                }

                if (
                    data.type ===
                    "KELSIE_EXTENSION_PAGE_CONTEXT"
                ) {
                    activatePageContext(
                        data.context
                    );

                    return;
                }

                if (
                    data.type ===
                    "KELSIE_EXTENSION_REMINDER_ACTION"
                ) {
                    const requestedId =
                        Number(
                            data.reminderId
                        );

                    const activeId =
                        Number(
                            reminderPopover
                                .dataset
                                .reminderId
                        );

                    if (
                        !Number.isFinite(
                            requestedId
                        ) ||
                        requestedId <= 0 ||
                        requestedId !==
                            activeId
                    ) {
                        publishReminderState();
                        return;
                    }

                    if (
                        data.action ===
                        "done"
                    ) {
                        reminderDone.click();
                        return;
                    }

                    if (
                        data.action ===
                        "dismiss"
                    ) {
                        reminderDismiss.click();
                    }
                }
            }
        );

        postToParent(
            "KELSIE_EXTENSION_READY"
        );

        if (extensionVisible) {
            ensureOpen();
        } else {
            ensureClosed();
        }

        window.setTimeout(
            publishReminderState,
            100
        );
    }

    waitForWidget();
})();