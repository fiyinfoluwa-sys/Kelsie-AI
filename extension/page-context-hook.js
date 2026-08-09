(() => {
  const params = new URLSearchParams(window.location.search);

  if (params.get("kelsie_extension_embed") !== "1") {
    return;
  }

  if (window.__KELSIE_PAGE_CONTEXT_HOOK__) {
    return;
  }

  window.__KELSIE_PAGE_CONTEXT_HOOK__ = true;

  const MAX_RECENT_CONVERSATION = 10;
  const MAX_PAGE_TURNS = 8;
  const MAX_PAGE_TEXT_CHARS = 18000;

  let pageContext = null;
  let latestSocket = null;
  let lastReplySource = "general";
  let recentConversation = [];
  let pageTurns = [];
  let contextualRequestBusy = false;

  const originalSend = WebSocket.prototype.send;

  function clean(value, limit = 1000) {
    return String(value || "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, limit);
  }

  function trimHistory(list, limit) {
    if (list.length > limit) {
      list.splice(
        0,
        list.length - limit
      );
    }
  }

  function addRecent(role, content) {
    const text = clean(
      content,
      1400
    );

    if (!text) {
      return;
    }

    recentConversation.push({
      role,
      content: text,
    });

    trimHistory(
      recentConversation,
      MAX_RECENT_CONVERSATION
    );
  }

  function addPageTurn(role, content) {
    const text = clean(
      content,
      1400
    );

    if (!text) {
      return;
    }

    pageTurns.push({
      role,
      content: text,
    });

    trimHistory(
      pageTurns,
      MAX_PAGE_TURNS
    );
  }

  function isObviousCasual(message) {
    const text = clean(
      message,
      200
    )
      .toLowerCase()
      .replace(/[!?.,]+$/g, "")
      .trim();

    return /^(?:hi|hello|hey|hey there|thanks|thank you|thank you so much|okay|ok|okay thanks|ok thanks|oh okay|oh ok|got it|makes sense|cool|nice|lol|haha|nevermind|never mind|bye|goodbye|good night|goodnight|sure|yep|yeah|yes|nope|no)$/.test(
      text
    );
  }

  function isNormalKelsieAction(message) {
    const text = clean(
      message,
      1200
    ).toLowerCase();

    return (
      /\bremind me\b/.test(text) ||
      /\bset (?:a |an |the )?reminder\b/.test(text) ||
      /\bcreate (?:a |an |the )?reminder\b/.test(text) ||
      /\bdon['’]?t let me forget\b/.test(text) ||
      /\bwhat time is it\b/.test(text) ||
      /\bwhat(?:'s| is) the time\b/.test(text) ||
      /\bwhat(?:'s| is) the date\b/.test(text) ||
      /\bwhat day is it\b/.test(text)
    );
  }

  function explicitlyReferencesPage(
    message
  ) {
    const text = clean(
      message,
      1400
    ).toLowerCase();

    return (
      /\b(?:this|the) (?:page|article|guide|tutorial|study|story|post|document|section|website)\b/.test(
        text
      ) ||
      /\baccording to (?:this|the)\b/.test(
        text
      ) ||
      /\b(?:the|this) author\b/.test(
        text
      ) ||
      /\b(?:what|where|when|why|how) does (?:it|this) (?:say|mean|explain|recommend|describe|define|show)\b/.test(
        text
      ) ||
      /\bwhat (?:does|did) (?:it|this) (?:say|recommend|mean)\b/.test(
        text
      ) ||
      /\bfrom (?:this|the) (?:page|article|guide|study|tutorial)\b/.test(
        text
      ) ||
      /\bback to (?:the|this) (?:page|article|guide|study|tutorial)\b/.test(
        text
      )
    );
  }

  function isTopicShift(message) {
    const text = clean(
      message,
      300
    ).toLowerCase();

    return /^(?:anyway|anyways|separate question|unrelated|different question|changing topics|new question)\b/.test(
      text
    );
  }

  function looksLikeIndependentQuestion(
    message
  ) {
    const text = clean(
      message,
      500
    ).toLowerCase();

    if (
      explicitlyReferencesPage(
        text
      )
    ) {
      return false;
    }

    return /^(?:what|who|where|when) (?:is|are|was|were) (?!that\b|this\b|it\b)/.test(
      text
    );
  }

  function looksLikePageFollowUp(
    message
  ) {
    const text = clean(
      message,
      500
    ).toLowerCase();

    if (
      text.split(/\s+/).length > 20
    ) {
      return false;
    }

    return (
      /^(?:why|how|what about|what does that|what do you mean|is that|does that|would that|could that|so |then |which |where |when )/.test(
        text
      ) ||
      /\b(?:that|this|it)\b/.test(
        text
      )
    );
  }

  function routeMessage(message) {
    if (!pageContext) {
      return "general";
    }

    /*
     * Greetings, acknowledgements and conversational closings
     * must never become "page questions".
     */
    if (isObviousCasual(message)) {
      if (
        pageContext.mode ===
        "scaffold"
      ) {
        pageContext.awaitingScaffoldAnswer =
          false;
      }

      return "general";
    }

    /*
     * Reminders and ordinary Kelsie system actions always stay
     * with the original Kelsie backend.
     */
    if (
      isNormalKelsieAction(
        message
      )
    ) {
      return "general";
    }

    /*
     * Explicit topic change releases the scaffold.
     */
    if (isTopicShift(message)) {
      pageContext.awaitingScaffoldAnswer =
        false;

      return "general";
    }

    /*
     * Strong explicit page reference.
     */
    if (
      explicitlyReferencesPage(
        message
      )
    ) {
      return "page";
    }

    /*
     * If Kelsie just asked a scaffold question, a short answer is
     * treated as an answer to that question unless it is clearly
     * an entirely different question.
     */
    if (
      pageContext.mode ===
        "scaffold" &&
      pageContext.awaitingScaffoldAnswer
    ) {
      if (
        looksLikeIndependentQuestion(
          message
        )
      ) {
        pageContext.awaitingScaffoldAnswer =
          false;

        return "general";
      }

      return "page";
    }

    /*
     * Short references immediately after a page-grounded answer
     * can continue that page conversation.
     */
    if (
      lastReplySource === "page" &&
      looksLikePageFollowUp(
        message
      )
    ) {
      /*
       * These usually require both the page and broader knowledge.
       *
       * Example:
       * Page says the ratio is 1:1.75.
       * User asks "is that normal?"
       */
      if (
        /\b(?:is that normal|is that true|is that accurate|does that usually|would that normally)\b/i.test(
          message
        )
      ) {
        return "blended";
      }

      return "page";
    }

    /*
     * Crucial default:
     * Having a page available does not mean Kelsie should use it.
     */
    return "general";
  }

  function parseSocketMessage(
    raw
  ) {
    try {
      const parsed =
        JSON.parse(raw);

      if (
        parsed &&
        typeof parsed ===
          "object"
      ) {
        return parsed;
      }
    } catch (_error) {
      return null;
    }

    return null;
  }

  function appendTemporaryAssistantBubble(
    message
  ) {
    const messageList =
      document.getElementById(
        "message-list"
      );

    const typingRow =
      document.getElementById(
        "typing-row"
      );

    if (!messageList) {
      return false;
    }

    const row =
      document.createElement(
        "div"
      );

    row.className =
      "message-row assistant";

    const bubble =
      document.createElement(
        "div"
      );

    bubble.className =
      "message-bubble";

    bubble.textContent =
      clean(
        message,
        2200
      );

    row.appendChild(
      bubble
    );

    if (typingRow) {
      messageList.insertBefore(
        row,
        typingRow
      );
    } else {
      messageList.appendChild(
        row
      );
    }

    messageList.scrollTop =
      messageList.scrollHeight;

    return true;
  }

  function dispatchAssistant(
    socket,
    message,
    extra = {}
  ) {
    const payload = {
      type: "assistant",
      message: clean(
        message,
        2200
      ),
      ...extra,
    };

    socket.dispatchEvent(
      new MessageEvent(
        "message",
        {
          data:
            JSON.stringify(
              payload
            ),
        }
      )
    );
  }

  function installSocketTracking(
    socket
  ) {
    if (
      socket
        .__kelsiePageTracked
    ) {
      return;
    }

    socket
      .__kelsiePageTracked =
      true;

    socket.addEventListener(
      "message",
      (event) => {
        const parsed =
          parseSocketMessage(
            event.data
          );

        if (!parsed) {
          return;
        }

        const assistantText =
          clean(
            parsed.message ||
              parsed.reply,
            2200
          );

        if (!assistantText) {
          return;
        }

        if (
          parsed.page_context ===
          true
        ) {
          lastReplySource =
            "page";

          return;
        }

        lastReplySource =
          "general";

        addRecent(
          "assistant",
          assistantText
        );
      }
    );
  }

  async function sendContextualRequest(
    socket,
    message,
    route
  ) {
    if (
      !pageContext ||
      contextualRequestBusy
    ) {
      return;
    }

    contextualRequestBusy =
      true;

    addPageTurn(
      "user",
      message
    );

    try {
      const response =
        await fetch(
          "/api/reading/chat",
          {
            method: "POST",
            headers: {
              Accept:
                "application/json",
              "Content-Type":
                "application/json",
            },
            body:
              JSON.stringify({
                title:
                  pageContext.title,

                url:
                  pageContext.url,

                text:
                  pageContext.text,

                content_type:
                  pageContext
                    .contentType,

                structure:
                  pageContext
                    .structure,

                message,

                route,

                mode:
                  pageContext.mode,

                active_question:
                  pageContext
                    .activeQuestion,

                recent_conversation:
                  recentConversation,

                page_turns:
                  pageTurns,
              }),
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
        throw new Error(
          typeof data?.detail ===
            "string"
            ? data.detail
            : "Kelsie could not use the page context right now."
        );
      }

      const reply =
        clean(
          data?.message,
          1800
        );

      const nextQuestion =
        clean(
          data?.next_question,
          700
        );

      if (!reply) {
        throw new Error(
          "Kelsie returned an empty page-context response."
        );
      }

      let visibleReply =
        reply;

      if (
        nextQuestion &&
        !reply.endsWith("?")
      ) {
        visibleReply =
          `${reply}\n\n${nextQuestion}`;
      }

      pageContext.activeQuestion =
        nextQuestion;

      pageContext.awaitingScaffoldAnswer =
        Boolean(
          pageContext.mode ===
            "scaffold" &&
          nextQuestion
        );

      addPageTurn(
        "assistant",
        visibleReply
      );

      addRecent(
        "user",
        message
      );

      addRecent(
        "assistant",
        visibleReply
      );

      lastReplySource =
        "page";

      dispatchAssistant(
        socket,
        visibleReply,
        {
          page_context: true,

          page_route:
            data?.route ||
            route,
        }
      );
    } catch (error) {
      console.error(
        "[Kelsie Page Context] Request failed:",
        error
      );

      dispatchAssistant(
        socket,
        error instanceof Error
          ? error.message
          : "Kelsie could not use the page context right now.",
        {
          page_context: true,
        }
      );
    } finally {
      contextualRequestBusy =
        false;
    }
  }

  /*
   * Kelsie's existing widget already sends normal chat through a
   * WebSocket. We intercept only messages that genuinely need page
   * context. Everything else continues through the original Kelsie
   * conversation pipeline untouched.
   */
  WebSocket.prototype.send =
    function kelsiePageAwareSend(
      data
    ) {
      latestSocket = this;

      installSocketTracking(
        this
      );

      if (
        typeof data !==
          "string" ||
        !pageContext
      ) {
        return originalSend.call(
          this,
          data
        );
      }

      const message =
        clean(
          data,
          1800
        );

      if (!message) {
        return originalSend.call(
          this,
          data
        );
      }

      const route =
        routeMessage(
          message
        );

      if (
        route === "general"
      ) {
        addRecent(
          "user",
          message
        );

        return originalSend.call(
          this,
          data
        );
      }

      sendContextualRequest(
        this,
        message,
        route
      );

      return undefined;
    };

  function createChip() {
    let chip =
      document.getElementById(
        "kelsie-page-context-chip"
      );

    if (chip) {
      return chip;
    }

    const chatView =
      document.getElementById(
        "chat-view"
      );

    const messageList =
      document.getElementById(
        "message-list"
      );

    if (
      !chatView ||
      !messageList
    ) {
      return null;
    }

    if (
      !document.getElementById(
        "kelsie-page-context-style"
      )
    ) {
      const style =
        document.createElement(
          "style"
        );

      style.id =
        "kelsie-page-context-style";

      style.textContent = `
        #kelsie-page-context-chip {
          display: none;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
          margin: 9px 13px 0;
          padding: 6px 8px 6px 10px;
          border: 1px solid rgba(83, 91, 101, 0.08);
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.52);
          color: #737a83;
          box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.72);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          flex: 0 0 auto;
        }

        #kelsie-page-context-chip.visible {
          display: flex;
        }

        .kelsie-page-context-copy {
          min-width: 0;
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 9px;
          font-weight: 580;
          line-height: 1;
        }

        .kelsie-page-context-dot {
          width: 5px;
          height: 5px;
          flex: 0 0 auto;
          border-radius: 50%;
          background: #808899;
          box-shadow:
            0 0 0 2px
            rgba(128, 136, 153, 0.09);
        }

        #kelsie-page-context-title {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          max-width: 208px;
        }

        #kelsie-page-context-close {
          width: 22px;
          height: 22px;
          min-width: 22px;
          display: grid;
          place-items: center;
          padding: 0;
          border: 0;
          border-radius: 50%;
          background: transparent;
          color: #9a9fa6;
          cursor: pointer;
          font-size: 15px;
          font-weight: 300;
          line-height: 1;
        }

        #kelsie-page-context-close:hover {
          background:
            rgba(
              69,
              76,
              84,
              0.05
            );

          color: #626970;
        }
      `;

      document.head.appendChild(
        style
      );
    }

    chip =
      document.createElement(
        "div"
      );

    chip.id =
      "kelsie-page-context-chip";

    chip.innerHTML = `
      <span
        class="kelsie-page-context-copy"
      >
        <span
          class="kelsie-page-context-dot"
          aria-hidden="true"
        ></span>

        <span
          id="kelsie-page-context-title"
        >
          Page available
        </span>
      </span>

      <button
        id="kelsie-page-context-close"
        type="button"
        aria-label="Remove page context"
        title="Remove page context"
      >
        ×
      </button>
    `;

    chatView.insertBefore(
      chip,
      messageList
    );

    chip
      .querySelector(
        "#kelsie-page-context-close"
      )
      ?.addEventListener(
        "click",
        clearPageContext
      );

    return chip;
  }

  function renderChip() {
    const chip =
      createChip();

    if (!chip) {
      window.setTimeout(
        renderChip,
        120
      );

      return;
    }

    const title =
      chip.querySelector(
        "#kelsie-page-context-title"
      );

    if (pageContext) {
      chip.classList.add(
        "visible"
      );

      const pageTitle =
        clean(
          pageContext.title,
          90
        );

      if (title) {
        title.textContent =
          pageTitle
            ? `Page available · ${pageTitle}`
            : "Page available";
      }

      return;
    }

    chip.classList.remove(
      "visible"
    );

    if (title) {
      title.textContent =
        "Page available";
    }
  }

  function clearPageContext() {
    pageContext = null;
    pageTurns = [];
    lastReplySource =
      "general";

    contextualRequestBusy =
      false;

    renderChip();

    try {
      window.parent.postMessage(
        {
          source:
            "kelsie-page-context-hook",

          type:
            "KELSIE_PAGE_CONTEXT_CLEARED",
        },
        "*"
      );
    } catch (_error) {
      // Best effort only.
    }
  }

  function activatePageContext(
    data
  ) {
    const context =
      data?.context;

    if (
      !context ||
      typeof context.text !==
        "string"
    ) {
      return;
    }

    pageContext = {
      title:
        clean(
          context.title,
          300
        ),

      url:
        clean(
          context.url,
          1200
        ),

      text:
        String(
          context.text || ""
        ).slice(
          0,
          MAX_PAGE_TEXT_CHARS
        ),

      contentType:
        clean(
          context.content_type ||
            "other",
          40
        ),

      structure:
        Array.isArray(
          context.structure
        )
          ? context.structure.slice(
              0,
              6
            )
          : [],

      mode:
        context.mode ===
          "scaffold"
          ? "scaffold"
          : "questions",

      activeQuestion:
        clean(
          context.active_question,
          700
        ),

      awaitingScaffoldAnswer:
        Boolean(
          context.mode ===
            "scaffold" &&
          context.active_question
        ),
    };

    pageTurns =
      Array.isArray(
        context.page_turns
      )
        ? context.page_turns
            .slice(
              -MAX_PAGE_TURNS
            )
            .map(
              (turn) => ({
                role:
                  turn?.role ===
                    "assistant"
                    ? "assistant"
                    : "user",

                content:
                  clean(
                    turn?.content,
                    1400
                  ),
              })
            )
            .filter(
              (turn) =>
                turn.content
            )
        : [];

    renderChip();

    const introMessage =
      clean(
        data?.intro_message,
        1100
      );

    if (!introMessage) {
      return;
    }

    const socket =
      latestSocket;

    addPageTurn(
      "assistant",
      introMessage
    );

    addRecent(
      "assistant",
      introMessage
    );

    lastReplySource =
      "page";

    if (socket) {
      dispatchAssistant(
        socket,
        introMessage,
        {
          page_context: true,
        }
      );
    } else {
      appendTemporaryAssistantBubble(
        introMessage
      );
    }
  }

  window.addEventListener(
    "message",
    (event) => {
      const data =
        event.data;

      if (
        !data ||
        data.source !==
          "kelsie-reading-assist"
      ) {
        return;
      }

      if (
        data.type ===
        "KELSIE_PAGE_CONTEXT_ACTIVATE"
      ) {
        activatePageContext(
          data
        );
      }

      if (
        data.type ===
        "KELSIE_PAGE_CONTEXT_CLEAR"
      ) {
        clearPageContext();
      }
    }
  );

  document.addEventListener(
    "DOMContentLoaded",
    renderChip,
    {
      once: true,
    }
  );

  window.setTimeout(
    renderChip,
    500
  );

  console.log(
    "[Kelsie Page Context] Main-world hook ready."
  );
})();