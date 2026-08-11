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

  if (
    window
      .__KELSIE_KEEPING_IN_MIND_HOOK__
  ) {
    return;
  }

  window
    .__KELSIE_KEEPING_IN_MIND_HOOK__ =
    true;

  const PARENT_SOURCE =
    "kelsie-context-assist";

  const HOOK_SOURCE =
    "kelsie-keeping-hook";

  let activeCount = 0;
  let overlayOpen = false;
  let latestItems = [];

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

  function postToParent(
    type,
    payload = {}
  ) {
    try {
      window.parent.postMessage(
        {
          source:
            HOOK_SOURCE,

          type,

          ...payload,
        },
        "*"
      );
    } catch (_error) {
      // Best effort only.
    }
  }

  async function requestJson(
    path,
    payload
  ) {
    const response =
      await fetch(
        path,
        {
          method:
            "POST",

          headers: {
            Accept:
              "application/json",

            "Content-Type":
              "application/json",
          },

          body:
            JSON.stringify(
              payload
            ),
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
          : (
              "Kelsie context "
              + "request failed "
              + `(${response.status}).`
            );

      throw new Error(
        detail
      );
    }

    return data;
  }

  function rpcReply(
    requestId,
    ok,
    data = null,
    error = ""
  ) {
    postToParent(
      "KELSIE_CONTEXT_RPC_RESPONSE",
      {
        request_id:
          requestId,

        ok,

        data,

        error,
      }
    );
  }

  /*
   * Observe outgoing normal Kelsie chat.
   *
   * This does NOT block, replace or rewrite
   * the existing chat message.
   *
   * It only lets the parent extension know
   * that a message was sent so it can decide
   * locally whether it even looks like a
   * "keep this in mind" candidate.
   */
  const previousSend =
    WebSocket.prototype.send;

  WebSocket.prototype.send =
    function kelsieKeepingInMindSend(
      data
    ) {
      if (
        typeof data ===
        "string"
      ) {
        const message =
          clean(
            data,
            1800
          );

        if (message) {
          postToParent(
            "KELSIE_CONTEXT_CAPTURE_CANDIDATE",
            {
              message,
            }
          );
        }
      }

      return previousSend.call(
        this,
        data
      );
    };

  function formatSource(
    item
  ) {
    const domain =
      clean(
        item?.sourceDomain,
        120
      );

    if (domain) {
      return domain;
    }

    try {
      return item?.sourceUrl
        ? new URL(
            item.sourceUrl
          )
            .hostname
            .replace(
              /^www\./,
              ""
            )
        : "";
    } catch (_error) {
      return "";
    }
  }

  function installUI() {
    if (
      document.getElementById(
        "keeping-in-mind-button"
      )
    ) {
      return true;
    }

    const headerActions =
      document.querySelector(
        ".header-actions"
      );

    const chatCard =
      document.getElementById(
        "chat-card"
      );

    if (
      !headerActions
      || !chatCard
    ) {
      return false;
    }

    const style =
      document.createElement(
        "style"
      );

    style.id =
      "keeping-in-mind-style";

    style.textContent = `
      #keeping-in-mind-button {
        position: relative;
      }

      #keeping-in-mind-count {
        position: absolute;
        top: -2px;
        right: -2px;

        min-width: 13px;
        height: 13px;

        display: none;
        align-items: center;
        justify-content: center;

        padding: 0 3px;

        border:
          2px solid
          rgba(
            250,
            252,
            251,
            0.98
          );

        border-radius: 999px;

        background:
          linear-gradient(
            145deg,
            #9ca4b2,
            #737c8d
          );

        color: #ffffff;

        box-shadow:
          0 2px 7px
          rgba(
            62,
            69,
            83,
            0.24
          );

        font-size: 7px;
        font-weight: 760;
        line-height: 1;

        pointer-events: none;
      }

      #keeping-in-mind-count.visible {
        display: flex;
      }

      #keeping-in-mind-overlay {
        position: absolute;
        inset: 61px 0 0;

        z-index: 40;

        display: none;
        flex-direction: column;

        overflow: hidden;

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
              242,
              244,
              246,
              0.97
            )
          );

        backdrop-filter:
          blur(24px)
          saturate(1.03);

        -webkit-backdrop-filter:
          blur(24px)
          saturate(1.03);
      }

      #keeping-in-mind-overlay.visible {
        display: flex;
      }

      .kim-header {
        min-height: 52px;

        display: flex;
        align-items: center;
        justify-content:
          space-between;

        gap: 10px;

        padding:
          12px
          13px
          10px
          15px;

        border-bottom:
          1px solid
          rgba(
            31,
            35,
            38,
            0.07
          );
      }

      .kim-title-wrap {
        display: grid;
        gap: 2px;
        min-width: 0;
      }

      .kim-title {
        color: #30363d;

        font-size: 12px;
        font-weight: 660;

        letter-spacing:
          -0.012em;
      }

      .kim-subtitle {
        color: #92989f;

        font-size: 8.5px;
        line-height: 1.3;
      }

      #keeping-in-mind-close {
        width: 27px;
        height: 27px;

        display: grid;
        place-items: center;

        padding: 0;

        border: 0;
        border-radius: 50%;

        background:
          transparent;

        color: #979da4;

        cursor: pointer;

        font-size: 18px;
        font-weight: 300;
        line-height: 1;
      }

      #keeping-in-mind-close:hover {
        color: #555d65;

        background:
          rgba(
            57,
            64,
            72,
            0.045
          );
      }

      #keeping-in-mind-list {
        flex: 1;
        min-height: 0;

        overflow-y: auto;

        padding:
          12px
          13px
          16px;

        scrollbar-width: thin;

        scrollbar-color:
          rgba(
            120,
            126,
            132,
            0.28
          )
          transparent;
      }

      #keeping-in-mind-list::-webkit-scrollbar {
        width: 4px;
      }

      #keeping-in-mind-list::-webkit-scrollbar-thumb {
        border-radius: 99px;

        background:
          rgba(
            120,
            126,
            132,
            0.28
          );
      }

      .kim-empty {
        padding:
          56px
          24px;

        color: #8a9098;

        text-align: center;

        font-size: 10px;
        line-height: 1.55;
      }

      .kim-empty strong {
        display: block;

        margin-bottom: 6px;

        color: #515862;

        font-size: 11px;
        font-weight: 640;
      }

      .kim-item {
        padding:
          12px
          11px
          10px;

        border:
          1px solid
          rgba(
            55,
            62,
            70,
            0.065
          );

        border-radius: 16px;

        background:
          rgba(
            255,
            255,
            255,
            0.54
          );

        box-shadow:
          inset
          0 1px 0
          rgba(
            255,
            255,
            255,
            0.76
          );
      }

      .kim-item + .kim-item {
        margin-top: 8px;
      }

      .kim-item-title {
        color: #353b44;

        font-size: 10.5px;
        line-height: 1.35;
        font-weight: 640;
      }

      .kim-item-reason {
        margin-top: 4px;

        color: #6f7680;

        font-size: 9px;
        line-height: 1.45;
      }

      .kim-item-source {
        margin-top: 6px;

        color: #a0a5ab;

        font-size: 7.8px;
        line-height: 1.3;
      }

      .kim-item-actions {
        display: flex;
        align-items: center;

        gap: 5px;

        margin-top: 9px;
      }

      .kim-item-action {
        min-height: 25px;

        padding:
          0
          7px;

        border: 0;
        border-radius: 8px;

        background:
          transparent;

        color: #818892;

        cursor: pointer;

        font: inherit;
        font-size: 7.8px;
        font-weight: 610;
      }

      .kim-item-action:hover {
        color: #505861;

        background:
          rgba(
            61,
            68,
            77,
            0.045
          );
      }

      .kim-item-action.done {
        color: #66766f;
      }

      #keeping-in-mind-toast {
        position: absolute;

        left: 50%;
        bottom: 62px;

        z-index: 60;

        display: none;

        transform:
          translateX(-50%);

        padding:
          7px
          10px;

        border:
          1px solid
          rgba(
            53,
            60,
            68,
            0.07
          );

        border-radius: 999px;

        background:
          rgba(
            248,
            249,
            250,
            0.96
          );

        color: #646b74;

        box-shadow:
          0 8px 22px
          rgba(
            30,
            35,
            41,
            0.11
          );

        font-size: 8.5px;
        font-weight: 610;

        white-space: nowrap;

        pointer-events: none;
      }

      #keeping-in-mind-toast.visible {
        display: block;

        animation:
          kimToastIn
          160ms
          ease-out;
      }

      @keyframes kimToastIn {
        from {
          opacity: 0;

          transform:
            translateX(-50%)
            translateY(4px);
        }

        to {
          opacity: 1;

          transform:
            translateX(-50%)
            translateY(0);
        }
      }
    `;

    document.head.appendChild(
      style
    );

    const button =
      document.createElement(
        "button"
      );

    button.id =
      "keeping-in-mind-button";

    button.className =
      "header-icon-button";

    button.type =
      "button";

    button.setAttribute(
      "aria-label",
      (
        "Open things Kelsie "
        + "is keeping in mind"
      )
    );

    button.title =
      "Keeping in mind";

    button.innerHTML = `
      <svg
        viewBox="0 0 24 24"
        fill="none"
        aria-hidden="true"
      >
        <path
          d="
            M7.3 4.7h9.4v14.6
            L12 16.5l-4.7 2.8
            V4.7Z
          "
          stroke="currentColor"
          stroke-width="1.45"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
      </svg>

      <span
        id="keeping-in-mind-count"
        aria-hidden="true"
      ></span>
    `;

    const settingsButton =
      document.getElementById(
        "settings-button"
      );

    if (settingsButton) {
      headerActions.insertBefore(
        button,
        settingsButton
      );
    } else {
      headerActions.appendChild(
        button
      );
    }

    const overlay =
      document.createElement(
        "section"
      );

    overlay.id =
      "keeping-in-mind-overlay";

    overlay.setAttribute(
      "aria-label",
      (
        "Things Kelsie is "
        + "keeping in mind"
      )
    );

    overlay.innerHTML = `
      <div class="kim-header">
        <div class="kim-title-wrap">
          <div class="kim-title">
            Keeping in mind
          </div>

          <div class="kim-subtitle">
            Things you asked Kelsie
            to carry forward
          </div>
        </div>

        <button
          id="keeping-in-mind-close"
          type="button"
          aria-label="Close keeping in mind"
        >
          ×
        </button>
      </div>

      <div
        id="keeping-in-mind-list"
      ></div>
    `;

    const toast =
      document.createElement(
        "div"
      );

    toast.id =
      "keeping-in-mind-toast";

    toast.textContent =
      "Kept in mind";

    chatCard.append(
      overlay,
      toast
    );

    button.addEventListener(
      "click",
      () => {
        overlayOpen =
          !overlayOpen;

        overlay.classList.toggle(
          "visible",
          overlayOpen
        );

        if (overlayOpen) {
          postToParent(
            "KELSIE_CONTEXT_ITEMS_REQUEST"
          );
        }
      }
    );

    overlay
      .querySelector(
        "#keeping-in-mind-close"
      )
      ?.addEventListener(
        "click",
        () => {
          overlayOpen = false;

          overlay.classList.remove(
            "visible"
          );
        }
      );

    /*
     * If another normal header control is used,
     * get the context layer out of the way.
     */
    headerActions.addEventListener(
      "click",
      (event) => {
        const clicked =
          event.target.closest(
            "button"
          );

        if (
          !clicked
          || clicked.id ===
            "keeping-in-mind-button"
        ) {
          return;
        }

        overlayOpen = false;

        overlay.classList.remove(
          "visible"
        );
      }
    );

    document
      .getElementById(
        "close-widget"
      )
      ?.addEventListener(
        "click",
        () => {
          overlayOpen = false;

          overlay.classList.remove(
            "visible"
          );
        }
      );

    renderItems(
      latestItems
    );

    updateBadge(
      activeCount
    );

    return true;
  }

  function updateBadge(
    count
  ) {
    activeCount =
      Math.max(
        0,
        Number(
          count || 0
        )
      );

    const badge =
      document.getElementById(
        "keeping-in-mind-count"
      );

    if (!badge) {
      return;
    }

    badge.textContent =
      activeCount > 9
        ? "9+"
        : String(
            activeCount
          );

    badge.classList.toggle(
      "visible",
      activeCount > 0
    );
  }

  function showToast(
    text =
      "Kept in mind"
  ) {
    const toast =
      document.getElementById(
        "keeping-in-mind-toast"
      );

    if (!toast) {
      return;
    }

    toast.textContent =
      clean(
        text,
        80
      )
      || "Kept in mind";

    toast.classList.add(
      "visible"
    );

    window.clearTimeout(
      showToast.timer
    );

    showToast.timer =
      window.setTimeout(
        () => {
          toast.classList.remove(
            "visible"
          );
        },
        1800
      );
  }

  function renderItems(
    items
  ) {
    latestItems =
      Array.isArray(
        items
      )
        ? items
        : [];

    const list =
      document.getElementById(
        "keeping-in-mind-list"
      );

    if (!list) {
      return;
    }

    list.replaceChildren();

    if (
      latestItems.length ===
      0
    ) {
      const empty =
        document.createElement(
          "div"
        );

      empty.className =
        "kim-empty";

      const strong =
        document.createElement(
          "strong"
        );

      strong.textContent =
        "Nothing here yet.";

      const copy =
        document.createElement(
          "span"
        );

      copy.textContent =
        (
          "Try telling Kelsie things "
          + "like “I want to come back "
          + "to this” or “I need this "
          + "for my presentation.”"
        );

      empty.append(
        strong,
        copy
      );

      list.appendChild(
        empty
      );

      return;
    }

    latestItems.forEach(
      (item) => {
        const card =
          document.createElement(
            "article"
          );

        card.className =
          "kim-item";

        const title =
          document.createElement(
            "div"
          );

        title.className =
          "kim-item-title";

        title.textContent =
          clean(
            item.thing,
            220
          )
          || "Saved context";

        const reason =
          document.createElement(
            "div"
          );

        reason.className =
          "kim-item-reason";

        reason.textContent =
          clean(
            item.reason,
            420
          )
          || (
            "Kelsie is keeping "
            + "this available "
            + "for later."
          );

        const source =
          document.createElement(
            "div"
          );

        source.className =
          "kim-item-source";

        source.textContent =
          formatSource(
            item
          );

        if (
          !source.textContent
        ) {
          source.hidden = true;
        }

        const actions =
          document.createElement(
            "div"
          );

        actions.className =
          "kim-item-actions";

        if (
          item.sourceUrl
        ) {
          const open =
            document.createElement(
              "button"
            );

          open.className =
            "kim-item-action";

          open.type =
            "button";

          open.textContent =
            "Open source";

          open.addEventListener(
            "click",
            () => {
              window.open(
                item.sourceUrl,
                "_blank",
                "noopener,noreferrer"
              );
            }
          );

          actions.appendChild(
            open
          );
        }

        const done =
          document.createElement(
            "button"
          );

        done.className =
          "kim-item-action done";

        done.type =
          "button";

        done.textContent =
          "Done";

        done.addEventListener(
          "click",
          () => {
            postToParent(
              "KELSIE_CONTEXT_ITEM_ACTION",
              {
                action:
                  "complete",

                item_id:
                  item.id,
              }
            );
          }
        );

        const forget =
          document.createElement(
            "button"
          );

        forget.className =
          "kim-item-action";

        forget.type =
          "button";

        forget.textContent =
          "Forget";

        forget.addEventListener(
          "click",
          () => {
            postToParent(
              "KELSIE_CONTEXT_ITEM_ACTION",
              {
                action:
                  "forget",

                item_id:
                  item.id,
              }
            );
          }
        );

        actions.append(
          done,
          forget
        );

        card.append(
          title,
          reason,
          source,
          actions
        );

        list.appendChild(
          card
        );
      }
    );
  }

  function ensureUI() {
    if (
      installUI()
    ) {
      return;
    }

    window.setTimeout(
      ensureUI,
      120
    );
  }

  window.addEventListener(
    "message",
    async (
      event
    ) => {
      if (
        event.source !==
        window.parent
      ) {
        return;
      }

      const data =
        event.data;

      if (
        !data
        || data.source !==
          PARENT_SOURCE
      ) {
        return;
      }

      if (
        data.type ===
        "KELSIE_CONTEXT_INTERPRET_REQUEST"
      ) {
        try {
          const result =
            await requestJson(
              "/api/context/interpret",
              data.payload || {}
            );

          rpcReply(
            data.request_id,
            true,
            result
          );
        } catch (error) {
          rpcReply(
            data.request_id,
            false,
            null,
            error instanceof
              Error
              ? error.message
              : (
                  "Kelsie could not "
                  + "interpret that context."
                )
          );
        }

        return;
      }

      if (
        data.type ===
        "KELSIE_CONTEXT_MATCH_REQUEST"
      ) {
        try {
          const result =
            await requestJson(
              "/api/context/match",
              data.payload || {}
            );

          rpcReply(
            data.request_id,
            true,
            result
          );
        } catch (error) {
          rpcReply(
            data.request_id,
            false,
            null,
            error instanceof
              Error
              ? error.message
              : (
                  "Kelsie could not "
                  + "match context "
                  + "right now."
                )
          );
        }

        return;
      }

      if (
        data.type ===
        "KELSIE_CONTEXT_ITEMS_RESPONSE"
      ) {
        renderItems(
          data.items
        );

        updateBadge(
          data.active_count
        );

        return;
      }

      if (
        data.type ===
        "KELSIE_CONTEXT_CAPTURED"
      ) {
        updateBadge(
          data.active_count
        );

        showToast(
          "Kept in mind"
        );

        if (overlayOpen) {
          postToParent(
            "KELSIE_CONTEXT_ITEMS_REQUEST"
          );
        }

        return;
      }

      if (
        data.type ===
        "KELSIE_CONTEXT_ITEMS_CHANGED"
      ) {
        updateBadge(
          data.active_count
        );

        if (overlayOpen) {
          postToParent(
            "KELSIE_CONTEXT_ITEMS_REQUEST"
          );
        }

        return;
      }

      if (
        data.type ===
        "KELSIE_CONTEXT_OPEN_KEEPING"
      ) {
        const overlay =
          document.getElementById(
            "keeping-in-mind-overlay"
          );

        if (overlay) {
          overlayOpen =
            true;

          overlay.classList.add(
            "visible"
          );

          postToParent(
            "KELSIE_CONTEXT_ITEMS_REQUEST"
          );
        }
      }
    }
  );

  ensureUI();

  postToParent(
    "KELSIE_KEEPING_HOOK_READY"
  );

  console.log(
    "[Kelsie Context] "
    + "Keeping-in-mind hook ready."
  );
})();