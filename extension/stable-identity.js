(() => {
  const ROOT_ID =
    "kelsie-extension-root";

  const FRAME_ID =
    "kelsie-widget-frame";

  const STORAGE_KEY =
    "kelsieStableUserIdV1";

  const USER_ID_PARAM =
    "kelsie_user_id";

  const WIDGET_ORIGIN =
    "http://127.0.0.1:8000";

  const HOOK_SOURCE =
    "kelsie-stable-identity-hook";

  if (
    window.top !==
    window.self
  ) {
    return;
  }

  if (
    window.__KELSIE_STABLE_IDENTITY__
  ) {
    return;
  }

  window.__KELSIE_STABLE_IDENTITY__ =
    true;

  let canonicalUserId =
    "";

  let widgetFrame =
    null;

  let srcObserver =
    null;

  let framePollTimer =
    null;

  function cleanUserId(
    value
  ) {
    const cleaned =
      String(
        value || ""
      ).trim();

    if (!cleaned) {
      return "";
    }

    if (
      !/^[A-Za-z0-9._:-]{8,180}$/.test(
        cleaned
      )
    ) {
      return "";
    }

    return cleaned;
  }

  async function readCanonicalUserId() {
    try {
      const result =
        await chrome
          .storage
          .local
          .get({
            [STORAGE_KEY]:
              "",
          });

      return cleanUserId(
        result[
          STORAGE_KEY
        ]
      );
    } catch (error) {
      console.error(
        "[Kelsie Identity] Could not read extension identity:",
        error
      );

      return "";
    }
  }

  async function saveCanonicalUserId(
    userId
  ) {
    const cleaned =
      cleanUserId(
        userId
      );

    if (!cleaned) {
      return false;
    }

    try {
      await chrome
        .storage
        .local
        .set({
          [STORAGE_KEY]:
            cleaned,
        });

      canonicalUserId =
        cleaned;

      return true;
    } catch (error) {
      console.error(
        "[Kelsie Identity] Could not save extension identity:",
        error
      );

      return false;
    }
  }

  function iframeUrlWithCanonicalId(
    rawUrl
  ) {
    if (
      !canonicalUserId
    ) {
      return "";
    }

    try {
      const url =
        new URL(
          rawUrl ||
            (
              "http://127.0.0.1:8000/"
              + "static/widget.html"
              + "?kelsie_extension_embed=1"
            )
        );

      if (
        url.origin !==
        WIDGET_ORIGIN
      ) {
        return "";
      }

      url.searchParams.set(
        USER_ID_PARAM,
        canonicalUserId
      );

      return url.toString();
    } catch (_error) {
      return "";
    }
  }

  function frameAlreadyUsesCanonicalId() {
    if (
      !widgetFrame ||
      !canonicalUserId
    ) {
      return false;
    }

    const raw =
      widgetFrame
        .getAttribute(
          "src"
        ) ||
      "";

    if (!raw) {
      return false;
    }

    try {
      const url =
        new URL(
          raw,
          window.location.href
        );

      return (
        url.origin ===
          WIDGET_ORIGIN &&
        cleanUserId(
          url.searchParams.get(
            USER_ID_PARAM
          )
        ) ===
          canonicalUserId
      );
    } catch (_error) {
      return false;
    }
  }

  function enforceCanonicalIdentity() {
    if (
      !widgetFrame ||
      !canonicalUserId
    ) {
      return;
    }

    const currentSrc =
      widgetFrame
        .getAttribute(
          "src"
        ) ||
      "";

    if (!currentSrc) {
      return;
    }

    if (
      frameAlreadyUsesCanonicalId()
    ) {
      return;
    }

    const nextSrc =
      iframeUrlWithCanonicalId(
        currentSrc
      );

    if (!nextSrc) {
      return;
    }

    console.debug(
      "[Kelsie Identity] Applying the stable extension identity."
    );

    widgetFrame.setAttribute(
      "src",
      nextSrc
    );
  }

  function attachToFrame(
    frame
  ) {
    if (
      !frame ||
      frame === widgetFrame
    ) {
      return;
    }

    widgetFrame =
      frame;

    srcObserver
      ?.disconnect();

    /*
     * This observer watches ONE attribute on ONE iframe.
     *
     * It is not the expensive subtree observer we removed
     * from contextual resurfacing.
     */
    srcObserver =
      new MutationObserver(
        () => {
          if (
            canonicalUserId
          ) {
            enforceCanonicalIdentity();
          }
        }
      );

    srcObserver.observe(
      widgetFrame,
      {
        attributes:
          true,

        attributeFilter:
          [
            "src",
          ],
      }
    );

    if (
      canonicalUserId
    ) {
      enforceCanonicalIdentity();
    }
  }

  function findWidgetFrame() {
    const host =
      document.getElementById(
        ROOT_ID
      );

    const frame =
      host
        ?.shadowRoot
        ?.getElementById(
          FRAME_ID
        );

    if (frame) {
      attachToFrame(
        frame
      );

      return true;
    }

    return false;
  }

  function beginFrameDiscovery() {
    if (
      findWidgetFrame()
    ) {
      return;
    }

    let attempts =
      0;

    framePollTimer =
      window.setInterval(
        () => {
          attempts +=
            1;

          if (
            findWidgetFrame() ||
            attempts >=
              80
          ) {
            window.clearInterval(
              framePollTimer
            );

            framePollTimer =
              null;
          }
        },
        100
      );
  }

  window.addEventListener(
    "message",
    async (
      event
    ) => {
      if (
        !widgetFrame ||
        event.source !==
          widgetFrame
            .contentWindow
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
          HOOK_SOURCE
      ) {
        return;
      }

      if (
        data.type !==
        "KELSIE_IDENTITY_CANDIDATE"
      ) {
        return;
      }

      const candidate =
        cleanUserId(
          data.user_id
        );

      if (!candidate) {
        return;
      }

      /*
       * First migration:
       *
       * Adopt the profile that already exists on
       * the page the user is currently using.
       *
       * This lets us preserve existing profile,
       * conversation history and reminder ownership
       * instead of generating yet another identity.
       */
      if (
        !canonicalUserId
      ) {
        const saved =
          await saveCanonicalUserId(
            candidate
          );

        if (saved) {
          console.log(
            "[Kelsie Identity] Existing Kelsie profile adopted as the stable extension identity."
          );
        }

        return;
      }

      /*
       * If this page's partition had an old/different
       * ID, reload only the Kelsie iframe using the
       * canonical extension ID.
       */
      if (
        candidate !==
        canonicalUserId
      ) {
        enforceCanonicalIdentity();
      }
    }
  );

  window.addEventListener(
    "pagehide",
    () => {
      srcObserver
        ?.disconnect();

      if (
        framePollTimer
      ) {
        window.clearInterval(
          framePollTimer
        );
      }
    },
    {
      once:
        true,
    }
  );

  (async () => {
    canonicalUserId =
      await readCanonicalUserId();

    beginFrameDiscovery();

    if (
      canonicalUserId
    ) {
      console.debug(
        "[Kelsie Identity] Stable extension identity loaded."
      );
    } else {
      console.debug(
        "[Kelsie Identity] No stable identity yet; the current Kelsie profile will be adopted."
      );
    }
  })();
})();