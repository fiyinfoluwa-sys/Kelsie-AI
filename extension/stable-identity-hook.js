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
      .__KELSIE_STABLE_IDENTITY_HOOK__
  ) {
    return;
  }

  window
    .__KELSIE_STABLE_IDENTITY_HOOK__ =
    true;

  const STORAGE_KEY =
    "kelsie_user_id";

  const USER_ID_PARAM =
    "kelsie_user_id";

  const SOURCE =
    "kelsie-stable-identity-hook";

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

  function readLocalUserId() {
    try {
      return cleanUserId(
        window
          .localStorage
          .getItem(
            STORAGE_KEY
          )
      );
    } catch (_error) {
      return "";
    }
  }

  function writeLocalUserId(
    userId
  ) {
    const cleaned =
      cleanUserId(
        userId
      );

    if (!cleaned) {
      return "";
    }

    try {
      window
        .localStorage
        .setItem(
          STORAGE_KEY,
          cleaned
        );

      return cleaned;
    } catch (_error) {
      return "";
    }
  }

  function announce(
    userId
  ) {
    const cleaned =
      cleanUserId(
        userId
      );

    if (!cleaned) {
      return;
    }

    try {
      window.parent
        .postMessage(
          {
            source:
              SOURCE,

            type:
              "KELSIE_IDENTITY_CANDIDATE",

            user_id:
              cleaned,
          },
          "*"
        );
    } catch (_error) {
      // Best effort only.
    }
  }

  /*
   * If the extension already has its one canonical
   * user ID, stable-identity.js places it in the
   * iframe URL.
   *
   * This script runs at document_start and writes
   * the ID BEFORE profile.js initializes.
   */
  const suppliedUserId =
    cleanUserId(
      params.get(
        USER_ID_PARAM
      )
    );

  if (
    suppliedUserId
  ) {
    writeLocalUserId(
      suppliedUserId
    );

    announce(
      suppliedUserId
    );
  } else {
    /*
     * Migration path:
     *
     * On the first page after this update,
     * preserve whichever Kelsie identity is
     * already stored for that page.
     */
    announce(
      readLocalUserId()
    );
  }

  /*
   * profile.js may generate the ID slightly later
   * on a completely fresh page.
   *
   * Poll briefly so the extension can capture it.
   */
  let attempts =
    0;

  let lastAnnounced =
    suppliedUserId ||
    "";

  const timer =
    window.setInterval(
      () => {
        attempts +=
          1;

        const current =
          readLocalUserId();

        if (
          current &&
          current !==
            lastAnnounced
        ) {
          lastAnnounced =
            current;

          announce(
            current
          );
        } else if (
          current &&
          attempts % 4 ===
            0
        ) {
          /*
           * Repeat occasionally in case the parent
           * listener was not ready for the first
           * document_start message.
           */
          announce(
            current
          );
        }

        if (
          attempts >=
          24
        ) {
          window.clearInterval(
            timer
          );
        }
      },
      250
    );
})();