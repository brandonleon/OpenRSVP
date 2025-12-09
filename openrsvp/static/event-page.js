(() => {
  let listenersBound = false;

  const copyLink = (text) => {
    if (!text) return Promise.reject(new Error("Missing share URL"));
    if (
      typeof navigator !== "undefined" &&
      navigator.clipboard &&
      typeof navigator.clipboard.writeText === "function"
    ) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise((resolve, reject) => {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "absolute";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      const selection = document.getSelection();
      const selected =
        selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null;
      textarea.select();
      try {
        document.execCommand("copy");
        resolve();
      } catch (error) {
        reject(error);
      } finally {
        document.body.removeChild(textarea);
        if (selected && selection) {
          selection.removeAllRanges();
          selection.addRange(selected);
        }
      }
    });
  };

  const renderQr = (container) => {
    const target = container.getAttribute("data-rsvp-url");
    if (!target || typeof QRCode === "undefined") return;
    const styles = getComputedStyle(document.documentElement);
    const colorDark = styles.getPropertyValue("--accent").trim() || "#1e66f5";
    const colorLight = styles.getPropertyValue("--surface").trim() || "#ffffff";
    container.innerHTML = "";
    /* global QRCode */
    new QRCode(container, {
      text: target,
      width: 168,
      height: 168,
      colorDark,
      colorLight,
      correctLevel: QRCode.CorrectLevel.H,
    });
  };

  const setupQr = () => {
    const container = document.getElementById("rsvp-qr");
    if (!container) return;
    renderQr(container);
  };

  const shareViaOsSheet = async (data) => {
    if (
      typeof navigator === "undefined" ||
      typeof navigator.share !== "function"
    ) {
      return false;
    }
    try {
      await navigator.share(data);
      return true;
    } catch (error) {
      if (error && error.name === "AbortError") {
        return true;
      }
      return false;
    }
  };

  const getShareCapabilityState = () => {
    if (typeof window !== "undefined" && !window.isSecureContext) {
      return "insecure";
    }
    if (
      typeof navigator === "undefined" ||
      typeof navigator.share !== "function"
    ) {
      return "unsupported";
    }
    return "available";
  };

  const bindShareButtons = () => {
    document.querySelectorAll("[data-share-button]").forEach((button) => {
      if (!button || button.dataset.shareBound === "true") return;
      const shareData = {
        title:
          button.getAttribute("data-share-title") || document.title || "OpenRSVP",
        text:
          button.getAttribute("data-share-text") ||
          `Check out ${document.title || "this event"} on OpenRSVP.`,
        url: button.getAttribute("data-share-url") || window.location.href,
      };
      const defaultMarkup = button.innerHTML;
      const successLabel =
        button.getAttribute("data-share-success-label") || "Link copied!";
      const failureLabel =
        button.getAttribute("data-share-error-label") || "Copy blocked";
      const insecureLabel =
        button.getAttribute("data-share-insecure-label") ||
        "Share needs HTTPS (link copied)";
      const unsupportedLabel =
        button.getAttribute("data-share-unsupported-label") ||
        "Sharing unavailable (link copied)";
      const restoreDefault = () => {
        if (typeof defaultMarkup === "string" && defaultMarkup.length > 0) {
          button.innerHTML = defaultMarkup;
        } else {
          button.textContent =
            button.getAttribute("data-share-default-label") || "Share event";
        }
      };
      button.dataset.shareBound = "true";

      button.addEventListener("click", async () => {
        const capability = getShareCapabilityState();
        if (capability === "available" && (await shareViaOsSheet(shareData))) {
          return;
        }
        try {
          await copyLink(shareData.url);
          if (capability === "insecure") {
            button.textContent = insecureLabel;
          } else if (capability === "unsupported") {
            button.textContent = unsupportedLabel;
          } else {
            button.textContent = successLabel;
          }
        } catch (error) {
          button.textContent = failureLabel;
        }
        setTimeout(() => {
          restoreDefault();
        }, 1600);
      });
    });
  };

  const init = () => {
    setupQr();
    bindShareButtons();
    if (!listenersBound) {
      document.addEventListener("themechange", setupQr);
      document.addEventListener("htmx:afterSwap", () => {
        setupQr();
        bindShareButtons();
      });
      listenersBound = true;
    }
  };

  document.addEventListener("DOMContentLoaded", init);
})();
