(() => {
  const THEME_STORAGE_KEY = "openrsvp-theme";
  const THEME_OPTIONS = ["latte", "mocha", "frappe", "macchiato"];
  const KEY_STORAGE_KEY = "openrsvp-keys";
  let eventTimeIntervalId = null;

  const safeGetStoredTheme = () => {
    try {
      const value = localStorage.getItem(THEME_STORAGE_KEY);
      return THEME_OPTIONS.includes(value) ? value : null;
    } catch (error) {
      return null;
    }
  };

  const safeSetStoredTheme = (theme) => {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch (error) {
      /* ignore storage errors */
    }
  };

  const safeReadKeyStore = () => {
    try {
      const raw = localStorage.getItem(KEY_STORAGE_KEY);
      if (!raw) return { events: {} };
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" ? parsed : { events: {} };
    } catch (error) {
      return { events: {} };
    }
  };

  const safeWriteKeyStore = (store) => {
    try {
      localStorage.setItem(KEY_STORAGE_KEY, JSON.stringify(store));
    } catch (error) {
      /* ignore storage errors */
    }
  };

  const saveAdminKey = (eventId, adminToken) => {
    if (!eventId || !adminToken) return;
    const store = safeReadKeyStore();
    if (!store.events) store.events = {};
    store.events[eventId] = store.events[eventId] || {};
    store.events[eventId].adminToken = adminToken;
    safeWriteKeyStore(store);
  };

  const saveRsvpKey = (eventId, rsvpId, rsvpToken, rsvpName) => {
    if (!eventId || !rsvpId || !rsvpToken) return;
    const store = safeReadKeyStore();
    if (!store.events) store.events = {};
    store.events[eventId] = store.events[eventId] || {};
    store.events[eventId].rsvp = {
      id: String(rsvpId),
      token: rsvpToken,
      name: rsvpName || "",
    };
    safeWriteKeyStore(store);
  };

  const getStoredEventKeys = (eventId) => {
    const store = safeReadKeyStore();
    return (store.events && store.events[eventId]) || null;
  };

  const clearStoredKeys = () => {
    try {
      localStorage.removeItem(KEY_STORAGE_KEY);
    } catch (error) {
      /* ignore storage errors */
    }
    renderStoredActions();
  };

  const showToast = (message, variant = "info") => {
    const stack = document.getElementById("toast-stack");
    if (!stack || !message) return;
    const toast = document.createElement("div");
    toast.className = `toast-banner toast-${variant}`;
    toast.textContent = message;
    stack.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
    }, 2000);
    setTimeout(() => {
      toast.remove();
    }, 2600);
  };

  const removeEventKeys = (eventId) => {
    if (!eventId) return;
    const store = safeReadKeyStore();
    if (store.events && store.events[eventId]) {
      delete store.events[eventId];
      safeWriteKeyStore(store);
    }
    renderStoredActions();
  };

  const copyToClipboard = (text) => {
    if (!text) return Promise.reject(new Error("Nothing to copy"));
    if (navigator.clipboard && navigator.clipboard.writeText) {
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

  const initCopyLinks = () => {
    document.querySelectorAll("[data-copy-link]").forEach((button) => {
      if (button.dataset.copyBound === "true") return;
      button.dataset.copyBound = "true";
      button.addEventListener("click", async () => {
        const target = button.getAttribute("data-copy-link");
        if (!target) return;
        const original = button.textContent;
        try {
          await copyToClipboard(target);
          button.textContent = "Copied!";
          showToast("Link copied to clipboard", "success");
          setTimeout(() => (button.textContent = original), 1600);
        } catch (error) {
          button.textContent = "Copy failed";
          showToast(`Copy blocked. Link: ${target}`, "danger");
          setTimeout(() => (button.textContent = original), 1600);
        }
      });
    });
  };

  const getSystemTheme = () =>
    window.matchMedia("(prefers-color-scheme: dark)").matches ? "mocha" : "latte";

  const updateThemeControls = (theme) => {
    const badge = document.getElementById("theme-badge");
    const formatted = theme ? `${theme.charAt(0).toUpperCase()}${theme.slice(1)}` : "";
    if (badge) {
      badge.textContent = formatted;
    }

    document.querySelectorAll("[data-theme-option]").forEach((option) => {
      const value = option.getAttribute("data-theme-option");
      const active = value === theme;
      option.classList.toggle("active", active);
      option.setAttribute("aria-pressed", active ? "true" : "false");
    });
  };

  const applyTheme = (theme, persist = true) => {
    const normalized = THEME_OPTIONS.includes(theme) ? theme : getSystemTheme();
    document.documentElement.dataset.theme = normalized;
    if (persist) {
      safeSetStoredTheme(normalized);
    }
    updateThemeControls(normalized);
    document.dispatchEvent(
      new CustomEvent("themechange", { detail: { theme: normalized } }),
    );
  };

  const bindThemeButtons = () => {
    document.querySelectorAll("[data-theme-option]").forEach((option) => {
      if (option.dataset.themeBound === "true") return;
      option.dataset.themeBound = "true";
      option.addEventListener("click", () => {
        const choice = option.getAttribute("data-theme-option");
        applyTheme(choice);
      });
    });
  };

  const bindForgetAllButton = () => {
    const clearBtn = document.getElementById("clear-keys-btn");
    if (!clearBtn || clearBtn.dataset.clearBound === "true") return;
    clearBtn.dataset.clearBound = "true";
    clearBtn.addEventListener("click", () => {
      const confirmed = window.confirm(
        "This removes stored admin and RSVP keys on this device. You will need your magic links to get back into admin views. Continue?",
      );
      if (!confirmed) return;
      clearStoredKeys();
      showToast("All stored keys forgotten on this device.", "info");
    });
  };

  const bindForgetEventButtons = () => {
    document.querySelectorAll("[data-clear-event-keys]").forEach((button) => {
      if (button.dataset.clearEventBound === "true") return;
      button.dataset.clearEventBound = "true";
      button.addEventListener("click", () => {
        const eventId = button.getAttribute("data-clear-event-keys");
        if (!eventId) return;
        const redirect = button.getAttribute("data-clear-redirect");
        const customMessage = button.getAttribute("data-clear-message");
        const confirmed = window.confirm(
          customMessage ||
            "Remove stored admin/RSVP keys for this event on this device? You'll need the magic link to get back.",
        );
        if (!confirmed) return;
        removeEventKeys(eventId);
        showToast("Stored key forgotten for this event.", "info");
        if (redirect) {
          window.location.href = redirect;
        }
      });
    });
  };

  const initThemeControls = () => {
    const savedTheme = safeGetStoredTheme();
    const systemTheme = getSystemTheme();
    const initialTheme =
      savedTheme || document.documentElement.dataset.theme || systemTheme;
    applyTheme(initialTheme, Boolean(savedTheme));
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");
    prefersDark.addEventListener("change", (event) => {
      const stored = safeGetStoredTheme();
      if (stored) return;
      applyTheme(event.matches ? "mocha" : "latte", false);
    });
    bindThemeButtons();
    bindForgetAllButton();
    bindForgetEventButtons();
  };

  const initKeyPersistence = () => {
    document.querySelectorAll("[data-event-admin-token]").forEach((node) => {
      const eventId = node.getAttribute("data-event-id");
      const token = node.getAttribute("data-admin-token");
      saveAdminKey(eventId, token);
    });

    document.querySelectorAll("[data-rsvp-token]").forEach((node) => {
      const eventId = node.getAttribute("data-event-id");
      const rsvpId = node.getAttribute("data-rsvp-id");
      const token = node.getAttribute("data-rsvp-token-value");
      const name = node.getAttribute("data-rsvp-name") || "";
      saveRsvpKey(eventId, rsvpId, token, name);
    });
  };

  const renderStoredActions = () => {
    const hero = document.querySelector("[data-event-page]");
    if (!hero) return;
    const eventId = hero.getAttribute("data-event-id");
    if (!eventId) return;
    const stored = getStoredEventKeys(eventId);
    const actionArea = document.getElementById("event-action-area");
    if (!stored) {
      if (actionArea) {
        actionArea
          .querySelectorAll("[data-stored-admin-link], [data-stored-rsvp-link]")
          .forEach((el) => el.remove());
      }
      document.querySelectorAll("[data-stored-rsvp-badge]").forEach((el) => el.remove());
      document.querySelectorAll("[data-stored-rsvp-placeholder]").forEach((el) =>
        el.remove(),
      );
      return;
    }
    if (actionArea) {
      if (stored.adminToken && !actionArea.querySelector("[data-stored-admin-link]")) {
        const adminLink = document.createElement("a");
        adminLink.className = "btn btn-outline-light";
        adminLink.href = `/e/${eventId}/admin/${stored.adminToken}`;
        adminLink.textContent = "Open Admin";
        adminLink.setAttribute("data-stored-admin-link", "true");
        actionArea.prepend(adminLink);
      }
      if (
        stored.rsvp &&
        stored.rsvp.token &&
        !actionArea.querySelector("[data-stored-rsvp-link]")
      ) {
        const rsvpLink = document.createElement("a");
        rsvpLink.className = "btn btn-outline-secondary";
        rsvpLink.href = `/e/${eventId}/rsvp/${stored.rsvp.token}`;
        rsvpLink.textContent = "Edit your RSVP";
        rsvpLink.setAttribute("data-stored-rsvp-link", "true");
        actionArea.appendChild(rsvpLink);
      }
    }

    if (stored.rsvp && stored.rsvp.id) {
      const listItems = document.querySelectorAll(`[data-rsvp-id='${stored.rsvp.id}']`);
      if (!listItems.length) {
        const listGroup = document.getElementById("event-rsvp-list");
        if (listGroup) {
          listGroup.querySelectorAll("[data-empty-rsvp-list]").forEach((el) => el.remove());
          const item = document.createElement("div");
          item.className = "list-group-item px-0";
          item.setAttribute("data-stored-rsvp-placeholder", "true");
          const name = stored.rsvp.name || "Your RSVP";
          item.innerHTML = `
                <div class="d-flex flex-wrap justify-content-between align-items-start gap-2">
                  <div>
                    <div class="fw-semibold"></div>
                    <div class="small text-muted">Private RSVP (visible only to you)</div>
                  </div>
                  <div class="text-end">
                    <div class="d-flex flex-wrap justify-content-end gap-1">
                      <span class="badge bg-dark text-uppercase">Private</span>
                      <a class="badge bg-primary text-decoration-none" href="/e/${eventId}/rsvp/${stored.rsvp.token}">Manage</a>
                    </div>
                  </div>
                </div>
              `;
          const nameEl = item.querySelector(".fw-semibold");
          if (nameEl) nameEl.textContent = name;
          listGroup.prepend(item);
        }
      }
      listItems.forEach((item) => {
        const nameEl = item.querySelector(".fw-semibold");
        if (nameEl && !item.querySelector("[data-stored-rsvp-badge]")) {
          const badge = document.createElement("a");
          badge.href = `/e/${eventId}/rsvp/${stored.rsvp.token}`;
          badge.className = "badge bg-primary text-decoration-none ms-2";
          badge.textContent = "Manage";
          badge.setAttribute("data-stored-rsvp-badge", "true");
          nameEl.appendChild(badge);
        }
      });
    }
  };

  const updateTimezoneInputs = () => {
    const offset = new Date().getTimezoneOffset();
    document.querySelectorAll("[data-tz-input]").forEach((input) => {
      input.value = offset;
    });
    document.querySelectorAll("[data-utc-input]").forEach((input) => {
      const iso = input.getAttribute("data-utc-input");
      if (!iso) return;
      const date = new Date(iso);
      if (Number.isNaN(date.getTime())) return;
      input.value = formatLocalInput(date);
    });
  };

  const updateEventTimes = () => {
    document.querySelectorAll("[data-event-utc]").forEach((element) => {
      const iso = element.getAttribute("data-event-utc");
      if (!iso) return;
      const date = new Date(iso);
      if (Number.isNaN(date.getTime())) return;
      const absoluteEl = element.querySelector(".js-event-absolute");
      if (absoluteEl) {
        absoluteEl.textContent = date.toLocaleString(undefined, {
          dateStyle: "medium",
          timeStyle: "short",
        });
      }
      const relativeEl = element.querySelector(".js-event-relative");
      if (relativeEl) {
        relativeEl.textContent = formatRelative(date);
      }
    });
  };

  const formatLocalInput = (date) => {
    const pad = (value) => String(value).padStart(2, "0");
    const year = date.getFullYear();
    const month = pad(date.getMonth() + 1);
    const day = pad(date.getDate());
    const hours = pad(date.getHours());
    const minutes = pad(date.getMinutes());
    return `${year}-${month}-${day}T${hours}:${minutes}`;
  };

  const formatRelative = (targetDate) => {
    const now = new Date();
    const seconds = Math.floor((targetDate.getTime() - now.getTime()) / 1000);
    const past = seconds < 0;
    const absSeconds = Math.abs(seconds);
    const units = [
      ["year", 365 * 24 * 3600],
      ["month", 30 * 24 * 3600],
      ["week", 7 * 24 * 3600],
      ["day", 24 * 3600],
      ["hour", 3600],
      ["minute", 60],
    ];
    for (const [label, size] of units) {
      const value = Math.floor(absSeconds / size);
      if (value >= 1) {
        const unit = value === 1 ? label : `${label}s`;
        return past ? `${value} ${unit} ago` : `in ${value} ${unit}`;
      }
    }
    return past ? "moments ago" : "in moments";
  };

  const initChannelVisibilityAlerts = () => {
    document.querySelectorAll("[data-channel-alert]").forEach((alert) => {
      if (alert.dataset.channelAlertBound === "true") return;
      const group = alert.getAttribute("data-channel-alert");
      if (!group) return;
      const inputs = document.querySelectorAll(
        `[data-channel-visibility='${group}']`,
      );
      if (!inputs.length) return;
      const toggle = () => {
        const active = Array.from(inputs).some(
          (input) => input.checked && input.value === "private",
        );
        alert.classList.toggle("d-none", !active);
      };
      inputs.forEach((input) => input.addEventListener("change", toggle));
      alert.dataset.channelAlertBound = "true";
      toggle();
    });
  };

  const initChannelEventAlerts = () => {
    document.querySelectorAll("[data-channel-event-alert]").forEach((alert) => {
      if (alert.dataset.channelEventAlertBound === "true") return;
      const group = alert.getAttribute("data-channel-event-alert");
      const toggleSelector = alert.getAttribute("data-event-private-toggle");
      if (!group || !toggleSelector) return;
      const visibilityInputs = document.querySelectorAll(
        `[data-channel-visibility='${group}']`,
      );
      const privateToggle = document.querySelector(toggleSelector);
      if (!visibilityInputs.length || !privateToggle) return;
      let userOverride = false;
      const update = () => {
        const channelIsPrivate = Array.from(visibilityInputs).some(
          (input) => input.checked && input.value === "private",
        );
        const eventIsPrivate = privateToggle.checked;
        if (channelIsPrivate && !eventIsPrivate && !userOverride) {
          privateToggle.checked = true;
        }
        const shouldShow = channelIsPrivate;
        alert.classList.toggle("d-none", !shouldShow);
        userOverride = false;
      };
      visibilityInputs.forEach((input) =>
        input.addEventListener("change", () => update()),
      );
      privateToggle.addEventListener("change", () => {
        userOverride = true;
        update();
      });
      alert.dataset.channelEventAlertBound = "true";
      update();
    });
  };

  const handleDomReady = () => {
    initThemeControls();
    initKeyPersistence();
    renderStoredActions();
    initCopyLinks();
    updateTimezoneInputs();
    updateEventTimes();
    initChannelVisibilityAlerts();
    initChannelEventAlerts();
    bindForgetEventButtons();
    if (!eventTimeIntervalId) {
      eventTimeIntervalId = window.setInterval(updateEventTimes, 60000);
    }
  };

  const handleHtmxSwap = () => {
    updateEventTimes();
    initChannelVisibilityAlerts();
    initChannelEventAlerts();
    initCopyLinks();
    renderStoredActions();
    initKeyPersistence();
    bindForgetEventButtons();
  };

  document.addEventListener("DOMContentLoaded", handleDomReady);
  document.addEventListener("htmx:afterSwap", handleHtmxSwap);
})();
