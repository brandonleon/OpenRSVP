(() => {
  const FILTER_BUTTON_SELECTOR = "[data-rsvp-filter]";
  const ALT_ROW_CLASSES = ["bg-dark", "bg-opacity-10", "alt-row"];
  let currentFilter = "all";
  let filterInitialized = false;

  const applyFilter = (target = currentFilter) => {
    currentFilter = target;
    const items = document.querySelectorAll("#admin-rsvp-list [data-approval-status]");
    items.forEach((item) => {
      const status = (item.getAttribute("data-approval-status") || "").toLowerCase();
      const shouldShow = target === "all" || status === target;
      item.style.display = shouldShow ? "" : "none";
    });
    document.querySelectorAll(FILTER_BUTTON_SELECTOR).forEach((btn) => {
      const value = btn.getAttribute("data-rsvp-filter") || "all";
      btn.classList.toggle("active", value === target);
    });
  };

  const bindFilterButtons = () => {
    document.querySelectorAll(FILTER_BUTTON_SELECTOR).forEach((btn) => {
      if (btn.dataset.rsvpFilterBound === "true") return;
      btn.dataset.rsvpFilterBound = "true";
      btn.addEventListener("click", () => {
        const target = btn.getAttribute("data-rsvp-filter") || "all";
        applyFilter(target);
      });
    });
  };

  const initFilterControls = () => {
    const list = document.getElementById("admin-rsvp-list");
    if (!list) return;
    if (!filterInitialized) {
      const active =
        document
          .querySelector(`${FILTER_BUTTON_SELECTOR}.active`)
          ?.getAttribute("data-rsvp-filter") || "all";
      currentFilter = active;
      filterInitialized = true;
    }
    applyFilter(currentFilter);
    bindFilterButtons();
  };

  const bindPendingLinks = () => {
    document.querySelectorAll("[data-rsvp-pending-link]").forEach((link) => {
      if (link.dataset.pendingLinkBound === "true") return;
      link.dataset.pendingLinkBound = "true";
      link.addEventListener("click", (event) => {
        event.preventDefault();
        applyFilter("pending");
        const hash = link.getAttribute("href") || "#rsvp-section";
        if (hash.startsWith("#")) {
          const targetEl = document.querySelector(hash);
          if (targetEl && targetEl.scrollIntoView) {
            targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
            return;
          }
        }
        window.location.hash = "#rsvp-section";
      });
    });
  };

  const applyAltRows = () => {
    const rows = document.querySelectorAll("#admin-rsvp-list [data-rsvp-card]");
    rows.forEach((item, index) => {
      const card = item.querySelector(".rsvp-card");
      if (!card) return;
      ALT_ROW_CLASSES.forEach((cls) => card.classList.remove(cls));
      if (index % 2 === 1) {
        ALT_ROW_CLASSES.forEach((cls) => card.classList.add(cls));
      }
    });
  };

  const init = () => {
    applyAltRows();
    initFilterControls();
    bindPendingLinks();
  };

  document.addEventListener("DOMContentLoaded", init);
  document.addEventListener("htmx:afterSwap", init);
  document.addEventListener("htmx:afterSettle", init);
})();
