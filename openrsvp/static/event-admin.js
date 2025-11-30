(() => {
  const FILTER_BUTTON_SELECTOR = "[data-rsvp-filter]";

  const initFilterControls = () => {
    const list = document.getElementById("admin-rsvp-list");
    const buttons = document.querySelectorAll(FILTER_BUTTON_SELECTOR);
    if (!list || !buttons.length) return;
    const items = Array.from(list.querySelectorAll("[data-approval-status]"));
    if (!items.length) return;

    const update = (target) => {
      items.forEach((item) => {
        const status = (item.getAttribute("data-approval-status") || "").toLowerCase();
        const shouldShow = target === "all" || status === target;
        item.style.display = shouldShow ? "" : "none";
      });
      buttons.forEach((btn) => {
        const isActive = (btn.getAttribute("data-rsvp-filter") || "all") === target;
        btn.classList.toggle("active", isActive);
      });
    };

    const initialTarget =
      document
        .querySelector(`${FILTER_BUTTON_SELECTOR}.active`)
        ?.getAttribute("data-rsvp-filter") || "all";
    update(initialTarget);

    buttons.forEach((btn) => {
      if (btn.dataset.rsvpFilterBound === "true") return;
      btn.dataset.rsvpFilterBound = "true";
      btn.addEventListener("click", () =>
        update(btn.getAttribute("data-rsvp-filter") || "all"),
      );
    });

    document.querySelectorAll("[data-rsvp-pending-link]").forEach((link) => {
      if (link.dataset.pendingLinkBound === "true") return;
      link.dataset.pendingLinkBound = "true";
      link.addEventListener("click", (event) => {
        event.preventDefault();
        update("pending");
        const hash = link.getAttribute("href") || "#rsvp-section";
        if (hash.startsWith("#")) {
          const targetEl = document.querySelector(hash);
          if (targetEl && targetEl.scrollIntoView) {
            targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
          } else {
            window.location.hash = hash;
          }
        } else {
          window.location.hash = "#rsvp-section";
        }
      });
    });
  };

  const init = () => {
    initFilterControls();
  };

  document.addEventListener("DOMContentLoaded", init);
  document.addEventListener("htmx:afterSwap", init);
})();
