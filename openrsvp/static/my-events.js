(() => {
  const KEY_STORAGE_KEY = "openrsvp-keys";

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

  const normalizeIso = (value) => {
    if (!value) return null;
    if (value.endsWith("Z") || /[+-]\d{2}:\d{2}$/.test(value)) {
      return value;
    }
    return `${value}Z`;
  };

  const formatAbsolute = (date) =>
    date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

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
    for (let index = 0; index < units.length; index += 1) {
      const [label, size] = units[index];
      let value = Math.floor(absSeconds / size);
      if (value >= 1) {
        let unitLabel = label;
        if (label === "week" && value === 1 && index + 1 < units.length) {
          const [nextLabel, nextSize] = units[index + 1];
          value = Math.max(1, Math.floor(absSeconds / nextSize));
          unitLabel = nextLabel;
        }
        const unit = value === 1 ? unitLabel : `${unitLabel}s`;
        return past ? `${value} ${unit} ago` : `in ${value} ${unit}`;
      }
    }
    return past ? "moments ago" : "in moments";
  };

  const makeEl = (tag, className, text) => {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined && text !== null) {
      el.textContent = text;
    }
    return el;
  };

  const buildLocationText = (event) => {
    if (!event) return "Location unavailable";
    if (event.location) return event.location;
    if (event.admin_approval_required) return "Location hidden";
    return "Location TBD";
  };

  const getStoredEntries = (view) => {
    const store = safeReadKeyStore();
    const events = store.events || {};
    return Object.entries(events)
      .map(([eventId, data]) => ({
        eventId,
        adminToken: data.adminToken || null,
        rsvp: data.rsvp || null,
      }))
      .filter((entry) => {
        if (view === "rsvps") {
          return entry.rsvp && entry.rsvp.token;
        }
        return entry.adminToken;
      });
  };

  const fetchEvent = async (eventId, token) => {
    const headers = {};
    if (token) headers.Authorization = `Bearer ${token}`;
    try {
      const response = await fetch(`/api/v1/events/${eventId}`, { headers });
      if (!response.ok) return null;
      const payload = await response.json();
      return payload.event || null;
    } catch (error) {
      return null;
    }
  };

  const removeEvent = (eventId) => {
    const store = safeReadKeyStore();
    if (store.events && store.events[eventId]) {
      delete store.events[eventId];
      safeWriteKeyStore(store);
    }
  };

  const buildEventItem = (entry, event, view) => {
    const item = makeEl("div", "list-group-item py-3");
    const row = makeEl(
      "div",
      "d-flex w-100 justify-content-between align-items-start flex-wrap gap-3",
    );
    const left = makeEl("div");
    const right = makeEl("div", "text-end");

    const titleRow = makeEl("div", "d-flex align-items-center gap-2 flex-wrap");
    const titleLink = makeEl("a", "h5 mb-0 text-decoration-none");
    titleLink.href = `/e/${entry.eventId}`;
    titleLink.textContent = event ? event.title : `Event ${entry.eventId}`;
    titleRow.appendChild(titleLink);
    if (event && event.is_private) {
      titleRow.appendChild(makeEl("span", "badge bg-dark text-uppercase", "Private"));
    }
    left.appendChild(titleRow);

    if (event && event.channel) {
      const channelLine = makeEl(
        "div",
        "text-muted small",
        `Channel: ${event.channel.name}`,
      );
      left.appendChild(channelLine);
    }

    const locationLine = makeEl(
      "div",
      "text-muted small",
      buildLocationText(event),
    );
    left.appendChild(locationLine);

    if (view === "rsvps" && entry.rsvp && entry.rsvp.name) {
      left.appendChild(
        makeEl("div", "text-muted small", `RSVP name: ${entry.rsvp.name}`),
      );
    }

    if (event && event.rsvps_closed) {
      left.appendChild(
        makeEl(
          "span",
          "badge bg-danger text-uppercase d-inline-block mt-1",
          "RSVPs closed",
        ),
      );
    } else if (event && event.rsvp_close_at) {
      const closeIso = normalizeIso(event.rsvp_close_at);
      const closeDate = closeIso ? new Date(closeIso) : null;
      if (closeDate && !Number.isNaN(closeDate.getTime())) {
        left.appendChild(
          makeEl(
            "span",
            "badge bg-info text-dark text-uppercase d-inline-block mt-1",
            `Auto closes ${formatRelative(closeDate)}`,
          ),
        );
      }
    }

    if (event && event.start_time) {
      const startIso = normalizeIso(event.start_time);
      const startDate = startIso ? new Date(startIso) : null;
      if (startDate && !Number.isNaN(startDate.getTime())) {
        right.appendChild(
          makeEl("span", "badge bg-light text-dark", formatAbsolute(startDate)),
        );
        right.appendChild(makeEl("div", "text-muted small", formatRelative(startDate)));
      }
    }

    row.appendChild(left);
    row.appendChild(right);
    item.appendChild(row);

    if (!event) {
      item.appendChild(
        makeEl(
          "div",
          "text-muted small mt-2",
          "Event details are unavailable. It may have expired or been deleted.",
        ),
      );
    }

    const actions = makeEl("div", "d-flex flex-wrap gap-2 mt-3");
    const eventLink = makeEl("a", "btn btn-outline-secondary btn-sm", "Open event");
    eventLink.href = `/e/${entry.eventId}`;
    actions.appendChild(eventLink);

    if (entry.adminToken) {
      const adminLink = makeEl("a", "btn btn-outline-primary btn-sm", "Open admin");
      adminLink.href = `/e/${entry.eventId}/admin/${entry.adminToken}`;
      actions.appendChild(adminLink);
    }

    if (entry.rsvp && entry.rsvp.token) {
      const rsvpLink = makeEl("a", "btn btn-outline-primary btn-sm", "Manage RSVP");
      rsvpLink.href = `/e/${entry.eventId}/rsvp/${entry.rsvp.token}`;
      actions.appendChild(rsvpLink);
    }

    const forgetButton = makeEl("button", "btn btn-outline-danger btn-sm", "Forget");
    forgetButton.type = "button";
    forgetButton.addEventListener("click", () => {
      const confirmed = window.confirm(
        "Remove saved access for this event on this device?",
      );
      if (!confirmed) return;
      removeEvent(entry.eventId);
      item.remove();
      if (!document.querySelector("#my-events-list .list-group-item")) {
        const empty = document.getElementById("my-events-empty");
        if (empty) empty.classList.remove("d-none");
      }
    });
    actions.appendChild(forgetButton);

    item.appendChild(actions);
    return item;
  };

  const renderEntries = async () => {
    const list = document.getElementById("my-events-list");
    if (!list) return;
    const loading = document.getElementById("my-events-loading");
    const empty = document.getElementById("my-events-empty");
    const view = list.getAttribute("data-my-events-view") || "hosted";
    const entries = getStoredEntries(view);

    if (!entries.length) {
      if (loading) loading.classList.add("d-none");
      if (empty) empty.classList.remove("d-none");
      return;
    }

    const results = await Promise.all(
      entries.map(async (entry) => {
        const token =
          view === "hosted"
            ? entry.adminToken
            : entry.rsvp
              ? entry.rsvp.token
              : null;
        const event = await fetchEvent(entry.eventId, token);
        return { entry, event };
      }),
    );

    const now = Date.now();
    results.sort((a, b) => {
      const aDate = a.event ? normalizeIso(a.event.start_time) : null;
      const bDate = b.event ? normalizeIso(b.event.start_time) : null;
      const aTime = aDate ? new Date(aDate).getTime() : Number.POSITIVE_INFINITY;
      const bTime = bDate ? new Date(bDate).getTime() : Number.POSITIVE_INFINITY;
      const aPast = Number.isFinite(aTime) && aTime < now;
      const bPast = Number.isFinite(bTime) && bTime < now;
      if (aPast !== bPast) return aPast ? 1 : -1;
      const aSort = aPast ? -aTime : aTime;
      const bSort = bPast ? -bTime : bTime;
      if (aSort !== bSort) return aSort - bSort;
      return a.entry.eventId.localeCompare(b.entry.eventId);
    });

    list.innerHTML = "";
    results.forEach(({ entry, event }) => {
      list.appendChild(buildEventItem(entry, event, view));
    });
    if (loading) loading.classList.add("d-none");
  };

  document.addEventListener("DOMContentLoaded", renderEntries);
})();
