(() => {
  const THEMES = ["latte", "mocha", "frappe", "macchiato"];
  const STORAGE_KEY = "openrsvp-theme";

  const getSystemTheme = () =>
    window.matchMedia("(prefers-color-scheme: dark)").matches ? "mocha" : "latte";

  const readStoredTheme = () => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved && THEMES.includes(saved) ? saved : null;
    } catch (error) {
      return null;
    }
  };

  const fallbackTheme = getSystemTheme();
  const theme = readStoredTheme() || fallbackTheme;
  document.documentElement.dataset.theme = theme;
})();
