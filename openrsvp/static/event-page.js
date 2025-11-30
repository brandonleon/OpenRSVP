(() => {
  let listenersBound = false;

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

  const init = () => {
    setupQr();
    if (!listenersBound) {
      document.addEventListener("themechange", setupQr);
      document.addEventListener("htmx:afterSwap", setupQr);
      listenersBound = true;
    }
  };

  document.addEventListener("DOMContentLoaded", init);
})();
