"use strict";

// A short installation path, shared by the demo entry and workspace links.
(() => {
  if (window.IRISRuntimeAdapter?.mode !== "demo") return;

  const docs = "https://github.com/victormonnot/iris/blob/main/docs/";
  const element = (tag, className, text) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text) item.textContent = text;
    return item;
  };
  const link = (text, href) => {
    const item = element("a", "", text);
    item.href = href;
    item.target = "_blank";
    item.rel = "noopener noreferrer";
    return item;
  };
  const dialog = element("dialog", "modal studio-dialog iris-demo-install-dialog");
  dialog.id = "iris-demo-install-dialog";
  dialog.setAttribute("aria-labelledby", "iris-demo-install-title");
  dialog.setAttribute("aria-describedby", "iris-demo-install-intro");

  const heading = element("div", "modal-heading");
  const headingText = element("div");
  headingText.append(element("div", "eyebrow", "Your own workspace"));
  const title = element("h2", "", "Run Iris on your computer.");
  title.id = "iris-demo-install-title";
  headingText.append(title);
  const close = element("button", "icon-button", "×");
  close.id = "iris-demo-install-close";
  close.type = "button";
  close.setAttribute("aria-label", "Close installation instructions");
  heading.append(headingText, close);

  const intro = element("p", "iris-demo-install-intro", "Use the same interface with your own files. Iris runs a local server; your browser opens the workbench and the server stores your work and runs local computation on your computer.");
  intro.id = "iris-demo-install-intro";
  const requirements = element("p", "iris-demo-install-requirements");
  requirements.append(
    element("strong", "", "Linux or WSL · Python 3.12 or 3.13 · Git · "),
    link("uv", "https://docs.astral.sh/uv/getting-started/installation/"),
    document.createTextNode(". On Windows, run the commands inside WSL."),
  );

  const steps = element("ol", "iris-demo-install-steps");
  const start = element("li");
  start.append(element("h3", "", "Start Iris"));
  const code = element("code", "", [
    "git clone https://github.com/victormonnot/iris.git",
    "cd iris",
    "uv sync --locked",
    "uv run iris",
  ].join("\n"));
  code.id = "iris-demo-install-commands";
  const commands = element("pre", "iris-demo-install-commands");
  commands.tabIndex = 0;
  commands.setAttribute("aria-label", "Terminal commands to install and start Iris");
  commands.append(code);
  const copy = element("button", "button button-secondary", "Copy commands");
  copy.id = "iris-demo-install-copy";
  copy.type = "button";
  copy.setAttribute("aria-describedby", "iris-demo-install-copy-status");
  const copyStatus = element("p", "iris-demo-install-copy-status");
  copyStatus.id = "iris-demo-install-copy-status";
  copyStatus.setAttribute("role", "status");
  copyStatus.setAttribute("aria-live", "polite");
  const commandActions = element("div", "iris-demo-install-command-actions");
  commandActions.append(copy, copyStatus);
  start.append(commands, commandActions);

  const open = element("li");
  open.append(element("h3", "", "Open the workbench"));
  const address = element("p");
  address.append(
    document.createTextNode("Visit "),
    link("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
    document.createTextNode(" in your browser. Keep the terminal running; Ctrl+C stops Iris."),
  );
  const storage = element("p", "field-hint");
  storage.append(
    document.createTextNode("Your workspace is saved in "),
    element("code", "", ".iris/"),
    document.createTextNode(" in the directory where you start Iris. Importing files and reviewing annotations work without a GPU, model weights or an API key."),
  );
  open.append(address, storage);

  const sample = element("li");
  sample.append(element("h3", "", "Try the included photos"));
  const tour = element("p");
  tour.append(
    document.createTextNode("Follow the "),
    link("two-photo walkthrough", docs + "first-run.md"),
    document.createTextNode(" to import the included street scenes and review their labels in a separate local workspace, or create a project and import your own images."),
  );
  sample.append(tour);
  steps.append(start, open, sample);

  const models = element("section", "iris-demo-install-models");
  models.setAttribute("aria-labelledby", "iris-demo-install-models-title");
  const modelsTitle = element("h3", "", "Model inference is an optional next step");
  modelsTitle.id = "iris-demo-install-models-title";
  const modelsText = element("p");
  modelsText.append(
    document.createTextNode("Stop Iris, then follow "),
    link("Add local detectors", docs + "setup.md#add-local-detectors"),
    document.createTextNode(" to install the CPU runtime and download model weights. For NVIDIA CUDA, follow the separate "),
    link("compute setup", docs + "compute-targets.md"),
    document.createTextNode("; choosing a GPU in the interface does not install its runtime."),
  );
  models.append(modelsTitle, modelsText);
  const footer = element("div", "iris-demo-install-footer");
  footer.append(link("Full installation guide ↗", docs + "setup.md"));
  dialog.append(heading, intro, requirements, steps, models, footer);
  document.body.append(dialog);

  let returnFocus = null;
  let copyAttempt = 0;
  function show(trigger) {
    if (dialog.open) return;
    returnFocus = trigger;
    copyStatus.textContent = "";
    dialog.showModal();
    dialog.scrollTop = 0;
    close.focus({ preventScroll: true });
  }
  close.addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    copyAttempt++;
    copy.disabled = false;
    copyStatus.textContent = "";
    const owner = returnFocus?.closest("dialog");
    if (returnFocus?.isConnected && (!owner || owner.open) && !returnFocus.closest("[hidden], [inert]"))
      returnFocus.focus({ preventScroll: true });
  });
  copy.addEventListener("click", async () => {
    const attempt = ++copyAttempt;
    copy.disabled = true;
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(code.textContent);
      if (dialog.open && attempt === copyAttempt)
        copyStatus.textContent = "Copied. Paste these commands into your terminal.";
    } catch {
      if (!dialog.open || attempt !== copyAttempt) return;
      commands.focus({ preventScroll: true });
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection?.removeAllRanges();
      selection?.addRange(range);
      copyStatus.textContent = "Automatic copying is unavailable. Copy the selected commands manually with Ctrl+C, ⌘C or your device’s Copy action.";
    } finally {
      if (attempt === copyAttempt) copy.disabled = false;
    }
  });
  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    const trigger = event.target instanceof Element ? event.target.closest("a.iris-demo-install") : null;
    if (!trigger) return;
    event.preventDefault();
    show(trigger);
  });
})();
