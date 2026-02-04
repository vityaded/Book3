(() => {
  const processing = document.getElementById("processing");
  const uploadForm = document.getElementById("upload-form");
  const cvDebugToggle = document.getElementById("cv-debug-toggle");
  const cvDebugInputs = document.querySelectorAll("[data-cv-debug-hidden]");

  const pasteButton = document.getElementById("paste-submit");

  const worksheetPasteArea = document.getElementById("paste-area");
  const worksheetPreview = document.getElementById("paste-preview");
  const worksheetStatus = document.getElementById("clipboard-status");

  const keysPasteArea = document.getElementById("keys-paste-area");
  const keysPreview = document.getElementById("keys-preview");
  const keysStatus = document.getElementById("keys-status");

  const uploadKeysArea = document.getElementById("upload-keys-area");
  const uploadKeysPreview = document.getElementById("upload-keys-preview");
  const uploadKeysStatus = document.getElementById("upload-keys-status");

  const CLIPBOARD_MAX_DIM = 1600;
  const CLIPBOARD_OUTPUT_TYPE = "image/png";
  const CLIPBOARD_OUTPUT_QUALITY = 0.92;

  const showProcessing = () => {
    if (processing) {
      processing.classList.add("active");
    }
  };

  const hideProcessing = () => {
    if (processing) {
      processing.classList.remove("active");
    }
  };

  const isCvDebugEnabled = () => Boolean(cvDebugToggle && cvDebugToggle.checked);

  const syncCvDebugHidden = () => {
    if (!cvDebugInputs.length) return;
    const value = isCvDebugEnabled() ? "1" : "0";
    cvDebugInputs.forEach((input) => {
      input.value = value;
    });
  };

  if (cvDebugToggle) {
    cvDebugToggle.addEventListener("change", syncCvDebugHidden);
    syncCvDebugHidden();
  }

  const preprocessClipboardImage = (file) =>
    new Promise((resolve, reject) => {
      if (!file) {
        reject(new Error("No clipboard image."));
        return;
      }

      const url = URL.createObjectURL(file);
      const image = new Image();
      image.onload = () => {
        URL.revokeObjectURL(url);
        const width = image.naturalWidth || image.width;
        const height = image.naturalHeight || image.height;
        const maxDim = Math.max(width, height);

        if (!Number.isFinite(maxDim) || maxDim <= CLIPBOARD_MAX_DIM) {
          resolve({
            blob: file,
            name: file.name || "clipboard.png",
            width,
            height,
            resized: false,
          });
          return;
        }

        const scale = CLIPBOARD_MAX_DIM / maxDim;
        const targetWidth = Math.max(1, Math.round(width * scale));
        const targetHeight = Math.max(1, Math.round(height * scale));
        const canvas = document.createElement("canvas");
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        const ctx = canvas.getContext("2d");

        if (!ctx) {
          resolve({
            blob: file,
            name: file.name || "clipboard.png",
            width,
            height,
            resized: false,
          });
          return;
        }

        ctx.drawImage(image, 0, 0, targetWidth, targetHeight);
        canvas.toBlob(
          (blob) => {
            if (!blob) {
              resolve({
                blob: file,
                name: file.name || "clipboard.png",
                width,
                height,
                resized: false,
              });
              return;
            }
            resolve({
              blob,
              name: "clipboard.png",
              width: targetWidth,
              height: targetHeight,
              resized: true,
            });
          },
          CLIPBOARD_OUTPUT_TYPE,
          CLIPBOARD_OUTPUT_QUALITY,
        );
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Failed to decode clipboard image."));
      };
      image.src = url;
    });

  const createPasteTarget = ({
    areaEl,
    previewEl,
    statusEl,
    label,
    defaultName,
    onStateChange,
  }) => {
    let file = null;
    let blob = null;
    let fileName = defaultName || "clipboard.png";
    let previewUrl = null;

    const setStatus = (message) => {
      if (statusEl) {
        statusEl.textContent = message;
      }
    };

    const clearPreviewUrl = () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
      }
    };

    const setPreviewBlob = (previewBlob) => {
      if (!previewEl || !previewBlob) return;
      clearPreviewUrl();
      previewUrl = URL.createObjectURL(previewBlob);
      previewEl.src = previewUrl;
      previewEl.style.display = "block";
    };

    const setReady = (ready) => {
      if (typeof onStateChange === "function") {
        onStateChange(Boolean(ready));
      }
    };

    const handlePaste = async (event) => {
      const items = event.clipboardData ? event.clipboardData.items : [];
      let found = null;

      for (const item of items) {
        if (item.type && item.type.startsWith("image/")) {
          found = item.getAsFile();
          break;
        }
      }

      if (!found) {
        setStatus(`${label} not found in clipboard.`);
        return;
      }

      file = found;
      blob = null;
      fileName = (file && file.name) || defaultName || "clipboard.png";
      setReady(false);

      try {
        setStatus(`Preprocessing ${label.toLowerCase()}...`);
        const result = await preprocessClipboardImage(file);
        blob = result.blob;
        fileName = (result && result.name) || fileName;
        const resizedLabel = result.resized
          ? ` (resized to ${result.width}×${result.height})`
          : "";
        setStatus(`${label} ready${resizedLabel}.`);
        setPreviewBlob(result.blob);
      } catch (_error) {
        blob = file;
        fileName = (file && file.name) || defaultName || "clipboard.png";
        setStatus(`${label} ready (preprocess skipped).`);
        if (file) setPreviewBlob(file);
      }

      setReady(true);
    };

    if (areaEl) {
      areaEl.addEventListener("click", () => {
        areaEl.focus();
      });
      areaEl.addEventListener("paste", handlePaste);
    }

    return {
      hasImage: () => Boolean(blob || file),
      getUploadBlob: () => blob || file,
      getUploadName: () => fileName,
      setStatus,
      setReady,
    };
  };

  const worksheetTarget = createPasteTarget({
    areaEl: worksheetPasteArea,
    previewEl: worksheetPreview,
    statusEl: worksheetStatus,
    label: "Clipboard image",
    defaultName: "clipboard.png",
    onStateChange: (ready) => {
      if (pasteButton) {
        pasteButton.disabled = !ready;
      }
    },
  });

  const keysTarget = createPasteTarget({
    areaEl: keysPasteArea,
    previewEl: keysPreview,
    statusEl: keysStatus,
    label: "Keys image",
    defaultName: "keys.png",
  });

  const uploadKeysTarget = createPasteTarget({
    areaEl: uploadKeysArea,
    previewEl: uploadKeysPreview,
    statusEl: uploadKeysStatus,
    label: "Keys image",
    defaultName: "keys.png",
  });

  if (pasteButton) {
    pasteButton.addEventListener("click", async () => {
      const uploadBlob = worksheetTarget.getUploadBlob();
      if (!uploadBlob) {
        worksheetTarget.setStatus("Paste an image first.");
        return;
      }

      const formData = new FormData();
      formData.append("image", uploadBlob, worksheetTarget.getUploadName());
      if (keysTarget.hasImage()) {
        formData.append("keys_image", keysTarget.getUploadBlob(), keysTarget.getUploadName());
      }
      formData.append("cv_debug", isCvDebugEnabled() ? "1" : "0");

      pasteButton.disabled = true;
      worksheetTarget.setStatus("Uploading clipboard image...");
      showProcessing();

      try {
        const response = await fetch("/paste", {
          method: "POST",
          body: formData,
        });

        if (response.redirected) {
          window.location.href = response.url;
          return;
        }

        const html = await response.text();
        document.open();
        document.write(html);
        document.close();
      } catch (_error) {
        worksheetTarget.setStatus("Failed to upload clipboard image.");
        hideProcessing();
        pasteButton.disabled = !worksheetTarget.hasImage();
      }
    });
  }

  if (uploadForm) {
    uploadForm.addEventListener("submit", async (event) => {
      syncCvDebugHidden();
      if (!uploadKeysTarget.hasImage()) {
        showProcessing();
        return;
      }

      event.preventDefault();
      showProcessing();

      const submitBtn = uploadForm.querySelector("button[type=\"submit\"]");
      if (submitBtn) submitBtn.disabled = true;

      const formData = new FormData(uploadForm);
      formData.append(
        "keys_image",
        uploadKeysTarget.getUploadBlob(),
        uploadKeysTarget.getUploadName(),
      );

      try {
        const response = await fetch("/upload", {
          method: "POST",
          body: formData,
        });

        if (response.redirected) {
          window.location.href = response.url;
          return;
        }

        const html = await response.text();
        document.open();
        document.write(html);
        document.close();
      } catch (_error) {
        uploadKeysTarget.setStatus("Failed to upload file.");
        hideProcessing();
        if (submitBtn) submitBtn.disabled = false;
      }
    });
  }
})();
