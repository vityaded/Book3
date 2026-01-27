(() => {
  const pasteArea = document.getElementById("paste-area");
  const pastePreview = document.getElementById("paste-preview");
  const pasteButton = document.getElementById("paste-submit");
  const clipboardStatus = document.getElementById("clipboard-status");
  const processing = document.getElementById("processing");
  const uploadForm = document.getElementById("upload-form");

  let clipboardFile = null;

  const setStatus = (message) => {
    if (clipboardStatus) {
      clipboardStatus.textContent = message;
    }
  };

  const showProcessing = () => {
    if (processing) {
      processing.classList.add("active");
    }
  };

  if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
      showProcessing();
    });
  }

  if (pasteArea) {
    pasteArea.addEventListener("click", () => {
      pasteArea.focus();
    });

    pasteArea.addEventListener("paste", (event) => {
      const items = event.clipboardData ? event.clipboardData.items : [];
      let found = false;

      for (const item of items) {
        if (item.type && item.type.startsWith("image/")) {
          clipboardFile = item.getAsFile();
          found = true;
          break;
        }
      }

      if (!found) {
        setStatus("Clipboard does not contain an image.");
        return;
      }

      setStatus("Clipboard image ready.");
      if (pasteButton) {
        pasteButton.disabled = false;
      }

      if (pastePreview && clipboardFile) {
        const reader = new FileReader();
        reader.onload = () => {
          pastePreview.src = reader.result;
          pastePreview.style.display = "block";
        };
        reader.readAsDataURL(clipboardFile);
      }
    });
  }

  if (pasteButton) {
    pasteButton.addEventListener("click", async () => {
      if (!clipboardFile) {
        setStatus("Paste an image first.");
        return;
      }

      const formData = new FormData();
      formData.append("image", clipboardFile, clipboardFile.name || "clipboard.png");

      pasteButton.disabled = true;
      setStatus("Uploading clipboard image...");
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
      } catch (error) {
        setStatus("Failed to upload clipboard image.");
        pasteButton.disabled = false;
      }
    });
  }
})();
