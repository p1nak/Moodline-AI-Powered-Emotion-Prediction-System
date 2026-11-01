(() => {
  "use strict";

  /* ============================================================
     EMOTION CONFIGURATION
     ============================================================ */
  const EMOTIONS = {
    sadness: { emoji: "😢", label: "Sadness", color: "#4789DC" },
    joy: { emoji: "😄", label: "Joy", color: "#E5A817" },
    love: { emoji: "❤️", label: "Love", color: "#E75F87" },
    anger: { emoji: "😠", label: "Anger", color: "#E45C57" },
    fear: { emoji: "😨", label: "Fear", color: "#9270D8" },
    surprise: { emoji: "😲", label: "Surprise", color: "#159D9A" },
  };

  /* ============================================================
     DOM ELEMENTS
     ============================================================ */
  const el = {
    statusDot: document.getElementById("statusDot"),
    serverStatusText: document.getElementById("serverStatusText"),
    textInput: document.getElementById("textInput"),
    charCount: document.getElementById("charCount"),
    analyzeBtn: document.getElementById("analyzeBtn"),
    clearBtn: document.getElementById("clearBtn"),
    errorMsg: document.getElementById("errorMsg"),
    errorMsgText: document.getElementById("errorMsgText"),
    orb: document.getElementById("orb"),
    orbGlow: document.getElementById("orbGlow"),
    orbEmoji: document.getElementById("orbEmoji"),
    orbStatusText: document.getElementById("orbStatusText"),
    resultSection: document.getElementById("resultSection"),
    revealCard: document.getElementById("revealCard"),
    emotionWord: document.getElementById("emotionWord"),
    emotionEmoji: document.getElementById("emotionEmoji"),
    confidenceText: document.getElementById("confidenceText"),
    confidenceNumber: document.getElementById("confidenceNumber"),
    radialFill: document.getElementById("radialFill"),
    echoedText: document.getElementById("echoedText"),
    barsContainer: document.getElementById("barsContainer"),
    copyResultBtn: document.getElementById("copyResultBtn"),
    copyBtnText: document.getElementById("copyBtnText"),
    exampleChips: document.querySelectorAll(".chip-btn"),
  };

  let modelReady = false;
  let activeAnimationId = null;

  /* ============================================================
     HEALTH CHECK — Model Status Polling
     ============================================================ */
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("Status " + res.status);
      const data = await res.json();

      modelReady = !!data.model_loaded;
      if (modelReady) {
        setStatus("live", "Model online & ready");
        if (el.orbStatusText && !document.body.dataset.emotion) {
          el.orbStatusText.textContent = "Ready to analyze";
        }
      } else {
        setStatus("warming", "Warming up BiGRU model…");
        if (el.orbStatusText) {
          el.orbStatusText.textContent = "Model warming up…";
        }
        setTimeout(checkHealth, 3000);
      }
    } catch (e) {
      modelReady = false;
      setStatus("down", "Connecting to model…");
      if (el.orbStatusText) {
        el.orbStatusText.textContent = "Waiting for backend…";
      }
      setTimeout(checkHealth, 5000);
    }
    syncButtonState();
  }

  function setStatus(kind, text) {
    if (el.statusDot) {
      el.statusDot.className = "brand-mark status-dot " + kind;
    }
    if (el.serverStatusText) {
      el.serverStatusText.textContent = text;
    }
  }

  /* ============================================================
     INPUT HANDLING & VALIDATION
     ============================================================ */
  function updateCharCount() {
    const len = el.textInput.value.length;
    el.charCount.textContent = len;

    if (len >= 2000) {
      el.charCount.parentElement.classList.add("at-limit");
      el.charCount.parentElement.classList.remove("near-limit");
    } else if (len >= 1800) {
      el.charCount.parentElement.classList.add("near-limit");
      el.charCount.parentElement.classList.remove("at-limit");
    } else {
      el.charCount.parentElement.classList.remove("near-limit", "at-limit");
    }
  }

  el.textInput.addEventListener("input", () => {
    updateCharCount();
    syncButtonState();
    if (el.errorMsg && !el.errorMsg.hidden) {
      hideError();
    }
  });

  el.textInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runAnalysis();
    }
  });

  function syncButtonState() {
    const hasText = el.textInput.value.trim().length > 0;
    el.analyzeBtn.disabled = !hasText || !modelReady;
  }

  // Clear button
  if (el.clearBtn) {
    el.clearBtn.addEventListener("click", () => {
      el.textInput.value = "";
      updateCharCount();
      syncButtonState();
      hideError();
      el.textInput.focus();
    });
  }

  // Example chips click
  if (el.exampleChips) {
    el.exampleChips.forEach((chip) => {
      chip.addEventListener("click", () => {
        const text = chip.getAttribute("data-text");
        if (text) {
          el.textInput.value = text;
          updateCharCount();
          syncButtonState();
          hideError();
          el.textInput.focus();
        }
      });
    });
  }

  el.analyzeBtn.addEventListener("click", runAnalysis);

  /* ============================================================
     ANALYSIS FLOW
     ============================================================ */
  async function runAnalysis() {
    const rawText = el.textInput.value;
    const text = rawText.trim();

    if (!text) {
      showError("Please enter a sentence to analyze.");
      el.textInput.focus();
      return;
    }

    if (!modelReady) {
      showError("The BiGRU model is still initializing. Please wait a moment.");
      return;
    }

    hideError();
    enterThinking();

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const detail =
          body && body.detail
            ? typeof body.detail === "string"
              ? body.detail
              : JSON.stringify(body.detail)
            : `Analysis failed (${res.status}).`;
        throw new Error(detail);
      }

      const data = await res.json();
      renderResult(data, text);
    } catch (err) {
      exitThinking(false);
      showError(err.message || "An unexpected error occurred during prediction. Please try again.");
    }
  }

  /* ============================================================
     STATE TRANSITIONS
     ============================================================ */
  function enterThinking() {
    el.analyzeBtn.classList.add("loading");
    el.analyzeBtn.disabled = true;
    const labelSpan = el.analyzeBtn.querySelector(".btn-label");
    if (labelSpan) labelSpan.textContent = "Analyzing…";

    if (el.orb) {
      el.orb.classList.remove("settled");
      el.orb.classList.add("thinking");
    }
    if (el.orbEmoji) {
      el.orbEmoji.style.opacity = "0";
    }
    if (el.orbStatusText) {
      el.orbStatusText.textContent = "Processing BiGRU Gates…";
    }
  }

  function exitThinking(success) {
    el.analyzeBtn.classList.remove("loading");
    const labelSpan = el.analyzeBtn.querySelector(".btn-label");
    if (labelSpan) labelSpan.textContent = "Analyze Emotion";
    syncButtonState();

    if (el.orb) {
      el.orb.classList.remove("thinking");
    }

    if (!success) {
      if (el.orbEmoji) {
        el.orbEmoji.textContent = "✎";
        el.orbEmoji.style.opacity = "1";
      }
      if (el.orbStatusText) {
        el.orbStatusText.textContent = modelReady ? "Ready to analyze" : "Model offline";
      }
    }
  }

  /* ============================================================
     RENDER PREDICTION RESULTS
     ============================================================ */
  function renderResult(data, originalText) {
    const emotion = (data.predicted_emotion || "joy").toLowerCase();
    const config = EMOTIONS[emotion] || { emoji: data.emoji || "✨", label: emotion, color: "#7357E8" };
    const emoji = data.emoji || config.emoji;

    // Apply emotion dynamic theme to body
    document.body.setAttribute("data-emotion", emotion);

    // Update Orb visualization
    if (el.orb) {
      el.orb.classList.add("settled");
    }
    if (el.orbEmoji) {
      el.orbEmoji.textContent = emoji;
      el.orbEmoji.style.opacity = "1";
    }
    if (el.orbStatusText) {
      el.orbStatusText.textContent = `${capitalize(emotion)} Detected`;
    }

    exitThinking(true);

    // Update reveal card text
    if (el.emotionWord) {
      el.emotionWord.textContent = capitalize(emotion);
    }
    if (el.emotionEmoji) {
      el.emotionEmoji.textContent = emoji;
      // Re-trigger emoji pop animation
      el.emotionEmoji.style.animation = "none";
      void el.emotionEmoji.offsetWidth;
      el.emotionEmoji.style.animation = "";
    }

    // Confidence metric calculation
    const rawConfidence = typeof data.confidence === "number" ? data.confidence : 0;
    const targetPct = rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence;

    // Smooth count-up animation for confidence score
    animateConfidenceCounter(targetPct);

    // Radial gauge fill
    if (el.radialFill) {
      el.radialFill.setAttribute("stroke-dasharray", `${targetPct.toFixed(1)}, 100`);
    }

    // Fallback confidence text if present
    if (el.confidenceText) {
      el.confidenceText.textContent = `${targetPct.toFixed(1)}% confidence`;
    }

    // Echo original sentence
    if (el.echoedText) {
      el.echoedText.textContent = `“${originalText}”`;
    }

    // Support both all_probabilities and previous all_probabilites
    const probabilities = data.all_probabilities || data.all_probabilites || {};
    renderBars(probabilities, emotion);

    // Reveal result section with animation
    if (el.resultSection) {
      el.resultSection.hidden = false;
      el.resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  /* ============================================================
     ANIMATE CONFIDENCE COUNTER
     ============================================================ */
  function animateConfidenceCounter(targetValue) {
    if (!el.confidenceNumber) return;

    if (activeAnimationId) {
      cancelAnimationFrame(activeAnimationId);
      activeAnimationId = null;
    }

    const duration = 800; // ms
    const startTime = performance.now();
    const startValue = 0;

    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = startValue + (targetValue - startValue) * ease;

      el.confidenceNumber.textContent = current.toFixed(1);

      if (progress < 1) {
        activeAnimationId = requestAnimationFrame(step);
      } else {
        el.confidenceNumber.textContent = targetValue.toFixed(1);
        activeAnimationId = null;
      }
    }

    activeAnimationId = requestAnimationFrame(step);
  }

  /* ============================================================
     RENDER PROBABILITY BARS (All 6 Emotions)
     ============================================================ */
  function renderBars(probs, topEmotion) {
    if (!el.barsContainer) return;

    // Ensure all 6 emotions are included even if zero
    const allEmotionsList = ["sadness", "joy", "love", "anger", "fear", "surprise"];
    const normalizedProbs = {};

    allEmotionsList.forEach((label) => {
      normalizedProbs[label] = typeof probs[label] === "number" ? probs[label] : 0;
    });

    // Sort descending by probability value
    const sortedEntries = Object.entries(normalizedProbs).sort((a, b) => b[1] - a[1]);

    el.barsContainer.innerHTML = "";

    sortedEntries.forEach(([label, value], index) => {
      const pct = (value <= 1 ? value * 100 : value);
      const isTop = label.toLowerCase() === topEmotion.toLowerCase() || index === 0;
      const cfg = EMOTIONS[label] || { emoji: "•", label };

      const row = document.createElement("div");
      row.className = `bar-row bar-${label}${isTop ? " top-match" : ""}`;
      row.innerHTML = `
        <div class="bar-label-group">
          <span class="bar-emoji">${cfg.emoji}</span>
          <span class="bar-name">${cfg.label}</span>
        </div>
        <div class="bar-track" title="${pct.toFixed(2)}%">
          <div class="bar-fill" style="width: 0%;"></div>
        </div>
        <div class="bar-pct-badge">${pct.toFixed(1)}%</div>
      `;

      el.barsContainer.appendChild(row);

      // Smooth staggered entrance for progress bar fill
      const fillEl = row.querySelector(".bar-fill");
      setTimeout(() => {
        if (fillEl) {
          fillEl.style.width = `${Math.max(pct, 0.5).toFixed(1)}%`;
        }
      }, 50 + index * 60);
    });
  }

  /* ============================================================
     COPY RESULT SUMMARY
     ============================================================ */
  if (el.copyResultBtn) {
    el.copyResultBtn.addEventListener("click", async () => {
      const emotion = el.emotionWord ? el.emotionWord.textContent : "";
      const confidence = el.confidenceNumber ? el.confidenceNumber.textContent : "";
      const text = el.echoedText ? el.echoedText.textContent : "";

      const summary = `Moodline AI Prediction:\nEmotion: ${emotion}\nConfidence: ${confidence}%\nInput: ${text}`;

      try {
        await navigator.clipboard.writeText(summary);
        if (el.copyBtnText) {
          const original = el.copyBtnText.textContent;
          el.copyBtnText.textContent = "Copied!";
          setTimeout(() => {
            el.copyBtnText.textContent = original;
          }, 2000);
        }
      } catch (err) {
        // Fallback for older browsers
        const textarea = document.createElement("textarea");
        textarea.value = summary;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
        if (el.copyBtnText) {
          el.copyBtnText.textContent = "Copied!";
          setTimeout(() => {
            el.copyBtnText.textContent = "Copy";
          }, 2000);
        }
      }
    });
  }

  /* ============================================================
     ERROR HELPERS
     ============================================================ */
  function showError(msg) {
    if (el.errorMsgText) {
      el.errorMsgText.textContent = msg;
    }
    if (el.errorMsg) {
      el.errorMsg.hidden = false;
    }
  }

  function hideError() {
    if (el.errorMsg) {
      el.errorMsg.hidden = true;
    }
  }

  function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  /* ============================================================
     INITIALIZE
     ============================================================ */
  updateCharCount();
  checkHealth();
})();
