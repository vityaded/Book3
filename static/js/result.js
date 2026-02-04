(() => {
  const answerWraps = document.querySelectorAll(".answer-wrap");
  if (!answerWraps.length) return;

  const pageCards = document.querySelectorAll(".page-card");

  const FIXED_CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "shan't": "shall not",
    "let's": "let us",
  };
  const GENERAL_SUFFIXES = {
    "n't": "not",
    "'re": "are",
    "'ve": "have",
    "'ll": "will",
    "'m": "am",
  };
  const AMBIGUOUS_SUFFIXES = {
    "'s": ["is", "has"],
    "'d": ["would", "had"],
  };

  const normalizeApostrophes = (value) =>
    String(value || "").replace(/[\u2019\u2018\u02BC\u2032`´]/g, "'");

  const isAllUpper = (value) =>
    Boolean(value) && value === value.toUpperCase() && /[A-Z]/.test(value);
  const isCapitalized = (value) =>
    Boolean(value) &&
    value[0] === value[0].toUpperCase() &&
    value.slice(1) === value.slice(1).toLowerCase();

  const applyCase = (value, original) => {
    if (isAllUpper(original)) return value.toUpperCase();
    if (isCapitalized(original)) {
      return value.charAt(0).toUpperCase() + value.slice(1);
    }
    return value;
  };

  const normalize = (value) =>
    normalizeApostrophes(value)
      .toLowerCase()
      .replace(/'/g, "")
      .replace(/[^a-z0-9]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();

  const expandCoreVariants = (core) => {
    if (!core) return [""];
    const lower = core.toLowerCase();
    if (FIXED_CONTRACTIONS[lower]) {
      return [FIXED_CONTRACTIONS[lower]];
    }
    for (const [suffix, options] of Object.entries(AMBIGUOUS_SUFFIXES)) {
      if (lower.endsWith(suffix) && lower.length > suffix.length) {
        const base = lower.slice(0, -suffix.length);
        return options.map((option) => `${base} ${option}`);
      }
    }
    for (const [suffix, expansion] of Object.entries(GENERAL_SUFFIXES)) {
      if (lower.endsWith(suffix) && lower.length > suffix.length) {
        const base = lower.slice(0, -suffix.length);
        return [`${base} ${expansion}`];
      }
    }
    return [lower];
  };

  const expandContractionsVariants = (text) => {
    const tokens = normalizeApostrophes(text).split(/\s+/).filter(Boolean);
    if (!tokens.length) return [""];
    let variants = [""];
    tokens.forEach((token) => {
      const core = token.replace(/^[^A-Za-z0-9']+|[^A-Za-z0-9']+$/g, "");
      if (!core) return;
      const expansions = expandCoreVariants(core);
      const next = [];
      variants.forEach((prefix) => {
        expansions.forEach((expansion) => {
          next.push(prefix ? `${prefix} ${expansion}` : expansion);
        });
      });
      variants = next;
    });
    return variants.length ? variants : [""];
  };

  const expandTokenDisplay = (token) => {
    token = normalizeApostrophes(token);
    const match = token.match(/^([^A-Za-z0-9']*)([A-Za-z0-9']+)([^A-Za-z0-9']*)$/);
    if (!match) return token;
    const [, prefix, core, suffix] = match;
    const lower = core.toLowerCase();
    if (FIXED_CONTRACTIONS[lower]) {
      const replacement = applyCase(FIXED_CONTRACTIONS[lower], core);
      return `${prefix}${replacement}${suffix}`;
    }
    for (const [suffixText, options] of Object.entries(AMBIGUOUS_SUFFIXES)) {
      if (lower.endsWith(suffixText) && core.length > suffixText.length) {
        const baseRaw = core.slice(0, -suffixText.length);
        const expansionWord = options[0];
        let combined = `${baseRaw} ${expansionWord}`;
        if (isAllUpper(core)) {
          combined = combined.toUpperCase();
        }
        return `${prefix}${combined}${suffix}`;
      }
    }
    for (const [suffixText, expansionWord] of Object.entries(GENERAL_SUFFIXES)) {
      if (lower.endsWith(suffixText) && core.length > suffixText.length) {
        const baseRaw = core.slice(0, -suffixText.length);
        let combined = `${baseRaw} ${expansionWord}`;
        if (isAllUpper(core)) {
          combined = combined.toUpperCase();
        }
        return `${prefix}${combined}${suffix}`;
      }
    }
    return token;
  };

  const expandContractionsDisplay = (text) => {
    const parts = String(text || "").split(/(\s+)/);
    const expanded = parts
      .map((part) => {
        if (!part || /\s+/.test(part)) return part;
        return expandTokenDisplay(part);
      })
      .join("");
    return expanded.replace(/\s+/g, " ").trim();
  };

  const buildNormalizedVariants = (text) => {
    const variants = new Set();
    const rawNormalized = normalize(text);
    if (rawNormalized) variants.add(rawNormalized);
    expandContractionsVariants(text).forEach((variant) => {
      const normalized = normalize(variant);
      if (normalized) variants.add(normalized);
    });
    return variants;
  };

  const setState = (wrap, isCorrect) => {
    wrap.classList.toggle("answer-correct", isCorrect === "correct");
    wrap.classList.toggle("answer-wrong", isCorrect === "wrong");
    wrap.classList.toggle("answer-given", isCorrect === "given");
    wrap.dataset.state = isCorrect || "";
  };

  const wrapMeta = new Map();

  answerWraps.forEach((wrap) => {
    const expectedRaw = String(wrap.dataset.answer || "").trim();
    const hasExpected = expectedRaw.length > 0;
    const expectedVariants = hasExpected ? buildNormalizedVariants(expectedRaw) : new Set();
    const expandedAnswer = hasExpected ? expandContractionsDisplay(expectedRaw) : "";
    wrapMeta.set(wrap, {
      expectedVariants,
      expandedAnswer,
      attempts: 0,
      giveUpPrompted: false,
      hasExpected,
    });
  });

  const updateScores = () => {
    if (!pageCards.length) return;
    pageCards.forEach((pageEl) => {
      const scoreCurrentEl = pageEl.querySelector("[data-score-current]");
      const scoreTotalEl = pageEl.querySelector("[data-score-total]");
      if (!scoreCurrentEl || !scoreTotalEl) return;

      const pageWraps = pageEl.querySelectorAll(".answer-wrap");
      let correctCount = 0;
      let totalCount = 0;
      pageWraps.forEach((wrap) => {
        const meta = wrapMeta.get(wrap);
        if (!meta || !meta.hasExpected) return;
        totalCount += 1;
        if (wrap.dataset.state === "correct") {
          correctCount += 1;
        }
      });

      scoreCurrentEl.textContent = String(correctCount);
      scoreTotalEl.textContent = String(totalCount);
    });
  };

  const isAnswerCorrect = (entered, expectedVariants) => {
    if (!expectedVariants || expectedVariants.size === 0) return false;
    const enteredVariants = buildNormalizedVariants(entered);
    for (const variant of enteredVariants) {
      if (expectedVariants.has(variant)) return true;
    }
    return false;
  };

  const applyGiveUp = (wrap, input, meta) => {
    setState(wrap, "given");
    input.value = meta.expandedAnswer || "";
    input.readOnly = true;
  };

  const checkAnswer = (input) => {
    const wrap = input.closest(".answer-wrap");
    if (!wrap) return;
    const meta = wrapMeta.get(wrap);
    if (!meta || !meta.hasExpected) return;
    if (wrap.dataset.state === "given") return;

    const entered = input.value || "";
    const correct = isAnswerCorrect(entered, meta.expectedVariants);
    if (correct) {
      setState(wrap, "correct");
      updateScores();
      return;
    }

    meta.attempts += 1;
    setState(wrap, "wrong");

    if (meta.attempts >= 4 && !meta.giveUpPrompted) {
      meta.giveUpPrompted = true;
      if (window.confirm("Give up?")) {
        applyGiveUp(wrap, input, meta);
      }
    }
    updateScores();
  };

  updateScores();

  answerWraps.forEach((wrap) => {
    const input = wrap.querySelector(".answer-input");
    if (!input) return;

    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      if (input.tagName === "TEXTAREA" && event.shiftKey) return;
      event.preventDefault();
      checkAnswer(input);
    });
  });
})();
