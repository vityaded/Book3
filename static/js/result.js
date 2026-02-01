(() => {
  const answerWraps = document.querySelectorAll(".answer-wrap");
  if (!answerWraps.length) return;

  const normalize = (value) =>
    String(value || "")
      .trim()
      .replace(/\s+/g, " ")
      .toLowerCase();

  const setState = (wrap, isCorrect) => {
    wrap.classList.toggle("answer-correct", isCorrect);
    wrap.classList.toggle("answer-wrong", !isCorrect);
  };

  const checkAnswer = (input) => {
    const wrap = input.closest(".answer-wrap");
    if (!wrap) return;
    const expected = String(wrap.dataset.answer || "").trim();
    if (!expected) return;

    const entered = input.value || "";
    const correct = normalize(entered) === normalize(expected);
    setState(wrap, correct);

    if (!correct) {
      input.dataset.lastGuess = entered;
      input.value = expected;
    }
  };

  answerWraps.forEach((wrap) => {
    const input = wrap.querySelector(".answer-input");
    if (!input) return;

    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      if (input.tagName === "TEXTAREA" && event.shiftKey) return;
      event.preventDefault();
      checkAnswer(input);
    });

    input.addEventListener("input", () => {
      wrap.classList.remove("answer-correct", "answer-wrong");
    });
  });
})();
