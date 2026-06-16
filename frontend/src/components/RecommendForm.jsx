import { useRef, useEffect, useState } from "react";
import { MagicWand, PaperPlaneTilt, DownloadSimple, ArrowCounterClockwise } from "@phosphor-icons/react";
import MagneticButton from "./MagneticButton";

const REFINEMENT_CHIPS = [
  "Actually, show me cheaper options.",
  "Prefer better battery life.",
  "Focus more on comfort than brand.",
  "Show me something more premium.",
  "I need something more compact.",
];

function formatCategory(value) {
  return value
    ? value
      .split(/\s+/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ")
    : "—";
}

export default function RecommendForm({ setRecommendState, selectedUserId, selectedUser, guestUser, recommendState }) {
  const [loadingSubmit, setLoadingSubmit] = useState(false);
  const [formError, setFormError] = useState("");
  const [formData, setFormData] = useState({
    n: 5,
    context: "I need a pair of wireless earbuds for the gym. Prioritize comfort, battery life, and value.",
    persona_description: "budget-conscious tech lover who wants dependable products with strong value for money",
    domain_filter: "electronics",
  });

  // Session history is now exclusively user messages + auto-generated assistant summaries
  const [sessionHistory, setSessionHistory] = useState([]);
  const [draftMessage, setDraftMessage] = useState("");
  const chatEndRef = useRef(null);

  const isReady = !!(selectedUserId || guestUser);
  const hasResults = recommendState?.status === "success" && recommendState?.data?.length > 0;

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [sessionHistory]);

  // When results come back, auto-append an assistant summary turn
  useEffect(() => {
    if (recommendState?.status === "success" && recommendState.data?.length > 0) {
      const names = recommendState.data.slice(0, 3).map((r) => r.name).join(", ");
      const summary = `Found ${recommendState.data.length} recommendation${recommendState.data.length !== 1 ? "s" : ""}: ${names}${recommendState.data.length > 3 ? ", and more." : "."}`;
      setSessionHistory((prev) => {
        // Avoid double-appending if last turn is already an assistant summary
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && last?.content === summary) return prev;
        return [...prev, { role: "assistant", content: summary }];
      });
    }
  }, [recommendState]);

  const handleChange = (e) => setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));

  const submitWithHistory = async (history) => {
    if (!isReady) {
      setRecommendState({ status: "error", data: null, error: "Select a user or enter a new user before running Task B." });
      setFormError("Select a user or enter a new user before running Task B.");
      return;
    }

    setLoadingSubmit(true);
    setRecommendState({ status: "loading", data: null, error: null });
    setFormError("");

    const payload = {
      user_id: selectedUserId || "guest_anonymous",
      context: formData.context,
      n: Number(formData.n),
      persona_description: guestUser?.persona || formData.persona_description,
      session_history: history.filter((h) => h.content.trim()),
      domain_filter: formData.domain_filter || null,
    };

    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setRecommendState({ status: "success", data: data.recommendations || [], error: null });
    } catch (err) {
      setRecommendState({ status: "error", data: null, error: err.message });
    } finally {
      setLoadingSubmit(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    await submitWithHistory(sessionHistory);
  };

  // User sends a follow-up message and immediately re-submits
  const handleSendMessage = async () => {
    const msg = draftMessage.trim();
    if (!msg) return;
    const newHistory = [...sessionHistory, { role: "user", content: msg }];
    setSessionHistory(newHistory);
    setDraftMessage("");
    await submitWithHistory(newHistory);
  };

  const handleChip = async (text) => {
    const newHistory = [...sessionHistory, { role: "user", content: text }];
    setSessionHistory(newHistory);
    await submitWithHistory(newHistory);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const resetConversation = () => {
    setSessionHistory([]);
    setRecommendState({ status: "idle", data: null, error: null });
  };

  const loadSample = () => {
    setFormData({
      n: 5,
      context: "I need a pair of wireless earbuds for the gym. Prioritize comfort, battery life, and value.",
      persona_description: "budget-conscious tech lover who wants dependable products with strong value for money",
      domain_filter: "electronics",
    });
    setSessionHistory([]);
    setDraftMessage("");
  };

  return (
    <form onSubmit={handleSubmit} className="mt-4 flex w-full flex-col gap-5">
      {/* Selected user strip */}
      <div className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-zinc-50/70 dark:bg-zinc-950/70 p-4">
        <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Selected user</p>
        <p className="mt-2 text-sm font-medium text-zinc-950 dark:text-white">
          {selectedUser ? selectedUser.name : guestUser ? guestUser.name : "Select a user above"}
        </p>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          {guestUser
            ? "New user — cold-start mode active"
            : selectedUser
              ? `${selectedUser.review_count} past reviews · ${selectedUser.mean_rating.toFixed(2)} avg rating`
              : "Task B uses the selected profile automatically."}
        </p>
        {formError && !isReady && (
          <p className="mt-2 text-xs font-medium text-rose-600 dark:text-rose-400">{formError}</p>
        )}
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.12fr)_minmax(320px,0.88fr)]">

        {/* ── Left: Conversation panel ── */}
        <article className="flex flex-col rounded-[2rem] border border-zinc-200/70 dark:border-zinc-800/80 bg-white/80 dark:bg-zinc-900/80 backdrop-blur overflow-hidden">
          <div className="flex items-start justify-between gap-4 p-5 pb-0 sm:p-6 sm:pb-0">
            <div>
              <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Multiturn chat</p>
              <h3 className="mt-3 text-2xl font-semibold tracking-tight text-zinc-950 dark:text-white">
                Conversation
              </h3>
              <p className="mt-1 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
                Type a follow-up after seeing results to refine your recommendations.
              </p>
            </div>
            {sessionHistory.length > 0 && (
              <button
                type="button"
                onClick={resetConversation}
                className="shrink-0 flex items-center gap-1.5 rounded-full border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-950 px-3 py-1.5 text-xs text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white transition"
              >
                <ArrowCounterClockwise size={12} weight="bold" />
                Reset
              </button>
            )}
          </div>

          {/* Chat bubbles */}
          <div className="mt-4 flex-1 min-h-[18rem] max-h-[26rem] overflow-y-auto px-5 pb-3 sm:px-6 space-y-3">
            {sessionHistory.length === 0 && (
              <div className="flex h-full min-h-[16rem] flex-col items-center justify-center text-center text-sm text-zinc-400 dark:text-zinc-500 gap-2">
                <MagicWand size={28} className="opacity-40" />
                <p>Submit a request to start the conversation.</p>
                <p className="text-xs text-zinc-500 dark:text-zinc-550">Follow-up messages will appear here after your first result.</p>
              </div>
            )}
            {sessionHistory.map((turn, idx) => (
              <div key={idx} className={`flex ${turn.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`relative max-w-[82%] rounded-[1.25rem] px-4 py-3 text-sm leading-relaxed shadow-sm ${
                    turn.role === "user"
                      ? "bg-zinc-950 dark:bg-zinc-50 text-white dark:text-zinc-950"
                      : "border border-zinc-200/70 dark:border-zinc-800/80 bg-zinc-50 dark:bg-zinc-900/80 text-zinc-700 dark:text-zinc-300"
                  }`}
                >
                  <p className={`mb-1 text-[9px] uppercase tracking-[0.2em] ${turn.role === "user" ? "text-white/50 dark:text-zinc-950/50" : "text-zinc-400 dark:text-zinc-500"}`}>
                    {turn.role === "user" ? "You" : "Ego"}
                  </p>
                  {turn.content}
                </div>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          {/* Refinement chips — only shown when there are already results */}
          {hasResults && (
            <div className="px-5 pb-2 sm:px-6">
              <p className="mb-2 text-[10px] uppercase tracking-[0.18em] text-zinc-400 dark:text-zinc-500">Refine</p>
              <div className="flex flex-wrap gap-2">
                {REFINEMENT_CHIPS.map((chip) => (
                  <button
                    key={chip}
                    type="button"
                    disabled={loadingSubmit}
                    onClick={() => handleChip(chip)}
                    className="rounded-full border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-3 py-1.5 text-xs text-zinc-650 dark:text-zinc-350 transition hover:bg-zinc-50 dark:hover:bg-zinc-850 hover:text-zinc-950 dark:hover:text-white disabled:opacity-50"
                  >
                    {chip}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message input — only shown after first submit */}
          {hasResults && (
            <div className="border-t border-zinc-200/70 dark:border-zinc-800 bg-zinc-50/80 dark:bg-zinc-950/80 px-4 py-3 sm:px-5">
              <div className="flex items-end gap-2">
                <textarea
                  value={draftMessage}
                  onChange={(e) => setDraftMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={2}
                  placeholder="Refine your request… (Enter to send)"
                  disabled={loadingSubmit}
                  className="input-base flex-1 resize-none text-sm disabled:opacity-60"
                />
                <button
                  type="button"
                  onClick={handleSendMessage}
                  disabled={loadingSubmit || !draftMessage.trim()}
                  className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-zinc-950 dark:bg-zinc-50 text-white dark:text-zinc-950 transition hover:-translate-y-0.5 disabled:opacity-40 disabled:translate-y-0"
                >
                  {loadingSubmit
                    ? <span className="h-2 w-2 rounded-full bg-white/80 dark:bg-zinc-950/80 animate-pulse" />
                    : <PaperPlaneTilt size={16} weight="fill" />
                  }
                </button>
              </div>
            </div>
          )}
        </article>

        {/* ── Right: Controls ── */}
        <article className="space-y-5">
          <div className="rounded-[2rem] border border-zinc-200/70 dark:border-zinc-800/80 bg-white/80 dark:bg-zinc-900/80 p-5 backdrop-blur sm:p-6">
            <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Current request</p>
            <h3 className="mt-3 text-2xl font-semibold tracking-tight text-zinc-950 dark:text-white">
              Run the recommendation endpoint
            </h3>
            <p className="mt-2 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
              This section sends the current request into <code className="rounded bg-zinc-100 dark:bg-zinc-850 px-1 py-0.5 text-[0.85em] text-zinc-800 dark:text-zinc-200">/api/recommend</code>.
            </p>

            <label className="mt-5 flex flex-col gap-2">
              <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">Shopping context</span>
              <textarea
                name="context"
                rows={4}
                value={formData.context}
                onChange={handleChange}
                required
                className="input-base resize-none"
              />
              <span className="text-xs text-zinc-500 dark:text-zinc-450">Explain the current shopping goal in plain language.</span>
            </label>
          </div>

          <div className="rounded-[2rem] border border-zinc-200/70 dark:border-zinc-800/80 bg-white/80 dark:bg-zinc-900/80 p-5 backdrop-blur sm:p-6">
            <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Context controls</p>

            <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <label className="flex flex-col gap-2">
                <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">Number of results</span>
                <input
                  type="number"
                  name="n"
                  min="1"
                  max="10"
                  value={formData.n}
                  onChange={handleChange}
                  required
                  className="input-base font-mono"
                />
                <span className="text-xs text-zinc-500 dark:text-zinc-450">Between 1 and 10 results.</span>
              </label>

              <label className="flex flex-col gap-2">
                <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">Domain filter</span>
                <input
                  type="text"
                  name="domain_filter"
                  value={formData.domain_filter}
                  onChange={handleChange}
                  placeholder="electronics"
                  className="input-base"
                />
                <span className="text-xs text-zinc-500 dark:text-zinc-450">Optional, leave blank for all domains.</span>
              </label>
            </div>

            <label className="mt-4 flex flex-col gap-2">
              <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">Persona hint</span>
              <textarea
                name="persona_description"
                rows={3}
                value={guestUser?.persona || formData.persona_description}
                onChange={guestUser ? undefined : handleChange}
                readOnly={!!guestUser}
                className={`input-base resize-none ${guestUser ? "opacity-70 cursor-default" : ""}`}
              />
              <span className="text-xs text-zinc-500 dark:text-zinc-450">
                {guestUser ? "Taken from new user persona above." : "Describe the shopper's intent or constraints."}
              </span>
            </label>

            <div className="mt-5 rounded-2xl border border-zinc-200/70 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 p-4">
              <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Session summary</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-4 py-3 text-xs text-zinc-600 dark:text-zinc-450">
                  {selectedUser ? `${formatCategory(selectedUser.top_category)} audience` : guestUser ? "New user (cold-start)" : "No user selected"}
                </div>
                <div className="rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-4 py-3 text-xs text-zinc-600 dark:text-zinc-450">
                  {sessionHistory.filter(h => h.role === "user").length} user turn{sessionHistory.filter(h => h.role === "user").length !== 1 ? "s" : ""}
                </div>
              </div>
            </div>
          </div>
        </article>
      </div>

      <div className="flex items-center gap-4">
        <MagneticButton
          type="submit"
          disabled={loadingSubmit || !isReady}
          className="flex-1 rounded-xl bg-zinc-950 dark:bg-zinc-50 px-6 py-4 font-semibold text-white dark:text-zinc-950 shadow-[0_10px_20px_-14px_rgba(9,9,11,0.55)] dark:shadow-[0_10px_20px_-14px_rgba(255,255,255,0.15)] transition-transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <span className="inline-flex items-center justify-center gap-2">
            {loadingSubmit ? <span className="h-2 w-2 rounded-full bg-white/80 dark:bg-zinc-950/80 animate-pulse" /> : <MagicWand weight="fill" />}
            {loadingSubmit ? "Generating..." : hasResults ? "Re-run recommendations" : "Generate recommendations"}
          </span>
        </MagneticButton>
        <button
          type="button"
          onClick={loadSample}
          className="flex items-center justify-center gap-2 rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-6 py-4 font-semibold text-zinc-700 dark:text-zinc-300 transition-all hover:-translate-y-0.5 hover:border-zinc-300/70 dark:hover:border-zinc-700 hover:text-zinc-950 dark:hover:text-white active:translate-y-[1px]"
        >
          <DownloadSimple weight="bold" />
          Sample
        </button>
      </div>
    </form>
  );
}
