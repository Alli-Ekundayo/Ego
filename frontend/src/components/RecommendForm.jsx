import { useState } from "react";
import { MagicWand, Sparkle, DownloadSimple } from "@phosphor-icons/react";
import MagneticButton from "./MagneticButton";

const QUICK_TURNS = [
  { role: "user", content: "Actually, show me cheaper options." },
  { role: "user", content: "Prefer better battery life." },
  { role: "user", content: "Focus more on comfort than brand." },
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

export default function RecommendForm({ setRecommendState, selectedUserId, selectedUser }) {
  const [loadingSubmit, setLoadingSubmit] = useState(false);
  const [formError, setFormError] = useState("");
  const [formData, setFormData] = useState({
    n: 5,
    context: "I need a pair of wireless earbuds for the gym. Prioritize comfort, battery life, and value.",
    persona_description: "budget-conscious tech lover who wants dependable products with strong value for money",
    domain_filter: "electronics",
  });
  const [sessionHistory, setSessionHistory] = useState([
    { role: "user", content: "Looking for audio gear" },
    { role: "assistant", content: "Focus on comfort and battery life." },
  ]);
  const [draftRole, setDraftRole] = useState("user");
  const [draftContent, setDraftContent] = useState("");

  const handleChange = (e) => setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));

  const addDraftTurn = () => {
    const content = draftContent.trim();
    if (!content) return;
    setSessionHistory((prev) => [...prev, { role: draftRole, content }]);
    setDraftContent("");
    setDraftRole("user");
  };

  const addQuickTurn = (turn) => {
    setSessionHistory((prev) => [...prev, turn]);
  };

  const clearHistory = () => setSessionHistory([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedUserId) {
      setRecommendState({ status: "error", data: null, error: "Select a user before running Task B." });
      setFormError("Select a user before running Task B.");
      return;
    }

    setLoadingSubmit(true);
    setRecommendState({ status: "loading", data: null, error: null });
    setFormError("");

    const payload = {
      user_id: selectedUserId,
      context: formData.context,
      n: Number(formData.n),
      persona_description: formData.persona_description,
      session_history: sessionHistory.filter((h) => h.content.trim()),
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

  const loadSample = () => {
    setFormData({
      n: 5,
      context: "I need a pair of wireless earbuds for the gym. Prioritize comfort, battery life, and value.",
      persona_description: "budget-conscious tech lover who wants dependable products with strong value for money",
      domain_filter: "electronics",
    });
    setSessionHistory([
      { role: "user", content: "Looking for audio gear" },
      { role: "assistant", content: "Focus on comfort and battery life." },
    ]);
    setDraftContent("");
    setDraftRole("user");
  };

  return (
    <form onSubmit={handleSubmit} className="mt-4 flex w-full flex-col gap-5">
      <div className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-zinc-50/70 dark:bg-zinc-950/70 p-4">
        <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Selected user</p>
        <p className="mt-2 text-sm font-medium text-zinc-950 dark:text-white">
          {selectedUser ? selectedUser.name : "Select a user above"}
        </p>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          {selectedUser ? `${selectedUser.review_count} past reviews · ${selectedUser.mean_rating.toFixed(2)} avg rating` : "Task B uses the selected profile automatically."}
        </p>
        {formError && !selectedUserId && (
          <p className="mt-2 text-xs font-medium text-rose-600 dark:text-rose-400">{formError}</p>
        )}
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.12fr)_minmax(320px,0.88fr)]">
        <article className="rounded-[2rem] border border-zinc-200/70 dark:border-zinc-800/80 bg-white/80 dark:bg-zinc-900/80 p-5 backdrop-blur sm:p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Multiturn chat</p>
              <h3 className="mt-3 text-2xl font-semibold tracking-tight text-zinc-950 dark:text-white">
                Session history
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-650 dark:text-zinc-400">
                Add user and assistant turns to guide the recommendation engine as the
                conversation changes.
              </p>
            </div>

            <span className="rounded-full border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-950 px-3 py-1 text-xs text-zinc-600 dark:text-zinc-350">
              {sessionHistory.length} turns
            </span>
          </div>

          <div className="mt-5 max-h-[24rem] space-y-3 overflow-y-auto pr-1">
            {sessionHistory.length === 0 && (
              <div className="rounded-2xl border border-dashed border-zinc-200/70 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 px-4 py-8 text-center text-sm text-zinc-500 dark:text-zinc-450">
                Start the conversation with a new turn.
              </div>
            )}

            {sessionHistory.map((turn, idx) => (
              <div key={`${turn.role}-${idx}`} className={`flex ${turn.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`group relative max-w-[86%] rounded-[1.35rem] px-4 py-3 shadow-sm ${
                    turn.role === "user"
                      ? "bg-zinc-950 dark:bg-zinc-50 text-white dark:text-zinc-950"
                      : "border border-zinc-200/70 dark:border-zinc-800/85 bg-white dark:bg-zinc-900 text-zinc-850 dark:text-zinc-200"
                  }`}
                >
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <span className={`text-[10px] uppercase tracking-[0.22em] ${turn.role === "user" ? "text-white/60 dark:text-zinc-950/60" : "text-zinc-400 dark:text-zinc-500"}`}>
                      {turn.role}
                    </span>
                    <button
                      type="button"
                      onClick={() => setSessionHistory(sessionHistory.filter((_, i) => i !== idx))}
                      className={`text-xs transition ${turn.role === "user" ? "text-white/45 dark:text-zinc-950/45 hover:text-white dark:hover:text-zinc-950" : "text-zinc-450 dark:text-zinc-400 hover:text-zinc-650 dark:hover:text-white"}`}
                    >
                      ×
                    </button>
                  </div>
                  <p className="text-sm leading-relaxed">{turn.content}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-5 rounded-2xl border border-zinc-200/70 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Add a turn</p>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">Build the session history before submitting.</p>
              </div>

              <div className="inline-flex rounded-full border border-zinc-200/70 dark:border-zinc-850 bg-white dark:bg-zinc-900 p-1">
                {["user", "assistant"].map((role) => (
                  <button
                    key={role}
                    type="button"
                    onClick={() => setDraftRole(role)}
                    className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                      draftRole === role
                        ? "bg-zinc-950 dark:bg-zinc-50 text-white dark:text-zinc-950"
                        : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white"
                    }`}
                  >
                    {role}
                  </button>
                ))}
              </div>
            </div>

            <textarea
              value={draftContent}
              onChange={(e) => setDraftContent(e.target.value)}
              rows={3}
              placeholder="Type the next message in the conversation..."
              className="input-base mt-3 w-full resize-none"
            />

            <div className="mt-3 flex flex-wrap gap-2">
              {QUICK_TURNS.map((turn) => (
                <button
                  key={turn.content}
                  type="button"
                  onClick={() => addQuickTurn(turn)}
                  className="rounded-full border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-3 py-1.5 text-xs text-zinc-650 dark:text-zinc-350 transition hover:bg-zinc-50 dark:hover:bg-zinc-850 hover:text-zinc-950 dark:hover:text-white"
                >
                  {turn.content}
                </button>
              ))}
            </div>

            <div className="mt-4 flex items-center gap-3">
              <button
                type="button"
                onClick={addDraftTurn}
                className="inline-flex items-center gap-2 rounded-xl bg-zinc-950 dark:bg-zinc-50 px-4 py-3 text-sm font-medium text-white dark:text-zinc-950 transition-all hover:-translate-y-0.5"
              >
                <Sparkle size={14} weight="bold" />
                Add turn
              </button>
              <button
                type="button"
                onClick={clearHistory}
                className="rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-4 py-3 text-sm font-medium text-zinc-600 dark:text-zinc-350 transition-all hover:-translate-y-0.5 hover:text-zinc-950 dark:hover:text-white"
              >
                Clear history
              </button>
            </div>
          </div>
        </article>        <article className="space-y-5">
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
                value={formData.persona_description}
                onChange={handleChange}
                className="input-base resize-none"
              />
              <span className="text-xs text-zinc-500 dark:text-zinc-450">Describe the shopper's intent or constraints.</span>
            </label>

            <div className="mt-5 rounded-2xl border border-zinc-200/70 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 p-4">
              <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Session summary</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-4 py-3 text-xs text-zinc-600 dark:text-zinc-450">
                  {selectedUser ? `${formatCategory(selectedUser.top_category)} audience` : "No user selected"}
                </div>
                <div className="rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-4 py-3 text-xs text-zinc-600 dark:text-zinc-450">
                  {sessionHistory.length} conversation turns
                </div>
              </div>
            </div>
          </div>
        </article>
      </div>

      <div className="flex items-center gap-4">
        <MagneticButton
          type="submit"
          disabled={loadingSubmit || !selectedUserId}
          className="flex-1 rounded-xl bg-zinc-950 dark:bg-zinc-50 px-6 py-4 font-semibold text-white dark:text-zinc-950 shadow-[0_10px_20px_-14px_rgba(9,9,11,0.55)] dark:shadow-[0_10px_20px_-14px_rgba(255,255,255,0.15)] transition-transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <span className="inline-flex items-center justify-center gap-2">
            {loadingSubmit ? <span className="h-2 w-2 rounded-full bg-white/80 dark:bg-zinc-950/80 animate-pulse" /> : <MagicWand weight="fill" />}
            {loadingSubmit ? "Generating..." : "Generate recommendations"}
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
