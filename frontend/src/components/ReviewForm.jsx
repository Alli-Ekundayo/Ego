import { useState } from "react";
import { PaperPlaneRight, DownloadSimple } from "@phosphor-icons/react";
import MagneticButton from "./MagneticButton";

export default function ReviewForm({ setReviewState, selectedUserId, selectedUser }) {
  const [loadingSubmit, setLoadingSubmit] = useState(false);
  const [formError, setFormError] = useState("");
  const [formData, setFormData] = useState({
    name: "Infinix Hot 40 Pro",
    category: "Smartphones",
    description: "6.78-inch display, 108MP camera, 5000mAh battery, smooth performance for everyday use.",
  });

  const handleChange = (e) => setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedUserId) {
      setReviewState({ status: "error", data: null, error: "Select a user before running Task A." });
      setFormError("Select a user before running Task A.");
      return;
    }

    setLoadingSubmit(true);
    setReviewState({ status: "loading", data: null, error: null });
    setFormError("");

    const payload = {
      user_id: selectedUserId,
      item: {
        name: formData.name,
        category: formData.category,
        description: formData.description || null,
      },
    };

    try {
      const res = await fetch("/api/simulate-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setReviewState({ status: "success", data, error: null });
    } catch (err) {
      setReviewState({ status: "error", data: null, error: err.message });
    } finally {
      setLoadingSubmit(false);
    }
  };

  const loadSample = () => {
    setFormData({
      name: "Infinix Hot 40 Pro",
      category: "Smartphones",
      description: "6.78-inch display, 108MP camera, 5000mAh battery, smooth performance for everyday use.",
    });
  };

  return (
    <form onSubmit={handleSubmit} className="mt-4 flex w-full flex-col gap-5">
      <div className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-zinc-50/70 dark:bg-zinc-950/70 p-4">
        <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Selected user</p>
        <p className="mt-2 text-sm font-medium text-zinc-950 dark:text-white">
          {selectedUser ? selectedUser.name : "Select a user above"}
        </p>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          {selectedUser ? `${selectedUser.review_count} past reviews` : "Task A uses the selected profile automatically."}
        </p>
        {formError && !selectedUserId && (
          <p className="mt-2 text-xs font-medium text-rose-600 dark:text-rose-455">{formError}</p>
        )}
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        <label className="flex flex-col gap-2">
          <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">Product name</span>
          <input
            type="text"
            name="name"
            value={formData.name}
            onChange={handleChange}
            required
            className="input-base"
          />
          <span className="text-xs text-zinc-500 dark:text-zinc-450">Use the catalog name seen by shoppers.</span>
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">Category</span>
          <input
            type="text"
            name="category"
            value={formData.category}
            onChange={handleChange}
            required
            className="input-base"
          />
          <span className="text-xs text-zinc-500 dark:text-zinc-450">Example: Smartphones, Home Audio, Skin Care.</span>
        </label>
      </div>

      <label className="flex flex-col gap-2">
        <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">Product description</span>
        <textarea
          name="description"
          rows={4}
          value={formData.description}
          onChange={handleChange}
          required
          className="input-base resize-none"
        />
        <span className="text-xs text-zinc-500 dark:text-zinc-450">Keep it factual and under three sentences.</span>
      </label>

      <div className="mt-2 flex items-center gap-4">
        <MagneticButton
          type="submit"
          disabled={loadingSubmit || !selectedUserId}
          className="flex-1 rounded-xl bg-zinc-950 dark:bg-zinc-50 px-6 py-4 font-semibold text-white dark:text-zinc-950 shadow-[0_10px_20px_-14px_rgba(9,9,11,0.55)] dark:shadow-[0_10px_20px_-14px_rgba(255,255,255,0.15)] transition-transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <span className="inline-flex items-center justify-center gap-2">
            {loadingSubmit ? <span className="h-2 w-2 rounded-full bg-white/80 dark:bg-zinc-950/80 animate-pulse" /> : <PaperPlaneRight weight="fill" />}
            {loadingSubmit ? "Simulating..." : "Simulate review"}
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
