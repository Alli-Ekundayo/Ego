import { memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { WarningCircle, ChatCircleText, Quotes, Sparkle, Star } from "@phosphor-icons/react";

const ShimmerBlock = memo(function ShimmerBlock({ className = "" }) {
  return (
    <div className={`relative overflow-hidden rounded-2xl bg-zinc-100/80 dark:bg-zinc-800/80 ${className}`}>
      <motion.div
        className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 dark:via-white/5 to-transparent"
        animate={{ x: ["-100%", "100%"] }}
        transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
});

export default function ResultsGrid({ activeTab, recommendState = { status: "idle" }, reviewState = { status: "idle" } }) {
  return (
    <section className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-white/80 dark:bg-zinc-900/80 p-6 shadow-[0_24px_40px_-30px_rgba(15,23,42,0.3)] dark:shadow-[0_24px_40px_-30px_rgba(0,0,0,0.5)] min-h-[420px] flex flex-col relative overflow-hidden">
      <div>
        <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400 mb-1">Live output</p>
        <h3 className="text-xl font-semibold tracking-tight text-zinc-950 dark:text-white m-0">
          {activeTab === "recommend" ? "Recommendations" : "Simulated Review"}
        </h3>
      </div>

      <div className="mt-6 flex-1 relative w-full">
        <AnimatePresence mode="wait">
          {activeTab === "recommend" && (
            <motion.div
              key="rec-container"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full flex flex-col gap-4"
            >
              {recommendState.status === "idle" && (
                <div className="flex flex-col items-center justify-center py-20 text-zinc-400 dark:text-zinc-500">
                  <Sparkle size={32} className="mb-2 opacity-50" />
                  <p className="text-sm">Waiting for a request.</p>
                  <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-450">Submit a context to see ranked products.</p>
                </div>
              )}
              
              {recommendState.status === "loading" && (
                <div className="flex flex-col gap-4">
                  <ShimmerBlock className="h-20" />
                  <ShimmerBlock className="h-20" />
                  <ShimmerBlock className="h-20" />
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">Generating recommendations...</p>
                </div>
              )}

              {recommendState.status === "error" && (
                <div className="flex items-start gap-3 p-4 bg-rose-55 dark:bg-rose-950/40 border border-rose-100 dark:border-rose-900/50 rounded-xl text-rose-600 dark:text-rose-400">
                  <WarningCircle size={24} weight="fill" className="shrink-0" />
                  <p className="text-sm font-medium">{recommendState.error}</p>
                </div>
              )}

              {recommendState.status === "success" && recommendState.data?.length === 0 && (
                <div className="p-4 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800 rounded-xl text-zinc-650 dark:text-zinc-405 text-center text-sm">
                  No recommendations found for this context. Try adjusting the domain filter or session history.
                </div>
              )}

              {recommendState.status === "success" && recommendState.data?.length > 0 && (
                <motion.div 
                  className="flex flex-col gap-4"
                  variants={{ show: { transition: { staggerChildren: 0.12 } } }}
                  initial="hidden"
                  animate="show"
                >
                  {recommendState.data.map((item, i) => (
                    <motion.div 
                      key={item.item_id || i}
                      layoutId={`rec-${item.item_id || i}`}
                      variants={{
                        hidden: { opacity: 0, y: 10 },
                        show: { opacity: 1, y: 0 }
                      }}
                      layout
                      className="p-5 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800/80 rounded-xl flex flex-col gap-3"
                    >
                      {/* Header row: rank badge + item id */}
                      <div className="flex items-center justify-between">
                        <span className="w-8 h-8 rounded-full border border-orange-200/70 dark:border-orange-900/50 bg-orange-50 dark:bg-orange-950/40 text-orange-700 dark:text-orange-400 flex items-center justify-center font-mono text-xs">
                          {String(i + 1).padStart(2, "00")}
                        </span>
                        <span className="px-3 py-1 rounded-full bg-white dark:bg-zinc-900 text-zinc-650 dark:text-zinc-400 text-[10px] font-semibold uppercase tracking-wider border border-zinc-200/70 dark:border-zinc-800">
                          {item.item_id || "item"}
                        </span>
                      </div>

                      {/* Product name */}
                      <h4 className="text-lg font-semibold text-zinc-950 dark:text-white m-0">{item.name || "Unknown product"}</h4>

                      {/* Price + rating row — shown only when data is available */}
                      {(item.price_value > 0 || item.rating_stats?.mean > 0) && (
                        <div className="flex flex-wrap items-center gap-2">
                          {/* Price badge */}
                          {item.price_value > 0 && (
                            <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200/70 dark:border-emerald-900/50 px-3 py-1 text-xs font-semibold text-emerald-700 dark:text-emerald-400">
                              ₦{item.price_value.toLocaleString("en-NG")}
                            </span>
                          )}
                          {/* Old price + discount */}
                          {item.old_price_value > 0 && (
                            <span className="text-[11px] text-zinc-400 dark:text-zinc-500 line-through font-mono">
                              ₦{item.old_price_value.toLocaleString("en-NG")}
                            </span>
                          )}
                          {item.discount_percent > 0 && (
                            <span className="inline-flex items-center rounded-full bg-rose-50 dark:bg-rose-950/40 border border-rose-200/70 dark:border-rose-900/50 px-2 py-0.5 text-[10px] font-bold text-rose-600 dark:text-rose-400">
                              -{item.discount_percent}%
                            </span>
                          )}
                          {/* Star rating */}
                          {item.rating_stats?.mean > 0 && (
                            <span className="inline-flex items-center gap-1 ml-auto text-[11px] text-zinc-500 dark:text-zinc-400">
                              {Array.from({ length: 5 }).map((_, idx) => {
                                const filled = idx < Math.round(item.rating_stats.mean);
                                return (
                                  <Star
                                    key={idx}
                                    size={12}
                                    weight={filled ? "fill" : "regular"}
                                    className={filled ? "text-orange-400" : "text-zinc-300 dark:text-zinc-700"}
                                  />
                                );
                              })}
                              <span className="font-mono ml-0.5">{item.rating_stats.mean.toFixed(1)}</span>
                              <span className="text-zinc-400 dark:text-zinc-600">({item.rating_stats.count})</span>
                            </span>
                          )}
                        </div>
                      )}

                      {/* Reason */}
                      <p className="text-sm text-zinc-650 dark:text-zinc-300 leading-relaxed m-0 bg-white/70 dark:bg-zinc-900/70 p-3 rounded-lg border border-zinc-200/70 dark:border-zinc-800/70">
                        {item.reason || "No explanation provided."}
                      </p>
                    </motion.div>
                  ))}
                </motion.div>
              )}

            </motion.div>
          )}

          {activeTab === "review" && (
            <motion.div
              key="rev-container"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full flex flex-col gap-4"
            >
              {reviewState.status === "idle" && (
                <div className="flex flex-col items-center justify-center py-20 text-zinc-400 dark:text-zinc-500">
                  <ChatCircleText size={32} className="mb-2 opacity-50" />
                  <p className="text-sm">Waiting for a request.</p>
                  <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-450">Submit a product to generate a review.</p>
                </div>
              )}
              
              {reviewState.status === "loading" && (
                <div className="flex flex-col gap-4">
                  <div className="grid grid-cols-2 gap-4">
                    <ShimmerBlock className="h-24" />
                    <ShimmerBlock className="h-24" />
                  </div>
                  <ShimmerBlock className="h-32" />
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">Simulating review...</p>
                </div>
              )}

              {reviewState.status === "error" && (
                <div className="flex items-start gap-3 p-4 bg-rose-55 dark:bg-rose-950/40 border border-rose-100 dark:border-rose-900/50 rounded-xl text-rose-600 dark:text-rose-455">
                  <WarningCircle size={24} weight="fill" className="shrink-0" />
                  <p className="text-sm font-medium">{reviewState.error}</p>
                </div>
              )}

              {reviewState.status === "success" && reviewState.data && (
                <motion.div 
                   initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ type: "spring", stiffness: 100, damping: 20 }}
                  className="flex flex-col gap-4"
                >
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-5 bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-100 dark:border-emerald-900/50 rounded-xl flex flex-col gap-2">
                      <span className="text-xs font-semibold uppercase tracking-widest text-emerald-700 dark:text-emerald-400">Predicted rating</span>
                      <strong className="text-2xl font-semibold text-emerald-900 dark:text-emerald-300 font-mono">
                        {Number(reviewState.data.rating || 0).toFixed(1)} / 5
                      </strong>
                      <div className="flex items-center gap-1">
                        {Array.from({ length: 5 }).map((_, idx) => {
                          const filled = idx < Math.max(1, Math.round(Number(reviewState.data.rating || 0)));
                          return (
                            <Star
                              key={`star-${idx}`}
                              size={16}
                              weight={filled ? "fill" : "regular"}
                              className={filled ? "text-emerald-600 dark:text-emerald-400" : "text-emerald-200 dark:text-emerald-900/40"}
                            />
                          );
                        })}
                      </div>
                    </div>
                    <div className="p-5 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800 rounded-xl flex flex-col gap-1">
                      <span className="text-xs font-semibold uppercase tracking-widest text-zinc-500 dark:text-zinc-400">Voice profile</span>
                      <strong className="text-lg font-semibold text-zinc-900 dark:text-white mt-1">Naija-grounded</strong>
                      <span className="text-sm text-zinc-600 dark:text-zinc-400">Culturally aligned output.</span>
                    </div>
                  </div>

                  <div className="p-6 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800/80 rounded-xl relative">
                    <Quotes size={32} weight="fill" className="absolute top-4 right-4 text-zinc-200 dark:text-zinc-800" />
                    <span className="text-xs font-semibold uppercase tracking-widest text-zinc-500 dark:text-zinc-400 mb-3 block">Simulated review</span>
                    <p className="text-zinc-800 dark:text-zinc-200 leading-relaxed relative z-10 font-medium">
                      "{reviewState.data.review || "No review returned."}"
                    </p>
                  </div>

                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
