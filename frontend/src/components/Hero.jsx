import { motion } from "framer-motion";
import { ArrowRight, Lightning, CheckCircle, ShieldCheck } from "@phosphor-icons/react";

export default function Hero() {
  return (
    <section className="bg-white/80 dark:bg-zinc-900/90 border border-zinc-200/70 dark:border-zinc-800/80 p-8 sm:p-12 shadow-[0_24px_40px_-30px_rgba(15,23,42,0.3)] dark:shadow-[0_24px_40px_-30px_rgba(0,0,0,0.5)] overflow-hidden relative min-h-[400px] flex items-center z-10">
      {/* Decorative Background Blob */}
      <div className="absolute top-0 right-0 -mr-32 -mt-32 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
      
      <div className="grid grid-cols-1 md:grid-cols-[1.2fr_1fr] gap-12 relative z-10 w-full">
        <div className="flex flex-col items-start justify-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
          >
            <p className="text-sm font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-emerald-500 rounded-full"></span>
              Transparent AI Signals
            </p>
            <h2 className="text-5xl md:text-6xl font-bold tracking-tighter leading-[1.05] text-zinc-950 dark:text-white max-w-[14ch]">
              Explore context with pristine clarity.
            </h2>
            <p className="text-lg text-zinc-500 dark:text-zinc-400 leading-relaxed max-w-[45ch] mt-6">
              Test recommendation contexts, simulate culturally grounded reviews, and inspect the reasoning behind each result in a focused workspace.
            </p>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.1 }}
            className="flex items-center gap-4 mt-10"
          >
            <button className="flex items-center gap-2 bg-zinc-950 dark:bg-zinc-50 text-white dark:text-zinc-950 px-6 py-3.5 rounded-full font-medium shadow-[0_10px_20px_rgba(0,0,0,0.15)] dark:shadow-[0_10px_20px_rgba(255,255,255,0.05)] hover:-translate-y-1 transition-transform duration-300">
              Launch Studio <ArrowRight weight="bold" />
            </button>
            <button className="flex items-center gap-2 bg-zinc-50 dark:bg-zinc-900 text-zinc-650 dark:text-zinc-350 px-6 py-3.5 rounded-full font-medium border border-zinc-200 dark:border-zinc-800 hover:-translate-y-1 hover:bg-zinc-100 dark:hover:bg-zinc-850 transition-all duration-300">
              View Insights
            </button>
          </motion.div>
        </div>

        {/* Right side metrics bento */}
        <div className="grid grid-cols-1 gap-4">
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.2 }}
            className="bg-zinc-50 dark:bg-zinc-950/60 p-6 rounded-3xl border border-zinc-200/70 dark:border-zinc-800/80"
          >
            <div className="w-10 h-10 rounded-xl bg-white dark:bg-zinc-900 shadow-sm border border-zinc-100 dark:border-zinc-800 flex items-center justify-center text-emerald-500 mb-4">
              <Lightning weight="fill" size={20} />
            </div>
            <p className="text-xs font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-1">Latency Target</p>
            <h3 className="text-2xl font-bold text-zinc-900 dark:text-white font-mono tracking-tight">Sub-1s</h3>
            <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">Designed for instant feedback loops.</p>
          </motion.div>

          <div className="grid grid-cols-2 gap-4">
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.3 }}
              className="bg-zinc-50 dark:bg-zinc-950/60 p-6 rounded-3xl border border-zinc-200/70 dark:border-zinc-800/80"
            >
              <div className="w-10 h-10 rounded-xl bg-white dark:bg-zinc-900 shadow-sm border border-zinc-100 dark:border-zinc-800 flex items-center justify-center text-zinc-700 dark:text-zinc-300 mb-4">
                <CheckCircle weight="fill" size={20} />
              </div>
              <p className="text-xs font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-1">Surfaces</p>
              <h3 className="text-xl font-bold text-zinc-900 dark:text-white">2 Flows</h3>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.4 }}
              className="bg-zinc-50 dark:bg-zinc-950/60 p-6 rounded-3xl border border-zinc-200/70 dark:border-zinc-800/80"
            >
              <div className="w-10 h-10 rounded-xl bg-white dark:bg-zinc-900 shadow-sm border border-zinc-100 dark:border-zinc-800 flex items-center justify-center text-zinc-700 dark:text-zinc-300 mb-4">
                <ShieldCheck weight="fill" size={20} />
              </div>
              <p className="text-xs font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-1">Signals</p>
              <h3 className="text-xl font-bold text-zinc-900 dark:text-white">Explainable</h3>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}
