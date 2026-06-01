import { motion } from "framer-motion";
import { CheckCircle, ChartBar, Users } from "@phosphor-icons/react";

export default function LiveStatusRail() {
  const stats = [
    { label: "Recommendation Quality", value: 87.4, icon: ChartBar, color: "text-emerald-600", bg: "bg-emerald-500" },
    { label: "Explainability", value: 92.6, icon: CheckCircle, color: "text-emerald-600", bg: "bg-emerald-500" },
    { label: "Naija Voice Fidelity", value: 90.8, icon: Users, color: "text-emerald-600", bg: "bg-emerald-500" },
  ];

  return (
    <section className="bg-white/80 dark:bg-zinc-900/90 rounded-[2rem] border border-zinc-200/70 dark:border-zinc-800/80 p-8 shadow-[0_24px_40px_-30px_rgba(15,23,42,0.3)] dark:shadow-[0_24px_40px_-30px_rgba(0,0,0,0.5)] flex flex-col gap-6 relative z-10">
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400 mb-1">Insight Rail</p>
        <h3 className="text-xl font-semibold tracking-tight text-zinc-950 dark:text-white m-0">Live Platform Metrics</h3>
      </div>

      <div className="flex flex-col gap-4">
        {stats.map((stat, i) => (
          <motion.div 
            key={stat.label}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.1, type: "spring", stiffness: 100, damping: 20 }}
            className="flex flex-col gap-3 p-4 rounded-2xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <stat.icon weight="fill" size={18} className={stat.color} />
                <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{stat.label}</span>
              </div>
              <span className="font-mono text-sm font-bold text-zinc-900 dark:text-white">{stat.value.toFixed(1)}%</span>
            </div>
            
            <div className="h-1.5 w-full bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: stat.value / 100 }}
                transition={{ type: "spring", stiffness: 120, damping: 18, delay: 0.5 + i * 0.1 }}
                className={`h-full ${stat.bg} rounded-full origin-left`}
              />
            </div>
          </motion.div>
        ))}
      </div>
      
      <div className="mt-2 bg-zinc-50 dark:bg-zinc-950 p-4 rounded-2xl border border-zinc-200/70 dark:border-zinc-800 flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">API Endpoints</span>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500/80 animate-pulse" />
          <span className="text-xs font-mono text-zinc-650 dark:text-zinc-400">/health, /recommend</span>
        </div>
      </div>
    </section>
  );
}
