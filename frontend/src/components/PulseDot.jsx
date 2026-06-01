import { memo } from "react";
import { motion } from "framer-motion";

const PulseDot = memo(function PulseDot({ className = "" }) {
  return (
    <span className={`relative inline-flex h-2.5 w-2.5 ${className}`} aria-hidden="true">
      <motion.span
        className="absolute inset-0 rounded-full bg-emerald-500/60"
        animate={{ scale: [1, 1.6, 1], opacity: [0.7, 0, 0.7] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
      />
      <span className="relative h-full w-full rounded-full bg-emerald-600" />
    </span>
  );
});

export default PulseDot;
