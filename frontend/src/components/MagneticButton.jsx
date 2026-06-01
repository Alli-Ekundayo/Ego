import { forwardRef } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";

const MagneticButton = forwardRef(function MagneticButton(
  { className = "", onPointerMove, onPointerLeave, children, ...rest },
  ref,
) {
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const dampenedX = useTransform(x, (value) => value * 0.22);
  const dampenedY = useTransform(y, (value) => value * 0.22);
  const springX = useSpring(dampenedX, { stiffness: 160, damping: 18, mass: 0.4 });
  const springY = useSpring(dampenedY, { stiffness: 160, damping: 18, mass: 0.4 });

  const handleMove = (event) => {
    if (event.pointerType === "touch") return;
    const rect = event.currentTarget.getBoundingClientRect();
    const offsetX = event.clientX - rect.left - rect.width / 2;
    const offsetY = event.clientY - rect.top - rect.height / 2;
    x.set(offsetX);
    y.set(offsetY);
    onPointerMove?.(event);
  };

  const handleLeave = (event) => {
    x.set(0);
    y.set(0);
    onPointerLeave?.(event);
  };

  return (
    <motion.button
      ref={ref}
      onPointerMove={handleMove}
      onPointerLeave={handleLeave}
      style={{ x: springX, y: springY }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: "spring", stiffness: 180, damping: 16 }}
      className={className}
      {...rest}
    >
      {children}
    </motion.button>
  );
});

export default MagneticButton;
