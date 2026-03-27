import { motion } from "motion/react";
import { ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  onClick?: () => void;
}

export function GlassCard({ children, className = "", hover = false, onClick }: GlassCardProps) {
  return (
    <motion.div
      className={`
        backdrop-blur-xl rounded-2xl p-6
        border shadow-[0_8px_32px_rgba(0,0,0,0.3)]
        ${hover ? "cursor-pointer transition-all duration-200 hover:translate-y-[-4px] hover:shadow-[0_12px_48px_rgba(59,130,246,0.2)] hover:border-primary/30" : ""}
        ${className}
      `}
      style={{
        backgroundColor: 'var(--glass-bg)',
        borderColor: 'var(--glass-border)',
      }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      onClick={onClick}
    >
      {children}
    </motion.div>
  );
}
