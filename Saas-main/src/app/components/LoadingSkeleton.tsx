import { motion } from "motion/react";

interface LoadingSkeletonProps {
  type?: "card" | "table" | "chart";
}

export function LoadingSkeleton({ type = "card" }: LoadingSkeletonProps) {
  if (type === "card") {
    return (
      <div className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10">
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <motion.div
              className="w-12 h-12 rounded-xl bg-white/10"
              animate={{ opacity: [0.3, 0.6, 0.3] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
            <div className="flex-1 space-y-2">
              <motion.div
                className="h-4 bg-white/10 rounded w-3/4"
                animate={{ opacity: [0.3, 0.6, 0.3] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
              <motion.div
                className="h-3 bg-white/10 rounded w-1/2"
                animate={{ opacity: [0.3, 0.6, 0.3] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
              />
            </div>
          </div>
          <motion.div
            className="h-24 bg-white/10 rounded-lg"
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 0.4 }}
          />
        </div>
      </div>
    );
  }

  if (type === "chart") {
    return (
      <div className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10 h-[400px] flex items-center justify-center">
        <div className="text-center">
          <motion.div
            className="w-16 h-16 border-4 border-primary/20 border-t-primary rounded-full mx-auto mb-4"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          <p className="text-sm text-foreground-secondary">Loading visualization...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10">
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <motion.div
            key={i}
            className="h-12 bg-white/10 rounded-lg"
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.1 }}
          />
        ))}
      </div>
    </div>
  );
}
