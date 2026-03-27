import { motion } from "motion/react";
import { TrendingUp, TrendingDown } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string | number;
  change?: number;
  format?: "number" | "percentage" | "currency";
  size?: "sm" | "md" | "lg";
}

export function MetricCard({ label, value, change, format = "number", size = "md" }: MetricCardProps) {
  const isPositive = change !== undefined && change >= 0;
  
  const valueSize = {
    sm: "text-2xl",
    md: "text-3xl",
    lg: "text-4xl",
  };

  const formatValue = (val: string | number) => {
    if (format === "currency") return `$${val}`;
    if (format === "percentage") return `${val}%`;
    return val;
  };

  return (
    <motion.div
      className="backdrop-blur-xl bg-white/5 rounded-xl p-4 border border-white/10"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.02 }}
    >
      <p className="text-sm text-foreground-secondary mb-2">{label}</p>
      <div className="flex items-baseline gap-2 mb-2">
        <span className={`${valueSize[size]} font-bold font-mono text-foreground`}>
          {formatValue(value)}
        </span>
      </div>
      {change !== undefined && (
        <div className={`flex items-center gap-1 text-sm ${isPositive ? "text-success" : "text-critical"}`}>
          {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          <span className="font-medium">{Math.abs(change)}%</span>
        </div>
      )}
    </motion.div>
  );
}
