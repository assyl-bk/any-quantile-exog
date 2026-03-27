import { motion } from "motion/react";
import { LucideIcon } from "lucide-react";
import { LineChart, Line, ResponsiveContainer } from "recharts";

interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  trend?: {
    value: string;
    isPositive: boolean;
  };
  icon: LucideIcon;
  iconGradient: string;
  sparklineData?: number[];
}

export function StatCard({
  label,
  value,
  unit,
  trend,
  icon: Icon,
  iconGradient,
  sparklineData,
}: StatCardProps) {
  return (
    <motion.div
      className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.3)] relative overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      whileHover={{ y: -4, boxShadow: "0 12px 48px rgba(59,130,246,0.2)" }}
    >
      {/* Background sparkline */}
      {sparklineData && (
        <div className="absolute inset-0 opacity-10">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sparklineData.map((val, i) => ({ value: val, index: i }))}>
              <Line type="monotone" dataKey="value" stroke="white" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div
            className={`w-12 h-12 rounded-xl bg-gradient-to-br ${iconGradient} flex items-center justify-center shadow-lg`}
          >
            <Icon className="w-6 h-6 text-white" />
          </div>
          {trend && (
            <div
              className={`flex items-center gap-1 px-2 py-1 rounded-md border ${
                trend.isPositive
                  ? "bg-success/10 border-success/20 text-success"
                  : "bg-critical/10 border-critical/20 text-critical"
              }`}
            >
              <span className="text-xs font-semibold">{trend.value}</span>
            </div>
          )}
        </div>

        <div className="space-y-1">
          <p className="text-sm text-foreground-secondary">{label}</p>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-bold text-white font-['JetBrains_Mono']">{value}</span>
            {unit && <span className="text-lg text-foreground-secondary">{unit}</span>}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
