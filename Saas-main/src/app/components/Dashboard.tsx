import { GlassCard } from "./GlassCard";
import {
  TrendingUp,
  Zap,
  Target,
  Activity,
  Database,
  AlertTriangle,
  ArrowUp,
  CheckCircle,
  Upload,
} from "lucide-react";
import { motion } from "motion/react";
import { useTheme } from "../context/ThemeContext";
import { useData } from "../context/Datacontext";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  LineChart,
  Line,
} from "recharts";

// ── Helpers ───────────────────────────────────────────────────────────────────

const CustomTooltip = ({ active, payload, theme }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      className={`backdrop-blur-xl rounded-lg p-3 shadow-xl ${
        theme === "dark"
          ? "bg-[#111827]/95 border border-white/20"
          : "bg-white/95 border border-gray-300"
      }`}
    >
      <p className="text-xs text-foreground-secondary mb-2">
        {payload[0].payload.time}
      </p>
      {payload.map(
        (entry: any, i: number) =>
          entry.value != null && (
            <p
              key={i}
              className="text-sm font-mono"
              style={{ color: entry.color }}
            >
              {entry.name}: {Number(entry.value).toFixed(1)} MW
            </p>
          ),
      )}
    </div>
  );
};

// ── Empty state helper ────────────────────────────────────────────────────────

function EmptyCard({ message }: { message: string }) {
  return (
    <div className="h-full flex flex-col items-center justify-center py-10 text-center">
      <Activity className="w-10 h-10 text-foreground-secondary/30 mb-3" />
      <p className="text-sm text-foreground-secondary">{message}</p>
    </div>
  );
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export function Dashboard() {
  const { theme } = useTheme();
  const { dataset, forecastResult } = useData();

  // ── Dataset summary stats ─────────────────────────────────────────────────

  const totalValidPoints = dataset
    ? dataset.variables.reduce((s, v) => s + v.validCount, 0)
    : null;

  const overallQuality = dataset
    ? Math.round(
        (dataset.variables.reduce((s, v) => s + v.validCount, 0) /
          dataset.totalRows) *
          100,
      )
    : null;

  // ── Forecast stats ────────────────────────────────────────────────────────

  const forecastData = forecastResult?.chartData ?? [];
  const hasForecast = forecastData.length > 0;
  const capacityThreshold = 11000; // could be made configurable

  const hasCapacityAlert = forecastData.some(
    (d) =>
      d.customQuantile !== undefined && d.customQuantile > capacityThreshold,
  );

  // Build sparkline data from forecast
  const allValidForecast = forecastData.filter(
    (d) => d.customQuantile !== undefined,
  );
  const peakDemand = forecastResult?.peakDemand ?? null;
  const avgDemand = forecastResult?.avgDemand ?? null;

  // For the chart: show forecast (Q50) and custom quantile
  const chartData = forecastData.map((d) => ({
    time: d.time,
    forecast: d.forecast,
    customQuantile: d.customQuantile,
    upper90: d.upper90,
    lower5: d.lower5,
  }));

  // KPI cards — only show real numbers
  const kpiCards = [
    {
      label: "Variables Loaded",
      value: dataset ? String(dataset.variables.length) : null,
      unit: "series",
      icon: Database,
      color: "from-purple-500 to-pink-500",
      note: dataset
        ? `${dataset.totalRows.toLocaleString()} rows`
        : "Upload CSV to see data",
    },
    {
      label: "Data Quality",
      value: overallQuality !== null ? String(overallQuality) : null,
      unit: "%",
      icon: CheckCircle,
      color: "from-emerald-500 to-teal-500",
      note: dataset
        ? `${totalValidPoints?.toLocaleString()} valid values`
        : "No data",
    },
    {
      label: "Peak Forecast",
      value: peakDemand !== null ? peakDemand.toFixed(0) : null,
      unit: "MW",
      icon: TrendingUp,
      color: "from-blue-500 to-cyan-500",
      note: forecastResult
        ? `${forecastResult.horizon}H horizon`
        : "Run a forecast first",
    },
    {
      label: "Avg Forecast",
      value: avgDemand !== null ? avgDemand.toFixed(0) : null,
      unit: "MW",
      icon: Target,
      color: "from-orange-500 to-red-500",
      note: forecastResult
        ? `Variable: ${forecastResult.selectedVariable}`
        : "No forecast yet",
    },
  ];

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-8">
      {/* ── Getting Started / Status Banner ── */}
      {!dataset && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <GlassCard className="p-6 border-l-4 border-primary">
            <div className="flex items-start gap-4">
              <Upload className="w-6 h-6 text-primary mt-0.5 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-foreground mb-2">
                  Getting Started
                </h3>
                <ol className="text-sm text-foreground-secondary space-y-1.5 list-decimal list-inside">
                  <li>
                    Go to <strong>Data</strong> and upload your CSV (format:{" "}
                    <code className="bg-white/10 px-1 rounded">
                      ds, unique_id, y
                    </code>
                    )
                  </li>
                  <li>
                    Go to <strong>Forecast</strong>, select a variable, adjust
                    the quantile slider, and click Generate
                  </li>
                  <li>Return here to see your results summarised</li>
                </ol>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* ── KPI Cards ── */}
      <div className="grid grid-cols-4 gap-6">
        {kpiCards.map((card, index) => {
          const Icon = card.icon;
          return (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: index * 0.08 }}
            >
              <GlassCard className="relative overflow-hidden">
                <div className="relative z-10">
                  <div className="flex items-start justify-between mb-4">
                    <div
                      className={`w-12 h-12 rounded-xl bg-gradient-to-br ${card.color} flex items-center justify-center shadow-lg`}
                    >
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    {card.value !== null && (
                      <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-success/10 border border-success/20">
                        <ArrowUp className="w-3 h-3 text-success" />
                        <span className="text-xs font-semibold text-success">
                          Live
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="space-y-1">
                    <p className="text-sm text-foreground-secondary">
                      {card.label}
                    </p>
                    {card.value !== null ? (
                      <div className="flex items-baseline gap-1.5">
                        <span className="text-3xl font-bold text-foreground font-mono">
                          {Number(card.value).toLocaleString()}
                        </span>
                        <span className="text-base text-foreground-secondary">
                          {card.unit}
                        </span>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground italic">
                        {card.note}
                      </p>
                    )}
                    {card.value !== null && (
                      <p className="text-xs text-muted-foreground">
                        {card.note}
                      </p>
                    )}
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          );
        })}
      </div>

      {/* ── Forecast Chart ── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <GlassCard className="p-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2
                className={`text-2xl font-bold mb-1 ${theme === "dark" ? "text-white" : "text-gray-900"}`}
              >
                Energy Demand Forecast
              </h2>
              <p
                className={`text-sm ${theme === "dark" ? "text-foreground-secondary" : "text-gray-600"}`}
              >
                {hasForecast
                  ? `${forecastResult!.selectedVariable} · ${forecastResult!.horizon}H · Generated ${forecastResult!.generatedAt.toLocaleTimeString()}`
                  : "No forecast generated yet — go to the Forecast page to create one"}
              </p>
            </div>

            {hasForecast && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/30">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-xs text-green-400 font-semibold">
                  Result Available
                </span>
              </div>
            )}
          </div>

          <div className="h-[380px]">
            {hasForecast ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={chartData}
                  margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="dbForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop
                        offset="5%"
                        stopColor="#10B981"
                        stopOpacity={0.25}
                      />
                      <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="dbCustom" x1="0" y1="0" x2="0" y2="1">
                      <stop
                        offset="5%"
                        stopColor="#3B82F6"
                        stopOpacity={0.25}
                      />
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={
                      theme === "dark"
                        ? "rgba(255,255,255,0.1)"
                        : "rgba(0,0,0,0.1)"
                    }
                  />
                  <XAxis
                    dataKey="time"
                    stroke={theme === "dark" ? "#9CA3AF" : "#6B7280"}
                    tick={{
                      fill: theme === "dark" ? "#9CA3AF" : "#4B5563",
                      fontSize: 11,
                    }}
                    tickLine={false}
                    interval={Math.max(1, Math.floor(chartData.length / 8))}
                  />
                  <YAxis
                    stroke={theme === "dark" ? "#9CA3AF" : "#6B7280"}
                    tick={{
                      fill: theme === "dark" ? "#9CA3AF" : "#4B5563",
                      fontSize: 11,
                    }}
                    tickLine={false}
                    label={{
                      value: "Demand (MW)",
                      angle: -90,
                      position: "insideLeft",
                      fill: theme === "dark" ? "#9CA3AF" : "#4B5563",
                    }}
                  />
                  <Tooltip content={<CustomTooltip theme={theme} />} />

                  <ReferenceLine
                    y={capacityThreshold}
                    stroke="#EF4444"
                    strokeDasharray="5 5"
                    strokeWidth={2}
                    label={{
                      value: "Capacity Limit",
                      fill: "#EF4444",
                      fontSize: 11,
                    }}
                  />

                  <Area
                    type="monotone"
                    dataKey="upper90"
                    stroke="none"
                    fill="rgba(59,130,246,0.07)"
                    strokeWidth={0}
                    name="Q90"
                  />
                  <Area
                    type="monotone"
                    dataKey="lower5"
                    stroke="none"
                    fill="rgba(59,130,246,0.07)"
                    strokeWidth={0}
                    name="Q5"
                  />
                  <Area
                    type="monotone"
                    dataKey="forecast"
                    stroke="#10B981"
                    strokeWidth={2}
                    fill="url(#dbForecast)"
                    name="Q50 (Median)"
                    dot={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="customQuantile"
                    stroke="#3B82F6"
                    strokeWidth={3}
                    fill="url(#dbCustom)"
                    name={`Custom Quantile`}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <EmptyCard message="Generate a forecast on the Forecast page to see it here." />
            )}
          </div>

          {hasForecast && (
            <div
              className={`flex items-center gap-6 mt-4 pt-4 ${
                theme === "dark"
                  ? "border-t border-white/10"
                  : "border-t border-gray-200"
              }`}
            >
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#10B981]" />
                <span className="text-sm text-foreground-secondary">
                  Median (Q50)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#3B82F6]" />
                <span className="text-sm text-foreground-secondary">
                  Custom Quantile
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-3 bg-blue-500/15 rounded" />
                <span className="text-sm text-foreground-secondary">
                  Q5–Q90 Band
                </span>
              </div>
              <div className="flex items-center gap-2 ml-auto">
                <div className="w-4 h-0.5 bg-[#EF4444]" />
                <span className="text-sm text-foreground-secondary">
                  Capacity Threshold
                </span>
              </div>
            </div>
          )}
        </GlassCard>
      </motion.div>

      {/* ── Capacity Alert (only when forecast exists and breaches limit) ── */}
      {hasForecast && hasCapacityAlert && (
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
        >
          <GlassCard className="border-l-4 border-warning bg-warning/5">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-lg bg-warning/20 flex items-center justify-center flex-shrink-0">
                <AlertTriangle className="w-5 h-5 text-warning" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-white mb-1">
                  Grid Capacity Alert
                </h3>
                <p className="text-sm text-foreground-secondary">
                  The forecast for{" "}
                  <strong>{forecastResult!.selectedVariable}</strong> predicts
                  demand exceeding the capacity threshold (
                  {capacityThreshold.toLocaleString()} MW). Consider initiating
                  load balancing protocols or demand response programs.
                </p>
                <div className="flex items-center gap-2 mt-3 text-xs text-muted-foreground">
                  <span>Detected at {new Date().toLocaleTimeString()}</span>
                  <span>·</span>
                  <span>Variable: {forecastResult!.selectedVariable}</span>
                </div>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* ── Dataset overview (when data is loaded) ── */}
      {dataset && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <GlassCard>
            <h3
              className={`text-lg font-bold mb-4 ${theme === "dark" ? "text-white" : "text-gray-900"}`}
            >
              Loaded Dataset · {dataset.fileName}
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {dataset.variables.slice(0, 8).map((v) => {
                const qPct = Math.round((v.validCount / v.rows.length) * 100);
                return (
                  <div
                    key={v.unique_id}
                    className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-foreground">
                        {v.unique_id}
                      </span>
                      <span
                        className={`text-xs font-mono ${qPct >= 90 ? "text-success" : qPct >= 75 ? "text-warning" : "text-critical"}`}
                      >
                        {qPct}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${qPct >= 90 ? "bg-success" : qPct >= 75 ? "bg-warning" : "bg-critical"}`}
                        style={{ width: `${qPct}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground font-mono">
                      {v.validCount} / {v.rows.length} pts
                    </p>
                  </div>
                );
              })}
              {dataset.variables.length > 8 && (
                <div className="p-3 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center">
                  <span className="text-sm text-foreground-secondary">
                    +{dataset.variables.length - 8} more
                  </span>
                </div>
              )}
            </div>
          </GlassCard>
        </motion.div>
      )}
    </div>
  );
}
