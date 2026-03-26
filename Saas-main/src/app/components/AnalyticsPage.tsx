import { useState, useMemo } from "react";
import { GlassCard } from "./GlassCard";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart2,
  Download,
} from "lucide-react";
import { motion } from "motion/react";
import { useTheme } from "../context/ThemeContext";
import { useData } from "../context/Datacontext";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

const TABS = ["Overview", "Patterns", "Trends", "Comparisons"];

// ── Helpers ───────────────────────────────────────────────────────────────────

function EmptyChart({ message }: { message: string }) {
  return (
    <div className="h-full flex flex-col items-center justify-center py-16 text-center">
      <Activity className="w-12 h-12 text-foreground-secondary/30 mb-4" />
      <p className="text-sm text-foreground-secondary max-w-xs">{message}</p>
    </div>
  );
}

function tooltipStyle(theme: string) {
  return {
    backgroundColor:
      theme === "dark" ? "rgba(17,24,39,0.95)" : "rgba(255,255,255,0.95)",
    border:
      theme === "dark"
        ? "1px solid rgba(255,255,255,0.15)"
        : "1px solid rgba(0,0,0,0.15)",
    borderRadius: 8,
    color: theme === "dark" ? "#fff" : "#111",
  };
}

// ── AnalyticsPage ─────────────────────────────────────────────────────────────

export function AnalyticsPage() {
  const { theme } = useTheme();
  const { dataset, forecastResult } = useData();
  const [activeTab, setActiveTab] = useState("Overview");
  const [selectedVar, setSelectedVar] = useState<string>("");

  // Auto-select first variable
  const variables = dataset?.variables ?? [];
  const activeVarId = selectedVar || variables[0]?.unique_id || "";
  const activeVar = variables.find((v) => v.unique_id === activeVarId);

  // ── Derived analytics ─────────────────────────────────────────────────────

  // Hourly average (group by hour-of-day index)
  const hourlyAvgData = useMemo(() => {
    if (!activeVar) return [];
    const buckets: { sum: number; count: number }[] = Array.from(
      { length: 24 },
      () => ({ sum: 0, count: 0 }),
    );
    activeVar.rows.forEach((r, idx) => {
      if (r.y === null) return;
      const hour = idx % 24; // fallback: use position mod 24 if we can't parse timestamp easily
      buckets[hour].sum += r.y;
      buckets[hour].count += 1;
    });
    return buckets.map((b, h) => ({
      hour: `${String(h).padStart(2, "0")}:00`,
      demand: b.count > 0 ? Math.round(b.sum / b.count) : 0,
    }));
  }, [activeVar]);

  // Rolling 7-point moving average (use as "trend")
  const trendData = useMemo(() => {
    if (!activeVar) return [];
    const valid = activeVar.rows
      .map((r, i) => ({ index: i, ds: r.ds, y: r.y }))
      .filter((r) => r.y !== null);

    const window = 7;
    return valid
      .map((r, i) => {
        const slice = valid.slice(Math.max(0, i - window + 1), i + 1);
        const avg =
          slice.reduce((s, x) => s + (x.y as number), 0) / slice.length;
        return { index: i, ds: r.ds, actual: r.y, trend: Math.round(avg) };
      })
      .filter((_, i) => i % Math.max(1, Math.floor(valid.length / 200)) === 0); // downsample for perf
  }, [activeVar]);

  // All-variables comparison (mean of each)
  const comparisonData = useMemo(
    () =>
      variables.map((v) => ({
        name: v.unique_id,
        mean: Math.round(v.mean),
        max: Math.round(v.max),
        missing: v.missingCount,
      })),
    [variables],
  );

  // Quantile comparison (from forecast)
  const forecastQuantileData = useMemo(() => {
    if (!forecastResult) return [];
    return forecastResult.chartData.map((d) => ({
      time: d.time,
      q5: d.lower5,
      q25: d.lower25,
      q50: d.forecast,
      q75: d.upper75,
      q90: d.upper90,
      custom: d.customQuantile,
    }));
  }, [forecastResult]);

  // ── Metric cards ──────────────────────────────────────────────────────────

  const metricsData = activeVar
    ? [
        {
          name: "Min Value",
          value: activeVar.min.toFixed(1),
          unit: "MW",
          trend: "—",
          good: true,
        },
        {
          name: "Max Value",
          value: activeVar.max.toFixed(1),
          unit: "MW",
          trend: "—",
          good: true,
        },
        {
          name: "Mean",
          value: activeVar.mean.toFixed(1),
          unit: "MW",
          trend: "—",
          good: true,
        },
        {
          name: "Std Dev",
          value: activeVar.stdDev.toFixed(1),
          unit: "MW",
          trend: "—",
          good: activeVar.stdDev / activeVar.mean < 0.25,
        },
        {
          name: "Missing",
          value: String(activeVar.missingCount),
          unit: "pts",
          trend: `${Math.round((activeVar.missingCount / activeVar.rows.length) * 100)}%`,
          good: activeVar.missingCount === 0,
        },
        {
          name: "Data Quality",
          value: `${Math.round((activeVar.validCount / activeVar.rows.length) * 100)}`,
          unit: "%",
          trend: "—",
          good: true,
        },
      ]
    : [];

  // ── Export helpers ────────────────────────────────────────────────────────

  const exportAnalyticsCSV = () => {
    if (!activeVar) return;
    const header = "ds,y\n";
    const rows = activeVar.rows.map((r) => `${r.ds},${r.y ?? ""}`).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analytics_${activeVarId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-1">
            Analytics & Insights
          </h2>
          <p className="text-sm text-foreground-secondary">
            {dataset
              ? `Analysing ${dataset.variables.length} variables · ${dataset.totalRows.toLocaleString()} total rows`
              : "Upload a dataset to unlock analytics"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Variable selector */}
          {variables.length > 1 && (
            <select
              value={activeVarId}
              onChange={(e) => setSelectedVar(e.target.value)}
              className={`rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 ${
                theme === "dark"
                  ? "bg-white/5 border border-white/10 text-white"
                  : "bg-white border border-gray-300 text-gray-900"
              }`}
            >
              {variables.map((v) => (
                <option key={v.unique_id} value={v.unique_id}>
                  {v.unique_id}
                </option>
              ))}
            </select>
          )}
          <button
            onClick={exportAnalyticsCSV}
            disabled={!activeVar}
            className="px-5 py-2.5 rounded-lg bg-white/5 hover:bg-white/10 text-white font-medium transition-colors border border-white/10 flex items-center gap-2 text-sm disabled:opacity-40"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      {/* Metric cards */}
      {metricsData.length > 0 && (
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
          {metricsData.map((m, i) => (
            <motion.div
              key={m.name}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
            >
              <GlassCard className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <p className="text-xs text-foreground-secondary">{m.name}</p>
                  <div
                    className={`p-1.5 rounded-lg ${m.good ? "bg-success/20" : "bg-critical/20"}`}
                  >
                    {m.good ? (
                      <TrendingUp className="w-3 h-3 text-success" />
                    ) : (
                      <TrendingDown className="w-3 h-3 text-critical" />
                    )}
                  </div>
                </div>
                <p className="text-xl font-bold font-mono text-foreground">
                  {m.value}
                </p>
                <p className="text-xs text-foreground-secondary">{m.unit}</p>
              </GlassCard>
            </motion.div>
          ))}
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex gap-2">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-5 py-2.5 rounded-xl font-medium text-sm transition-all ${
              activeTab === tab
                ? "bg-primary text-white shadow-lg shadow-primary/30"
                : "bg-white/5 text-foreground-secondary hover:bg-white/10"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* ── Tab: Overview ── */}
      {activeTab === "Overview" && (
        <div className="grid grid-cols-2 gap-6">
          {/* Hourly average */}
          <GlassCard>
            <h3 className="text-lg font-bold text-foreground mb-4">
              Average Demand by Hour-of-Day
              {activeVar && (
                <span className="text-sm font-normal text-foreground-secondary ml-2">
                  · {activeVarId}
                </span>
              )}
            </h3>
            <div className="h-[380px]">
              {hourlyAvgData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={hourlyAvgData}
                    margin={{ top: 8, right: 16, bottom: 16, left: 16 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke={
                        theme === "dark"
                          ? "rgba(255,255,255,0.08)"
                          : "rgba(0,0,0,0.08)"
                      }
                    />
                    <XAxis
                      dataKey="hour"
                      tick={{
                        fill: theme === "dark" ? "#9CA3AF" : "#6B7280",
                        fontSize: 10,
                      }}
                      tickLine={false}
                      interval={3}
                    />
                    <YAxis
                      tick={{
                        fill: theme === "dark" ? "#9CA3AF" : "#6B7280",
                        fontSize: 10,
                      }}
                      tickLine={false}
                      label={{
                        value: "Demand (MW)",
                        angle: -90,
                        position: "insideLeft",
                        fill: theme === "dark" ? "#9CA3AF" : "#4B5563",
                      }}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle(theme)}
                      formatter={(v: any) => [`${v} MW`, "Avg Demand"]}
                    />
                    <Bar
                      dataKey="demand"
                      fill="#3B82F6"
                      radius={[6, 6, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart message="Upload a dataset to see demand by hour." />
              )}
            </div>
          </GlassCard>

          {/* Variables comparison */}
          <GlassCard>
            <h3 className="text-lg font-bold text-foreground mb-4">
              Mean Demand per Variable
            </h3>
            <div className="h-[380px]">
              {comparisonData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={comparisonData}
                    margin={{ top: 8, right: 16, bottom: 16, left: 16 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke={
                        theme === "dark"
                          ? "rgba(255,255,255,0.08)"
                          : "rgba(0,0,0,0.08)"
                      }
                    />
                    <XAxis
                      dataKey="name"
                      tick={{
                        fill: theme === "dark" ? "#9CA3AF" : "#6B7280",
                        fontSize: 10,
                      }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{
                        fill: theme === "dark" ? "#9CA3AF" : "#6B7280",
                        fontSize: 10,
                      }}
                      tickLine={false}
                      label={{
                        value: "Mean (MW)",
                        angle: -90,
                        position: "insideLeft",
                        fill: theme === "dark" ? "#9CA3AF" : "#4B5563",
                      }}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle(theme)}
                      formatter={(v: any) => [`${v} MW`, ""]}
                    />
                    <Bar
                      dataKey="mean"
                      fill="#10B981"
                      radius={[6, 6, 0, 0]}
                      name="Mean"
                    />
                    <Bar
                      dataKey="max"
                      fill="#3B82F6"
                      radius={[6, 6, 0, 0]}
                      name="Max"
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart message="Upload a dataset to compare variables." />
              )}
            </div>
          </GlassCard>
        </div>
      )}

      {/* ── Tab: Patterns ── */}
      {activeTab === "Patterns" && (
        <GlassCard>
          <h3 className="text-lg font-bold text-foreground mb-4">
            Raw Time Series
            {activeVar && (
              <span className="text-sm font-normal text-foreground-secondary ml-2">
                · {activeVarId}
              </span>
            )}
          </h3>
          <div className="h-[480px]">
            {trendData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={trendData}
                  margin={{ top: 12, right: 24, bottom: 16, left: 16 }}
                >
                  <defs>
                    <linearGradient id="actualGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={
                      theme === "dark"
                        ? "rgba(255,255,255,0.08)"
                        : "rgba(0,0,0,0.08)"
                    }
                  />
                  <XAxis
                    dataKey="ds"
                    tick={{
                      fill: theme === "dark" ? "#9CA3AF" : "#6B7280",
                      fontSize: 10,
                    }}
                    tickLine={false}
                    interval={Math.max(1, Math.floor(trendData.length / 8))}
                  />
                  <YAxis
                    tick={{
                      fill: theme === "dark" ? "#9CA3AF" : "#6B7280",
                      fontSize: 10,
                    }}
                    tickLine={false}
                    label={{
                      value: "Value (MW)",
                      angle: -90,
                      position: "insideLeft",
                      fill: theme === "dark" ? "#9CA3AF" : "#4B5563",
                    }}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle(theme)}
                    formatter={(v: any) => [Number(v).toFixed(1), ""]}
                  />
                  <Legend
                    wrapperStyle={{
                      color: theme === "dark" ? "#D1D5DB" : "#4B5563",
                      fontSize: 12,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="actual"
                    stroke="#3B82F6"
                    strokeWidth={1.5}
                    fill="url(#actualGrad)"
                    dot={false}
                    name="Actual"
                  />
                  <Line
                    type="monotone"
                    dataKey="trend"
                    stroke="#F59E0B"
                    strokeWidth={2}
                    dot={false}
                    name="7-pt Moving Avg"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart message="Upload a dataset to see time-series patterns." />
            )}
          </div>
        </GlassCard>
      )}

      {/* ── Tab: Trends ── */}
      {activeTab === "Trends" && (
        <GlassCard>
          <h3 className="text-lg font-bold text-foreground mb-4">
            Forecast Quantile Trends
          </h3>
          <div className="h-[480px]">
            {forecastQuantileData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={forecastQuantileData}
                  margin={{ top: 12, right: 24, bottom: 16, left: 16 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={
                      theme === "dark"
                        ? "rgba(255,255,255,0.08)"
                        : "rgba(0,0,0,0.08)"
                    }
                  />
                  <XAxis
                    dataKey="time"
                    tick={{
                      fill: theme === "dark" ? "#9CA3AF" : "#6B7280",
                      fontSize: 10,
                    }}
                    tickLine={false}
                    interval={Math.max(
                      1,
                      Math.floor(forecastQuantileData.length / 8),
                    )}
                  />
                  <YAxis
                    tick={{
                      fill: theme === "dark" ? "#9CA3AF" : "#6B7280",
                      fontSize: 10,
                    }}
                    tickLine={false}
                    label={{
                      value: "Demand (MW)",
                      angle: -90,
                      position: "insideLeft",
                      fill: theme === "dark" ? "#9CA3AF" : "#4B5563",
                    }}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle(theme)}
                    formatter={(v: any) => [`${Number(v).toFixed(1)} MW`, ""]}
                  />
                  <Legend
                    wrapperStyle={{
                      color: theme === "dark" ? "#D1D5DB" : "#4B5563",
                      fontSize: 12,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="q5"
                    stroke="#6B7280"
                    strokeWidth={1}
                    dot={false}
                    strokeDasharray="3 3"
                    name="Q5"
                  />
                  <Line
                    type="monotone"
                    dataKey="q25"
                    stroke="#06B6D4"
                    strokeWidth={1.5}
                    dot={false}
                    strokeDasharray="2 2"
                    name="Q25"
                  />
                  <Line
                    type="monotone"
                    dataKey="q50"
                    stroke="#10B981"
                    strokeWidth={2.5}
                    dot={false}
                    name="Q50 (Median)"
                  />
                  <Line
                    type="monotone"
                    dataKey="q75"
                    stroke="#8B5CF6"
                    strokeWidth={1.5}
                    dot={false}
                    strokeDasharray="2 2"
                    name="Q75"
                  />
                  <Line
                    type="monotone"
                    dataKey="q90"
                    stroke="#F59E0B"
                    strokeWidth={1}
                    dot={false}
                    strokeDasharray="3 3"
                    name="Q90"
                  />
                  <Line
                    type="monotone"
                    dataKey="custom"
                    stroke="#3B82F6"
                    strokeWidth={3}
                    dot={false}
                    name="Custom Quantile"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart message="Generate a forecast on the Forecast page to see quantile trends here." />
            )}
          </div>
          {!forecastResult && (
            <p className="text-center text-xs text-muted-foreground mt-3">
              Forecast results will appear here after you generate one.
            </p>
          )}
        </GlassCard>
      )}

      {/* ── Tab: Comparisons ── */}
      {activeTab === "Comparisons" && (
        <GlassCard>
          <h3 className="text-lg font-bold text-foreground mb-4">
            Missing Data per Variable
          </h3>
          <div className="h-[480px]">
            {comparisonData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={comparisonData}
                  layout="vertical"
                  margin={{ top: 8, right: 24, bottom: 8, left: 48 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={
                      theme === "dark"
                        ? "rgba(255,255,255,0.08)"
                        : "rgba(0,0,0,0.08)"
                    }
                  />
                  <XAxis
                    type="number"
                    tick={{
                      fill: theme === "dark" ? "#9CA3AF" : "#6B7280",
                      fontSize: 10,
                    }}
                    tickLine={false}
                    label={{
                      value: "Missing values",
                      position: "insideBottom",
                      offset: -4,
                      fill: theme === "dark" ? "#9CA3AF" : "#4B5563",
                    }}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{
                      fill: theme === "dark" ? "#9CA3AF" : "#6B7280",
                      fontSize: 11,
                    }}
                    tickLine={false}
                    width={50}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle(theme)}
                    formatter={(v: any) => [`${v} missing`, ""]}
                  />
                  <Bar
                    dataKey="missing"
                    fill="#EF4444"
                    radius={[0, 6, 6, 0]}
                    name="Missing pts"
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart message="Upload a dataset to compare data completeness across variables." />
            )}
          </div>
        </GlassCard>
      )}
    </div>
  );
}
