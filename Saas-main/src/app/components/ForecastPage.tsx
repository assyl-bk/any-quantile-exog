import { useState, useEffect } from "react";
import { GlassCard } from "./GlassCard";
import {
  Calendar,
  Download,
  Settings,
  Zap,
  AlertTriangle,
  TrendingUp,
  Database,
} from "lucide-react";
import { motion } from "motion/react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
  ReferenceLine,
} from "recharts";
import { useTheme } from "../context/ThemeContext";
import { useAuth } from "../context/AuthContext";
import { useData, QuantilePoint, ForecastResult } from "../context/Datacontext";

// ── Quantile options shown as checkboxes ──────────────────────────────────────
const QUANTILE_OPTIONS = ["5%", "25%", "50%", "75%", "90%"];

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Export chart data as CSV */
function exportForecastCSV(result: ForecastResult) {
  const header = "time,hour,q5,q25,q50,q75,q90,custom\n";
  const rows = result.chartData
    .map(
      (d) =>
        `${d.time},${d.hour},${d.lower5 ?? ""},${d.lower25 ?? ""},${d.forecast ?? ""},${d.upper75 ?? ""},${d.upper90 ?? ""},${d.customQuantile ?? ""}`,
    )
    .join("\n");
  const blob = new Blob([header + rows], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `forecast_${result.selectedVariable}_q${result.quantiles.join("_")}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Component ──────────────────────────────────────────────────────────────────

export function ForecastPage() {
  const { theme } = useTheme();
  const { user } = useAuth();
  const { dataset, forecastResult, setForecastResult } = useData();

  // Config state
  const [horizon, setHorizon] = useState<24 | 48 | 72>(48);
  const [selectedQuantiles, setSelectedQuantiles] = useState(["50%", "90%"]);
  const [customQuantile, setCustomQuantile] = useState(75);
  const [selectedVariable, setSelectedVariable] = useState<string>("");
  const [startDate, setStartDate] = useState(
    new Date().toISOString().slice(0, 16),
  );
  const [capacityThreshold, setCapacityThreshold] = useState(11000);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Request state
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-select first variable when dataset loads
  useEffect(() => {
    if (dataset?.variables.length && !selectedVariable) {
      setSelectedVariable(dataset.variables[0].unique_id);
    }
  }, [dataset, selectedVariable]);

  // Derived
  const forecastData = forecastResult?.chartData ?? [];
  const generated = !!forecastResult;

  const hasCapacityAlert = forecastData.some(
    (d) =>
      d.customQuantile !== undefined && d.customQuantile > capacityThreshold,
  );

  const handleGenerate = async () => {
    setError(null);

    if (!dataset) {
      setError("No dataset loaded. Please upload a CSV file first.");
      return;
    }
    if (!selectedVariable) {
      setError("Please select a variable to forecast.");
      return;
    }

    const variable = dataset.variables.find(
      (v) => v.unique_id === selectedVariable,
    );
    if (!variable) {
      setError(`Variable "${selectedVariable}" not found.`);
      return;
    }

    const validValues = variable.rows
      .filter((r) => r.y !== null)
      .map((r) => r.y as number);

    if (validValues.length < 24) {
      setError(
        `Variable "${selectedVariable}" has only ${validValues.length} valid points. Need at least 24 (168 preferred).`,
      );
      return;
    }

    setGenerating(true);

    // api_key is provided by the Token response at login/signup and stored in AuthContext
    const apiKey: string | null =
      (user as any)?.api_key ?? (user as any)?.apiKey ?? null;

    if (!apiKey) {
      setError(
        "No API key found. Please log out and log in again to refresh your session.",
      );
      setGenerating(false);
      return;
    }

    try {
      // Limit historical data to the most recent 10,000 points (API constraint)
      const historicalData = validValues.slice(
        Math.max(0, validValues.length - 10000),
      );

      const response = await fetch("/api/forecast/forecast", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": apiKey,
        },
        body: JSON.stringify({
          historical_data: historicalData,
          quantiles: [0.05, 0.25, 0.5, 0.75, 0.9],
          apply_cqr: true,
          horizon,
        }),
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        const detail = errBody.detail;
        const message = Array.isArray(detail)
          ? detail.map((e: any) => e.msg ?? JSON.stringify(e)).join("; ")
          : typeof detail === "string"
            ? detail
            : `HTTP ${response.status}`;
        throw new Error(message);
      }

      const result = await response.json();
      // result: { forecasts: [{quantile, values}], forecast_horizon, model_version, quantiles }

      const chartData: QuantilePoint[] = [];
      const startMs = new Date(startDate).getTime();

      for (let i = 0; i < (result.forecast_horizon ?? horizon); i++) {
        const time = new Date(startMs + i * 3_600_000).toLocaleTimeString(
          "en-US",
          {
            hour: "2-digit",
            minute: "2-digit",
          },
        );
        const point: QuantilePoint = { time, hour: i };

        for (const f of result.forecasts ?? []) {
          const v = f.values[i];
          if (f.quantile === 0.05) point.lower5 = v;
          if (f.quantile === 0.25) point.lower25 = v;
          if (f.quantile === 0.5) point.forecast = v;
          if (f.quantile === 0.75) point.upper75 = v;
          if (f.quantile === 0.9) point.upper90 = v;
        }

        // Custom quantile: pick closest available or interpolate
        const cq = customQuantile / 100;
        const byDistance = [
          { q: 0.05, v: point.lower5 },
          { q: 0.25, v: point.lower25 },
          { q: 0.5, v: point.forecast },
          { q: 0.75, v: point.upper75 },
          { q: 0.9, v: point.upper90 },
        ]
          .filter((x) => x.v !== undefined)
          .sort((a, b) => Math.abs(a.q - cq) - Math.abs(b.q - cq));
        point.customQuantile = byDistance[0]?.v;

        chartData.push(point);
      }

      const validChart = chartData.filter(
        (d) => d.customQuantile !== undefined,
      );
      const peak = validChart.length
        ? Math.max(...validChart.map((d) => d.customQuantile!))
        : 0;
      const avg = validChart.length
        ? validChart.reduce((s, d) => s + d.customQuantile!, 0) /
          validChart.length
        : 0;

      setForecastResult({
        generatedAt: new Date(),
        horizon,
        quantiles: result.quantiles ?? [0.05, 0.25, 0.5, 0.75, 0.9],
        selectedVariable,
        chartData,
        modelVersion: result.model_version,
        peakDemand: peak,
        avgDemand: avg,
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unknown error generating forecast.",
      );
    } finally {
      setGenerating(false);
    }
  };

  const toggleQuantile = (q: string) =>
    setSelectedQuantiles((prev) =>
      prev.includes(q) ? prev.filter((x) => x !== q) : [...prev, q],
    );

  // ── Y-axis domain from data ──────────────────────────────────────────────────
  const allValues = forecastData.flatMap(
    (d) =>
      [
        d.lower5,
        d.lower25,
        d.forecast,
        d.upper75,
        d.upper90,
        d.customQuantile,
      ].filter((v) => v !== undefined) as number[],
  );
  const yMin = allValues.length ? Math.floor(Math.min(...allValues) * 0.97) : 0;
  const yMax = allValues.length
    ? Math.ceil(
        Math.max(Math.max(...allValues) * 1.03, capacityThreshold * 1.01),
      )
    : capacityThreshold + 1000;

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="grid grid-cols-[35%_1fr] gap-6 h-[calc(100vh-120px)]">
      {/* ── Left panel ── */}
      <div className="space-y-6 overflow-y-auto pr-2">
        <GlassCard>
          <h2 className="text-xl font-bold text-foreground mb-4">
            Energy Demand Forecast
          </h2>
          <p className="text-sm text-foreground-secondary mb-7">
            Configure quantile parameters for probabilistic predictions
          </p>

          {/* No dataset warning */}
          {!dataset && (
            <div className="mb-6 flex items-start gap-3 p-3 rounded-lg bg-warning/10 border border-warning/30">
              <Database className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
              <p className="text-sm text-warning">
                No data loaded. Upload a CSV on the <strong>Data</strong> page
                first.
              </p>
            </div>
          )}

          {/* API error */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 flex items-start gap-3 p-3 rounded-lg bg-red-500/10 border border-red-500/30"
            >
              <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-300">{error}</p>
            </motion.div>
          )}

          {/* Capacity alert */}
          {hasCapacityAlert && generated && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 flex items-start gap-3 p-3 rounded-lg bg-red-500/10 border border-red-500/30"
            >
              <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-red-400 mb-0.5">
                  Capacity Threshold Alert
                </p>
                <p className="text-xs text-red-300/80">
                  Predicted demand may exceed grid capacity at quantile{" "}
                  {customQuantile}%.
                </p>
              </div>
            </motion.div>
          )}

          {/* Custom Quantile Slider */}
          <div className="space-y-4 mb-7 p-5 rounded-lg bg-gradient-to-br from-primary/10 to-purple-500/10 border border-primary/30">
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm font-semibold text-foreground">
                Custom Quantile
              </label>
              <span className="text-3xl font-bold text-primary">
                {customQuantile}%
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="99"
              step="1"
              value={customQuantile}
              onChange={(e) => setCustomQuantile(Number(e.target.value))}
              className="w-full h-2 rounded-lg appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${customQuantile}%, rgba(255,255,255,0.1) ${customQuantile}%, rgba(255,255,255,0.1) 100%)`,
              }}
            />
            <div className="flex justify-between text-xs text-muted-foreground mb-3">
              <span>1% (lower)</span>
              <span>50% (median)</span>
              <span>99% (upper)</span>
            </div>
            <p className="text-xs text-foreground-secondary">
              {customQuantile < 50
                ? `Conservative estimate — ${customQuantile}% chance demand stays below this level`
                : customQuantile === 50
                  ? "Median — equal chance above or below"
                  : `Optimistic estimate — ${100 - customQuantile}% chance demand exceeds this level`}
            </p>
          </div>

          {/* Variable selector */}
          <div className="space-y-2 mb-6">
            <label className="text-sm font-semibold text-foreground">
              Variable to Forecast
            </label>
            {dataset ? (
              <select
                value={selectedVariable}
                onChange={(e) => setSelectedVariable(e.target.value)}
                className={`w-full rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 ${
                  theme === "dark"
                    ? "bg-white/5 border border-white/10 text-white"
                    : "bg-white border border-gray-300 text-gray-900"
                }`}
              >
                {dataset.variables.map((v) => (
                  <option key={v.unique_id} value={v.unique_id}>
                    {v.unique_id} ({v.validCount} valid pts)
                  </option>
                ))}
              </select>
            ) : (
              <div className="w-full rounded-lg px-3 py-2.5 text-sm bg-white/5 border border-white/10 text-muted-foreground">
                Upload data first
              </div>
            )}
          </div>

          {/* Horizon */}
          <div className="space-y-2 mb-6">
            <label className="text-sm font-semibold text-foreground">
              Forecast Horizon
            </label>
            <div className="grid grid-cols-3 gap-2">
              {([24, 48, 72] as const).map((h) => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={`py-2.5 px-4 rounded-lg font-medium text-sm transition-all ${
                    horizon === h
                      ? "bg-primary text-white shadow-lg shadow-primary/30 border border-primary/50"
                      : "bg-white/5 text-foreground-secondary hover:bg-white/10 border border-white/10"
                  }`}
                >
                  {h}H
                </button>
              ))}
            </div>
          </div>

          {/* Start date */}
          <div className="space-y-2 mb-6">
            <label className="text-sm font-semibold text-foreground">
              Forecast Start
            </label>
            <div className="relative">
              <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="datetime-local"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className={`w-full rounded-lg pl-10 pr-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 ${
                  theme === "dark"
                    ? "bg-white/5 border border-white/10 text-white"
                    : "bg-white border border-gray-300 text-gray-900"
                }`}
              />
            </div>
          </div>

          {/* Quantile overlays */}
          <div className="space-y-2 mb-6">
            <label className="text-sm font-semibold text-foreground">
              Additional Quantile Overlays
            </label>
            <div className="grid grid-cols-5 gap-1.5">
              {QUANTILE_OPTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => toggleQuantile(q)}
                  className={`py-2 px-2 rounded-lg text-xs font-medium transition-all ${
                    selectedQuantiles.includes(q)
                      ? "bg-primary/20 text-primary border border-primary/50"
                      : "bg-white/5 text-foreground-secondary hover:bg-white/10 border border-white/10"
                  }`}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Advanced */}
          <div className="space-y-2 mb-6">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-sm font-semibold text-primary hover:text-primary/80 transition-colors"
            >
              <Settings className="w-4 h-4" />
              Advanced Options
              <span className="ml-auto">{showAdvanced ? "−" : "+"}</span>
            </button>

            {showAdvanced && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                className="space-y-3 pt-2"
              >
                <div>
                  <label className="text-xs text-foreground-secondary mb-1.5 block">
                    Grid Capacity Threshold (MW)
                  </label>
                  <input
                    type="number"
                    value={capacityThreshold}
                    onChange={(e) =>
                      setCapacityThreshold(Number(e.target.value))
                    }
                    className={`w-full rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 ${
                      theme === "dark"
                        ? "bg-white/5 border border-white/10 text-white"
                        : "bg-white border border-gray-300 text-gray-900"
                    }`}
                  />
                </div>
              </motion.div>
            )}
          </div>

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={generating || !dataset}
            className="w-full py-4 px-6 rounded-xl bg-gradient-to-r from-primary to-[#2563EB] hover:shadow-lg hover:shadow-primary/30 text-white font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {generating ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Generate Forecast
              </>
            )}
          </button>
        </GlassCard>

        {/* System info */}
        <GlassCard className="bg-gradient-to-br from-primary/5 to-purple-500/5 border-primary/20">
          <h3 className="text-sm font-semibold text-foreground mb-3">
            System Status
          </h3>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-foreground-secondary">Data Loaded:</span>
              <span
                className={
                  dataset
                    ? "text-green-400 font-mono"
                    : "text-warning font-mono"
                }
              >
                {dataset ? `● ${dataset.variables.length} variables` : "○ None"}
              </span>
            </div>
            {dataset && (
              <div className="flex justify-between">
                <span className="text-foreground-secondary">
                  Selected Variable:
                </span>
                <span className="text-foreground font-mono">
                  {selectedVariable || "—"}
                </span>
              </div>
            )}
            {forecastResult && (
              <>
                <div className="flex justify-between">
                  <span className="text-foreground-secondary">
                    Last Forecast:
                  </span>
                  <span className="text-foreground font-mono">
                    {forecastResult.generatedAt.toLocaleTimeString()}
                  </span>
                </div>
                {forecastResult.modelVersion && (
                  <div className="flex justify-between">
                    <span className="text-foreground-secondary">Model:</span>
                    <span className="text-foreground font-mono">
                      {forecastResult.modelVersion}
                    </span>
                  </div>
                )}
              </>
            )}
            <div className="flex justify-between">
              <span className="text-foreground-secondary">API Endpoint:</span>
              <span className="text-green-400 font-mono">● /api/forecast</span>
            </div>
          </div>
        </GlassCard>
      </div>

      {/* ── Right panel: Chart ── */}
      <GlassCard className="flex flex-col h-full overflow-hidden">
        <div className="flex items-center justify-between mb-4 flex-shrink-0">
          <div>
            <h3 className="text-xl font-bold text-foreground mb-0.5">
              Forecast — {horizon}H
              {forecastResult && ` · ${forecastResult.selectedVariable}`}
            </h3>
            <p className="text-sm text-foreground-secondary">
              {generated
                ? `Quantile ${customQuantile}% · Generated at ${forecastResult!.generatedAt.toLocaleTimeString()}`
                : "Configure and generate a forecast to see results"}
            </p>
          </div>

          {generated && (
            <button
              onClick={() => exportForecastCSV(forecastResult!)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-sm text-white border border-white/10 transition-colors"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </button>
          )}
        </div>

        {/* Stats bar (when generated) */}
        {generated && (
          <div className="grid grid-cols-4 gap-3 mb-4 flex-shrink-0">
            {[
              {
                label: "Peak Demand",
                value: `${forecastResult!.peakDemand.toFixed(0)} MW`,
              },
              {
                label: "Avg Demand",
                value: `${forecastResult!.avgDemand.toFixed(0)} MW`,
              },
              { label: "Horizon", value: `${forecastResult!.horizon}H` },
              {
                label: "Quantile",
                value: `${customQuantile}%`,
                highlight: true,
              },
            ].map((s) => (
              <div
                key={s.label}
                className="bg-white/5 rounded-lg p-3 border border-white/10"
              >
                <div className="text-xs text-foreground-secondary mb-1">
                  {s.label}
                </div>
                <div
                  className={`text-lg font-bold ${s.highlight ? "text-primary" : "text-foreground"}`}
                >
                  {s.value}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Chart area */}
        <div className="flex-1 flex items-center justify-center min-h-0">
          {!generated && !generating && (
            <div className="text-center">
              <div className="w-24 h-24 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-primary/20 to-purple-500/20 flex items-center justify-center border border-primary/30">
                <Zap className="w-12 h-12 text-primary" />
              </div>
              <h4 className="text-lg font-semibold text-foreground mb-2">
                Ready to Forecast
              </h4>
              <p className="text-sm text-foreground-secondary max-w-sm mx-auto">
                {dataset
                  ? "Select a variable and click Generate Forecast"
                  : "Upload a CSV file on the Data page first"}
              </p>
            </div>
          )}

          {generating && (
            <div className="text-center">
              <div className="w-24 h-24 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-primary/20 to-purple-500/20 flex items-center justify-center border border-primary/30 animate-pulse">
                <Zap className="w-12 h-12 text-primary" />
              </div>
              <h4 className="text-lg font-semibold text-foreground mb-2">
                Generating Forecast…
              </h4>
              <p className="text-sm text-foreground-secondary">
                Calling model API for {selectedVariable}
              </p>
            </div>
          )}

          {generated && (
            <div className="w-full h-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={forecastData}
                  margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
                >
                  <defs>
                    <linearGradient id="gCustom" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10B981" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
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
                      fontSize: 10,
                    }}
                    tickLine={false}
                    interval={Math.max(1, Math.floor(forecastData.length / 8))}
                    angle={-35}
                    textAnchor="end"
                    height={55}
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
                    domain={[yMin, yMax]}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor:
                        theme === "dark"
                          ? "rgba(17,24,39,0.95)"
                          : "rgba(255,255,255,0.95)",
                      border:
                        theme === "dark"
                          ? "1px solid rgba(255,255,255,0.2)"
                          : "1px solid rgba(0,0,0,0.15)",
                      borderRadius: 8,
                      color: theme === "dark" ? "#fff" : "#111",
                    }}
                    formatter={(v: any, name: string) => [
                      `${Number(v).toFixed(1)} MW`,
                      name,
                    ]}
                  />
                  <Legend
                    wrapperStyle={{
                      color: theme === "dark" ? "#D1D5DB" : "#4B5563",
                      fontSize: 12,
                    }}
                  />

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

                  {/* Quantile overlays */}
                  {selectedQuantiles.includes("90%") && (
                    <Line
                      type="monotone"
                      dataKey="upper90"
                      stroke="#F59E0B"
                      strokeWidth={1}
                      dot={false}
                      strokeDasharray="3 3"
                      name="Q90"
                    />
                  )}
                  {selectedQuantiles.includes("75%") && (
                    <Line
                      type="monotone"
                      dataKey="upper75"
                      stroke="#8B5CF6"
                      strokeWidth={1}
                      dot={false}
                      strokeDasharray="2 2"
                      name="Q75"
                    />
                  )}
                  {selectedQuantiles.includes("50%") && (
                    <Area
                      type="monotone"
                      dataKey="forecast"
                      stroke="#10B981"
                      strokeWidth={2}
                      fill="url(#gForecast)"
                      dot={false}
                      name="Q50 (Median)"
                    />
                  )}
                  {selectedQuantiles.includes("25%") && (
                    <Line
                      type="monotone"
                      dataKey="lower25"
                      stroke="#06B6D4"
                      strokeWidth={1}
                      dot={false}
                      strokeDasharray="2 2"
                      name="Q25"
                    />
                  )}
                  {selectedQuantiles.includes("5%") && (
                    <Line
                      type="monotone"
                      dataKey="lower5"
                      stroke="#6B7280"
                      strokeWidth={1}
                      dot={false}
                      strokeDasharray="3 3"
                      name="Q5"
                    />
                  )}

                  {/* Primary: custom quantile */}
                  <Area
                    type="monotone"
                    dataKey="customQuantile"
                    stroke="#3B82F6"
                    strokeWidth={3}
                    fill="url(#gCustom)"
                    dot={false}
                    name={`Q${customQuantile} (Custom)`}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </GlassCard>
    </div>
  );
}
