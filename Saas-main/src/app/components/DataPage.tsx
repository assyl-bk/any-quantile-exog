import { useState, useRef, useCallback } from "react";
import { GlassCard } from "./GlassCard";
import {
  Upload,
  Search,
  Download,
  Eye,
  CheckCircle,
  AlertCircle,
  X,
  TrendingUp,
  FileText,
  AlertTriangle,
  BarChart2,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import {
  useData,
  parseEnergyCSV,
  VariableSeries,
} from "../context/Datacontext";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useTheme } from "../context/ThemeContext";

// ── Variable Detail Modal ──────────────────────────────────────────────────────

function VariableModal({
  variable,
  onClose,
}: {
  variable: VariableSeries;
  onClose: () => void;
}) {
  const { theme } = useTheme();
  const chartData = variable.rows
    .filter((r) => r.y !== null)
    .map((r, i) => ({ index: i, value: r.y as number, time: r.ds }));

  const qualityPct = variable.validCount
    ? Math.round((variable.validCount / variable.rows.length) * 100)
    : 0;

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-8 bg-black/60 backdrop-blur-sm"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.div
        className="w-full max-w-5xl max-h-[90vh] overflow-hidden"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
      >
        <GlassCard className="p-0">
          {/* Header */}
          <div className="px-8 py-6 border-b border-white/10 flex items-center justify-between">
            <div>
              <h3 className="text-xl font-bold text-foreground mb-1">
                {variable.unique_id}
              </h3>
              <p className="text-sm text-foreground-secondary">
                {variable.rows.length} time points · {variable.validCount} valid
                · {variable.missingCount} missing
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors"
            >
              <X className="w-5 h-5 text-foreground-secondary" />
            </button>
          </div>

          {/* Stats */}
          <div className="px-8 py-6 grid grid-cols-5 gap-4 border-b border-white/10">
            {[
              { label: "Min", value: variable.min.toFixed(1) },
              { label: "Max", value: variable.max.toFixed(1) },
              { label: "Mean", value: variable.mean.toFixed(1) },
              { label: "Std Dev", value: variable.stdDev.toFixed(1) },
              {
                label: "Quality",
                value: `${qualityPct}%`,
                highlight:
                  qualityPct >= 90
                    ? "text-success"
                    : qualityPct >= 75
                      ? "text-warning"
                      : "text-critical",
              },
            ].map((s) => (
              <div key={s.label}>
                <p className="text-xs text-foreground-secondary mb-1">
                  {s.label}
                </p>
                <p
                  className={`text-lg font-mono font-semibold ${s.highlight ?? "text-foreground"}`}
                >
                  {s.value}
                </p>
              </div>
            ))}
          </div>

          {/* Chart */}
          {chartData.length > 0 && (
            <div className="px-8 py-4 border-b border-white/10">
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={chartData}
                    margin={{ top: 4, right: 16, left: 0, bottom: 4 }}
                  >
                    <defs>
                      <linearGradient
                        id="modalAreaGrad"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor="#3B82F6"
                          stopOpacity={0.3}
                        />
                        <stop
                          offset="95%"
                          stopColor="#3B82F6"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="rgba(255,255,255,0.07)"
                    />
                    <XAxis
                      dataKey="index"
                      tick={{ fill: "#9CA3AF", fontSize: 10 }}
                      tickLine={false}
                      interval={Math.floor(chartData.length / 6)}
                    />
                    <YAxis
                      tick={{ fill: "#9CA3AF", fontSize: 10 }}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor:
                          theme === "dark"
                            ? "rgba(17,24,39,0.95)"
                            : "rgba(255,255,255,0.95)",
                        border: "1px solid rgba(255,255,255,0.15)",
                        borderRadius: 8,
                        color: theme === "dark" ? "#fff" : "#111",
                      }}
                      formatter={(v: any) => [v.toFixed(2), "Value"]}
                      labelFormatter={(l) => chartData[l]?.time ?? l}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#3B82F6"
                      strokeWidth={2}
                      fill="url(#modalAreaGrad)"
                      dot={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Raw table sample */}
          <div className="px-8 py-4 overflow-auto max-h-[220px]">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-[#111827]/95 backdrop-blur-xl">
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 px-3 text-xs text-foreground-secondary">
                    #
                  </th>
                  <th className="text-left py-2 px-3 text-xs text-foreground-secondary">
                    Timestamp (ds)
                  </th>
                  <th className="text-right py-2 px-3 text-xs text-foreground-secondary">
                    Value (y)
                  </th>
                  <th className="text-center py-2 px-3 text-xs text-foreground-secondary">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody>
                {variable.rows.slice(0, 30).map((row, i) => (
                  <tr
                    key={i}
                    className="border-b border-white/5 hover:bg-white/5"
                  >
                    <td className="py-2 px-3 text-foreground-secondary font-mono text-xs">
                      {i + 1}
                    </td>
                    <td className="py-2 px-3 text-foreground font-mono text-xs">
                      {row.ds}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-xs text-foreground">
                      {row.y !== null ? row.y.toFixed(2) : "—"}
                    </td>
                    <td className="py-2 px-3 text-center">
                      {row.y !== null ? (
                        <CheckCircle className="w-3 h-3 text-success inline" />
                      ) : (
                        <AlertCircle className="w-3 h-3 text-warning inline" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {variable.rows.length > 30 && (
              <p className="text-xs text-foreground-secondary text-center mt-2 py-2">
                Showing first 30 of {variable.rows.length} rows
              </p>
            )}
          </div>

          {/* Footer */}
          <div className="px-8 py-5 border-t border-white/10 flex gap-3 justify-end">
            <button
              onClick={onClose}
              className="px-6 py-2.5 rounded-lg bg-white/5 hover:bg-white/10 text-white font-medium transition-colors border border-white/10"
            >
              Close
            </button>
          </div>
        </GlassCard>
      </motion.div>
    </motion.div>
  );
}

// ── Main DataPage ──────────────────────────────────────────────────────────────

export function DataPage() {
  const { dataset, setDataset } = useData();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedVariable, setSelectedVariable] =
    useState<VariableSeries | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadWarning, setUploadWarning] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── File Handling ──────────────────────────────────────────────────────────

  const processFile = useCallback(
    (file: File) => {
      if (!file.name.endsWith(".csv")) {
        setUploadError("Please upload a .csv file.");
        return;
      }
      setUploadError(null);
      setUploadWarning(null);

      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        const { data, error } = parseEnergyCSV(text);
        if (data) {
          data.fileName = file.name;
          setDataset(data);
          if (error) setUploadWarning(error); // soft warning (skipped rows)
        } else {
          setUploadError(error ?? "Unknown parse error.");
        }
      };
      reader.readAsText(file);
    },
    [setDataset],
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    e.target.value = ""; // allow re-upload of same file
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  };

  // ── Derived Stats ──────────────────────────────────────────────────────────

  const filteredVariables = (dataset?.variables ?? []).filter((v) =>
    v.unique_id.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  const overallQuality = dataset
    ? Math.round(
        (dataset.variables.reduce((acc, v) => acc + v.validCount, 0) /
          dataset.totalRows) *
          100,
      )
    : 0;

  const getQualityColor = (pct: number) =>
    pct >= 90 ? "text-success" : pct >= 75 ? "text-warning" : "text-critical";
  const getQualityBg = (pct: number) =>
    pct >= 90
      ? "bg-success/20 border-success/30"
      : pct >= 75
        ? "bg-warning/20 border-warning/30"
        : "bg-critical/20 border-critical/30";
  const getQualityBar = (pct: number) =>
    pct >= 90 ? "bg-success" : pct >= 75 ? "bg-warning" : "bg-critical";

  const downloadCSVTemplate = () => {
    const template = [
      "ds,unique_id,y",
      "01-Jan-2006 00:00:00,Var1,8234.5",
      "01-Jan-2006 01:00:00,Var1,7980.2",
      "01-Jan-2006 00:00:00,Var2,6297.0",
      "01-Jan-2006 01:00:00,Var2,6450.1",
    ].join("\n");
    const blob = new Blob([template], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "energy_template.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-1">
            Time Series Management
          </h2>
          <p className="text-sm text-foreground-secondary">
            Upload and inspect your{" "}
            <code className="bg-white/10 px-1.5 py-0.5 rounded text-xs">
              ds, unique_id, y
            </code>{" "}
            CSV file
          </p>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        onChange={handleFileInput}
        className="hidden"
      />

      {/* Upload errors / warnings */}
      {uploadError && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/30"
        >
          <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-400 mb-0.5">
              Upload Failed
            </p>
            <p className="text-xs text-red-300/80">{uploadError}</p>
            <p className="text-xs text-red-300/60 mt-1">
              Expected format:{" "}
              <code className="bg-red-900/30 px-1 rounded">ds,unique_id,y</code>
            </p>
          </div>
        </motion.div>
      )}
      {uploadWarning && !uploadError && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start gap-3 p-4 rounded-lg bg-warning/10 border border-warning/30"
        >
          <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
          <p className="text-sm text-warning">{uploadWarning}</p>
        </motion.div>
      )}

      {/* ── No data: Drop Zone ── */}
      {!dataset && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`border-2 border-dashed rounded-2xl p-16 text-center transition-all cursor-pointer ${
            isDragging
              ? "border-primary bg-primary/10 scale-[1.01]"
              : "border-white/20 hover:border-primary/50 hover:bg-white/5"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-primary/20 to-purple-500/20 flex items-center justify-center border border-primary/30">
            <Upload className="w-10 h-10 text-primary" />
          </div>
          <h3 className="text-xl font-bold text-foreground mb-2">
            Drop your CSV here
          </h3>
          <p className="text-foreground-secondary mb-1">or click to browse</p>
          <p className="text-xs text-muted-foreground">
            Required columns:{" "}
            <code className="bg-white/10 px-1.5 py-0.5 rounded">ds</code>{" "}
            <code className="bg-white/10 px-1.5 py-0.5 rounded">unique_id</code>{" "}
            <code className="bg-white/10 px-1.5 py-0.5 rounded">y</code>
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Example timestamp:{" "}
            <code className="bg-white/10 px-1.5 py-0.5 rounded">
              01-Jan-2006 00:00:00
            </code>
          </p>
        </motion.div>
      )}

      {/* ── Data loaded ── */}
      {dataset && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-4 gap-4">
            <GlassCard className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-foreground-secondary">File</span>
                <FileText className="w-4 h-4 text-primary" />
              </div>
              <div className="text-sm font-semibold text-foreground truncate">
                {dataset.fileName}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Uploaded {dataset.uploadedAt.toLocaleTimeString()}
              </div>
            </GlassCard>

            <GlassCard className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-foreground-secondary">
                  Total Rows
                </span>
                <TrendingUp className="w-4 h-4 text-primary" />
              </div>
              <div className="text-2xl font-bold text-foreground">
                {dataset.totalRows.toLocaleString()}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {dataset.variables.length} variables
              </div>
            </GlassCard>

            <GlassCard className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-foreground-secondary">
                  Data Quality
                </span>
                <CheckCircle
                  className={`w-4 h-4 ${getQualityColor(overallQuality)}`}
                />
              </div>
              <div
                className={`text-2xl font-bold ${getQualityColor(overallQuality)}`}
              >
                {overallQuality}%
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {dataset.variables.reduce((a, v) => a + v.missingCount, 0)}{" "}
                missing values
              </div>
            </GlassCard>

            <GlassCard className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-foreground-secondary">
                  Variables
                </span>
                <BarChart2 className="w-4 h-4 text-primary" />
              </div>
              <div className="text-2xl font-bold text-foreground">
                {dataset.variables.length}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Unique time series
              </div>
            </GlassCard>
          </div>

          {/* Search + Replace button */}
          <GlassCard className="p-4">
            <div className="flex items-center gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Search variables..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-lg pl-10 pr-4 py-2 text-sm text-white placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                />
              </div>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-sm text-white border border-white/10 transition-colors"
              >
                <Upload className="w-4 h-4" />
                Replace File
              </button>
            </div>
          </GlassCard>

          {/* Variables Table */}
          <GlassCard className="p-0 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left px-6 py-4 text-xs font-semibold text-foreground-secondary">
                      Variable
                    </th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-foreground-secondary">
                      Points
                    </th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-foreground-secondary">
                      Valid
                    </th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-foreground-secondary">
                      Missing
                    </th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-foreground-secondary">
                      Min
                    </th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-foreground-secondary">
                      Max
                    </th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-foreground-secondary">
                      Mean
                    </th>
                    <th className="px-6 py-4 text-xs font-semibold text-foreground-secondary">
                      Quality
                    </th>
                    <th className="px-6 py-4 text-xs font-semibold text-foreground-secondary text-right">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {filteredVariables.map((v, idx) => {
                    const qualPct = Math.round(
                      (v.validCount / v.rows.length) * 100,
                    );
                    return (
                      <motion.tr
                        key={v.unique_id}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.03 }}
                        className="border-b border-white/5 hover:bg-white/5 transition-colors"
                      >
                        <td className="px-6 py-4">
                          <span className="font-semibold text-foreground">
                            {v.unique_id}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-sm text-foreground">
                          {v.rows.length.toLocaleString()}
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-sm text-success">
                          {v.validCount.toLocaleString()}
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-sm">
                          <span
                            className={
                              v.missingCount > 0
                                ? "text-warning"
                                : "text-foreground-secondary"
                            }
                          >
                            {v.missingCount}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-xs text-foreground-secondary">
                          {v.min.toFixed(1)}
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-xs text-foreground-secondary">
                          {v.max.toFixed(1)}
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-xs text-foreground-secondary">
                          {v.mean.toFixed(1)}
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2 min-w-[100px]">
                            <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                              <div
                                className={`h-full ${getQualityBar(qualPct)}`}
                                style={{ width: `${qualPct}%` }}
                              />
                            </div>
                            <span
                              className={`text-xs font-semibold ${getQualityColor(qualPct)}`}
                            >
                              {qualPct}%
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center justify-end gap-1">
                            <button
                              onClick={() => setSelectedVariable(v)}
                              className="p-2 rounded-lg hover:bg-white/10 transition-colors group"
                              title="Inspect"
                            >
                              <Eye className="w-4 h-4 text-foreground-secondary group-hover:text-primary" />
                            </button>
                          </div>
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>

              {filteredVariables.length === 0 && (
                <div className="py-12 text-center text-foreground-secondary text-sm">
                  No variables match "{searchQuery}"
                </div>
              )}
            </div>
          </GlassCard>
        </>
      )}

      {/* Variable Detail Modal */}
      <AnimatePresence>
        {selectedVariable && (
          <VariableModal
            variable={selectedVariable}
            onClose={() => setSelectedVariable(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
