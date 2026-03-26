import {
  createContext,
  useContext,
  useState,
  useCallback,
  ReactNode,
} from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface ParsedRow {
  ds: string; // raw timestamp string
  unique_id: string; // variable name e.g. "Var1"
  y: number | null; // demand value (null if missing)
}

export interface TimePoint {
  ds: string;
  y: number | null;
}

export interface VariableSeries {
  unique_id: string;
  rows: TimePoint[];
  validCount: number;
  missingCount: number;
  min: number;
  max: number;
  mean: number;
  stdDev: number;
}

export interface UploadedDataset {
  fileName: string;
  uploadedAt: Date;
  rawRows: ParsedRow[];
  variables: VariableSeries[];
  totalRows: number;
}

export interface QuantilePoint {
  time: string;
  hour: number;
  forecast?: number; // median (q50)
  lower5?: number;
  lower25?: number;
  upper75?: number;
  upper90?: number;
  customQuantile?: number;
}

export interface ForecastResult {
  generatedAt: Date;
  horizon: number;
  quantiles: number[];
  selectedVariable: string;
  chartData: QuantilePoint[];
  modelVersion?: string;
  peakDemand: number;
  avgDemand: number;
}

interface DataContextValue {
  dataset: UploadedDataset | null;
  setDataset: (d: UploadedDataset | null) => void;
  forecastResult: ForecastResult | null;
  setForecastResult: (r: ForecastResult | null) => void;
  clearAll: () => void;
}

// ── Context ───────────────────────────────────────────────────────────────────

const DataContext = createContext<DataContextValue | null>(null);

export function DataProvider({ children }: { children: ReactNode }) {
  const [dataset, setDataset] = useState<UploadedDataset | null>(null);
  const [forecastResult, setForecastResult] = useState<ForecastResult | null>(
    null,
  );

  const clearAll = useCallback(() => {
    setDataset(null);
    setForecastResult(null);
  }, []);

  return (
    <DataContext.Provider
      value={{
        dataset,
        setDataset,
        forecastResult,
        setForecastResult,
        clearAll,
      }}
    >
      {children}
    </DataContext.Provider>
  );
}

export function useData() {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error("useData must be used inside <DataProvider>");
  return ctx;
}

// ── CSV Parser ────────────────────────────────────────────────────────────────
// Supports format:  ds,unique_id,y
//                   01-Jan-2006 00:00:00,Var1,
//                   01-Jan-2006 00:00:00,Var2,6297.0

export function parseEnergyCSV(text: string): {
  data: UploadedDataset | null;
  error: string | null;
} {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2)
    return { data: null, error: "File is empty or has only a header row." };

  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());

  // Find column indices flexibly
  const dsIdx = header.findIndex((h) =>
    ["ds", "date", "datetime", "timestamp", "time"].includes(h),
  );
  const idIdx = header.findIndex((h) =>
    ["unique_id", "id", "variable", "series"].includes(h),
  );
  const yIdx = header.findIndex((h) =>
    ["y", "value", "demand", "load", "actual", "mw"].includes(h),
  );

  if (dsIdx === -1)
    return {
      data: null,
      error: 'Missing column "ds" (or date/datetime/timestamp).',
    };
  if (idIdx === -1)
    return {
      data: null,
      error: 'Missing column "unique_id" (or id/variable/series).',
    };
  if (yIdx === -1)
    return {
      data: null,
      error: 'Missing column "y" (or value/demand/load/mw).',
    };

  const rawRows: ParsedRow[] = [];
  const parseErrors: string[] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    // Split on comma, respecting potential quoted fields
    const parts = line.split(",");
    if (parts.length < Math.max(dsIdx, idIdx, yIdx) + 1) continue;

    const ds = parts[dsIdx]?.trim() ?? "";
    const unique_id = parts[idIdx]?.trim() ?? "";
    const rawY = parts[yIdx]?.trim() ?? "";
    const y = rawY === "" ? null : parseFloat(rawY);

    if (!ds || !unique_id) continue;
    if (rawY !== "" && y !== null && isNaN(y)) {
      parseErrors.push(`Row ${i + 1}: "${rawY}" is not a valid number`);
      continue;
    }

    rawRows.push({ ds, unique_id, y });
  }

  if (rawRows.length === 0)
    return { data: null, error: "No valid data rows found." };

  // Group by unique_id
  const grouped = new Map<string, TimePoint[]>();
  for (const row of rawRows) {
    if (!grouped.has(row.unique_id)) grouped.set(row.unique_id, []);
    grouped.get(row.unique_id)!.push({ ds: row.ds, y: row.y });
  }

  const variables: VariableSeries[] = [];
  for (const [unique_id, rows] of grouped) {
    const valid = rows.filter((r) => r.y !== null).map((r) => r.y as number);
    const missing = rows.length - valid.length;
    const min = valid.length ? Math.min(...valid) : 0;
    const max = valid.length ? Math.max(...valid) : 0;
    const mean = valid.length
      ? valid.reduce((a, b) => a + b, 0) / valid.length
      : 0;
    const variance = valid.length
      ? valid.reduce((acc, v) => acc + (v - mean) ** 2, 0) / valid.length
      : 0;
    const stdDev = Math.sqrt(variance);

    variables.push({
      unique_id,
      rows,
      validCount: valid.length,
      missingCount: missing,
      min,
      max,
      mean,
      stdDev,
    });
  }

  // Sort variables by name
  variables.sort((a, b) =>
    a.unique_id.localeCompare(b.unique_id, undefined, { numeric: true }),
  );

  return {
    data: {
      fileName: "",
      uploadedAt: new Date(),
      rawRows,
      variables,
      totalRows: rawRows.length,
    },
    error:
      parseErrors.length > 0
        ? `Parsed with ${parseErrors.length} skipped rows.`
        : null,
  };
}
