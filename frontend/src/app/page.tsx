"use client";
import { useState } from "react";

const API_URL = "http://localhost:8000/predict";
const WINDOW_SIZE = 100;
const MID_PRICE = 1000;

type Snapshot = [number, number, number][];
type WindowData = Snapshot[];
type Result = { Prediction: number; Confidence: number };
type InputMode = "generate" | "paste";

function generateSnapshot(midPrice: number, spoof = false): Snapshot {
  const snap: Snapshot = [];
  for (let i = 0; i < 3; i++) {
    const cSpeed = spoof ? 0.7 + Math.random() * 0.3 : Math.random() * 0.3;
    const price = Math.floor(Math.random() * midPrice);
    const quantity = spoof
      ? Math.floor(midPrice * 1.5 + Math.random() * midPrice * 0.5)
      : Math.floor(Math.random() * midPrice * 2);
    snap.push([price, quantity, parseFloat(cSpeed.toFixed(4))]);
  }
  for (let i = 0; i < 3; i++) {
    const cSpeed = Math.random() * 0.3;
    const price = Math.floor(midPrice + Math.random() * midPrice);
    const quantity = Math.floor(Math.random() * midPrice * 2);
    snap.push([-price, quantity, parseFloat(cSpeed.toFixed(4))]);
  }
  return snap;
}

function generateDummyWindow(): WindowData {
  return Array.from({ length: WINDOW_SIZE }, (_, i) =>
    generateSnapshot(MID_PRICE, i >= 40 && i < 60),
  );
}

function CancelSpeedBar({ value }: { value: number }) {
  const color =
    value > 0.6 ? "bg-red-400" : value > 0.3 ? "bg-amber-400" : "bg-green-400";
  return (
    <div className="flex items-center gap-2">
      <div className="w-12 h-1 rounded-full bg-stone-200 overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${(value * 100).toFixed(0)}%` }}
        />
      </div>
      <span className="text-stone-400 text-xs font-mono">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

function OrderBookTable({ data }: { data: Snapshot }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="border-b border-stone-100">
            {["#", "Side", "Price", "Qty", "Cancel Speed"].map((h) => (
              <th
                key={h}
                className="px-3 py-2 text-left text-stone-400 font-medium"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => {
            const isSell = row[0] < 0;
            return (
              <tr key={idx} className="border-b border-stone-50">
                <td className="px-3 py-1.5 text-stone-400">{idx + 1}</td>
                <td className="px-3 py-1.5">
                  <span
                    className={`text-xs px-2 py-0.5 rounded ${
                      isSell
                        ? "bg-red-50 text-red-600"
                        : "bg-green-50 text-green-700"
                    }`}
                  >
                    {isSell ? "sell" : "buy"}
                  </span>
                </td>
                <td className="px-3 py-1.5 text-stone-700">
                  {Math.abs(row[0])}
                </td>
                <td className="px-3 py-1.5 text-stone-700">{row[1]}</td>
                <td className="px-3 py-1.5">
                  <CancelSpeedBar value={row[2]} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function MetricCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "red" | "green" | "neutral";
}) {
  const valueColor =
    accent === "red"
      ? "text-red-600"
      : accent === "green"
        ? "text-green-700"
        : "text-stone-800";
  return (
    <div className="bg-stone-50 rounded-lg px-4 py-3 flex-1 min-w-[100px]">
      <p className="text-xs text-stone-400 mb-1 tracking-wide">{label}</p>
      <p className={`text-xl font-medium tabular-nums ${valueColor}`}>
        {value}
      </p>
    </div>
  );
}

export default function Dashboard() {
  const [windowData, setWindowData] = useState<WindowData | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeSnapshot, setActiveSnapshot] = useState(0);
  const [jsonInput, setJsonInput] = useState("");
  const [inputMode, setInputMode] = useState<InputMode>("generate");

  const handleGenerate = () => {
    setWindowData(generateDummyWindow());
    setResult(null);
    setError(null);
    setActiveSnapshot(0);
  };

  const handleJsonPaste = () => {
    try {
      const parsed = JSON.parse(jsonInput);
      if (!Array.isArray(parsed) || parsed.length !== WINDOW_SIZE) {
        setError(`Expected array of exactly ${WINDOW_SIZE} snapshots.`);
        return;
      }
      setWindowData(parsed);
      setResult(null);
      setError(null);
      setActiveSnapshot(0);
    } catch {
      setError("Invalid JSON — please check the format.");
    }
  };

  const handlePredict = async () => {
    if (!windowData) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: windowData }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const json: Result = await res.json();
      setResult(json);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Could not reach server.");
    } finally {
      setLoading(false);
    }
  };

  const isSpoofing = result?.Prediction === 1;
  const confidence = result ? (result.Confidence * 100).toFixed(1) : null;

  return (
    <div className="min-h-screen bg-stone-50 flex flex-col items-center justify-start py-12 px-4 ">
      <div className="w-full max-w-2xl flex flex-col items-center gap-6 pt-5">
        {/* Header */}
        <div className="text-center pb-10 pt-10">
          <h1 className="text-4xl font-medium text-stone-800 tracking-tight mb-1">
            Market Anomaly Detector
          </h1>
          <p className="text-md text-stone-400">
            LSTM-based spoofing and layering detection on order book data
          </p>
        </div>

        {/* Input card */}
        <div className="w-full bg-white rounded-xl border border-stone-200 p-6 ">
          <div className="flex gap-2 mb-5">
            {(["generate", "paste"] as InputMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => {
                  setInputMode(mode);
                  setWindowData(null);
                  setResult(null);
                  setError(null);
                }}
                className={`px-4 py-1.5 rounded-md text-sm border transition-colors ${
                  inputMode === mode
                    ? "bg-stone-800 text-white border-stone-800"
                    : "bg-transparent text-stone-500 border-stone-200 hover:border-stone-400"
                }`}
              >
                {mode === "generate" ? "Generate dummy data" : "Paste your own"}
              </button>
            ))}
          </div>

          {inputMode === "generate" && (
            <div className="flex flex-col items-center gap-3">
              <p className="text-sm text-stone-400 text-center">
                Generates a 100-snapshot window with a spoofing pattern planted
                at timesteps 40–60.
              </p>
              <button
                onClick={handleGenerate}
                className="px-5 py-2 rounded-md border border-stone-300 bg-stone-50 text-stone-700 text-sm hover:bg-stone-100 transition-colors"
              >
                Generate window
              </button>
            </div>
          )}

          {inputMode === "paste" && (
            <div className="flex flex-col gap-3">
              <p className="text-xs text-stone-400">
                Paste a JSON array of exactly 100 snapshots. Each snapshot is an
                array of 6 rows:{" "}
                <code className="bg-stone-100 px-1 rounded">
                  [price, quantity, cancel_speed]
                </code>
                . Sell-side prices should be negative.
              </p>
              <textarea
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                placeholder="[[[500, 200, 0.1], ...6 rows], ...100 snapshots]"
                className="w-full h-24 font-mono text-xs rounded-md border text-stone-400 border-stone-200 p-3 resize-y focus:outline-none focus:border-stone-400"
              />
              <button
                onClick={handleJsonPaste}
                className="self-start px-5 py-2 rounded-md border border-stone-300 bg-stone-50 text-stone-700 text-sm hover:bg-stone-100 transition-colors"
              >
                Load data
              </button>
            </div>
          )}

          {error && (
            <p className="mt-3 text-sm text-red-500 text-center">{error}</p>
          )}
        </div>

        {/* Snapshot browser */}
        {windowData && (
          <div className="w-full bg-white rounded-xl border border-stone-200 p-6">
            <div className="flex items-center gap-3 mb-4 flex-wrap">
              <span className="text-sm text-stone-400">Snapshot</span>
              <input
                type="range"
                min={0}
                max={WINDOW_SIZE - 1}
                step={1}
                value={activeSnapshot}
                onChange={(e) => setActiveSnapshot(Number(e.target.value))}
                className="flex-1 min-w-32 max-w-xs"
              />
              <span className="text-sm font-mono text-stone-700 min-w-[2rem] text-center">
                {activeSnapshot + 1}
              </span>
            </div>

            <OrderBookTable data={windowData[activeSnapshot]} />

            <div className="mt-5 flex justify-center">
              <button
                onClick={handlePredict}
                disabled={loading}
                className={`px-6 py-2.5 rounded-md border text-sm transition-colors ${
                  loading
                    ? "opacity-60 cursor-wait border-stone-300 text-stone-400"
                    : "bg-stone-800 text-white border-stone-800 hover:bg-stone-700"
                }`}
              >
                {loading ? "Analysing..." : "Run detection"}
              </button>
            </div>
          </div>
        )}

        {/* Result */}
        {result && (
          <div
            className={`w-full bg-white rounded-xl border-2 p-6 ${
              isSpoofing ? "border-red-200" : "border-green-200"
            }`}
          >
            <div className="flex items-center justify-center mb-5">
              <span
                className={`text-sm font-medium px-4 py-1.5 rounded-full ${
                  isSpoofing
                    ? "bg-red-50 text-red-600"
                    : "bg-green-50 text-green-700"
                }`}
              >
                {isSpoofing ? "Spoofing detected" : "Normal activity"}
              </span>
            </div>

            <div className="flex gap-3 mb-5 flex-wrap">
              <MetricCard
                label="verdict"
                value={isSpoofing ? "Spoof" : "Normal"}
                accent={isSpoofing ? "red" : "green"}
              />
              <MetricCard
                label="confidence"
                value={`${confidence}%`}
                accent={result.Confidence > 0.8 ? "neutral" : "neutral"}
              />
              <MetricCard label="snapshots" value={String(WINDOW_SIZE)} />
            </div>

            <div className="flex flex-col items-center gap-2">
              <p className="text-xs text-stone-400">Confidence</p>
              <div className="w-full max-w-xs h-2 rounded-full bg-stone-100 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isSpoofing ? "bg-red-400" : "bg-green-400"
                  }`}
                  style={{ width: `${confidence}%` }}
                />
              </div>
              <p className="text-xs text-stone-400 text-center max-w-sm">
                {isSpoofing
                  ? "Anomalous cancellation patterns detected — consistent with spoofing behavior."
                  : "No significant anomalies detected. Order book activity within normal parameters."}
              </p>
            </div>
          </div>
        )}

        <p className="text-xs text-stone-300 text-center">
          LSTM model trained on synthetic order book data
        </p>
      </div>
    </div>
  );
}
