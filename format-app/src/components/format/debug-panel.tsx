"use client";

import { Bug, Trash2, ChevronDown, ChevronRight } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export interface DebugLog {
  timestamp: string;
  type: "info" | "warn" | "error";
  message: string;
}

interface DebugPanelProps {
  logs: DebugLog[];
  onClear: () => void;
}

export function DebugPanel({ logs, onClear }: DebugPanelProps) {
  const [open, setOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, open]);

  const typeColor = { info: "text-gray-400", warn: "text-yellow-400", error: "text-red-400" };
  const typeLabel = { info: "INFO", warn: "WARN", error: "ERR " };

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-300 transition-colors font-mono"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <Bug className="h-3 w-3" />
        {open ? "Hide Debug" : "Show Debug"}
        {logs.length > 0 && <span className="text-gray-500">({logs.length})</span>}
      </button>
      {open && (
        <div className="mt-2 rounded-lg border border-gray-700 bg-gray-900 overflow-hidden">
          <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-700">
            <span className="text-xs text-gray-400 font-mono">Debug Log</span>
            <button onClick={onClear} className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors">
              <Trash2 className="h-3 w-3" /> Clear
            </button>
          </div>
          <div ref={scrollRef} className="max-h-64 overflow-y-auto p-2 font-mono text-xs leading-relaxed">
            {logs.length === 0 ? (
              <div className="text-gray-600 py-2 text-center">No activity yet</div>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-gray-600 shrink-0">[{log.timestamp}]</span>
                  <span className={`shrink-0 ${typeColor[log.type]}`}>[{typeLabel[log.type]}]</span>
                  <span className={log.type === "error" ? "text-red-300" : "text-gray-300"}>{log.message}</span>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
