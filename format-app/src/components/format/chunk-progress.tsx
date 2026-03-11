"use client";

import { Check, Clock, Loader2, X } from "lucide-react";
import type { ChunkResult, TranscriptChunk } from "./utils/types";

interface ChunkProgressProps {
  chunks: TranscriptChunk[];
  results: ChunkResult[];
}

export function ChunkProgress({ chunks, results }: ChunkProgressProps) {
  if (chunks.length === 0) return null;
  const doneCount = results.filter((r) => r.status === "done").length;
  const progress = chunks.length > 0 ? (doneCount / chunks.length) * 100 : 0;

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="px-5 py-3 border-b border-slate-100 bg-slate-50/60 flex justify-between items-center">
        <span className="text-sm font-semibold text-slate-800">Processing</span>
        <span className="text-xs font-medium text-slate-500 tabular-nums">{doneCount} / {chunks.length} chunks</span>
      </div>
      <div className="p-5 flex flex-col gap-3">
        <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
          <div className="h-full rounded-full bg-indigo-500 transition-all duration-500 ease-out" style={{ width: `${progress}%` }} />
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1.5">
          {chunks.map((chunk) => {
            const result = results.find((r) => r.index === chunk.index);
            const status = result?.status ?? "pending";
            const color = status === "done" ? "text-green-600" : status === "processing" ? "text-indigo-600" : status === "error" ? "text-red-500" : "text-slate-400";
            return (
              <div key={chunk.index} className={`inline-flex items-center gap-1 text-xs tabular-nums ${color}`}>
                {status === "pending" && <Clock className="h-3 w-3" />}
                {status === "processing" && <Loader2 className="h-3 w-3 animate-spin" />}
                {status === "done" && <Check className="h-3 w-3" />}
                {status === "error" && <X className="h-3 w-3" />}
                <span>{chunk.startLabel}&ndash;{chunk.endLabel}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
