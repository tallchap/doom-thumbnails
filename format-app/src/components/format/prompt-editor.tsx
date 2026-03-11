"use client";

import { ChevronDown, ChevronRight, RotateCcw } from "lucide-react";
import { useState } from "react";
import { DEFAULT_PROMPT } from "./utils/prompt";
import type { FormatConfig } from "./utils/types";

interface PromptEditorProps {
  config: FormatConfig;
  onChange: (config: FormatConfig) => void;
  disabled: boolean;
}

export function PromptEditor({ config, onChange, disabled }: PromptEditorProps) {
  const [showPrompt, setShowPrompt] = useState(false);

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="px-5 py-3 border-b border-slate-100 bg-slate-50/60">
        <span className="text-sm font-semibold text-slate-800">Settings</span>
      </div>
      <div className="p-5 flex flex-col gap-5">
        <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto] gap-5">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="speakers" className="text-xs font-medium text-slate-500 uppercase tracking-wider">Speaker Names</label>
            <input
              id="speakers"
              value={config.speakers}
              onChange={(e) => onChange({ ...config, speakers: e.target.value })}
              placeholder="Speaker One, Speaker Two"
              disabled={disabled}
              className="rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-sm text-slate-800 transition-colors disabled:opacity-50 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
            />
            <span className="text-xs text-slate-400">Comma-separated full names</span>
          </div>
          <div className="flex flex-col gap-1.5">
            <label htmlFor="chunkMinutes" className="text-xs font-medium text-slate-500 uppercase tracking-wider">Chunk Size</label>
            <div className="flex items-center gap-2">
              <input
                id="chunkMinutes"
                type="number"
                min={10}
                max={120}
                value={config.chunkMinutes}
                onChange={(e) => onChange({ ...config, chunkMinutes: parseInt(e.target.value) || 45 })}
                className="w-20 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-center text-slate-800 transition-colors disabled:opacity-50 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
                disabled={disabled}
              />
              <span className="text-sm text-slate-500">min</span>
            </div>
          </div>
        </div>
        <div className="border-t border-slate-100 pt-4">
          <button
            type="button"
            className="flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700 transition-colors uppercase tracking-wider"
            onClick={() => setShowPrompt(!showPrompt)}
          >
            {showPrompt ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            System Prompt
          </button>
          {showPrompt && (
            <div className="mt-3 flex flex-col gap-2">
              <textarea
                value={config.prompt}
                onChange={(e) => onChange({ ...config, prompt: e.target.value })}
                className="min-h-[12rem] w-full rounded-lg border border-slate-200 bg-slate-50 p-3.5 font-mono text-xs leading-relaxed text-slate-700 resize-y transition-colors disabled:opacity-50 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
                disabled={disabled}
              />
              <button
                type="button"
                className="self-end inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 hover:text-slate-800 transition-colors disabled:opacity-50"
                onClick={() => onChange({ ...config, prompt: DEFAULT_PROMPT })}
                disabled={disabled}
              >
                <RotateCcw className="h-3 w-3" />
                Reset to Default
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
