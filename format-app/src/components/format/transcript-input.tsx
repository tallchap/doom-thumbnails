"use client";

import { Upload } from "lucide-react";
import { useCallback, useState } from "react";
import { chunkTranscript } from "./utils/chunker";

interface TranscriptInputProps {
  value: string;
  onChange: (value: string) => void;
  chunkMinutes: number;
  disabled: boolean;
}

export function TranscriptInput({ value, onChange, chunkMinutes, disabled }: TranscriptInputProps) {
  const [isDragging, setIsDragging] = useState(false);

  const readFile = useCallback(async (file: File) => {
    onChange(await file.text());
  }, [onChange]);

  const handleFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) readFile(file);
  }, [readFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) readFile(file);
  }, [disabled, readFile]);

  const chunks = value.trim() ? chunkTranscript(value, chunkMinutes) : [];
  const showDropZone = !value.trim() && !disabled;

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="flex justify-between items-center px-5 py-3 border-b border-slate-100 bg-slate-50/60">
        <span className="text-sm font-semibold text-slate-800">Raw Transcript</span>
        <label className="cursor-pointer inline-flex items-center gap-1.5 text-xs font-medium text-indigo-600 hover:text-indigo-700 transition-colors">
          <Upload className="h-3.5 w-3.5" />
          Upload file
          <input type="file" accept=".txt,.srt,.vtt" className="hidden" onChange={handleFile} disabled={disabled} />
        </label>
      </div>
      <div
        className="p-4"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {showDropZone && !value ? (
          <div
            className={`w-full min-h-[13rem] rounded-lg border-2 border-dashed flex flex-col items-center justify-center gap-3 transition-colors ${
              isDragging
                ? "border-indigo-400 bg-indigo-50"
                : "border-slate-300 bg-slate-50 hover:border-slate-400"
            }`}
          >
            <Upload className={`h-8 w-8 ${isDragging ? "text-indigo-500" : "text-slate-400"}`} />
            <div className="text-center">
              <p className={`text-sm font-medium ${isDragging ? "text-indigo-600" : "text-slate-600"}`}>
                {isDragging ? "Drop file here" : "Drag & drop a transcript file"}
              </p>
              <p className="text-xs text-slate-400 mt-1">or paste text below</p>
            </div>
            <textarea
              value={value}
              onChange={(e) => onChange(e.target.value)}
              placeholder="Paste your transcript here..."
              className="w-full mt-2 mx-4 flex-1 min-h-[4rem] rounded-lg border border-slate-200 bg-white p-3 text-sm font-mono leading-relaxed resize-y placeholder:text-slate-400 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
            />
          </div>
        ) : (
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Paste your transcript here..."
            className={`w-full min-h-[13rem] rounded-lg border p-3.5 text-sm font-mono leading-relaxed resize-y placeholder:text-slate-400 disabled:opacity-50 transition-colors focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 ${
              isDragging
                ? "border-indigo-400 bg-indigo-50"
                : "border-slate-200 bg-slate-50"
            }`}
            disabled={disabled}
          />
        )}
      </div>
      {(value.length > 0 || chunks.length > 0) && (
        <div className="px-5 py-2.5 border-t border-slate-100 bg-slate-50/40 text-xs text-slate-500 flex gap-4">
          <span>{value.length.toLocaleString()} characters</span>
          {chunks.length > 0 && (
            <span>{chunks.length} chunk{chunks.length !== 1 && "s"} ({chunks[0].startLabel} &ndash; {chunks[chunks.length - 1].endLabel})</span>
          )}
        </div>
      )}
    </div>
  );
}
