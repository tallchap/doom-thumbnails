"use client";

import { Check, ClipboardCopy, Download, FileText, MessageSquare, RefreshCw } from "lucide-react";
import { useMemo, useState } from "react";
import { stripChapterTitles } from "./utils/chunker";

interface OutputViewerProps {
  output: string;
  isStreaming: boolean;
  chapterTitles: string;
  onChapterTitlesChange: (titles: string) => void;
  onChatAboutTitles: (instruction: string) => void;
  onReextractTitles: () => void;
  isChattingTitles: boolean;
  hasOutput: boolean;
  links: string;
  onReextractLinks: () => void;
  onReviseLinks: (instructions: string) => void;
  isExtractingLinks: boolean;
  finalDocument: string;
  onBuildFinalDocument: () => void;
}

function SmallBtn({ onClick, disabled, children }: { onClick: () => void; disabled?: boolean; children: React.ReactNode }) {
  return (
    <button
      className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 bg-white hover:bg-slate-50 hover:text-slate-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

function parseLinks(raw: string): { description: string; url: string }[] {
  return raw.split("\n").filter((l) => l.trim()).map((line) => {
    const match = line.match(/^(.+?)\s*[—–-]\s*(https?:\/\/\S+)$/);
    if (match) return { description: match[1].trim(), url: match[2].trim().replace(/\.+$/, "") };
    const urlOnly = line.match(/(https?:\/\/\S+)/);
    if (urlOnly) return { description: line.replace(urlOnly[0], "").replace(/[—–-]\s*$/, "").trim() || urlOnly[0], url: urlOnly[0].replace(/\.+$/, "") };
    return null;
  }).filter(Boolean) as { description: string; url: string }[];
}

export function OutputViewer({ output, isStreaming, chapterTitles, onChapterTitlesChange, onChatAboutTitles, onReextractTitles, isChattingTitles, hasOutput, links, onReextractLinks, onReviseLinks, isExtractingLinks, finalDocument, onBuildFinalDocument }: OutputViewerProps) {
  const [copiedTranscript, setCopiedTranscript] = useState(false);
  const [copiedChapters, setCopiedChapters] = useState(false);
  const [copiedLinks, setCopiedLinks] = useState(false);
  const [copiedFinal, setCopiedFinal] = useState(false);
  const [titlesChatInput, setTitlesChatInput] = useState("");
  const [linksRevisionInput, setLinksRevisionInput] = useState("");
  const strippedOutput = useMemo(() => stripChapterTitles(output), [output]);

  if (!output && !isStreaming && !hasOutput) return null;

  async function copyTranscript() {
    await navigator.clipboard.writeText(strippedOutput);
    setCopiedTranscript(true);
    setTimeout(() => setCopiedTranscript(false), 2000);
  }
  async function copyChapters() {
    await navigator.clipboard.writeText(chapterTitles);
    setCopiedChapters(true);
    setTimeout(() => setCopiedChapters(false), 2000);
  }
  async function copyLinks() {
    await navigator.clipboard.writeText(links);
    setCopiedLinks(true);
    setTimeout(() => setCopiedLinks(false), 2000);
  }
  function downloadMarkdown() {
    const blob = new Blob([strippedOutput], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "transcript.md";
    a.click();
    URL.revokeObjectURL(url);
  }
  async function copyFinal() {
    await navigator.clipboard.writeText(finalDocument);
    setCopiedFinal(true);
    setTimeout(() => setCopiedFinal(false), 2000);
  }
  function downloadFinal() {
    const blob = new Blob([finalDocument], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "final-document.md";
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleTitlesChat() {
    if (!titlesChatInput.trim() || isChattingTitles) return;
    onChatAboutTitles(titlesChatInput);
    setTitlesChatInput("");
  }

  function handleLinksRevise() {
    if (!linksRevisionInput.trim() || isExtractingLinks) return;
    onReviseLinks(linksRevisionInput);
    setLinksRevisionInput("");
  }

  const modelTag = "claude-opus-4-6 · adaptive thinking · max effort";
  const modelTagSearch = "claude-opus-4-6 · adaptive thinking · max effort · web search";

  return (
    <div className="flex flex-col gap-5">
      {/* Chapter Titles Pane */}
      {(chapterTitles || hasOutput) && (
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="flex justify-between items-center px-5 py-3 border-b border-slate-100 bg-slate-50/60">
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-sm font-semibold text-slate-800">Chapter Titles</span>
                {isChattingTitles && (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-2.5 py-0.5 text-[11px] font-medium text-indigo-600">
                    <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
                    Iterating
                  </span>
                )}
              </div>
              <p className="text-[10px] text-slate-400 mt-0.5">{modelTag}</p>
            </div>
            <div className="flex gap-2">
              <SmallBtn onClick={copyChapters} disabled={!chapterTitles}>
                {copiedChapters ? <Check className="h-3.5 w-3.5" /> : <ClipboardCopy className="h-3.5 w-3.5" />}
                {copiedChapters ? "Copied" : "Copy"}
              </SmallBtn>
              <SmallBtn onClick={onReextractTitles} disabled={!hasOutput}>
                <RefreshCw className="h-3.5 w-3.5" />
                Re-extract
              </SmallBtn>
            </div>
          </div>
          <div className="p-4">
            <textarea
              value={chapterTitles}
              onChange={(e) => onChapterTitlesChange(e.target.value)}
              className="min-h-[7rem] w-full rounded-lg border border-slate-200 bg-slate-50 p-3.5 font-mono text-sm leading-relaxed text-slate-700 resize-y transition-colors placeholder:text-slate-400 disabled:opacity-50 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
              placeholder="Chapter titles will appear here after formatting..."
              disabled={isChattingTitles}
            />
            <div className="mt-3 flex flex-col gap-2">
              <textarea
                value={titlesChatInput}
                onChange={(e) => setTitlesChatInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleTitlesChat(); } }}
                placeholder='e.g. "Make titles more descriptive" or "Combine the first two sections"'
                className="w-full min-h-[3.5rem] rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-sm text-slate-800 placeholder:text-slate-400 transition-colors disabled:opacity-50 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 resize-y"
                disabled={isChattingTitles || !chapterTitles}
                rows={2}
              />
              <div className="flex justify-end">
                <button
                  className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-500 px-3.5 py-2 text-xs font-medium text-white hover:bg-indigo-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  onClick={handleTitlesChat}
                  disabled={isChattingTitles || !titlesChatInput.trim() || !chapterTitles}
                >
                  <MessageSquare className={`h-3.5 w-3.5 ${isChattingTitles ? "animate-pulse" : ""}`} />
                  {isChattingTitles ? "Iterating..." : "Iterate with LLM"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Links Pane */}
      {(links || isExtractingLinks || hasOutput) && (
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="flex justify-between items-center px-5 py-3 border-b border-slate-100 bg-slate-50/60">
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-sm font-semibold text-slate-800">Links</span>
                {isExtractingLinks && (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-2.5 py-0.5 text-[11px] font-medium text-indigo-600">
                    <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
                    Extracting
                  </span>
                )}
              </div>
              <p className="text-[10px] text-slate-400 mt-0.5">{modelTagSearch}</p>
            </div>
            <div className="flex gap-2">
              <SmallBtn onClick={copyLinks} disabled={!links}>
                {copiedLinks ? <Check className="h-3.5 w-3.5" /> : <ClipboardCopy className="h-3.5 w-3.5" />}
                {copiedLinks ? "Copied" : "Copy"}
              </SmallBtn>
              <SmallBtn onClick={onReextractLinks} disabled={isExtractingLinks || !hasOutput}>
                <RefreshCw className={`h-3.5 w-3.5 ${isExtractingLinks ? "animate-spin" : ""}`} />
                Re-extract
              </SmallBtn>
            </div>
          </div>
          <div className="p-4">
            {links ? (
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5 space-y-2 resize-y overflow-auto min-h-[5rem] max-h-[50vh]">
                {parseLinks(links).map((link, i) => (
                  <div key={i} className="text-sm leading-relaxed">
                    <span className="text-slate-700">{link.description}</span>
                    {" — "}
                    <a href={link.url} target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:text-indigo-800 underline underline-offset-2 break-all">
                      {link.url}
                    </a>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-400">Links will appear here after formatting...</p>
            )}
            <div className="mt-3 flex flex-col gap-2">
              <textarea
                value={linksRevisionInput}
                onChange={(e) => setLinksRevisionInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleLinksRevise(); } }}
                placeholder='e.g. "Add a link for OpenAI" or "Remove the Amazon links"'
                className="w-full min-h-[3.5rem] rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-sm text-slate-800 placeholder:text-slate-400 transition-colors disabled:opacity-50 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 resize-y"
                disabled={isExtractingLinks || !links}
                rows={2}
              />
              <div className="flex justify-end">
                <button
                  className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-500 px-3.5 py-2 text-xs font-medium text-white hover:bg-indigo-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  onClick={handleLinksRevise}
                  disabled={isExtractingLinks || !linksRevisionInput.trim() || !links}
                >
                  <RefreshCw className={`h-3.5 w-3.5 ${isExtractingLinks ? "animate-spin" : ""}`} />
                  Revise Links
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Full Transcript Pane */}
      {(output || isStreaming) && (
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="flex justify-between items-center px-5 py-3 border-b border-slate-100 bg-slate-50/60">
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-sm font-semibold text-slate-800">Full Transcript</span>
                {isStreaming && (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-2.5 py-0.5 text-[11px] font-medium text-indigo-600">
                    <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
                    Streaming
                  </span>
                )}
              </div>
              <p className="text-[10px] text-slate-400 mt-0.5">{modelTag}</p>
            </div>
            <div className="flex gap-2">
              <SmallBtn onClick={copyTranscript} disabled={!output}>
                {copiedTranscript ? <Check className="h-3.5 w-3.5" /> : <ClipboardCopy className="h-3.5 w-3.5" />}
                {copiedTranscript ? "Copied" : "Copy"}
              </SmallBtn>
              <SmallBtn onClick={downloadMarkdown} disabled={!output}>
                <Download className="h-3.5 w-3.5" />
                Download
              </SmallBtn>
            </div>
          </div>
          <div className="p-4">
            <pre className="whitespace-pre-wrap text-sm font-mono leading-relaxed bg-slate-50 rounded-lg p-4 max-h-[70vh] overflow-y-auto border border-slate-200 text-slate-800">
              {strippedOutput}
              {isStreaming && <span className="animate-pulse text-indigo-500">|</span>}
            </pre>
          </div>
        </div>
      )}
      {/* Final Document Pane */}
      {hasOutput && (
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="flex justify-between items-center px-5 py-3 border-b border-slate-100 bg-slate-50/60">
            <div>
              <span className="text-sm font-semibold text-slate-800">Final Document</span>
              <p className="text-[10px] text-slate-400 mt-0.5">assembled from above components</p>
            </div>
            <div className="flex gap-2">
              <SmallBtn onClick={copyFinal} disabled={!finalDocument}>
                {copiedFinal ? <Check className="h-3.5 w-3.5" /> : <ClipboardCopy className="h-3.5 w-3.5" />}
                {copiedFinal ? "Copied" : "Copy"}
              </SmallBtn>
              <SmallBtn onClick={downloadFinal} disabled={!finalDocument}>
                <Download className="h-3.5 w-3.5" />
                Download
              </SmallBtn>
              <button
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-emerald-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={onBuildFinalDocument}
                disabled={!hasOutput}
              >
                <FileText className="h-3.5 w-3.5" />
                {finalDocument ? "Rebuild" : "Build Final Document"}
              </button>
            </div>
          </div>
          <div className="p-4">
            {finalDocument ? (
              <pre className="whitespace-pre-wrap text-sm font-mono leading-relaxed bg-slate-50 rounded-lg p-4 max-h-[70vh] overflow-y-auto border border-slate-200 text-slate-800">
                {finalDocument}
              </pre>
            ) : (
              <p className="text-sm text-slate-400">Click &ldquo;Build Final Document&rdquo; to assemble links + transcript with chapter titles.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
