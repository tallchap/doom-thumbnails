"use client";

import { ArrowDownToLine, Check, ClipboardCopy, Download, FileText, MessageSquare, RefreshCw, Send } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { stripChapterTitles } from "./utils/chunker";
import type { ChatMessage } from "./utils/types";

interface OutputViewerProps {
  output: string;
  isStreaming: boolean;
  chapterTitles: string;
  onChapterTitlesChange: (titles: string) => void;
  onChatAboutTitles: (instruction: string) => void;
  onReextractTitles: () => void;
  isChattingTitles: boolean;
  titlesChatMessages: ChatMessage[];
  titlesChatStreaming: string;
  onApplyTitlesFromChat: (content: string) => void;
  hasOutput: boolean;
  links: string;
  onLinksChange: (links: string) => void;
  onReextractLinks: () => void;
  onChatAboutLinks: (instruction: string) => void;
  onApplyLinksFromChat: (content: string) => void;
  isExtractingLinks: boolean;
  isChattingLinks: boolean;
  linksChatMessages: ChatMessage[];
  linksChatStreaming: string;
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

export function OutputViewer({ output, isStreaming, chapterTitles, onChapterTitlesChange, onChatAboutTitles, onReextractTitles, isChattingTitles, titlesChatMessages, titlesChatStreaming, onApplyTitlesFromChat, hasOutput, links, onLinksChange, onReextractLinks, onChatAboutLinks, onApplyLinksFromChat, isExtractingLinks, isChattingLinks, linksChatMessages, linksChatStreaming, finalDocument, onBuildFinalDocument }: OutputViewerProps) {
  const [copiedTranscript, setCopiedTranscript] = useState(false);
  const [copiedChapters, setCopiedChapters] = useState(false);
  const [copiedLinks, setCopiedLinks] = useState(false);
  const [copiedFinal, setCopiedFinal] = useState(false);
  const [titlesChatInput, setTitlesChatInput] = useState("");
  const [linksChatInput, setLinksChatInput] = useState("");
  const strippedOutput = useMemo(() => stripChapterTitles(output), [output]);
  const titlesChatScrollRef = useRef<HTMLDivElement>(null);
  const linksChatScrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chats to bottom on new messages
  useEffect(() => {
    if (titlesChatScrollRef.current) titlesChatScrollRef.current.scrollTop = titlesChatScrollRef.current.scrollHeight;
  }, [titlesChatMessages, titlesChatStreaming]);
  useEffect(() => {
    if (linksChatScrollRef.current) linksChatScrollRef.current.scrollTop = linksChatScrollRef.current.scrollHeight;
  }, [linksChatMessages, linksChatStreaming]);

  function hasApplyableTitles(content: string): boolean {
    if (/```titles\n[\s\S]*?```/.test(content)) return true;
    const lines = content.split("\n").filter((l) => /^\d{1,2}:\d{2}:\d{2}\s*[—–-]/.test(l.trim()));
    return lines.length >= 2;
  }

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

  function handleLinksChat() {
    if (!linksChatInput.trim() || isChattingLinks) return;
    onChatAboutLinks(linksChatInput);
    setLinksChatInput("");
  }

  function hasApplyableLinks(content: string): boolean {
    if (/```links\n[\s\S]*?```/.test(content)) return true;
    const lines = content.split("\n").filter((l) => /https?:\/\//.test(l));
    return lines.length >= 2;
  }

  const modelTag = "claude-opus-4-6 · adaptive thinking · max effort";
  const modelTagSearch = "claude-opus-4-6 · adaptive thinking · max effort · web search";

  return (
    <div className="flex flex-col gap-5">
      {/* Chapter Titles — 3-Pane Layout */}
      {(chapterTitles || hasOutput) && (
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="flex justify-between items-center px-5 py-3 border-b border-slate-100 bg-slate-50/60">
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-sm font-semibold text-slate-800">Chapter Titles</span>
                {isChattingTitles && (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-2.5 py-0.5 text-[11px] font-medium text-indigo-600">
                    <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
                    Thinking
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
          <div className="grid grid-cols-2 divide-x divide-slate-200">
            {/* Left: Document (editable titles) */}
            <div className="p-4">
              <p className="text-[11px] font-medium text-slate-500 mb-2 uppercase tracking-wide">Document</p>
              <textarea
                value={chapterTitles}
                onChange={(e) => onChapterTitlesChange(e.target.value)}
                className="min-h-[12rem] w-full rounded-lg border border-slate-200 bg-slate-50 p-3.5 font-mono text-sm leading-relaxed text-slate-700 resize-y transition-colors placeholder:text-slate-400 disabled:opacity-50 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
                placeholder="Chapter titles will appear here after formatting..."
              />
            </div>
            {/* Right: Chat (messages + prompt) */}
            <div className="flex flex-col">
              <div className="px-4 pt-4 pb-1">
                <p className="text-[11px] font-medium text-slate-500 mb-2 uppercase tracking-wide">Chat</p>
              </div>
              {/* Chat messages */}
              <div ref={titlesChatScrollRef} className="flex-1 overflow-y-auto px-4 space-y-3 min-h-[8rem] max-h-[30vh]">
                {titlesChatMessages.length === 0 && !titlesChatStreaming && (
                  <p className="text-xs text-slate-400 italic py-4">Ask questions, brainstorm titles, or request revisions. The LLM has access to the full transcript.</p>
                )}
                {titlesChatMessages.map((msg, i) => (
                  <div key={i} className={`text-sm leading-relaxed ${msg.role === "user" ? "text-slate-600" : "text-slate-800"}`}>
                    <span className={`text-[10px] font-semibold uppercase tracking-wide ${msg.role === "user" ? "text-slate-400" : "text-indigo-500"}`}>
                      {msg.role === "user" ? "You" : "Assistant"}
                    </span>
                    <div className="mt-0.5 whitespace-pre-wrap font-mono text-xs leading-relaxed">
                      {msg.content}
                    </div>
                    {msg.role === "assistant" && hasApplyableTitles(msg.content) && (
                      <button
                        className="mt-1.5 inline-flex items-center gap-1 rounded-md bg-emerald-50 border border-emerald-200 px-2 py-1 text-[10px] font-medium text-emerald-700 hover:bg-emerald-100 transition-colors"
                        onClick={() => onApplyTitlesFromChat(msg.content)}
                      >
                        <ArrowDownToLine className="h-3 w-3" />
                        Apply these titles
                      </button>
                    )}
                  </div>
                ))}
                {titlesChatStreaming && (
                  <div className="text-sm leading-relaxed text-slate-800">
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-indigo-500">Assistant</span>
                    <div className="mt-0.5 whitespace-pre-wrap font-mono text-xs leading-relaxed">
                      {titlesChatStreaming}
                      <span className="animate-pulse text-indigo-500">|</span>
                    </div>
                  </div>
                )}
              </div>
              {/* Prompt input */}
              <div className="p-3 border-t border-slate-100">
                <div className="flex gap-2">
                  <textarea
                    value={titlesChatInput}
                    onChange={(e) => setTitlesChatInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleTitlesChat(); } }}
                    placeholder="Ask about titles, brainstorm ideas..."
                    className="flex-1 min-h-[2.5rem] max-h-[6rem] rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 transition-colors disabled:opacity-50 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 resize-y"
                    disabled={isChattingTitles || !chapterTitles}
                    rows={2}
                  />
                  <button
                    className="self-end inline-flex items-center justify-center rounded-lg bg-indigo-500 p-2.5 text-white hover:bg-indigo-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    onClick={handleTitlesChat}
                    disabled={isChattingTitles || !titlesChatInput.trim() || !chapterTitles}
                  >
                    <Send className={`h-4 w-4 ${isChattingTitles ? "animate-pulse" : ""}`} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Links — 3-Pane Layout */}
      {(links || isExtractingLinks || hasOutput) && (
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="flex justify-between items-center px-5 py-3 border-b border-slate-100 bg-slate-50/60">
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-sm font-semibold text-slate-800">Links</span>
                {(isExtractingLinks || isChattingLinks) && (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-2.5 py-0.5 text-[11px] font-medium text-indigo-600">
                    <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
                    {isExtractingLinks ? "Extracting" : "Thinking"}
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
          <div className="grid grid-cols-2 divide-x divide-slate-200">
            {/* Left: Document (links list) */}
            <div className="p-4">
              <p className="text-[11px] font-medium text-slate-500 mb-2 uppercase tracking-wide">Document</p>
              {links ? (
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5 space-y-2 resize-y overflow-auto min-h-[8rem] max-h-[40vh]">
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
                <p className="text-sm text-slate-400 py-4">Links will appear here after formatting...</p>
              )}
            </div>
            {/* Right: Chat (messages + prompt) */}
            <div className="flex flex-col">
              <div className="px-4 pt-4 pb-1">
                <p className="text-[11px] font-medium text-slate-500 mb-2 uppercase tracking-wide">Chat</p>
              </div>
              {/* Chat messages */}
              <div ref={linksChatScrollRef} className="flex-1 overflow-y-auto px-4 space-y-3 min-h-[8rem] max-h-[30vh]">
                {linksChatMessages.length === 0 && !linksChatStreaming && (
                  <p className="text-xs text-slate-400 italic py-4">Ask about links, suggest additions, or request changes. The LLM has web search + full transcript access.</p>
                )}
                {linksChatMessages.map((msg, i) => (
                  <div key={i} className={`text-sm leading-relaxed ${msg.role === "user" ? "text-slate-600" : "text-slate-800"}`}>
                    <span className={`text-[10px] font-semibold uppercase tracking-wide ${msg.role === "user" ? "text-slate-400" : "text-indigo-500"}`}>
                      {msg.role === "user" ? "You" : "Assistant"}
                    </span>
                    <div className="mt-0.5 whitespace-pre-wrap font-mono text-xs leading-relaxed">
                      {msg.content}
                    </div>
                    {msg.role === "assistant" && hasApplyableLinks(msg.content) && (
                      <button
                        className="mt-1.5 inline-flex items-center gap-1 rounded-md bg-emerald-50 border border-emerald-200 px-2 py-1 text-[10px] font-medium text-emerald-700 hover:bg-emerald-100 transition-colors"
                        onClick={() => onApplyLinksFromChat(msg.content)}
                      >
                        <ArrowDownToLine className="h-3 w-3" />
                        Apply these links
                      </button>
                    )}
                  </div>
                ))}
                {linksChatStreaming && (
                  <div className="text-sm leading-relaxed text-slate-800">
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-indigo-500">Assistant</span>
                    <div className="mt-0.5 whitespace-pre-wrap font-mono text-xs leading-relaxed">
                      {linksChatStreaming}
                      <span className="animate-pulse text-indigo-500">|</span>
                    </div>
                  </div>
                )}
              </div>
              {/* Prompt input */}
              <div className="p-3 border-t border-slate-100">
                <div className="flex gap-2">
                  <textarea
                    value={linksChatInput}
                    onChange={(e) => setLinksChatInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleLinksChat(); } }}
                    placeholder="Ask about links, suggest additions..."
                    className="flex-1 min-h-[2.5rem] max-h-[6rem] rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 transition-colors disabled:opacity-50 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 resize-y"
                    disabled={isChattingLinks || !hasOutput}
                    rows={2}
                  />
                  <button
                    className="self-end inline-flex items-center justify-center rounded-lg bg-indigo-500 p-2.5 text-white hover:bg-indigo-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    onClick={handleLinksChat}
                    disabled={isChattingLinks || !linksChatInput.trim() || !hasOutput}
                  >
                    <Send className={`h-4 w-4 ${isChattingLinks ? "animate-pulse" : ""}`} />
                  </button>
                </div>
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
