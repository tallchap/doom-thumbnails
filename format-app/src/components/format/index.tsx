"use client";

import { Play, Square } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { ChunkProgress } from "./chunk-progress";
import { OutputViewer } from "./output-viewer";
import { PromptEditor } from "./prompt-editor";
import { TranscriptInput } from "./transcript-input";
import { chunkTranscript, extractChapterTitlesFormatted, extractSectionHeaders } from "./utils/chunker";
import { buildLinksChatSystemPrompt, buildLinksChatUserMessage, buildLinksSystemPrompt, buildLinksUserMessage, buildSystemPrompt, buildTitlesChatSystemPrompt, buildTitlesChatUserMessage, buildUserMessage, DEFAULT_PROMPT } from "./utils/prompt";
import type { ChunkResult, FormatConfig, ChatMessage, TranscriptChunk } from "./utils/types";

export function FormatTranscript() {
  const [rawTranscript, setRawTranscript] = useState("");
  const [config, setConfig] = useState<FormatConfig>({ speakers: "Liron Shapira, Moshe Vardi", chunkMinutes: 45, prompt: DEFAULT_PROMPT });
  const [chunks, setChunks] = useState<TranscriptChunk[]>([]);
  const [chunkResults, setChunkResults] = useState<ChunkResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [displayOutput, setDisplayOutput] = useState("");
  const [chapterTitles, setChapterTitles] = useState("");
  const [links, setLinks] = useState("");
  const [isChattingTitles, setIsChattingTitles] = useState(false);
  const [isExtractingLinks, setIsExtractingLinks] = useState(false);
  const [finalDocument, setFinalDocument] = useState("");
  const [titlesChatMessages, setTitlesChatMessages] = useState<ChatMessage[]>([]);
  const [titlesChatStreaming, setTitlesChatStreaming] = useState("");
  const [linksChatMessages, setLinksChatMessages] = useState<ChatMessage[]>([]);
  const [linksChatStreaming, setLinksChatStreaming] = useState("");
  const [isChattingLinks, setIsChattingLinks] = useState(false);
  const outputRef = useRef("");
  const abortRef = useRef<AbortController | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const titlesUserEdited = useRef(false);

  useEffect(() => {
    if (isProcessing) {
      intervalRef.current = setInterval(() => setDisplayOutput(outputRef.current), 100);
    } else {
      if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
      setDisplayOutput(outputRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isProcessing]);

  const updateChunkStatus = useCallback((index: number, status: ChunkResult["status"], output?: string) => {
    setChunkResults((prev) => {
      const existing = prev.find((r) => r.index === index);
      if (existing) return prev.map((r) => r.index === index ? { ...r, status, output: output ?? r.output } : r);
      return [...prev, { index, status, output: output ?? "" }];
    });
  }, []);

  const handleChapterTitlesChange = useCallback((titles: string) => {
    titlesUserEdited.current = true;
    setChapterTitles(titles);
  }, []);

  async function processAllChunks() {
    const parsedChunks = chunkTranscript(rawTranscript, config.chunkMinutes);
    if (parsedChunks.length === 0) { toast.error("No timestamps detected in transcript"); return; }
    setIsProcessing(true); setChunks(parsedChunks); setChunkResults([]); outputRef.current = ""; setDisplayOutput(""); setChapterTitles(""); setLinks(""); setFinalDocument(""); setTitlesChatMessages([]); setTitlesChatStreaming(""); setLinksChatMessages([]); setLinksChatStreaming("");
    titlesUserEdited.current = false;
    const abort = new AbortController(); abortRef.current = abort;
    let context: { sectionsCompleted: string[]; lastLines: string } | undefined;
    for (let i = 0; i < parsedChunks.length; i++) {
      if (abort.signal.aborted) break;
      updateChunkStatus(i, "processing");
      try {
        const response = await fetch("/api/format-chunk", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ systemPrompt: buildSystemPrompt(config, i, parsedChunks.length), userMessage: buildUserMessage(parsedChunks[i].rawText, context) }),
          signal: abort.signal,
        });
        if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
        if (!response.body) throw new Error("No response body");
        const reader = response.body.getReader(); const decoder = new TextDecoder(); let chunkOutput = "";
        while (true) { const { done, value } = await reader.read(); if (done) break; const text = decoder.decode(value, { stream: true }); chunkOutput += text; outputRef.current += text; }
        context = { sectionsCompleted: extractSectionHeaders(chunkOutput), lastLines: chunkOutput.slice(-500) };
        updateChunkStatus(i, "done", chunkOutput);
        // Progressive chapter title extraction (only if user hasn't manually edited)
        if (!titlesUserEdited.current) {
          setChapterTitles(extractChapterTitlesFormatted(outputRef.current));
        }
      } catch (err) {
        if (abort.signal.aborted) break;
        updateChunkStatus(i, "error");
        toast.error(`Chunk ${i + 1} failed: ${err instanceof Error ? err.message : "Unknown error"}`);
      }
    }
    if (!abort.signal.aborted) {
      if (!titlesUserEdited.current) {
        setChapterTitles(extractChapterTitlesFormatted(outputRef.current));
      }
      fetchLinks(outputRef.current);
    }
    setIsProcessing(false); abortRef.current = null;
  }

  async function fetchLinks(transcript: string) {
    setIsExtractingLinks(true); setLinks("");
    try {
      const response = await fetch("/api/format-chunk", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ systemPrompt: buildLinksSystemPrompt(), userMessage: buildLinksUserMessage(transcript), useWebSearch: true }),
      });
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let result = "";
      while (true) { const { done, value } = await reader.read(); if (done) break; result += decoder.decode(value, { stream: true }); setLinks(result); }
      setLinks(result);
    } catch (err) {
      toast.error(`Link extraction failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
    setIsExtractingLinks(false);
  }

  async function chatAboutLinks(instruction: string) {
    if (!instruction.trim()) return;
    const userMsg: ChatMessage = { role: "user", content: instruction };
    setLinksChatMessages((prev) => [...prev, userMsg]);
    setIsChattingLinks(true);
    setLinksChatStreaming("");
    try {
      const transcript = outputRef.current || displayOutput;
      const response = await fetch("/api/format-chunk", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          systemPrompt: buildLinksChatSystemPrompt(transcript, rawTranscript),
          userMessage: buildLinksChatUserMessage(linksChatMessages, links, instruction),
          useWebSearch: true,
        }),
      });
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let result = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        result += decoder.decode(value, { stream: true });
        setLinksChatStreaming(result);
      }
      setLinksChatMessages((prev) => [...prev, { role: "assistant", content: result.trim() }]);
      setLinksChatStreaming("");
    } catch (err) {
      toast.error(`Links chat failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
    setIsChattingLinks(false);
  }

  function applyLinksFromChat(content: string) {
    const fenceMatch = content.match(/```links\n([\s\S]*?)```/);
    const linksText = fenceMatch ? fenceMatch[1].trim() : content;
    const lines = linksText.split("\n").filter((l) => /https?:\/\//.test(l));
    if (lines.length === 0) { toast.error("No valid links found in this message"); return; }
    setLinks(lines.join("\n"));
    toast.success("Links applied from chat");
  }

  async function chatAboutTitles(instruction: string) {
    if (!chapterTitles.trim() || !instruction.trim()) return;
    const userMsg: ChatMessage = { role: "user", content: instruction };
    setTitlesChatMessages((prev) => [...prev, userMsg]);
    setIsChattingTitles(true);
    setTitlesChatStreaming("");
    try {
      const transcript = outputRef.current || displayOutput;
      const response = await fetch("/api/format-chunk", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          systemPrompt: buildTitlesChatSystemPrompt(transcript, rawTranscript),
          userMessage: buildTitlesChatUserMessage(titlesChatMessages, chapterTitles, instruction),
        }),
      });
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let result = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        result += decoder.decode(value, { stream: true });
        setTitlesChatStreaming(result);
      }
      setTitlesChatMessages((prev) => [...prev, { role: "assistant", content: result.trim() }]);
      setTitlesChatStreaming("");
    } catch (err) {
      toast.error(`Title chat failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
    setIsChattingTitles(false);
  }

  function applyTitlesFromChat(content: string) {
    // Extract titles from ```titles ... ``` block or from lines matching HH:MM:SS — Title
    const fenceMatch = content.match(/```titles\n([\s\S]*?)```/);
    const titlesText = fenceMatch ? fenceMatch[1].trim() : content;
    const newLines = titlesText.split("\n").filter((l) => /^\d{1,2}:\d{2}:\d{2}\s*[—–-]/.test(l.trim()));
    if (newLines.length === 0) { toast.error("No valid titles found in this message"); return; }
    // Build a map of timestamp -> new title from the chat message
    const newTitlesMap = new Map<string, string>();
    for (const line of newLines) {
      const match = line.trim().match(/^(\d{1,2}:\d{2}:\d{2})\s*[—–-]\s*(.+)$/);
      if (match) {
        const ts = match[1].split(":")[0].length === 1 ? "0" + match[1] : match[1];
        newTitlesMap.set(ts, match[2].trim());
      }
    }
    const existingLines = chapterTitles.split("\n").filter((l) => l.trim());
    // Full replacement if chat provides a complete set (>= existing count), merge if partial
    let result: string[];
    if (newTitlesMap.size >= existingLines.length) {
      // Full replacement — use the new titles as-is
      result = [];
      newTitlesMap.forEach((title, ts) => { result.push(`${ts} — ${title}`); });
    } else {
      // Partial merge — update matching timestamps, keep rest unchanged
      result = existingLines.map((line) => {
        const m = line.trim().match(/^(\d{1,2}:\d{2}:\d{2})\s*[—–-]\s*(.+)$/);
        if (m) {
          const ts = m[1].split(":")[0].length === 1 ? "0" + m[1] : m[1];
          if (newTitlesMap.has(ts)) return `${ts} — ${newTitlesMap.get(ts)}`;
        }
        return line;
      });
      // Append any new timestamps not in existing titles
      const existingTimestamps = new Set(existingLines.map((l) => {
        const m = l.trim().match(/^(\d{1,2}:\d{2}:\d{2})/);
        return m ? (m[1].split(":")[0].length === 1 ? "0" + m[1] : m[1]) : "";
      }));
      newTitlesMap.forEach((title, ts) => {
        if (!existingTimestamps.has(ts)) result.push(`${ts} — ${title}`);
      });
    }
    setChapterTitles(result.join("\n"));
    titlesUserEdited.current = true;
    toast.success(`${newTitlesMap.size} title${newTitlesMap.size > 1 ? "s" : ""} applied from chat`);
  }

  function buildFinalDocument() {
    const currentOutput = outputRef.current || displayOutput;
    if (!currentOutput) return;
    // Apply edited chapter titles to a copy of the transcript
    let transcript = currentOutput;
    if (chapterTitles.trim()) {
      const newTitles = chapterTitles.split("\n").filter((l) => l.trim()).map((l) => { const idx = l.indexOf(" — "); return idx >= 0 ? l.slice(idx + 3).trim() : l.trim(); });
      let titleIdx = 0;
      transcript = transcript.split("\n").map((line) => {
        if (line.startsWith("## ") && titleIdx < newTitles.length) return `## ${newTitles[titleIdx++]}`;
        return line;
      }).join("\n");
    }
    // Assemble final document: Links + Transcript + Footer
    let doc = "";
    if (links.trim()) {
      // Normalize: rejoin URL-only lines back onto previous description line
      const rawLines = links.trim().split("\n").filter((l) => l.trim());
      const normalized: string[] = [];
      for (const line of rawLines) {
        if (/^https?:\/\//.test(line.trim()) && normalized.length > 0) {
          normalized[normalized.length - 1] += line.trim();
        } else {
          normalized.push(line);
        }
      }
      const formattedLinks = normalized.join("\n\n");
      doc += `# Links\n\n${formattedLinks}\n\n`;
    }
    doc += `# Transcript\n\n${transcript}`;
    doc += `\n\n---\n\nDoom Debates\u2019 Mission is to raise mainstream awareness of imminent extinction from AGI and build the social infrastructure for high-quality debate.\n\nSupport the mission by subscribing to my Substack at [DoomDebates.com](https://doomdebates.com) and to [youtube.com/@DoomDebates](https://youtube.com/@DoomDebates), or to really take things to the next level: [Donate](https://doomdebates.com/donate) \uD83D\uDE4F`;
    setFinalDocument(doc);
    toast.success("Final document built");
  }

  function reextractTitles() {
    const current = outputRef.current || displayOutput;
    if (!current) return;
    setChapterTitles(extractChapterTitlesFormatted(current));
    titlesUserEdited.current = false;
  }

  function stopProcessing() { abortRef.current?.abort(); setIsProcessing(false); }

  const canStart = rawTranscript.trim().length > 0 && !isProcessing;
  const hasOutput = (displayOutput || outputRef.current).length > 0;

  return (
    <div className="flex flex-col gap-5">
      <TranscriptInput value={rawTranscript} onChange={setRawTranscript} chunkMinutes={config.chunkMinutes} disabled={isProcessing} />
      <PromptEditor config={config} onChange={setConfig} disabled={isProcessing} />
      <div>
        {!isProcessing ? (
          <button onClick={processAllChunks} disabled={!canStart} className="inline-flex items-center gap-2 rounded-xl bg-indigo-500 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            <Play className="h-4 w-4" /> Format Transcript
          </button>
        ) : (
          <button onClick={stopProcessing} className="inline-flex items-center gap-2 rounded-xl bg-red-500 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-red-600 transition-colors">
            <Square className="h-4 w-4" /> Stop
          </button>
        )}
      </div>
      <ChunkProgress chunks={chunks} results={chunkResults} />
      <OutputViewer
        output={displayOutput}
        isStreaming={isProcessing}
        chapterTitles={chapterTitles}
        onChapterTitlesChange={handleChapterTitlesChange}
        onChatAboutTitles={chatAboutTitles}
        onReextractTitles={reextractTitles}
        isChattingTitles={isChattingTitles}
        titlesChatMessages={titlesChatMessages}
        titlesChatStreaming={titlesChatStreaming}
        onApplyTitlesFromChat={applyTitlesFromChat}
        hasOutput={hasOutput}
        links={links}
        onLinksChange={setLinks}
        onReextractLinks={() => { const t = outputRef.current || displayOutput; if (t) fetchLinks(t); }}
        onChatAboutLinks={chatAboutLinks}
        onApplyLinksFromChat={applyLinksFromChat}
        isExtractingLinks={isExtractingLinks}
        isChattingLinks={isChattingLinks}
        linksChatMessages={linksChatMessages}
        linksChatStreaming={linksChatStreaming}
        finalDocument={finalDocument}
        onBuildFinalDocument={buildFinalDocument}
      />
    </div>
  );
}
