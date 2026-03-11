"use client";

import { Play, Square } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { ChunkProgress } from "./chunk-progress";
import { OutputViewer } from "./output-viewer";
import { PromptEditor } from "./prompt-editor";
import { TranscriptInput } from "./transcript-input";
import { chunkTranscript, extractChapterTitlesFormatted, extractSectionHeaders, stripChapterTitles } from "./utils/chunker";
import { buildLinksRevisionSystemPrompt, buildLinksRevisionUserMessage, buildLinksSystemPrompt, buildLinksUserMessage, buildSystemPrompt, buildTitlesChatSystemPrompt, buildTitlesChatUserMessage, buildUserMessage, DEFAULT_PROMPT } from "./utils/prompt";
import type { ChunkResult, FormatConfig, TitlesChatMessage, TranscriptChunk } from "./utils/types";

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
  const [titlesChatMessages, setTitlesChatMessages] = useState<TitlesChatMessage[]>([]);
  const [titlesChatStreaming, setTitlesChatStreaming] = useState("");
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
    setIsProcessing(true); setChunks(parsedChunks); setChunkResults([]); outputRef.current = ""; setDisplayOutput(""); setChapterTitles(""); setLinks(""); setFinalDocument(""); setTitlesChatMessages([]); setTitlesChatStreaming("");
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

  async function reviseLinks(instructions: string) {
    if (!displayOutput && !outputRef.current) return;
    if (!instructions.trim()) return;
    setIsExtractingLinks(true);
    const previousLinks = links;
    setLinks("");
    try {
      const response = await fetch("/api/format-chunk", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ systemPrompt: buildLinksRevisionSystemPrompt(), userMessage: buildLinksRevisionUserMessage(previousLinks, instructions, displayOutput || outputRef.current), useWebSearch: true }),
      });
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let result = "";
      while (true) { const { done, value } = await reader.read(); if (done) break; result += decoder.decode(value, { stream: true }); setLinks(result); }
      setLinks(result);
    } catch (err) {
      toast.error(`Link revision failed: ${err instanceof Error ? err.message : "Unknown error"}`);
      setLinks(previousLinks);
    }
    setIsExtractingLinks(false);
  }

  async function chatAboutTitles(instruction: string) {
    if (!chapterTitles.trim() || !instruction.trim()) return;
    const userMsg: TitlesChatMessage = { role: "user", content: instruction };
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
    const lines = titlesText.split("\n").filter((l) => /^\d{1,2}:\d{2}:\d{2}\s*[—–-]/.test(l.trim()));
    if (lines.length === 0) { toast.error("No valid titles found in this message"); return; }
    setChapterTitles(lines.join("\n"));
    titlesUserEdited.current = true;
    toast.success("Titles applied from chat");
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
    // Assemble final document: Links + Transcript
    let doc = "";
    if (links.trim()) {
      doc += `# Links\n\n${links.trim()}\n\n`;
    }
    doc += transcript;
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
        onReextractLinks={() => { const t = outputRef.current || displayOutput; if (t) fetchLinks(t); }}
        onReviseLinks={(instructions: string) => reviseLinks(instructions)}
        isExtractingLinks={isExtractingLinks}
        finalDocument={finalDocument}
        onBuildFinalDocument={buildFinalDocument}
      />
    </div>
  );
}
