import type { TranscriptChunk } from "./types";

export function parseTimestampToSeconds(line: string): number | null {
  const bracketMatch = line.match(/^\[(\d+):(\d{2})(?::(\d{2}))?\]/);
  if (bracketMatch) {
    const a = parseInt(bracketMatch[1]);
    const b = parseInt(bracketMatch[2]);
    const c = bracketMatch[3] ? parseInt(bracketMatch[3]) : null;
    return c !== null ? a * 3600 + b * 60 + c : a * 60 + b;
  }
  const decimalMatch = line.match(/^(\d+(?:\.\d+)):/);
  if (decimalMatch) return parseFloat(decimalMatch[1]);
  return null;
}

export function formatSecondsToLabel(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function chunkTranscript(rawText: string, chunkMinutes: number = 45): TranscriptChunk[] {
  const chunkSeconds = chunkMinutes * 60;
  const lines = rawText.split("\n");
  let lastKnownSeconds = 0;
  const annotatedLines = lines.map((text) => {
    const parsed = parseTimestampToSeconds(text);
    if (parsed !== null) lastKnownSeconds = parsed;
    return { text, seconds: lastKnownSeconds };
  });
  if (annotatedLines.length === 0) return [];
  const totalDuration = annotatedLines[annotatedLines.length - 1].seconds;
  const numChunks = Math.max(1, Math.ceil(totalDuration / chunkSeconds));
  const chunks: TranscriptChunk[] = [];
  for (let i = 0; i < numChunks; i++) {
    const targetStart = i * chunkSeconds;
    const targetEnd = (i + 1) * chunkSeconds;
    const chunkLines = i === numChunks - 1
      ? annotatedLines.filter((l) => l.seconds >= targetStart)
      : annotatedLines.filter((l) => l.seconds >= targetStart && l.seconds < targetEnd);
    if (chunkLines.length === 0) continue;
    chunks.push({
      index: i,
      startSeconds: chunkLines[0].seconds,
      endSeconds: chunkLines[chunkLines.length - 1].seconds,
      startLabel: formatSecondsToLabel(chunkLines[0].seconds),
      endLabel: formatSecondsToLabel(chunkLines[chunkLines.length - 1].seconds),
      rawText: chunkLines.map((l) => l.text).join("\n"),
    });
  }
  return chunks;
}

export function extractSectionHeaders(markdown: string): string[] {
  return markdown.split("\n").filter((line) => line.startsWith("## ")).map((line) => line.replace(/^## /, "").trim());
}

export function stripChapterTitles(markdown: string): string {
  const lines = markdown.split("\n");
  const result: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith("## ")) {
      // Skip the heading and any blank line immediately after it
      if (i + 1 < lines.length && lines[i + 1].trim() === "") i++;
      continue;
    }
    result.push(lines[i]);
  }
  // Collapse any resulting triple+ blank lines into double
  return result.join("\n").replace(/\n{3,}/g, "\n\n");
}

export function extractChapterTitlesFormatted(markdown: string): string {
  const lines = markdown.split("\n");
  const chapters: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].startsWith("## ")) continue;
    const title = lines[i].replace(/^## /, "").trim();
    let timestamp = "00:00:00";
    for (let j = i + 1; j < Math.min(i + 20, lines.length); j++) {
      const tsMatch = lines[j].match(/\*(\d{1,2}:\d{2}:\d{2})\*/);
      if (tsMatch) {
        timestamp = tsMatch[1];
        if (timestamp.split(":")[0].length === 1) timestamp = "0" + timestamp;
        break;
      }
    }
    chapters.push(`${timestamp} — ${title}`);
  }
  return chapters.join("\n");
}
