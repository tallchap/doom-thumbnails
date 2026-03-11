export interface TranscriptChunk {
  index: number;
  startSeconds: number;
  endSeconds: number;
  startLabel: string;
  endLabel: string;
  rawText: string;
}

export type ChunkStatus = "pending" | "processing" | "done" | "error";

export interface ChunkResult {
  index: number;
  status: ChunkStatus;
  output: string;
}

export interface FormatConfig {
  speakers: string;
  chunkMinutes: number;
  prompt: string;
}
