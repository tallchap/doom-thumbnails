import type { FormatConfig } from "./types";

export const DEFAULT_PROMPT = `When you read the enclosed transcript, you will constantly see speakers' full names being used to label dialogue. Give me a new Markdown transcript of THE ENTIRE RECORDING. The whole thing.

Break the conversation into logical sections; put each section title as an H2 (##). Within each section, in each line of dialogue, preface it with a speaker's FIRST name in bold, followed by the timestamp in italics that's hh:mm:ss format (no milliseconds), then a line break, and the dialogue on the next line. The bold speaker name should be just their first name, EXCEPT the first time a speaker speaks, use their full name in bold.

The transcript should be COMPLETE, and feel like it's an exact transcript, almost verbatim, BUT it should be lightly edited to get rid of unnecessary "ums", "likes", "right?"s, etc. It should read like all the people were speaking very naturally, but clearly and easy to read with no stumbles. Minimize use of "like" and "you know" and minimize unnecessarily repeated words.

Also make minor corrections using your best judgment, for instance "PDUM" in context should be corrected to "P(Doom)".

Each speaker's content should be broken up into short paragraphs, about 4 lines per paragraph max.

Internally plan out the major sections in your thinking before outputting, to make sure you don't skip anything. You MUST start from the very first line of dialogue in the transcript — do not skip any opening content, cold opens, or intro quotes. Every single line of dialogue must appear in your output. But do NOT include a title, section list, or table of contents in your output. Start directly with the first ## section heading and dialogue.`;

export function buildSystemPrompt(config: FormatConfig, chunkIndex: number, totalChunks: number): string {
  const isFirstChunk = chunkIndex === 0;
  const chunkInfo = totalChunks > 1
    ? `You are processing chunk ${chunkIndex + 1} of ${totalChunks} of a long transcript.`
    : "You are processing the full transcript.";
  return `You are a professional transcript editor. ${chunkInfo}

The speakers in this recording are: ${config.speakers}.

${config.prompt}

Additional formatting rules:
${isFirstChunk ? "- Do NOT include an H1 title or section list. Start directly with the first ## section heading." : "- Do NOT output a title heading. Continue from where the previous chunk left off."}
${isFirstChunk ? "- The first time each speaker speaks, bold their FULL name. After that, bold only their FIRST name." : "- All speakers have already been introduced. Use only their FIRST name in bold."}
- Format each dialogue line as:
  **FirstName** *hh:mm:ss*
  Dialogue text here.
- Use two spaces at the end of lines where you need line breaks.
${!isFirstChunk ? "- Do NOT repeat content from previous chunks." : ""}

Output only the formatted Markdown. No preamble, meta-commentary, title, or section list.`;
}

export function buildUserMessage(chunkText: string, context?: { sectionsCompleted: string[]; lastLines: string }): string {
  let message = "";
  if (context && context.sectionsCompleted.length > 0) {
    message += `[Context from previous chunks]\nSections already covered: ${context.sectionsCompleted.join(", ")}\n\nLast formatted lines:\n${context.lastLines}\n\n---\n\n`;
  }
  message += `[Raw transcript${context ? " for this chunk" : ""}]\n${chunkText}`;
  return message;
}

export function buildRevisionSystemPrompt(): string {
  return `You are a professional transcript editor. You will receive a formatted Markdown transcript and a list of revised chapter titles. Your task is to replace all ## section headings in the transcript with the revised titles provided, matching each by its timestamp.

Rules:
- Each revised title is in the format: HH:MM:SS — New Title
- Find the ## heading whose first dialogue timestamp matches (or is closest to) the given timestamp
- Replace ONLY the ## heading text. Keep all dialogue content, formatting, timestamps, and speaker names exactly identical.
- Output the COMPLETE revised transcript. Do not skip or summarize any content.
- Do not add any preamble or commentary. Output only the revised Markdown transcript.`;
}

export function buildRevisionUserMessage(chapterTitles: string, fullTranscript: string): string {
  return `[Revised chapter titles]\n${chapterTitles}\n\n---\n\n[Full transcript to revise]\n${fullTranscript}`;
}

export function buildLinksSystemPrompt(): string {
  return `You are a research assistant with web search capability. You will receive a formatted transcript. Your task is to identify every person, organization, book, paper, website, concept, or resource mentioned in the transcript that has a well-known URL, and produce a list of links.

Use your web search tool to verify URLs and find the most accurate, current links for people, organizations, books, papers, and resources mentioned. Search for each entity to find official pages, Wikipedia articles, Amazon book links, etc.

Rules:
- Output one link per line in the format: Short Description — https://url
- The description should be concise and identify what/who is being linked (e.g. "Moshe Vardi's Wikipedia", "Nick Bostrom, Deep Utopia: Life and Meaning in a Solved World")
- Only include links you have verified via web search or are highly confident are correct
- Include links for: people mentioned (Wikipedia or official pages), organizations, books (Amazon links), papers, websites explicitly mentioned, and other notable references
- Order links by first appearance in the transcript
- Output ONLY the link list, no preamble, headers, or commentary
- Do not include markdown formatting — just plain text lines`;
}

export function buildLinksUserMessage(fullTranscript: string): string {
  return `[Formatted transcript]\n${fullTranscript}`;
}

export function buildLinksRevisionSystemPrompt(): string {
  return `You are a research assistant. You will receive a current list of links extracted from a transcript, along with revision instructions from the user. Apply the requested changes to the links list.

Rules:
- Output one link per line in the format: Short Description — https://url
- Apply the user's revision instructions (e.g. add links, remove links, change descriptions, fix URLs)
- Keep all unchanged links exactly as they are
- Only include links you are confident are correct real URLs
- Order links by first appearance in the transcript
- Output ONLY the revised link list, no preamble, headers, or commentary
- Do not include markdown formatting — just plain text lines`;
}

export function buildLinksRevisionUserMessage(currentLinks: string, instructions: string, fullTranscript: string): string {
  return `[Current links]\n${currentLinks}\n\n[Revision instructions]\n${instructions}\n\n[Transcript for context]\n${fullTranscript}`;
}

export function buildLinksChatSystemPrompt(transcript: string, rawInput: string): string {
  return `You are a research assistant with web search capability helping curate links for a podcast/interview transcript. You have access to the full formatted transcript and original raw input.

You can:
- Suggest new links for people, organizations, books, papers, or resources mentioned
- Verify and fix existing URLs using web search
- Discuss which links are most relevant
- Brainstorm link descriptions
- Answer questions about what's mentioned in the transcript

When suggesting a set of links, format them as:
\`\`\`links
Short Description — https://url
Short Description — https://url
\`\`\`

This lets the user easily apply your suggestions. Use web search to verify URLs are correct.

Be conversational and helpful. You can mix discussion with link suggestions.

[Formatted transcript for reference]
${transcript}

[Original raw input for reference]
${rawInput.slice(0, 10000)}`;
}

export function buildLinksChatUserMessage(chatHistory: { role: "user" | "assistant"; content: string }[], currentLinks: string, newMessage: string): string {
  let msg = `[Current links]\n${currentLinks || "(none yet)"}\n\n`;
  if (chatHistory.length > 0) {
    msg += "[Conversation so far]\n";
    for (const m of chatHistory) {
      msg += `${m.role === "user" ? "User" : "Assistant"}: ${m.content}\n\n`;
    }
  }
  msg += `User: ${newMessage}`;
  return msg;
}

export function buildTitlesChatSystemPrompt(transcript: string, rawInput: string): string {
  return `You are a professional transcript editor helping brainstorm and refine chapter titles for a podcast/interview transcript. You have access to the full formatted transcript and the original raw input for context.

You can:
- Suggest improved titles
- Brainstorm alternatives
- Explain why certain titles work better
- Answer questions about the transcript content
- Discuss section boundaries

When suggesting revised titles, format them as:
\`\`\`titles
HH:MM:SS — Title
HH:MM:SS — Title
\`\`\`

This lets the user easily apply your suggestions. Keep timestamps exactly as they are — only modify title text.

Be conversational and helpful. You can mix discussion with title suggestions.

[Formatted transcript for reference]
${transcript}

[Original raw input for reference]
${rawInput.slice(0, 10000)}`;
}

export function buildTitlesChatUserMessage(chatHistory: { role: "user" | "assistant"; content: string }[], currentTitles: string, newMessage: string): string {
  let msg = `[Current chapter titles]\n${currentTitles}\n\n`;
  if (chatHistory.length > 0) {
    msg += "[Conversation so far]\n";
    for (const m of chatHistory) {
      msg += `${m.role === "user" ? "User" : "Assistant"}: ${m.content}\n\n`;
    }
  }
  msg += `User: ${newMessage}`;
  return msg;
}
