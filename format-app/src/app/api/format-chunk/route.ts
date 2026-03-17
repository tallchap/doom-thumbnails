export const runtime = "nodejs";

import Anthropic from "@anthropic-ai/sdk";
import { NextRequest } from "next/server";

export async function POST(req: NextRequest) {
  console.log("[format-chunk] Request received");
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("[format-chunk] ANTHROPIC_API_KEY not configured");
    return new Response(
      JSON.stringify({ error: "ANTHROPIC_API_KEY not configured" }),
      { status: 503, headers: { "Content-Type": "application/json" } }
    );
  }

  const { systemPrompt, userMessage, useWebSearch } = (await req.json()) as {
    systemPrompt: string;
    userMessage: string;
    useWebSearch?: boolean;
  };
  console.log(`[format-chunk] Chunk size: ${userMessage.length} chars, webSearch: ${!!useWebSearch}`);

  const client = new Anthropic();

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const createParams: Record<string, unknown> = {
    model: "claude-opus-4-6",
    max_tokens: 128000,
    thinking: {
      type: "adaptive",
    },
    output_config: {
      effort: "high",
    },
    system: systemPrompt,
    messages: [{ role: "user", content: userMessage }],
    stream: true,
  };
  if (useWebSearch) {
    createParams.tools = [{ type: "web_search_20250305", name: "web_search", max_uses: 20 }];
  }
  let stream;
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    stream = await (client.messages as any).create(createParams);
    console.log("[format-chunk] Claude stream started");
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Unknown API error";
    console.error(`[format-chunk] Claude API error: ${msg}`);
    return new Response(msg, { status: 500 });
  }

  const readable = new ReadableStream({
    async start(controller) {
      try {
        const enc = new TextEncoder();
        // Send immediate keepalive to establish the streaming connection
        controller.enqueue(enc.encode(" "));
        let lastKeepAlive = Date.now();
        let firstText = true;
        for await (const event of stream) {
          if (
            event.type === "content_block_delta" &&
            event.delta.type === "text_delta"
          ) {
            if (firstText) {
              console.log("[format-chunk] First text_delta received");
              firstText = false;
            }
            controller.enqueue(enc.encode(event.delta.text));
          } else {
            // Send keepalive space every 10s during thinking to prevent Render idle timeout
            const now = Date.now();
            if (now - lastKeepAlive > 10000) {
              controller.enqueue(enc.encode(" "));
              lastKeepAlive = now;
            }
          }
        }
        console.log("[format-chunk] Stream complete");
        controller.close();
      } catch (err) {
        console.error(`[format-chunk] Stream error: ${err instanceof Error ? err.message : err}`);
        controller.error(err);
      }
    },
  });

  return new Response(readable, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "X-Content-Type-Options": "nosniff",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
      "Connection": "keep-alive",
    },
  });
}
