export const runtime = "nodejs";

import Anthropic from "@anthropic-ai/sdk";
import { NextRequest } from "next/server";

export async function POST(req: NextRequest) {
  if (!process.env.ANTHROPIC_API_KEY) {
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
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Unknown API error";
    return new Response(msg, { status: 500 });
  }

  const readable = new ReadableStream({
    async start(controller) {
      try {
        for await (const event of stream) {
          if (
            event.type === "content_block_delta" &&
            event.delta.type === "text_delta"
          ) {
            controller.enqueue(new TextEncoder().encode(event.delta.text));
          }
        }
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });

  return new Response(readable, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
