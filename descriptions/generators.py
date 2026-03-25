"""Multi-model YouTube description generation (Gemini, Claude, GPT)."""

import json
import sys
import time
import requests

print(f"[LOAD] descriptions/generators.py loaded from {__file__}", file=sys.stderr, flush=True)

from config import (
    DESCRIPTION_MODEL, ANTHROPIC_API_KEY, CLAUDE_DESCRIPTION_MODEL,
    OPENAI_API_KEY, GPT_DESCRIPTION_MODEL,
)
from shared.helpers import _record_api_call

EXISTING_DESCRIPTIONS_TONE_REFERENCE = """EXISTING DESCRIPTIONS AND TITLES FOR TONE REFERENCE
TITLE: Episode 133 — Elon Musk's Insane Plan for Surviving AI Takeover (Feb 14, 2026)
DESCRIPTION: Elon Musk just made a stunning admission about the insane future he's steering us toward. In a new interview with Dwarkesh Patel and John Collison on the Cheeky Pint podcast, Elon said that humanity can't expect to be "in charge" of AI for long, because humans will soon only have 1% of the combined total human+AI intelligence. Then, he claimed to have a plan to build AI overlords that will naturally support humanity's flourishing. In this mini episode, I react to Elon's remarks and expose why his plan for humanity's survival in the age of AI is dangerously flimsy.
TITLE: Episode 132 — The Only Politician Thinking Clearly About Superintelligence — California Governor Candidate Zoltan Istvan (Feb 13, 2026)
DESCRIPTION: California gubernatorial candidate Zoltan Istvan reveals his P(Doom) and makes the case for universal basic income and radical life extension.
TITLE: Episode 131 — His P(Doom) Is Only 2.6% — AI Doom Debate with Bentham's Bulldog, a.k.a. Matthew Adelstein (Feb 10, 2026)
DESCRIPTION: Get ready for a rematch with the one & only Bentham's Bulldog, a.k.a. Matthew Adelstein! Our first debate covered a wide range of philosophical topics. Today's Debate #2 is all about Matthew's new argument against the inevitability of AI doom. He comes out swinging with a calculated P(Doom) of just 2.6%, based on a multi-step probability chain that I challenge as potentially falling into a "Type 2 Conjunction Fallacy" (a.k.a. Multiple Stage Fallacy). We clash on whether to expect "alignment by default" and the nature of future AI architectures. While Matthew sees current RLHF success as evidence that AIs will likely remain compliant, I argue that we're building "Goal Engines" — superhuman optimization modules that act like nuclear cores wrapped in friendly personalities. We debate whether these engines can be safely contained, or if the capability to map goals to actions is inherently dangerous and prone to exfiltration. Despite our different forecasts (my 50% vs his sub-10%), we actually land in the "sane zone" together on some key policy ideas, like the potential necessity of a global pause. While Matthew's case for low P(Doom) hasn't convinced me, I consider his post and his engagement with me to be super high quality and good faith. We're not here to score points, we just want to better predict how the intelligence explosion will play out.
TITLE: Episode 130 — What Dario Amodei Misses In "The Adolescence of Technology" — Reaction With MIRI's Harlan Stewart (Feb 4, 2026)
DESCRIPTION: Harlan Stewart works in communications for the Machine Intelligence Research Institute (MIRI). In this episode, Harlan and I give our honest opinions on Dario Amodei's new essay "The Adolescence of Technology".
TITLE: Episode 129 — Q&A: Is Liron too DISMISSIVE of AI Harms? + New Studio, Demis Would #PauseAI, AI Water Use Debate (Jan 27, 2026)
DESCRIPTION: Check out the new Doom Debates studio in this Q&A with special guest Producer Ori! Liron gets into a heated discussion about whether doomers must validate short-term risks, like data center water usage, in order to build a successful political coalition. Originally streamed on Saturday, January 24.
TITLE: Episode 128 — Taiwan's Cyber Ambassador-At-Large Says Humans & AI Can FOOM Together (Jan 20, 2026)
DESCRIPTION: Audrey Tang was the youngest minister in Taiwanese history. Now she's working to align AI with democratic principles as Taiwan's Cyber Ambassador. In this debate, I probe her P(doom) and stress-test her vision for safe AI development."""

DESCRIPTION_ARCHIVE_PROMPT = """YouTube Description Revision Prompt
You are revising ONE YouTube description for an AI-focused channel.

Goal:
- Produce exactly ONE revised description per run.
- Prioritize the HOW TO REVISE instructions over everything else.
- Keep the revised result faithful to transcript facts.

Output format (strict):
Description:
[one full revised description only]

RULES
- Accuracy first: do not invent facts; represent transcript faithfully.
- Voice & vibe: match existing channel samples — dramatic, high-contrast, slightly irreverent.
- If transcript does NOT cover a point, do not promise it.
- Front-load searchable keywords naturally; no keyword stuffing.
- Keep description between 800-1,500 characters unless content strongly warrants longer.
- Return ONLY one revised description, not multiple candidates."""


def _build_description_prompt(title, primary_description, revise_instructions, transcript, channel_samples):
    merged_samples = EXISTING_DESCRIPTIONS_TONE_REFERENCE
    if channel_samples:
        merged_samples = f"{merged_samples}\n\nADDITIONAL USER-PROVIDED CHANNEL SAMPLES:\n{channel_samples}"

    title_block = f"EPISODE TITLE:\n{title}\n\n" if title else ""
    return (
        f"{DESCRIPTION_ARCHIVE_PROMPT}\n\n"
        f"{title_block}"
        f"HOW TO REVISE (highest priority):\n{revise_instructions}\n\n"
        f"PRIMARY DESCRIPTION (existing draft):\n{primary_description}\n\n"
        f"FULL VIDEO TRANSCRIPT:\n{transcript}\n\n"
        f"EXISTING CHANNEL DESCRIPTION SAMPLES:\n{merged_samples}\n"
    )


def generate_description_gemini(client, prompt):
    _record_api_call(DESCRIPTION_MODEL, prompt, phase="description_generation_gemini")
    response = client.models.generate_content(
        model=DESCRIPTION_MODEL,
        contents=prompt,
        config={"http_options": {"timeout": 120_000}},
    )
    return (response.text or "").strip()


def generate_description_claude(prompt):
    if not ANTHROPIC_API_KEY:
        return "[Claude unavailable: ANTHROPIC_API_KEY not set]"
    _record_api_call(CLAUDE_DESCRIPTION_MODEL, prompt, phase="description_generation_claude")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_DESCRIPTION_MODEL,
            "max_tokens": 1800,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = data.get("content", [])
    text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return text.strip()


def _extract_gpt_text(data):
    """Extract text from an OpenAI Responses API result.

    Checks output_text (sync responses) then falls back to
    output[].content[].text (background responses).
    """
    text = data.get("output_text") or ""
    if text:
        return text.strip()
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text" and part.get("text"):
                    return part["text"].strip()
    return ""


def generate_description_gpt(prompt):
    if not OPENAI_API_KEY:
        return "[GPT unavailable: OPENAI_API_KEY not set]"
    # CANARY: if this string appears in output, our code is running
    import os
    if os.environ.get("GPT_CANARY_TEST") == "1":
        return "[GPT CANARY OK — generators.py v2 is loaded]"
    _record_api_call(GPT_DESCRIPTION_MODEL, prompt, phase="description_generation_gpt")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GPT_DESCRIPTION_MODEL,
        "input": prompt,
        "background": True,
    }
    # Create a background response (returns immediately with an ID)
    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=body,
            timeout=(15, 30),
        )
    except requests.exceptions.ReadTimeout:
        return "[GPT error] Background POST timed out at (15,30) — this should not happen with background=True"
    except requests.exceptions.ConnectTimeout:
        return "[GPT error] Could not connect to api.openai.com within 15s"
    resp.raise_for_status()
    data = resp.json()
    resp_status = data.get("status", "unknown")
    resp_id = data.get("id", "none")
    # If it completed immediately, return the text
    if resp_status == "completed":
        text = _extract_gpt_text(data)
        return text if text else json.dumps(data)[:4000]
    # Poll until completed
    if not resp_id or resp_id == "none":
        return json.dumps(data)[:4000]
    poll_url = f"https://api.openai.com/v1/responses/{resp_id}"
    poll_count = 0
    while True:
        time.sleep(10)
        poll_count += 1
        poll_resp = requests.get(poll_url, headers=headers, timeout=30)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        poll_status = poll_data.get("status", "")
        print(f"[GPT] Poll #{poll_count} status={poll_status}", file=sys.stderr, flush=True)
        if poll_status == "completed":
            text = _extract_gpt_text(poll_data)
            return text if text else json.dumps(poll_data)[:4000]
        elif poll_status in ("failed", "cancelled", "incomplete"):
            error = poll_data.get("error", {})
            return f"[GPT {poll_status}] {json.dumps(error)[:300]}"
        # Still in_progress or queued — keep polling
