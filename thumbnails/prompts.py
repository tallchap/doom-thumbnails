"""Prompt templates and brand guidelines for thumbnail generation."""

BRAND_GUIDE = """BRAND — "DOOM DEBATES":
YouTube channel by Liron Shapira about AI existential risk.
Study the attached brand reference thumbnails for VISUAL STYLE ONLY — colors, composition, typography, energy.
CRITICAL: The brand thumbnails contain images of people — COMPLETELY IGNORE all human faces in the brand references. Do NOT reproduce, copy, or be inspired by ANY faces from the brand references. Use ONLY the colors, layout, typography, and visual energy. The ONLY faces allowed in your output are from separately labeled speaker/host photo sections.
Key brand traits:
- Red-toned background gradient or texture as the base
- Large bold headline text in white or yellow, dramatic and punchy
- Guest/host photos with intense, exaggerated expressions (shock, concern, confrontation)
- Composite imagery — people combined with dramatic AI/apocalyptic visuals
- Avoid cliché AI iconography: NO microchips/chipsets/circuit-board "AI" icons or chip-with-AI symbols
- Overall feel: high-stakes, provocative, debate energy
- Color palette: deep reds, blacks, whites, with yellow accents for emphasis"""

IDEA_GENERATION_PROMPT = """You are a YouTube thumbnail strategist for "Doom Debates", a channel about AI existential risk hosted by Liron Shapira.

Generate exactly 10 thumbnail concept ideas for this episode.

EPISODE TITLE: {title}
{custom_prompt_section}
{transcript_section}

Each idea should be a short, vivid visual description (1-2 sentences) that would make a compelling, clickable thumbnail. Include specific imagery, people positioning, and a suggested headline (1-5 words).

{additional_instructions}

Return as a JSON array of exactly 10 strings. Example:
["Doomsday clock at 11:59 with a terminator looming behind Liron, headline: TIMES UP", "Split screen of human brain vs courtroom gavel, both glowing red, headline: WHO DECIDES?"]"""

SEARCH_QUERY_PROMPT = """Given this episode info, suggest 3-5 image search queries to find useful source images for a YouTube thumbnail. Return as a JSON array of strings.

EPISODE TITLE: {title}
{custom_prompt_section}

Focus on: guest headshots, topic-relevant imagery (logos, icons, dramatic visuals), anything that could be composited into a thumbnail.
Example: ["Daniel Kokotajlo headshot", "doomsday clock icon", "AI robot dramatic red lighting"]"""

IDEA_THUMBNAIL_PROMPT = """Generate a YouTube thumbnail image.

MOST IMPORTANT RULE — TEXT: The image must contain ONLY ONE text element. This single text element must be 1-5 words. No subtitles, no secondary text, no labels, no captions, no watermarks. Just ONE short punchy headline. If in doubt, use FEWER words.

THUMBNAIL CONCEPT:
{idea_text}

{custom_prompt_section}

BRAND STYLE:
Apply the Doom Debates brand style. Use the attached brand reference thumbnails for visual style ONLY (colors, composition, typography, energy). Do NOT copy any faces or people from the brand references.

{brand_guide}

{speaker_section}

{liron_section}

{additional_instructions}

RULES:
- 16:9 aspect ratio, photorealistic, sharp focus
- Large expressive faces (40-60% of frame)
- High contrast, clean composition, one focal point
- PEOPLE RULE (CRITICAL): The ONLY human faces/people allowed in this thumbnail are those whose reference photos are attached separately below. Do NOT generate, copy, or include ANY other human faces. The brand reference thumbnails contain people — IGNORE those people entirely, use ONLY the color/layout/typography style.
- Remember: ONLY 1-5 words of text total in the entire image. ONE text element only.
- DO NOT include AI chip/microchip/circuit-board iconography (including a square chip with "AI" text or similar symbols).

You are currently working on Variation #{variation_seed} out of #{variation_total} variations on this concept."""

REVISION_PROMPT = """Revise the following YouTube thumbnail for "Doom Debates" podcast, based on these REVISION INSTRUCTIONS:

{custom_prompt}

You are currently working on Variation #{variation_seed} out of #{variation_total} variations on this concept."""

REVISION_CONTEXT_PROMPT = """When designing the thumbnail keep this in mind:
- Keep the core composition but just apply the requested changes.
- Maintain 16:9 aspect ratio.
- Do NOT introduce AI chip/microchip/circuit-board iconography (including a square chip with "AI" text).
- TEXT FIDELITY (CRITICAL): If the revision instructions include text wrapped in quotes (single or double), preserve that quoted text EXACTLY as written — same words, order, punctuation, apostrophes, and capitalization. Do NOT paraphrase, normalize, or substitute synonyms for quoted text.

Match colors, layout, typography, and energy of the Doom Debates Podcast theme, which are enclosed as separate images, but WARNING: These images contain people — COMPLETELY IGNORE all faces/people in these images. Do NOT reproduce any human likeness from these references."""

VARIATION_PROMPT = """Create a variation of the attached YouTube thumbnail for "Doom Debates" podcast.
Keep the same general composition, mood, and subject, but vary:
- Color treatment and lighting
- Expression intensity
- Background details and atmosphere

The variation should feel like a sibling of the original, not a copy.
Maintain 16:9 aspect ratio. ONLY 1-5 words of text in the entire image — one short headline, nothing else.
Do NOT introduce AI chip/microchip/circuit-board iconography (including a square chip with "AI" text).
{speaker_section}
- The ONLY human faces allowed are from the attached speaker/host photos (if any). Do NOT generate faces from brand references.

You are currently working on Variation #{variation_seed} out of #{variation_total} variations on this concept."""

BORDER_PASS_PROMPT = """Composite the thumbnail image into the attached border-reference template exactly.

The border-reference template is a red dotted/textured frame with a white cutout area in the middle and a "DOOM DEBATES" wordmark badge in the bottom-left corner. Treat it as a literal overlay template, not a style guide.

Instructions:
- Place the thumbnail image INTO the white cutout region of the template, filling it edge-to-edge
- Preserve the red dotted border frame from the template EXACTLY — same color, same dotted texture, same thickness, same proportions
- Preserve the "DOOM DEBATES" wordmark badge from the template EXACTLY — same placement (bottom-left), same rounded black rectangle, same white text, same font weight, same size. Do NOT re-draw, re-render, re-type, or redesign the wordmark. Copy it pixel-for-pixel from the reference.
- Keep ALL existing content of the thumbnail image (text, faces, visuals) completely intact — do not alter, crop, resize, or recompose it
- No white gap or white space between the red frame and the thumbnail content
- Output at 1280x720 resolution"""


# ----- OpenAI gpt-image revision (engine toggle on /revision) -----
#
# The OpenAI path deliberately sends NO brand reference images. `images.edit` takes a flat
# image[] array with a single prompt string — there is no per-image channel to mark an input
# "style only", so every attached image is treated as content to composite. Attaching published
# episode thumbnails made gpt-image transcribe their headlines and faces into unrelated episodes
# (e.g. "GLOBAL WARMING SOLVED!" + an SO2 balloon appearing in every attempt). The brand look is
# conveyed as TEXT below instead. See thumbnail_gen.py commit ea8c7c7 for the same fix on Gemini.
#
# DO NOT add real episode headlines, guest names, or episode subjects to this text — naming them
# in prose reintroduces the exact leak this constant exists to prevent. Style attributes only.

OPENAI_BRAND_STYLE_TEXT = """DOOM DEBATES BRAND STYLE (text description — no style reference images are attached; \
the base image already carries the house look, so preserve it rather than restyling it):
- Background: deep red, from bright crimson to dark maroon, with a fine halftone dot texture and \
often a soft radial gradient that is lighter behind the subjects.
- Accents: flat stylized orange-and-yellow flame shapes rising from the bottom and side edges.
- Headline typography: very heavy condensed sans-serif, ALL CAPS, tight tracking, white with a \
thick dark outline and drop shadow. One key word may be gold/yellow instead of white, or the \
headline may sit on a solid black band. Short — a few words at most, one text element.
- Subjects: photographic people cut out from their backgrounds, large in frame (roughly 40-60% of \
the height), placed left and right, facing camera or turned toward each other, with strong \
intense expressions. Crisp cutout edges, subtle outer glow or shadow separating them from the red.
- Optional emphasis devices in this house style: a hand-drawn yellow curved arrow, a centered prop \
or object inside a thin yellow rectangular frame, and a gold "VS" lockup for head-to-head framings.
- Palette overall: deep reds, black, white, gold/yellow accents, orange flame tones.
- Feel: high-stakes, provocative, confrontational debate energy. Photorealistic faces, sharp focus, \
high contrast, one clear focal point, 16:9."""

OPENAI_REVISION_CONTEXT_PROMPT = """When designing the thumbnail keep this in mind:
- Keep the core composition but just apply the requested changes.
- Maintain 16:9 aspect ratio.
- Do NOT introduce AI chip/microchip/circuit-board iconography (including a square chip with "AI" text).
- TEXT FIDELITY (CRITICAL): If the revision instructions include text wrapped in quotes (single or \
double), preserve that quoted text EXACTLY as written — same words, order, punctuation, \
apostrophes, and capitalization. Do NOT paraphrase, normalize, or substitute synonyms for quoted text."""

OPENAI_NO_INVENTED_CONTENT_RULE = """NOTHING-NEW RULE (CRITICAL): Render no headline, word, number, \
date, name, logo, badge, prop, or graphic element that is not either already visible in Image 1 or \
explicitly requested in the revision instructions above. Do not invent additional text. Do not add \
people who are not in Image 1 or in a user-attached reference image. If the revision instructions \
do not ask for a headline change, reproduce the existing headline from Image 1 verbatim."""
