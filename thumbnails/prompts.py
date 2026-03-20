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

BORDER_PASS_PROMPT = """Take this thumbnail image and add a "Full Episode" border frame to it, matching the style of the attached border reference images exactly:
- Red border frame around the entire image (same thickness, color, and texture as the references)
- The border goes DIRECTLY around the thumbnail content with NO gap, NO white space, and NO inner border between the red frame and the image content
- "DOOM DEBATES" wordmark/badge — PREFERRED placement is BOTTOM-LEFT corner. Only use a different corner if bottom-left is heavily occluded by a face or critical text. Bottom-right is the LAST resort. Style should match the references, with a black drop shadow or dark glow behind the text to ensure it is clearly visible against any background
- CRITICAL: Keep ALL existing image content (text, faces, visuals) completely intact — only ADD the border frame and wordmark ON TOP of the existing image. Do NOT replace, blank out, or remove any part of the thumbnail content
- Do NOT alter, crop, resize, or recompose the underlying thumbnail in any way
- Do NOT add any white border, white gap, or white space anywhere
- Output at 1280x720 resolution"""
