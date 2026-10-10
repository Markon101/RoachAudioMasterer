"""
ROACH EARS: Multimodal Audio Reviewer (Layer D)
Sends true PCM audio excerpts to genuine audio-capable multimodal models
via OpenRouter (e.g. google/gemini-2.5-flash) under randomized blind A/B protocol.
"""

import os
import json
import base64
import random
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional


class MultimodalAudioReviewer:
    def __init__(
        self,
        model_name: str = "google/gemini-2.5-flash",
        api_key_path: Optional[str] = None
    ):
        self.model_name = model_name
        if api_key_path is None:
            api_key_path = os.path.expanduser("~/.openrouter_key")
        self.api_key_path = api_key_path
        self._key = None

    def _get_api_key(self) -> Optional[str]:
        if self._key:
            return self._key
        env_key = os.environ.get("OPENROUTER_API_KEY")
        if env_key:
            self._key = env_key.strip()
            return self._key
        if os.path.exists(self.api_key_path):
            with open(self.api_key_path, "r") as f:
                self._key = f.read().strip()
                return self._key
        return None

    def is_available(self) -> bool:
        return self._get_api_key() is not None

    def review_excerpt_pair(
        self,
        ref_wav_path: Path,
        cand_wav_path: Path,
        excerpt_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Conducts a randomized blind comparative listening review between reference and candidate clips.
        """
        key = self._get_api_key()
        if not key:
            return {
                "status": "skipped",
                "reason": "OpenRouter API key not configured. Multimodal listening disabled.",
                "model": self.model_name,
            }

        with open(ref_wav_path, "rb") as f:
            ref_bytes = f.read()
        with open(cand_wav_path, "rb") as f:
            cand_bytes = f.read()

        # Blind randomized presentation: swap Clip 1 and Clip 2 with 50% probability
        is_swapped = random.random() < 0.50
        if is_swapped:
            clip1_bytes, clip2_bytes = cand_bytes, ref_bytes
            clip1_identity, clip2_identity = "candidate", "reference"
        else:
            clip1_bytes, clip2_bytes = ref_bytes, cand_bytes
            clip1_identity, clip2_identity = "reference", "candidate"

        clip1_b64 = base64.b64encode(clip1_bytes).decode("utf-8")
        clip2_b64 = base64.b64encode(clip2_bytes).decode("utf-8")

        prompt = f"""You are an elite mastering acoustician and audio perception evaluator at the Roach Research Institute.
You are performing a rigorous blind A/B listening comparison between two time-aligned audio excerpts: Clip 1 and Clip 2.
Excerpt context: "{excerpt_context.get('selection_reason', 'General section')}".

LISTEN TO BOTH CLIPS CAREFULLY. Answer these explicit acoustic questions with complete precision:

1. Perceived Magnitude of Difference:
   Is the difference between Clip 1 and Clip 2:
   [Significant / Moderate / Subtle / Nearly Indistinguishable / No Reliable Audible Difference]

2. High-Frequency Texture & Grain:
   Which clip exhibits smoother, more natural high-frequency extension vs metallic grain, fizz, or digital harshness? Or are they identical?

3. Transient Attack & Dynamic Impact:
   Which clip has sharper, cleaner transient punch (drums, percussive attacks) without smearing or unnatural pumping? Or are they identical?

4. Bass & Low-Frequency Control:
   Which clip has tighter, more focused low-end resonance while preserving natural sub-bass weight? Or are they identical?

5. Stereo Field & Spatial Coherence:
   Which clip maintains more stable stereo imaging and natural room ambience?

6. Preferred Master & Confidence:
   Which clip do you evaluate as the technically superior and musically preferable master?
   Options: [Clip 1 / Clip 2 / Indistinguishable / Different but Equal]
   Confidence score: [0.0 to 1.0]

7. Crucial Audio Observation & Timestamps:
   Provide specific timestamps (in seconds from clip start) and audible details supporting your judgment. If no difference is heard, explicitly state so.

Format your output as valid JSON matching this schema:
{{
  "magnitude_of_difference": "Significant|Moderate|Subtle|Nearly Indistinguishable|No Reliable Audible Difference",
  "high_frequency_texture": {{"preferred_clip": "Clip 1|Clip 2|Identical", "observation": "..."}},
  "transient_attack": {{"preferred_clip": "Clip 1|Clip 2|Identical", "observation": "..."}},
  "low_frequency_control": {{"preferred_clip": "Clip 1|Clip 2|Identical", "observation": "..."}},
  "stereo_imaging": {{"preferred_clip": "Clip 1|Clip 2|Identical", "observation": "..."}},
  "overall_preferred_clip": "Clip 1|Clip 2|Indistinguishable|Different but Equal",
  "confidence": 0.85,
  "supporting_observation": "...",
  "timestamp_evidence": "..."
}}
Return ONLY the raw JSON object.
"""

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "=== AUDIO EXCERPT: CLIP 1 ==="},
                        {"type": "input_audio", "input_audio": {"data": clip1_b64, "format": "wav"}},
                        {"type": "text", "text": "=== AUDIO EXCERPT: CLIP 2 ==="},
                        {"type": "input_audio", "input_audio": {"data": clip2_b64, "format": "wav"}}
                    ]
                }
            ],
            "temperature": 0.2
        }

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8")
        )
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw_content = data["choices"][0]["message"]["content"].strip()

            # Parse JSON
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()

            parsed = json.loads(raw_content)

            # Invert blind randomization back to Candidate vs Reference
            def unblind(clip_label: str) -> str:
                if clip_label == "Clip 1":
                    return clip1_identity
                elif clip_label == "Clip 2":
                    return clip2_identity
                return clip_label

            unblinded_pref = unblind(parsed.get("overall_preferred_clip", "Indistinguishable"))

            result = {
                "status": "success",
                "model": self.model_name,
                "blind_presentation": {
                    "clip1_was": clip1_identity,
                    "clip2_was": clip2_identity,
                    "was_swapped": is_swapped,
                },
                "magnitude_of_difference": parsed.get("magnitude_of_difference"),
                "overall_verdict": unblinded_pref, # 'candidate', 'reference', 'Indistinguishable', etc.
                "confidence": parsed.get("confidence", 0.5),
                "supporting_observation": parsed.get("supporting_observation"),
                "timestamp_evidence": parsed.get("timestamp_evidence"),
                "dimensions": {
                    "high_frequency_texture": {
                        "winner": unblind(parsed.get("high_frequency_texture", {}).get("preferred_clip", "Identical")),
                        "observation": parsed.get("high_frequency_texture", {}).get("observation", "")
                    },
                    "transient_attack": {
                        "winner": unblind(parsed.get("transient_attack", {}).get("preferred_clip", "Identical")),
                        "observation": parsed.get("transient_attack", {}).get("observation", "")
                    },
                    "low_frequency_control": {
                        "winner": unblind(parsed.get("low_frequency_control", {}).get("preferred_clip", "Identical")),
                        "observation": parsed.get("low_frequency_control", {}).get("observation", "")
                    },
                    "stereo_imaging": {
                        "winner": unblind(parsed.get("stereo_imaging", {}).get("preferred_clip", "Identical")),
                        "observation": parsed.get("stereo_imaging", {}).get("observation", "")
                    },
                }
            }
            return result

        except Exception as e:
            return {
                "status": "error",
                "model": self.model_name,
                "error": str(e),
            }
