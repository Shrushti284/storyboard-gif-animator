# P44 · Storyboard-to-GIF Animator
Idea → 6-shot storyboard (Gemini) → fixed-seed frames → BLIP consistency check → cross-faded GIF/MP4 with music → "AI-generated" label.

**Live app:** <paste Streamlit URL>

## Run locally
pip install -r requirements.txt, add .streamlit/secrets.toml (see .example), then `streamlit run app.py`.

## Model card
- **LLM:** Gemini (free tier) for storyboard JSON. **Image:** FLUX.1-schnell / Pollinations, fixed seed. **Captioner:** BLIP-base. **Encoder:** MiniLM (BERT-family).
- **Intended use:** educational demo. **Not for:** deceptive media, real people, minors.
- **Results:** (fill in) JSON validity zero vs few-shot; consistency score; latency.
- **Failure case:** (fill in) e.g. character colour drifts between shots despite fixed seed.
- **Bias/safety:** pronoun-count test; keyword guardrail; AI disclosure label on every frame.
- **Licences:** see About tab in the app.
