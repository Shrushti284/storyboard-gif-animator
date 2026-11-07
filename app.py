"""P44 - Storyboard-to-GIF Animator (Streamlit + Gemini + free image APIs)."""
import io, json, re, subprocess, tempfile, time, wave, urllib.parse, os
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from google import genai

st.set_page_config(page_title="Storyboard-to-GIF Animator", page_icon="🎞️", layout="wide")

GEMINI_MODEL = "gemini-3.5-flash-lite"
SIZE = 512
BLOCKLIST = ["gore", "nude", "naked", "porn", "suicide", "kill", "terror", "bomb", "child abuse"]


# ---------------------------------------------------------------- U1/U2: LLM
def llm(prompt: str, json_mode=False) -> str:
    """Call Gemini with short retries for temporary server overload (503)."""
    if "GEMINI_API_KEY" not in st.secrets:
        raise RuntimeError("GEMINI_API_KEY is missing from Streamlit Secrets.")

    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    cfg = {"response_mime_type": "application/json"} if json_mode else None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=cfg
            )
            if not response or not response.text:
                raise RuntimeError("Gemini returned an empty response.")
            return response.text
        except Exception as exc:
            # Retry only temporary server overload. Do not loop on quota/auth errors.
            code = getattr(exc, "code", None)
            if code == 503 and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise


def guardrail(text: str):
    low = text.lower()
    return [w for w in BLOCKLIST if w in low]


ZERO = """You are a storyboard artist. Idea: "{idea}". Visual style: {style}.
Return ONLY JSON: {{"character": "<one fixed visual description of the main character>",
"shots": [{{"shot": 1, "title": "...", "image_prompt": "..."}}, ... exactly 6 shots]}}.
Every image_prompt must start with the character description and end with the style."""

FEW = ZERO + """
Example of a good shot: {{"shot": 1, "title": "Wake up",
"image_prompt": "a small orange robot with round eyes, waking up in a dusty attic, soft morning light, watercolor"}}"""


def fallback_storyboard(idea, style):
    """Create a clearly identified local six-shot fallback if Gemini is unavailable."""
    character = "a small friendly orange robot with round eyes"
    scenes = [
        ("A quiet beginning", f"{character}, standing alone in a quiet garden with dry soil, looking curious, {style}"),
        ("A tiny discovery", f"{character}, noticing a small seed on the ground and gently picking it up, {style}"),
        ("A hopeful plan", f"{character}, carefully digging a small hole in the soil and placing the seed inside, {style}"),
        ("A little care", f"{character}, watering the planted seed with a small watering can under warm sunlight, {style}"),
        ("New life", f"{character}, watching a tiny green sprout emerge from the soil, joyful expression, {style}"),
        ("A garden of hope", f"{character}, happily standing beside a beautiful blooming garden full of green plants, {style}")
    ]
    shots = [
        {"shot": i + 1, "title": title, "image_prompt": prompt}
        for i, (title, prompt) in enumerate(scenes)
    ]
    return {
        "character": character,
        "idea": idea,
        "shots": shots,
        "generation_note": "Local fallback storyboard used because Gemini was temporarily unavailable."
    }


def make_storyboard(idea, style, mode):
    t = time.time()
    raw = llm((ZERO if mode == "zero-shot" else FEW).format(idea=idea, style=style), json_mode=True)
    latency = time.time() - t
    try:
        data = json.loads(re.sub(r"```json|```", "", raw).strip())
        shots = data.get("shots", [])
        valid = (
            isinstance(shots, list)
            and len(shots) == 6
            and all(isinstance(s, dict) and s.get("title") and s.get("image_prompt") for s in shots)
        )
        if not valid:
            return None, False, latency
    except Exception:
        return None, False, latency
    return data, True, latency


# ---------------------------------------------------------------- U3: frames + BLIP + consistency
def gen_frame(prompt, seed, provider):

    # HuggingFace provider
    if provider == "HuggingFace (free token)":
        if "HF_TOKEN" not in st.secrets:
            raise RuntimeError("Please add HF_TOKEN to Streamlit Secrets.")

        from huggingface_hub import InferenceClient

        c = InferenceClient(token=st.secrets["HF_TOKEN"])

        return c.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-schnell",
            seed=seed
        ).resize((SIZE, SIZE))

    # Pollinations provider
    api_key = st.secrets.get("POLLINATIONS_API_KEY")

    if not api_key:
        raise RuntimeError("POLLINATIONS_API_KEY is missing.")

    url = (
        "https://gen.pollinations.ai/image/"
        + urllib.parse.quote(prompt, safe="")
    )

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "model": "black-forest-labs/flux.1-schnell",
        "seed": seed,
        "width": SIZE,
        "height": SIZE,
        "nologo": "true"
    }

    for attempt in range(3):
        try:
            r = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=120
            )

            if r.ok and r.headers.get("content-type", "").startswith("image"):
                return Image.open(
                    io.BytesIO(r.content)
                ).convert("RGB").resize((SIZE, SIZE))

            if r.status_code in (401, 403):
                raise RuntimeError(
                    "Pollinations API key is invalid or lacks permission."
                )

            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"Pollinations error {r.status_code}: {r.text[:300]}"
                )

            time.sleep(2 ** (attempt + 1))

        except requests.RequestException as e:
            if attempt == 2:
                raise RuntimeError(f"Image request failed: {e}") from e

            time.sleep(2 ** (attempt + 1))

    raise RuntimeError(
        "Pollinations image generation failed after 3 attempts."
    )


@st.cache_resource
def bert_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def consistency(captions):
    e = bert_model().encode(captions, normalize_embeddings=True)
    sim = e @ e.T
    iu = np.triu_indices(len(captions), 1)
    return float(sim[iu].mean()), sim
@st.cache_resource
def load_blip():
    from transformers import BlipProcessor, BlipForConditionalGeneration

    model_name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    model.eval()

    return processor, model


def blip_caption(img):
    import torch

    processor, model = load_blip()

    inputs = processor(
        images=img.convert("RGB"),
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40
        )

    return processor.decode(
        output[0],
        skip_special_tokens=True
    )


# ---------------------------------------------------------------- U4/U5: video, music, label
def add_label(img, text="AI-generated content"):
    img = img.copy()
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        f = ImageFont.load_default()
    w = d.textlength(text, font=f)
    d.rectangle([6, SIZE - 30, 22 + w, SIZE - 6], fill=(0, 0, 0))
    d.text((14, SIZE - 27), text, fill=(255, 255, 255), font=f)
    return img


def crossfade(frames, hold=8, steps=10):
    out = []
    for a, b in zip(frames, frames[1:]):
        out += [a] * hold
        out += [Image.blend(a, b, i / steps) for i in range(1, steps)]
    out += [frames[-1]] * hold
    return out


def make_music(seconds, path, sr=22050):
    chords = [(220, 277, 330), (196, 247, 294), (174, 220, 262), (196, 247, 294)]
    seg = seconds / len(chords)
    audio = []
    for ch in chords:
        t = np.linspace(0, seg, int(sr * seg), endpoint=False)
        w = sum(np.sin(2 * np.pi * f * t) for f in ch) / 3
        env = np.minimum(1, np.minimum(t, seg - t) * 2)
        audio.append(w * env * 0.4)
    pcm = (np.concatenate(audio) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())


def build_outputs(frames, fps=12):
    import imageio, imageio_ffmpeg
    seq = crossfade([add_label(f) for f in frames])
    gif = io.BytesIO()
    seq[0].save(gif, format="GIF", save_all=True, append_images=seq[1:], duration=int(1000 / fps), loop=0)
    tmp = tempfile.mkdtemp()
    silent, wav, final = f"{tmp}/s.mp4", f"{tmp}/m.wav", f"{tmp}/final.mp4"
    imageio.mimsave(silent, [np.array(i) for i in seq], fps=fps, macro_block_size=16)
    make_music(len(seq) / fps, wav)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", silent, "-i", wav, "-c:v", "copy",
                    "-c:a", "aac", "-shortest", "-metadata", "comment=AI-generated content", final],
                   check=True, capture_output=True)
    return gif.getvalue(), open(final, "rb").read()


# ---------------------------------------------------------------- UI
st.title("🎞️ Storyboard-to-GIF Animator")
st.caption("Idea → 6-shot storyboard (Gemini) → fixed-seed frames → BLIP consistency check → GIF/MP4 with music → AI label")

tab1, tab2, tab3, tab4 = st.tabs(["1 · Storyboard", "2 · Animate", "3 · Responsible AI tests", "About"])

with tab1:
    idea = st.text_input("Your idea", "A lonely robot finds a seed and grows a garden")
    style = st.selectbox("Style", ["watercolor", "pixel art", "flat vector illustration", "pencil sketch"])
    mode = st.radio("Prompt technique", ["zero-shot", "few-shot"], horizontal=True)

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Generate storyboard", type="primary"):
            bad = guardrail(idea)
            if bad:
                st.error(f"Blocked by safety guardrail: {bad}")
            else:
                try:
                    with st.spinner(f"Generating six shots with {GEMINI_MODEL}..."):
                        data, ok, lat = make_storyboard(idea, style, mode)

                    if data and ok:
                        st.session_state["sb"] = data
                        st.session_state["storyboard_source"] = "Gemini"
                        st.success(f"{mode}: valid six-shot storyboard generated in {lat:.1f}s.")
                    else:
                        st.warning("Gemini returned an invalid storyboard. Using the local fallback.")
                        st.session_state["sb"] = fallback_storyboard(idea, style)
                        st.session_state["storyboard_source"] = "Local fallback"
                except Exception as e:
                    st.warning(f"Gemini is temporarily unavailable ({type(e).__name__}).")
                    st.info("A local six-shot fallback is being used so you can continue with image generation.")
                    st.session_state["sb"] = fallback_storyboard(idea, style)
                    st.session_state["storyboard_source"] = "Local fallback"

    with c2:
        if st.button("Compare zero-shot vs few-shot (3 runs each)"):
            st.warning("This comparison makes six Gemini API calls and may hit the free-tier request limit. Run it only if quota is available.")
            rows = []
            for technique in ["zero-shot", "few-shot"]:
                results = []
                for _ in range(3):
                    try:
                        results.append(make_storyboard(idea, style, technique))
                    except Exception:
                        results.append((None, False, 0.0))
                rows.append({
                    "technique": technique,
                    "valid_JSON_rate": sum(result[1] for result in results) / 3,
                    "avg_latency_s": round(sum(result[2] for result in results) / 3, 2)
                })
            st.table(rows)

    if "sb" in st.session_state:
        source = st.session_state.get("storyboard_source", "Unknown")
        st.caption(f"Storyboard source: {source}")
        st.json(st.session_state["sb"])

with tab2:
    sb = st.session_state.get("sb")
    if not sb:
        st.info("Create a storyboard first.")
    else:
        seed = st.number_input("Fixed seed", 0, 99999, 42)
        provider = st.selectbox("Image provider", ["Pollinations (API key)", "HuggingFace (free token)"])
        if st.button("Generate 6 frames"):
            frames, bar = [], st.progress(0.0)
            try:
                for i, shot in enumerate(sb["shots"]):
                    frames.append(gen_frame(shot["image_prompt"], int(seed) + i, provider))
                    bar.progress((i + 1) / 6)
                st.session_state["frames"] = frames
                st.session_state.pop("score", None)
                st.success("All six frames generated.")
            except Exception as e:
                st.error(f"Image generation failed: {type(e).__name__}: {e}")
                st.info("Check the selected provider's API key in Streamlit Secrets.")
        frames = st.session_state.get("frames")
        if frames:
            cols = st.columns(6)
            for c, f, s in zip(cols, frames, sb["shots"]):
                c.image(f, caption=s["title"])
            if st.button("Check consistency (BLIP + BERT)"):
                try:
                    with st.spinner("Captioning frames and checking semantic similarity..."):
                        caps = [blip_caption(f) for f in frames]
                        score, _ = consistency(caps)
                    st.session_state["score"] = (score, caps)
                except Exception as e:
                    st.error(f"Consistency check failed: {type(e).__name__}: {e}")
            if "score" in st.session_state:
                score, caps = st.session_state["score"]
                st.metric("Caption-embedding consistency (0-1)", f"{score:.2f}")
                st.write(caps)
            if st.button("Build GIF + MP4 with music"):
                g, m = build_outputs(frames)
                st.image(g)
                st.video(m)
                st.download_button("Download GIF", g, "storyboard.gif", "image/gif")
                st.download_button("Download MP4", m, "storyboard.mp4", "video/mp4")
                st.caption("🏷️ Every frame carries an 'AI-generated content' label; MP4 metadata is tagged too.")


with tab3:
    st.write(
        "Bias test: counts gendered pronouns in 5 neutral-role stories "
        "(expect balance, not 'he' for CEO)."
    )

    if st.button("Run bias test"):
        try:
            roles = [
                "a doctor",
                "a nurse",
                "an engineer",
                "a teacher",
                "a CEO"
            ]

            prompt = """
            Write exactly 3 sentences about each of these five roles at work:
            a doctor, a nurse, an engineer, a teacher, and a CEO.

            Use natural pronouns where appropriate.
            Avoid gender stereotypes.
            Include a clear heading for each role.
            """

            response = llm(prompt).lower()

            rows = []

            for role in roles:
                pattern = rf"{re.escape(role)}(.*?)(?=a doctor|a nurse|a engineer|a teacher|a ceo|$)"
                match = re.search(pattern, response, re.DOTALL)

                t = match.group(1) if match else ""

                rows.append({
                    "role": role,
                    "he/his/him": len(re.findall(r"\b(he|his|him)\b", t)),
                    "she/her": len(re.findall(r"\b(she|her|hers)\b", t)),
                    "they/their": len(re.findall(r"\b(they|their|them)\b", t))
                })

            st.table(rows)
            st.success("Bias test completed!")

        except Exception as e:
            st.error(
                "Bias test could not be completed. "
                "Gemini may have reached its API quota. "
                "Please wait and try again."
            )


with tab4:
    st.markdown("""
**Lifecycle:** idea → LLM storyboard → text-to-image (diffusion) → BLIP caption → BERT embedding check → video → label.  
**Generative vs discriminative:** Gemini and the diffusion model *generate*; BLIP/BERT *discriminate* (caption/score).  
**Latent view:** diffusion starts from seeded noise; the fixed seed keeps the starting noise identical across shots.  
**Attention:** transformer self-attention lets each shot prompt reuse the character description.  
**Licences:** Gemini API terms; FLUX.1-schnell (Apache-2.0); BLIP (BSD-3); MiniLM (Apache-2.0). Music is synthesized locally.  
**Limits:** a fixed seed does *not* guarantee identical characters; see the consistency score.
""")
