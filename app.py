"""Streamlit demo: damage an image with a simulated telecom channel, then restore it.

Run locally:   streamlit run app.py
"""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from restoration import channel, pipeline  # noqa: E402
from restoration.infer import load_ai  # noqa: E402

MAX_SIDE = 320  # keeps the CPU-only free hosting responsive

st.set_page_config(page_title="Noisy Channel Image Rescue", page_icon="📡", layout="wide")


@st.cache_resource
def get_ai():
    return load_ai()


@st.cache_data(show_spinner=False)
def cached_demo(clean, kind, level, seed, _ai):
    return pipeline.run_demo(clean, kind, level, seed=seed, ai=_ai)


@st.cache_data(show_spinner=False)
def cached_sweep(clean, kind, seed, _ai):
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    return pipeline.sweep(clean, kind, levels, seed=seed, ai=_ai)


def png_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(pipeline.to_u8(img)).save(buf, format="PNG")
    return buf.getvalue()


ai = get_ai()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Image")
    samples = {p.stem: p for p in pipeline.sample_paths()}
    choice = st.selectbox("Sample image", list(samples)) if samples else None
    upload = st.file_uploader("...or upload your own", type=["png", "jpg", "jpeg"])

    st.header("2. Channel")
    kind = st.selectbox("Damage type", list(channel.CHANNELS),
                        format_func=lambda k: channel.LABELS[k])
    level = st.slider("Severity", 0.0, 1.0, 0.6, 0.05,
                      help="0 = nearly clean link, 1 = very bad link")
    seed = int(st.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1))

if upload is not None:
    clean = pipeline.load_image(upload, MAX_SIDE)
elif choice is not None:
    clean = pipeline.load_image(samples[choice], MAX_SIDE)
else:
    st.error("No sample images found. Upload an image to start.")
    st.stop()

# --------------------------------------------------------------------- main
st.title("📡 Noisy Channel Image Rescue")
st.caption("Send an image through a simulated bad telecom channel, then repair it with "
           "classic OpenCV filters and with a small neural network.")

if ai is None:
    st.info("The AI model (`models/model.onnx`) was not found, so only the classic baseline is "
            "shown. Train the model with `notebooks/train_colab.ipynb` and add the file to enable it.")

tab_demo, tab_curve, tab_about = st.tabs(["Live demo", "Quality curve", "How it works"])

with tab_demo:
    res = cached_demo(clean, kind, level, seed, ai)
    st.subheader(res["label"])

    panels = [("Original", "clean", clean), ("Damaged", "damaged", res["images"]["damaged"]),
              ("Classic OpenCV fix", "classic", res["images"]["classic"])]
    if ai is not None:
        panels.append(("AI fix (CNN)", "ai", res["images"]["ai"]))

    base = res["scores"]["damaged"]["psnr"]
    for col, (title, key, img) in zip(st.columns(len(panels)), panels):
        with col:
            st.markdown(f"**{title}**")
            st.image(pipeline.to_u8(img), use_container_width=True)
            if key != "clean":
                sc = res["scores"][key]
                if key == "damaged":
                    st.metric("PSNR", f"{sc['psnr']:.2f} dB")
                else:
                    st.metric("PSNR", f"{sc['psnr']:.2f} dB", delta=f"{sc['psnr'] - base:+.2f} dB")
                st.caption(f"SSIM {sc['ssim']:.3f}")

    best_key = "ai" if ai is not None else "classic"
    st.download_button("Download the restored image", data=png_bytes(res["images"][best_key]),
                       file_name="restored.png", mime="image/png")

with tab_curve:
    st.write("How does each method hold up as the channel gets worse? "
             "(Computed for the current image and damage type.)")
    if st.button("Compute quality curve"):
        with st.spinner("Running 6 severity levels..."):
            rows = cached_sweep(clean, kind, seed, ai)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {"damaged": "#9aa0a6", "classic": "#e8710a", "ai": "#1a73e8"}
        names = {"damaged": "Damaged", "classic": "Classic (OpenCV)", "ai": "AI (CNN)"}
        for method in ("damaged", "classic", "ai"):
            pts = sorted((r["x"], r["psnr"]) for r in rows if r["method"] == method)
            if pts:
                ax.plot(*zip(*pts), marker="o", lw=2, label=names[method], color=colors[method])
        ax.set_xlabel(rows[0]["xlabel"])
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

with tab_about:
    st.markdown(
        """
**Pipeline:** image → *simulated channel* → damaged image → *restoration* → quality metrics.

**Channel (the telecom part).** Gaussian noise at a chosen SNR (dB), impulse noise, lost 8×8
blocks (packets that never arrived), JPEG compression artifacts, or all of them combined.

**Classic baseline.** The standard tool for each damage type, told what went wrong:
bilateral / non-local-means denoising, median filter, and inpainting of lost blocks. It is
deliberately a strong baseline.

**AI model.** A small residual CNN (about 0.2 M parameters, dilated convolutions for a ~49 px
receptive field) trained once on mixed damage. It is *blind*: it is not told which damage
happened or how severe it is. Exported to ONNX and run on CPU here.

**Metrics.** PSNR (dB) and SSIM versus the original image. Higher is better.
"""
    )
