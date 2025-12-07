# app.py
from __future__ import annotations
import os
import json
import base64
import time
from typing import List

import gradio as gr
import numpy as np
import torch

from core.runtime import (
    analyze_video_basic,
    get_multitask_pred_with_attribution,
)

from core.attribution import (
    plot_emotion_probs_barchart,             # Bar chart of emotion probabilities
    compute_contributions_from_result,       # Contributions by (modality - type)
    format_contribution_summary,             # Text explanation of contributions
    visualize_all_task_heatmaps,             # All tasks combined into one strip
)

from core.media_utils import extract_keyframes_from_result

DEFAULT_CHECKPOINT = "best_ep9_emo0.6390_pkl0.8269.pt"

EMO_ORDER = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
EMO_ORDER_FOR_BARS = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
# OCEAN order
PERS_ORDER = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Non-Neuroticism"]

PERS_COLORS = {
    "Extraversion": "#F97316",
    "Agreeableness": "#22C55E",
    "Conscientiousness": "#0EA5E9",
    "Non-Neuroticism": "#8B5CF6",
    "Openness": "#EAB308",
}


# -------------------- Device choice --------------------
def choose_device(requested: str | None) -> str:
    if requested in ("cuda", "cpu"):
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- Helpers --------------------
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def render_gallery_html(paths, captions, uid: str = "g", thumb_h: int = 150) -> str:
    def _img_to_data_uri(path: str) -> str:
        ext = (os.path.splitext(path)[1].lower().lstrip(".") or "jpeg")
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/{ext};base64,{b64}"

    tiles = []
    for i, (p, cap) in enumerate(zip(paths, captions)):
        try:
            src = _img_to_data_uri(p)
        except Exception:
            src = ""
        fig_id = f"{uid}_img{i}"
        tiles.append(f"""
        <figure style="margin:0 12px 0 0; display:inline-block; text-align:center; vertical-align:top;">
          <a href="#{fig_id}" style="display:inline-block;">
            <img src="{src}" style="height:{thumb_h}px; width:auto; border-radius:12px; cursor:pointer; background:#000; box-shadow:0 2px 10px rgba(0,0,0,.35);" />
          </a>
          <figcaption style="font-size:11px; color:#ddd; opacity:0.9; margin-top:6px;">{cap}</figcaption>
        </figure>

        <!-- Lightbox -->
        <div id="{fig_id}" class="lightbox">
          <a href="#" class="lightbox-close"></a>
          <img src="{src}" class="lightbox-content" />
        </div>
        """)

    strip_id = f"{uid}_strip"
    style = f"""
    <style>
    /* Strip вЂ” single row */
    #{strip_id} {{
        white-space: nowrap;
        overflow-x: auto; overflow-y: hidden;
        padding: 6px 2px 10px 2px;
        border-radius: 10px;
        scrollbar-width: thin;
    }}
    /* (optionally hide scrollbar)
       #{strip_id}::-webkit-scrollbar {{ height: 0; }} */

    #{strip_id}::-webkit-scrollbar {{ height: 8px; }}
    #{strip_id}::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,.2); border-radius: 6px; }}

    /* Lightbox */
    .lightbox {{
        display:none; position:fixed; inset:0; z-index:9999;
        background:rgba(0,0,0,0.85);
        justify-content:center; align-items:center;
    }}
    .lightbox:target {{ display:flex; }}
    .lightbox-content {{
        max-width:90%; max-height:90%;
        border-radius:12px; box-shadow:0 0 24px rgba(0,0,0,.55);
        animation:zoomIn .2s ease;
    }}
    @keyframes zoomIn {{ from{{transform:scale(.9); opacity:.7;}} to{{transform:scale(1); opacity:1;}} }}
    .lightbox-close {{ position:absolute; inset:0; }}
    </style>
    """

    # JS: vertical wheel в†’ horizontal scroll
    script = f"""
    <script>
    (function(){{
      const el = document.getElementById("{strip_id}");
      if (!el) return;
      el.addEventListener('wheel', function(e){{
        if (e.deltaY === 0) return;
        e.preventDefault();
        el.scrollLeft += e.deltaY;
      }}, {{passive:false}});
    }})();
    </script>
    """

    return style + f'<div id="{strip_id}">{"".join(tiles)}</div>' + script


# -------------------- Inference (basic + split heatmap & explanation) --------------------
def run_basic_and_split_heatmap(
    video_path,
    checkpoint_path,
    device_choice,
    segment_length,
    out_dir,
    target_features,       # int (preferably even)
    inputs_choice          # str (for split heatmap it's better to use 'features')
):

    empty_txt = ""
    if video_path is None:
        return (
            empty_txt, None,  # transcript, osc
            None,  # bars
            None,  # combined heatmap
            "Please upload a video.",  # explanation
            "", "", "",  # body/face/scene html
            None,  # personality bars
            0.0,   # video duration
        )

    ckpt = (checkpoint_path or "").strip() or DEFAULT_CHECKPOINT
    dev = choose_device(device_choice)
    save_dir = _ensure_dir((out_dir or "").strip() or "outputs")

    # Basic pass
    res_basic = analyze_video_basic(
        video_path=video_path,
        checkpoint_path=ckpt,
        segment_length=int(segment_length),
        device=dev,
        save_dir=save_dir,
    )
    video_duration = float(res_basic.get("video_duration_sec", 0.0))

    transcript = res_basic.get("transcript", "")
    emo_prob = list(map(float, res_basic.get("emotion_logits", [0.0] * len(EMO_ORDER))))
    per_prob = list(map(float, res_basic.get("personality_scores", [0.0] * len(PERS_ORDER))))

    osc = res_basic.get("oscilloscope_path")
    osc_path = osc if (osc and os.path.exists(osc)) else None

    bars_png = os.path.join(save_dir, "emo_bars.png")
    remap = [EMO_ORDER.index(lbl) for lbl in EMO_ORDER_FOR_BARS]
    emo_prob_for_bars = [emo_prob[i] for i in remap]

    EMO_COLORS = {
        "Neutral": "#9CA3AF",
        "Anger": "#EF4444",
        "Disgust": "#10B981",
        "Fear": "#8B5CF6",
        "Happiness": "#F59E0B",
        "Sadness": "#3B82F6",
        "Surprise": "#EC4899",
    }
    palette = [EMO_COLORS[lbl] for lbl in EMO_ORDER_FOR_BARS]

    plot_emotion_probs_barchart(
        probs=emo_prob_for_bars,
        labels=EMO_ORDER_FOR_BARS,
        out_path=bars_png,
        colors=palette,
        figsize=(8, 3.7),
        auto_ylim=False,
    )

    pers_bars_png = os.path.join(save_dir, "pers_bars.png")
    plot_emotion_probs_barchart(
        probs=per_prob,
        labels=PERS_ORDER,
        out_path=pers_bars_png,
        title = "Score (%)",
        colors=[PERS_COLORS[lbl] for lbl in PERS_ORDER],
        figsize=(8, 4),
        auto_ylim=True,
        ylim_margin=3.0,
    )

    # Heatmaps for each task and contribution explanation
    res_attr = get_multitask_pred_with_attribution(
        video_path=video_path,
        checkpoint_path=ckpt,
        segment_length=int(segment_length),
        device=dev,
    )

    combined_heatmap = visualize_all_task_heatmaps(
        result=res_attr,
        name_video=os.path.basename(video_path),
        target_features=int(target_features),
        inputs=inputs_choice,     # usually 'features'
        out_dir="heatmaps_img",
    )

    contrib = compute_contributions_from_result(
        result=res_attr,
        inputs=inputs_choice,
        modality_order=['body', 'face', 'scene', 'audio', 'text'],
        target_features=int(target_features),
        visual_granularity="detailed",
    )
    explain_md = format_contribution_summary(contrib, emo_labels=EMO_ORDER)

    # Key frames (scene/face/body) вЂ” WITHOUT drawing on top of the image
    sample_dir = os.path.join(save_dir, "samples")
    frames_info = extract_keyframes_from_result(
        video_path=video_path,
        result=res_attr,
        out_dir=sample_dir,
        n_default=8,
    )

    indices = frames_info.get("indices", [])
    body_paths  = frames_info.get("body", [])
    face_paths  = frames_info.get("face", [])
    scene_paths = frames_info.get("scene", [])

    captions = [f"Frame {idx}" for idx in indices]

    # Single-row filmstrip galleries (by category)
    thumb_h = 120
    body_html  = render_gallery_html(body_paths,  captions, uid="body",  thumb_h=thumb_h)
    face_html  = render_gallery_html(face_paths,  captions, uid="face",  thumb_h=thumb_h)
    scene_html = render_gallery_html(scene_paths, captions,  uid="scene", thumb_h=thumb_h)

    return (
        transcript,
        osc_path,
        bars_png,
        combined_heatmap,
        explain_md,
        body_html,
        face_html,
        scene_html,
        pers_bars_png,
        video_duration,
    )


# -------------------- CSS --------------------
CUSTOM_CSS = """
:root{ --viz-h: 240px; --content-max-w: 1280px; }  /* one height for both images */

.divider {
    border-top: 3px solid rgba(0,0,0,0.2);
    margin: 1px 0;
}

.viz_row .gr-image { height: var(--viz-h) !important; }
.viz_row .gr-image img{
    height: 100% !important;
    width: 100% !important;
    object-fit: contain !important;
}
.viz_row {
    gap: 12px;
    margin-top: 12px;
}
.viz_row,
#gallery_row,
#audio_text_row {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

#osc_img {
    display: block;
    height: 80px !important;
    width: 100%;
    margin: 0;
}
#osc_img .gr-image,
#osc_img img {
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}

#page_wrap {
    max-width: var(--content-max-w);
    width: 100%;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

/* remove grey panels on results column */
#results_wrapper,
#results_group {
    background: transparent !important;
}
#results_wrapper .gr-row,
#results_wrapper .gr-column,
#results_wrapper .gr-group {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}
#results_group .gr-box,
#results_group .gr-panel,
#results_group .gr-block {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}
#results_group .gr-html,
#results_group .gr-image,
#results_group .gr-textbox,
#results_group .gr-markdown {
    background: transparent !important;
}
#results_wrapper .gradio-row {
    background: transparent !important;
}
.heat_img .gr-image { width: 100% !important; }
.heat_img img {
    width: 100% !important;
    max-height: 260px;
    object-fit: contain !important;
}

/* two-column block: audio+text on the left, bars on the right */
#audio_bars_row {
    align-items: flex-start;
    gap: 16px;
}
#audio_text_col .gr-image,
#audio_text_col .gr-textbox {
    width: 100% !important;
    max-width: 480px;
    margin: 0 auto;
}
#audio_text_col {
    border-right: 3px solid rgba(0, 0, 0, 0.2) !important;
    padding-right: 12px;
    margin-right: 10px;
}
#audio_bars_row {
    align-items: flex-start;
    gap: 20px;
}
#bars_col .gr-image {
    width: 100% !important;
    max-width: 420px;
    margin: 0 auto;
}
#bars_row_inner {
    gap: 12px;
    justify-content: center;
}

#emo_bar {
    height:200px;
}

#pers_bar {
    height:230px;
}

/* Kill gray background from Gradio styler wrapper (styler svelte-1ngupd) */
.styler {
    background: none !important;
    box-shadow: none !important;
    border: none !important;
}
#transcript_box textarea {
    background: transparent !important;
    border: 1px solid #555 !important;
    border-radius: 8px !important;
}
#results_group h2,
#results_group h3,
#results_group h4,
#results_group .gr-markdown p {
    text-align: center;
}
#results_group {
    padding-top: 20px;
    gap: 14px;
}

/* layout tuning for inputs */
#input_panel {
    display:flex;
    justify-content: space-between;
    align-items: flex-start;
    width: 100%;
    max-width: var(--content-max-w);
    margin: 0 auto;
}

#main_video video {
    object-fit: contain !important;
}
#main_video [data-testid="media-controls"] {
    justify-content: flex-start !important;
    padding-left: 6px;
}

#main_video > .video-container {
    height: 95% !important;
}

#settings_col {
    max-width: 544px !important;
}
#settings_col .gr-box,
#settings_col .gr-panel,
#settings_col .gr-block {
    padding: 8px 10px;
}
#run_btn_top {
    width: 100% !important;
    display: block !important;
}
#run_btn_top button {
    width: 100% !important;
}

#results_wrapper {
    max-width: var(--content-max-w);
    margin: 0 auto;
}
#results_group {
    padding-top: 20px;
    gap: 12px;
}
#results_group .gr-image img {
    object-fit: contain !important;
}

/* tighten gallery tiles */
.lightbox {
    align-items: center;
    justify-content: center;
}
.lightbox-content {
    max-height: 90vh;
}
#gallery_row {
    gap: 12px;
    margin-bottom: 12px;
    align-items: flex-start;
}
#gallery_row .gr-html {
    padding: 0;
}

#gallery_row > div {
    border: solid 0.5px #e4e4e7;
    border-radius: 3px;
}

#body_strip figcaption {
    color: black !important;
    opacity: 1.0 !important;
}

#face_strip figcaption {
    color: black !important;
    opacity: 1.0 !important;
}

#scene_strip figcaption {
    color: black !important;
    opacity: 1.0 !important;
}

#audio_text_row {
    gap: 12px;
    margin-bottom: 12px;
    align-items: stretch;
}
#demo_heat_row {
    width: 100%;
    max-width: var(--content-max-w);
    margin: 0 auto 8px;
    align-items: stretch;
    gap: 12px;
}
#demo_col {
    display: flex;
    flex-direction: column;
    justify-content: center;
}
#heatmap_col {
    display: flex;
    flex-direction: column;
    justify-content: center;
}
#demo_row {
    align-items: center;
    min-height: 87px;
    flex-wrap: wrap;
    gap: 10px;
}
#demo_row > button {
    height: 100%;
}

"""

# ---- Demo videos ----
DEMO_DIR = "demo_video"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def list_demo_videos():
    """Collect all video files from the `demo_video` folder."""
    if not os.path.isdir(DEMO_DIR):
        return []

    files = []
    for name in os.listdir(DEMO_DIR):
        ext = os.path.splitext(name)[1].lower()
        if ext in VIDEO_EXTS:
            files.append(os.path.join(DEMO_DIR, name))
    return sorted(files)

def load_demo_video(path: str):
    """Just return the path вЂ” Gradio will pass it to the Video component."""
    return path

# Wrapper that also controls the visibility of the right panel
def run_and_show(*args):
    start = time.time()
    result = run_basic_and_split_heatmap(*args)
    elapsed = time.time() - start
    *core_outputs, video_duration = result
    runtime_txt = f"Video duration: {video_duration:.1f} sec | Inference time: {elapsed:.1f} sec"
    # 10 values from run_basic_and_split_heatmap + status + group visibility + runtime text
    return (
        *core_outputs,
        gr.update(value="", visible=False),  # hide status text
        gr.update(visible=True),             # show results
        runtime_txt,
    )

# -------------------- UI --------------------
with gr.Blocks(title="MER", css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="page_wrap"):
        gr.Markdown("# Multimodal Emotion and Personality Recognition")

        demo_video_paths = list_demo_videos()

        # ---- Input panel: two columns ----
        with gr.Row(elem_id="input_panel"):
            # Left: video + demo buttons
            with gr.Column(scale=4, min_width=500, elem_id="video_col"):
                in_video = gr.Video(label="Video", height=337, elem_id="main_video")

            # Right: compact settings
            with gr.Column(scale=3, min_width=420, elem_id="settings_col"):
                in_ckpt  = gr.Textbox(label="Checkpoint path", value=DEFAULT_CHECKPOINT)
                in_device = gr.Dropdown(
                    label="Device",
                    choices=["auto (select automatically)", "cuda", "cpu"],
                    value="auto (select automatically)",
                )
                in_seglen = gr.Slider(5, 60, value=30, step=1, label="Segment length (sec.)")
                in_outdir = gr.Textbox(label="Output directory", value="outputs")

        # ---- Demos + heatmap settings row ----
        with gr.Row(elem_id="demo_heat_row"):
            with gr.Column(scale=4, min_width=500, elem_id="demo_col"):
                if demo_video_paths:
                    gr.Markdown("### Or select one of the preloaded videos:")
                    with gr.Row(elem_id="demo_row"):
                        for path in demo_video_paths:
                            label = os.path.basename(path)
                            btn = gr.Button(label, variant="secondary", size="sm")
                            btn.click(fn=load_demo_video, inputs=gr.State(path), outputs=in_video)
                else:
                    gr.Markdown(
                        "No files found in the `demo_video` folder. "
                        "Create the folder next to `app.py` and put some videos there."
                    )

            with gr.Column(scale=3, min_width=420, elem_id="heatmap_col"):
                gr.Markdown("### Heatmap display settings")
                with gr.Row(visible=True):
                    in_targetfeat = gr.Slider(8, 128, value=16, step=2, label="target_features")
                    in_inputs = gr.Dropdown(
                        label="Attribution source",
                        choices=["features", "emotion_logits", "personality_scores"],
                        value="features",
                    )

        # Full-width Run button under both columns
        with gr.Row():
            btn_run = gr.Button("Run", variant="primary", elem_id="run_btn_top")
        runtime_md = gr.Markdown("", elem_id="runtime_text")

        # ---- Results section below inputs ----
        gr.Markdown("", elem_classes=["divider"])
        with gr.Column(elem_id="results_wrapper"):
            right_status = gr.Markdown(
                "Analysis results will appear here after you run it.",
                elem_id="right_status"
            )

            with gr.Group(visible=False, elem_id="results_group") as results_group:

                with gr.Row(elem_id="gallery_row"):
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("### **Body - key gesture frames**")
                        out_body  = gr.HTML()
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("### **Face - key expression frames**")
                        out_face  = gr.HTML()
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("### **Scene - key context frames**")
                        out_scene = gr.HTML()

                gr.Markdown("", elem_classes=["divider"])
                with gr.Row(elem_id="audio_bars_row"):
                    with gr.Column(scale=2, min_width=0, elem_id="audio_text_col"):
                        gr.Markdown("## Input audio and text data")
                        gr.Markdown("<div align='center'><h4>Audio waveform</h4></div>")
                        out_osc = gr.Image(
                            label="Waveform",
                            elem_id="osc_img",
                            show_download_button=False,
                            interactive=False,
                            type="filepath",
                            container=False,
                            height=160,
                        )
                        gr.Markdown("<div align='center'><h4>Text transcript</h4></div>")
                        out_transcript = gr.Textbox(
                            label=None,
                            show_label=False,
                            lines=3,
                            container=False,
                            elem_id="transcript_box",
                        )

                    with gr.Column(scale=5, min_width=0, elem_id="bars_col"):
                        gr.Markdown("## Prediction result")
                        with gr.Row(elem_id="bars_row_inner"):
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("<div align='center'><h4>Probability distribution of emotions</h4></div>")
                                out_bars_png = gr.Image(
                                    label="Emotion Probabilities (Bars)",
                                    type="filepath",
                                    show_download_button=False,
                                    container=False,
                                    height=200,
                                    elem_id="emo_bar",
                                )
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("<div align='center'><h4>Personality Traits Scores</h4></div>")
                                out_pers_bars_png = gr.Image(
                                    label="Personality Traits Scores (Bars)",
                                    type="filepath",
                                    show_download_button=False,
                                    container=False,
                                    height=230,
                                    elem_id="pers_bar",
                                )

                gr.Markdown("", elem_classes=["divider"])
                gr.Markdown("## Visualization of Attention")
                with gr.Row(elem_classes=["viz_row", "heat_row"]):
                    out_heat_all = gr.Image(
                        label=None,
                        show_label=False,
                        type="filepath",
                        show_download_button=False,
                        container=False,
                        elem_classes=["heat_img"],
                        height=240,
                    )

                gr.Markdown("", elem_classes=["divider"])
                gr.Markdown("## Summary")
                out_explain = gr.Markdown(label="Explanation")
    # 1) before running: show "Processing...", hide results
    btn_run.click(
        fn=lambda: (
            gr.update(value="Processing...", visible=True),
            gr.update(visible=False),
            "",
        ),
        inputs=[],
        outputs=[right_status, results_group, runtime_md],
    ).then(
        # 2) run analysis + show results
        fn=run_and_show,
        inputs=[in_video, in_ckpt, in_device, in_seglen, in_outdir, in_targetfeat, in_inputs],
        outputs=[
            out_transcript,
            out_osc, out_bars_png,
            out_heat_all,
            out_explain,
            out_body, out_face, out_scene,
            out_pers_bars_png,
            right_status, results_group, runtime_md,
        ],
    )

if __name__ == "__main__":
    dev_auto = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[app] Auto device = {dev_auto}")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
