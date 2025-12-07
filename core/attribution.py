# core/attribution.py
from __future__ import annotations
import os
from typing import Sequence, Optional, Dict, List
import numpy as np
import matplotlib.pyplot as plt


EMOTION_NAMES = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
# OCEAN order
PERS_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Non-Neuroticism"]
TRAIT_THRESHOLDS = {"high": 0.50}



# ───────────────────────── basic utilities ─────────────────────────
def normalize_minus1_to_1(vec: np.ndarray) -> np.ndarray:
    v_min, v_max = vec.min(), vec.max()
    if v_max == v_min:
        return np.zeros_like(vec, dtype=np.uint8)
    normalized = (vec - v_min) / (v_max - v_min)
    return (normalized * 255).astype(np.uint8)


def resize_vector(vec: np.ndarray, target_len: int) -> np.ndarray:
    old_indices = np.linspace(0, 1, num=len(vec))
    new_indices = np.linspace(0, 1, num=target_len)
    return np.interp(new_indices, old_indices, vec)


# === build split heatmap matrix (emo | personality halves) ===
def _build_split_heatmap_matrix(
    result: dict,
    task: str,
    idx: int,
    modality_order: Optional[Sequence[str]] = None,
    target_features: int = 32,
    inputs: str = "features",
) -> np.ndarray:
    """
    Build a matrix [modality, features] in 0..255 where the X-axis is split
    into emotional and personality halves. Uses per-row min-max normalization.
    """
    if modality_order is None:
        modality_order = ["body", "face", "scene", "audio", "text"]

    def _resize(v, n):
        old = np.linspace(0, 1, num=len(v))
        new = np.linspace(0, 1, num=n)
        return np.interp(new, old, v)

    def _norm_0_255(v):
        v = np.asarray(v, dtype=float)
        v_min, v_max = v.min(), v.max()
        if v_max == v_min:
            return np.zeros_like(v, dtype=np.uint8)
        return ((v - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)

    rows = []
    for mod in modality_order:
        vec = np.asarray(result["attribution"][inputs][task][mod][idx], dtype=float)
        if inputs == "features":
            half = int(vec.shape[0] / 2)
            emo_part, per_part = vec[:half], vec[half:]
            emo_resized = _resize(emo_part, int(target_features / 2))
            per_resized = _resize(per_part, int(target_features / 2))
            combined = np.concatenate([emo_resized, per_resized])
        else:
            combined = _resize(vec, target_features)
        rows.append(_norm_0_255(combined))

    return np.vstack(rows)  # [M, F]

def visualize_all_task_heatmaps(
    result: dict,
    name_video: str,
    modality_order: Optional[Sequence[str]] = None,
    target_features: int = 30,
    inputs: str = "features",
    figsize=(15.5, 2.5),
    cmap: str = "coolwarm",
    out_dir: str = "heatmaps",
    include_personality: bool = True,
) -> str:

    if modality_order is None:
        modality_order = ["body", "face", "scene", "audio", "text"]

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(name_video))[0]

    scores_raw = np.asarray(result.get("personality_scores", []), dtype=float)
    if scores_raw.size and scores_raw.min() >= 0.0 and scores_raw.max() <= 1.0:
        scores = scores_raw
    else:
        scores = 1.0 / (1.0 + np.exp(-scores_raw))

    emo_idx = int(np.argmax(result["emotion_logits"]))
    emo_prob = float(result["emotion_logits"][emo_idx])

    def _build_matrix(task: str, idx: int) -> np.ndarray:
        return _build_split_heatmap_matrix(
            result=result,
            task=task,
            idx=idx,
            modality_order=modality_order,
            target_features=target_features,
            inputs=inputs,
        )

    if not include_personality:
        mat = _build_matrix("emotion", emo_idx)

        half = target_features / 2.0
        last_idx = max(0, int(target_features - 1))
        xticks = np.linspace(0, last_idx, num=4, dtype=int)
        xticks = sorted(set(int(t) for t in xticks))
        x_label = "The Number of Features\n(Emotional | Personality)"

        fig, ax = plt.subplots(1, 1, figsize=(5.5, figsize[1]), dpi=170)
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=255)

        ax.set_yticks(np.arange(len(modality_order)))
        ax.set_yticklabels([m.capitalize() for m in modality_order], fontsize=8)
        ax.set_ylabel("Modality", fontsize=8)
        ax.tick_params(axis="y", labelleft=True, left=True, labelsize=8)

        ax.set_xlabel(x_label, fontsize=8)
        ax.axvline(half - 0.5, color="k", linewidth=2.0, alpha=0.6)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks], fontsize=8, rotation=0)

        title_txt = f"Predicted Emotion:\n {EMOTION_NAMES[emo_idx]} (prob. {emo_prob:.1%})"
        ax.set_title(title_txt, fontsize=9, pad=4)
        ax.tick_params(axis="both", labelsize=8)

        fig.subplots_adjust(right=0.90, bottom=0.22, top=0.82, left=0.14)
        cbar_ax = fig.add_axes([0.915, 0.22, 0.015, 0.60])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_ticks([0, 255])
        cbar.set_label("Attention", rotation=90, fontsize=8, labelpad=2)

        out_path = os.path.join(out_dir, f"{stem}_all_heatmaps.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path

    mats = [_build_matrix("emotion", emo_idx)]
    for i in range(len(PERS_NAMES)):
        mats.append(_build_matrix("personality", i))

    fig, axes = plt.subplots(1, 6, figsize=figsize, dpi=170, sharey=True)
    half = target_features / 2.0
    im = None
    titles = ["Emotion"] + PERS_NAMES
    x_label = "The Number of Features\n(Emotional | Personality)"
    for j, ax in enumerate(axes):
        mat = mats[j] if j < len(mats) else np.zeros((len(modality_order), target_features))
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=255)
        if j == 0:
            ax.set_yticks(np.arange(len(modality_order)))
            ax.set_yticklabels([m.capitalize() for m in modality_order], fontsize=8)
            ax.set_ylabel("Modality", fontsize=8)
            ax.tick_params(axis="y", labelleft=True, left=True, labelsize=8)
        else:
            ax.tick_params(axis="y", labelleft=False, left=False)
        ax.set_xlabel(x_label, fontsize=8)
        ax.axvline(half - 0.5, color="k", linewidth=2.0, alpha=0.6)
        last_idx = max(0, int(target_features - 1))
        xticks = np.linspace(0, last_idx, num=4, dtype=int)
        xticks = sorted(set(int(t) for t in xticks))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks], fontsize=8, rotation=0)
        if j == 0:
            title_txt = f"Predicted Emotion:\n {EMOTION_NAMES[emo_idx]} (prob. {emo_prob:.1%})"
        else:
            prob = float(scores[j - 1]) if (j - 1) < len(scores) else 0.0
            title_txt = f"Predicted Score of\n {titles[j]}: {prob:.2f}"
        ax.set_title(title_txt, fontsize=9, pad=4)
        ax.tick_params(axis="both", labelsize=8)

    fig.subplots_adjust(right=0.90, wspace=0.05, bottom=0.18, top=0.80, left=0.12)
    cbar_ax = fig.add_axes([0.915, 0.18, 0.005, 0.62])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, 255])
    cbar.set_label("Attention", rotation=90, fontsize=8, labelpad=2)

    out_path = os.path.join(out_dir, f"{stem}_all_heatmaps.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_emotion_probs_barchart(
    probs: Sequence[float],
    labels: Sequence[str],
    out_path: str,
    title: str = "Probability (%)",
    emojis: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    figsize: tuple = (7.5, 3.5),
    dpi: int = 170,
    auto_ylim: bool = False,
    ylim_margin: float = 3.0,
) -> str:
    """
    Saves a bar chart of emotion probabilities to PNG and returns the path.
    """
    p = np.asarray(probs, dtype=np.float32)
    assert p.ndim == 1, "probs must be a vector"
    assert len(labels) == len(p), "labels and probs must have the same length"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    x = np.arange(len(p))

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    ax.bar(x, p * 100.0, color=(colors if colors is not None else None), width=0.6)
    ax.set_ylabel(title, fontsize=14)
    ax.set_xticks(x)

    if emojis and len(emojis) == len(labels):
        xt = [f"{lbl}\n{emo}" for lbl, emo in zip(labels, emojis)]
        ax.set_xticklabels(xt, rotation=35, ha="right")
    else:
        ax.set_xticklabels(labels, rotation=35, ha="right")

    if auto_ylim:
        y_vals = p * 100.0
        y_min = float(y_vals.min())
        y_max = float(y_vals.max())
        margin = float(max(0.0, ylim_margin))
        low = max(0.0, y_min - margin)
        high = y_max + margin
        if high <= low:
            high = low + 1.0
        ax.set_ylim(low, high)
    else:
        ax.set_ylim(0.0, max(35.0, float(p.max() * 100.0) + 5.0))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def compute_contributions_from_result(
    result: dict,
    inputs: str = "features",
    modality_order: Optional[List[str]] = None,
    target_features: int = 32,
    visual_granularity: str = "detailed",  # "grouped" | "detailed"
) -> Dict[str, any]:
    if modality_order is None:
        modality_order = ["body", "face", "scene", "audio", "text"]

    emo_logits = np.asarray(result.get("emotion_logits", []), dtype=float)
    pred_emo_idx = int(np.argmax(emo_logits)) if emo_logits.size else 0
    pred_emo_prob = float(emo_logits[pred_emo_idx]) if emo_logits.size else 0.0

    scores_raw = np.asarray(result.get("personality_scores", []), dtype=float)
    if scores_raw.size and scores_raw.min() >= 0.0 and scores_raw.max() <= 1.0:
        pers_scores = scores_raw
    else:
        pers_scores = 1.0 / (1.0 + np.exp(-scores_raw))

    def _compute_from_matrix(mat: np.ndarray) -> Dict[str, any]:
        """
        Sum over the same matrix that is drawn on the heatmap (already resized
        and normalized to 0..255). This keeps text shares aligned with visuals.
        """
        M, F = mat.shape
        half = F // 2
        emo_slice = mat[:, :half]
        per_slice = mat[:, half:]

        emo_sums_by_mod = {
            m: float(np.sum(emo_slice[i])) for i, m in enumerate(modality_order)
        }
        per_sums_by_mod = {
            m: float(np.sum(per_slice[i])) for i, m in enumerate(modality_order)
        }

        if visual_granularity == "detailed":
            grouped = {
                "body_emotional": emo_sums_by_mod.get("body", 0.0),
                "face_emotional": emo_sums_by_mod.get("face", 0.0),
                "scene_emotional": emo_sums_by_mod.get("scene", 0.0),
                "audio_emotional": emo_sums_by_mod.get("audio", 0.0),
                "text_emotional": emo_sums_by_mod.get("text", 0.0),
                "body_personality": per_sums_by_mod.get("body", 0.0),
                "face_personality": per_sums_by_mod.get("face", 0.0),
                "scene_personality": per_sums_by_mod.get("scene", 0.0),
                "audio_personality": per_sums_by_mod.get("audio", 0.0),
                "text_personality": per_sums_by_mod.get("text", 0.0),
            }
        else:
            visual_mods = ["body", "face", "scene"]
            grouped = {
                "visual_emotional": float(sum(emo_sums_by_mod[m] for m in visual_mods)),
                "audio_emotional": emo_sums_by_mod.get("audio", 0.0),
                "text_emotional": emo_sums_by_mod.get("text", 0.0),
                "visual_personality": float(sum(per_sums_by_mod[m] for m in visual_mods)),
                "audio_personality": per_sums_by_mod.get("audio", 0.0),
                "text_personality": per_sums_by_mod.get("text", 0.0),
            }

        totals_local = {
            "emotional": float(np.sum(emo_slice)),
            "personality": float(np.sum(per_slice)),
        }
        totals_local["all"] = totals_local["emotional"] + totals_local["personality"]

        # build top list across ALL 10 cells (5 mods x 2 halves), ranked by share of total_all
        total_all = max(1e-9, totals_local["all"])
        top_candidates = []
        for mod in modality_order:
            e_val = emo_sums_by_mod.get(mod, 0.0)
            p_val = per_sums_by_mod.get(mod, 0.0)
            top_candidates.append((f"{mod}_emotional", e_val, e_val / total_all * 100.0))
            top_candidates.append((f"{mod}_personality", p_val, p_val / total_all * 100.0))
        top_candidates.sort(key=lambda t: t[2], reverse=True)
        top3_overall = top_candidates[:3]

        return {
            "grouped": grouped,
            "emo_sums_by_mod": emo_sums_by_mod,
            "per_sums_by_mod": per_sums_by_mod,
            "totals": totals_local,
            "top3_overall": top3_overall,
        }

    # emotion contributions from the heatmap matrix
    mat_emo = _build_split_heatmap_matrix(
        result=result,
        task="emotion",
        idx=pred_emo_idx,
        modality_order=modality_order,
        target_features=target_features,
        inputs=inputs,
    )
    emo_contrib = _compute_from_matrix(mat_emo)

    # per-trait personality matrices and contributions
    personality_details: List[Dict[str, any]] = []
    for trait_idx, trait_name in enumerate(PERS_NAMES):
        mat_trait = _build_split_heatmap_matrix(
            result=result,
            task="personality",
            idx=trait_idx,
            modality_order=modality_order,
            target_features=target_features,
            inputs=inputs,
        )
        contrib_trait = _compute_from_matrix(mat_trait)
        personality_details.append(
            {
                "trait": trait_name,
                "score": float(pers_scores[trait_idx]) if trait_idx < len(pers_scores) else 0.0,
                "emo_sums_by_mod": contrib_trait["emo_sums_by_mod"],
                "per_sums_by_mod": contrib_trait["per_sums_by_mod"],
                "totals": contrib_trait["totals"],
                "top3_overall": contrib_trait["top3_overall"],
            }
        )

    return {
        "grouped": emo_contrib["grouped"],
        "emo_sums_by_mod": emo_contrib["emo_sums_by_mod"],
        "per_sums_by_mod": emo_contrib["per_sums_by_mod"],
        "top3_overall": emo_contrib["top3_overall"],
        "pred_emo_idx": pred_emo_idx,
        "pred_emo_prob": pred_emo_prob,
        "personality_scores": pers_scores.tolist() if isinstance(pers_scores, np.ndarray) else [],
        "totals": emo_contrib["totals"],
        "visual_granularity": visual_granularity,
        "target_features": target_features,
        "modality_order": modality_order,
        "personality_details": personality_details,
    }


def format_contribution_summary(contrib: dict, emo_labels: List[str]) -> str:
    total_emo = max(1e-9, float(contrib["totals"]["emotional"]))
    total_per = max(1e-9, float(contrib["totals"]["personality"]))
    total_all = total_emo + total_per
    emo_share = total_emo / total_all * 100.0
    per_share = total_per / total_all * 100.0

    emo_name = None
    emo_prob_pct = None
    if "pred_emo_idx" in contrib and isinstance(emo_labels, (list, tuple)):
        idx = int(contrib["pred_emo_idx"])
        if 0 <= idx < len(emo_labels):
            emo_name = emo_labels[idx]
            if "pred_emo_prob" in contrib:
                try:
                    emo_prob_pct = float(contrib["pred_emo_prob"]) * 100.0
                except Exception:
                    emo_prob_pct = None

    if per_share >= emo_share:
        dom_label = "personality"
    else:
        dom_label = "emotional"

    names = {
        "body": "Body",
        "face": "Face",
        "scene": "Scene",
        "audio": "Audio",
        "text": "Text",
        "visual": "Visual",
    }

    def _fmt_toplist(top_list):
        items = []
        for key, val, share in top_list:
            if key.startswith("body_"):
                name = names["body"]
            elif key.startswith("face_"):
                name = names["face"]
            elif key.startswith("scene_"):
                name = names["scene"]
            elif key.startswith("audio_"):
                name = names["audio"]
            elif key.startswith("text_"):
                name = names["text"]
            else:
                name = key
            src = "emotional" if "emotional" in key else "personality"
            items.append(f"{name} - {share:.1f}% ({src})")
        return items

    lines: List[str] = []
    lines.append('<div style="font-size:15px; line-height:1.45">\n')

    if emo_name is not None:
        if emo_prob_pct is not None:
            lines.append(f"Predicted emotion: <b>{emo_name}</b> ({emo_prob_pct:.1f}%).\n")
        else:
            lines.append(f"Predicted emotion: <b>{emo_name}</b>.\n")

    lines.append(
        f"Impact of features: emotional is {emo_share:.1f}%, "
        f"personality is {per_share:.1f}%."
    )

    top_mods_strs = _fmt_toplist(contrib.get("top3_overall", []))
    if top_mods_strs:
        lines.append("Top modalities: " + "; ".join(top_mods_strs) + ".\n")

    pers_scores = contrib.get("personality_scores", [])
    pers_details = contrib.get("personality_details", [])

    high_traits: List[str] = []
    low_traits: List[str] = []

    if isinstance(pers_scores, (list, tuple)) and len(pers_scores) == len(PERS_NAMES):
        thr = TRAIT_THRESHOLDS["high"]
        scores = [float(s) for s in pers_scores]

        for name, s in zip(PERS_NAMES, scores):
            if s >= thr:
                high_traits.append(name)
            else:
                low_traits.append(name)

        for detail in pers_details:
            t_name = detail.get("trait", "")
            t_score = float(detail.get("score", 0.0))
            t_totals = detail.get("totals", {})
            t_emo = float(t_totals.get("emotional", 0.0))
            t_per = float(t_totals.get("personality", 0.0))
            t_all = t_emo + t_per if (t_emo + t_per) > 0 else 1e-9
            t_emo_share = t_emo / t_all * 100.0
            t_per_share = t_per / t_all * 100.0

            t_top3 = detail.get("top3_overall", [])
            t_dom_label = "personality" if t_per_share >= t_emo_share else "emotional"
            t_dom_share = max(t_emo_share, t_per_share)

            lines.append(
                f"</br>Predicted score of <b>{t_name}</b>: {t_score:.2f}. "
                f"Impact of features: emotional is {t_emo_share:.1f}%, "
                f"personality is {t_per_share:.1f}%. "
            )
            trait_top_strs = _fmt_toplist(t_top3)
            if trait_top_strs:
                lines.append("Top modalities: " + "; ".join(trait_top_strs) + ".\n")

        def _join_list(lst: List[str]) -> str:
            if not lst:
                return ""
            if len(lst) == 1:
                return lst[0]
            if len(lst) == 2:
                return f"{lst[0]} and {lst[1]}"
            return f"{', '.join(lst[:-1])} and {lst[-1]}"

        def _narrative_for(emo: str, high: List[str], low: List[str]) -> str:
            e = (emo or "").lower()
            h = {x.lower() for x in high}
            l = {x.lower() for x in low}

            def has(name, bucket):
                return name.lower() in bucket

            if e == "anger" and has("agreeableness", l) and has("non-neuroticism", l):
                return "may reflect a tendency to express frustration directly and react strongly to perceived challenges, with less emphasis on social harmony."
            if e == "sadness" and has("extraversion", l) and has("non-neuroticism", l):
                return "is consistent with a reserved and introspective behavioral style, potentially accompanied by heightened sensitivity to negative emotional cues."
            if e == "happiness" and has("extraversion", h) and has("agreeableness", h):
                return "aligns with an outwardly expressive and socially engaged demeanor, often associated with positive interpersonal interactions."
            if e == "happiness" and has("openness", h) and has("conscientiousness", l):
                return "suggests enthusiasm for new experiences and ideas, though with less focus on structured or routine-oriented behavior."
            if e == "neutral" and has("conscientiousness", h) and has("openness", l):
                return "indicates a preference for predictability, order, and goal-directed actions, rather than novelty or ambiguity."
            if e == "fear" and has("non-neuroticism", l) and has("extraversion", l):
                return "may correspond to cautious and vigilant behavior in social or uncertain situations, with a tendency to avoid high-stimulation environments."
            if e == "disgust" and has("agreeableness", l) and has("openness", l):
                return "can be associated with skepticism toward unfamiliar people or ideas and a preference for clear social or moral boundaries."
            if e == "surprise" and has("openness", h) and has("extraversion", h):
                return "often reflects curiosity and responsiveness to unexpected events, especially in dynamic or socially rich contexts."
            if not high and len(low) == len(PERS_NAMES) and e in {"anger", "fear", "disgust", "sadness"}:
                return "may indicate a context-dependent reaction rather than a stable personality tendency, warranting cautious interpretation without longitudinal data."
            if has("agreeableness", h) and has("non-neuroticism", h) and e in {"happiness", "neutral"}:
                return "typically corresponds to calm, cooperative, and empathetic interaction patterns, even under mild stress."
            return "suggests a combination of the observed emotion and trait profile, and should be interpreted in the context of the situation."

        if emo_name:
            high_str_raw = _join_list(high_traits)
            low_str_raw = _join_list(low_traits)

            if high_traits and low_traits:
                traits_phrase = (
                    f"high scores for the traits of <b>{high_str_raw}</b>, "
                    f"low scores for the traits of <b>{low_str_raw}</b>"
                )
            elif high_traits:
                traits_phrase = f"high scores for the traits of {high_str_raw}"
            elif low_traits:
                traits_phrase = f"low scores for the traits of {low_str_raw}"
            else:
                traits_phrase = "no clearly defined high or low personality traits"

            main_sentence = (
                f"</br>Thus, a person experiences the emotion of <b>{emo_name}</b> and shows "
                f"{traits_phrase}."
            )
            narrative_clause = _narrative_for(emo_name, high_traits, low_traits)
            lines.append(main_sentence + f" Which indicates that the person {narrative_clause}")

    lines.append("</div>")
    return "".join(lines)
