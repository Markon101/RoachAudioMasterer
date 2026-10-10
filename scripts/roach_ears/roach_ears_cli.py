"""
ROACH EARS: Unified Autonomous Audio Evaluation CLI
Evidence-Grounded Autonomous Audio Evaluation and Research Selection.

Commands:
  compare     Aligns, measures, mines excerpts, and optionally auditions two audio versions.
  grade       Detailed single-file deterministic and psychoacoustic acoustic profile.
  preference  Inspects Bradley-Terry skill ratings and records human listener judgments.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np

from .audio_io import read_wav, write_wav
from .alignment import align_audio_pair
from .dsp_metrics import (
    compute_bs1770_lufs,
    compute_true_peak,
    compute_spectral_metrics,
    compute_stereo_acoustics,
    scan_localized_artifacts
)
from .excerpt_selector import select_informative_excerpts, export_excerpts
from .multimodal_reviewer import MultimodalAudioReviewer
from .preference_engine import PreferenceEngine


def grade_single_audio(audio_path: Path) -> Dict[str, Any]:
    audio = read_wav(audio_path)
    lufs_meta = compute_bs1770_lufs(audio)
    tp_dbtp, overs = compute_true_peak(audio.samples)
    spec_meta = compute_spectral_metrics(audio)
    stereo_meta = compute_stereo_acoustics(audio)
    artifacts = scan_localized_artifacts(audio)

    # Compute crest factor
    peak_linear = float(np.max(np.abs(audio.samples)))
    rms_linear = float(np.sqrt(np.mean(audio.samples ** 2))) + 1e-12
    crest_db = 20.0 * np.log10(peak_linear / rms_linear)

    return {
        "file": str(audio_path.name),
        "path": str(audio_path.resolve()),
        "duration_sec": round(audio.duration_sec, 2),
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
        "loudness": lufs_meta,
        "dynamics": {
            "true_peak_dbtp": round(tp_dbtp, 2),
            "intersample_overs": overs,
            "crest_factor_db": round(crest_db, 2),
            "dynamic_spread_db": lufs_meta.get("dynamic_spread_db", 0.0),
        },
        "spectral": spec_meta,
        "spatial": stereo_meta,
        "detected_artifacts_count": len(artifacts),
        "top_artifacts": artifacts[:10],
    }


def compare_audio_pair(
    ref_path: Path,
    cand_path: Path,
    out_dir: Path,
    run_multimodal: bool = True,
    excerpt_duration_sec: float = 10.0,
    model_name: str = "google/gemini-3.8-flash"
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"================================================================================")
    print(f"    ROACH EARS: MULTIMODAL ACOUSTIC EVALUATION & A/B SELECTION")
    print(f"================================================================================")
    print(f"Reference Audio:   {ref_path}")
    print(f"Candidate Audio:   {cand_path}")
    print(f"Output Directory:  {out_dir}")
    print(f"Audition Model:    {model_name}")

    # 1. Read files
    ref_raw = read_wav(ref_path)
    cand_raw = read_wav(cand_path)
    print(f"Raw Reference:     {ref_raw.channels}ch, {ref_raw.sample_rate}Hz, {ref_raw.duration_sec:.2f}s")
    print(f"Raw Candidate:     {cand_raw.channels}ch, {cand_raw.sample_rate}Hz, {cand_raw.duration_sec:.2f}s")

    # 2. High-precision sub-sample alignment
    print("\n[Step 1/5] Performing sub-sample cross-correlation alignment...")
    ref_aligned, cand_aligned, align_meta = align_audio_pair(ref_raw, cand_raw)
    print(f"  Alignment Lag:   {align_meta['subsample_lag_samples']} samples ({align_meta['lag_time_ms']} ms)")
    print(f"  Cross-Corr Peak: {align_meta['peak_cross_correlation']:.5f}")
    print(f"  Signal/Diff SDR: {align_meta['sdr_db']:.2f} dB (Delta RMS: {align_meta['delta_rms_dbfs']:.2f} dBFS)")

    # 3. Deterministic & Psychoacoustic Metric extraction
    print("\n[Step 2/5] Extracting Layer A & Layer B acoustic profiles...")
    ref_prof = grade_single_audio(ref_path)
    cand_prof = grade_single_audio(cand_path)

    # 4. Autonomous Excerpt Selection
    print("\n[Step 3/5] Mining high-information comparison excerpts...")
    excerpts = select_informative_excerpts(
        ref_aligned,
        cand_aligned,
        excerpt_duration_sec=excerpt_duration_sec,
        max_excerpts=4
    )
    excerpts_dir = out_dir / "excerpts"
    manifest = export_excerpts(ref_aligned, cand_aligned, excerpts, excerpts_dir)
    for entry in manifest:
        print(f"  * Excerpt '{entry['excerpt_id']}': {entry['start_sec']:.1f}s - {entry['end_sec']:.1f}s | {entry['selection_reason']}")

    # 5. Multimodal Audio Review
    reviewer_results = []
    if run_multimodal:
        reviewer = MultimodalAudioReviewer(model_name=model_name)
        if reviewer.is_available():
            print(f"\n[Step 4/5] Dispatching blind A/B audio listening to Multimodal Reviewer ({reviewer.model_name})...")
            # Review top 2 most informative excerpts (max_delta and dense_air or transient_punch)
            for entry in manifest[:2]:
                exc_id = entry["excerpt_id"]
                ref_wav = Path(entry["files"]["reference_wav"])
                cand_wav = Path(entry["files"]["candidate_wav"])
                delta_wav = Path(entry["files"]["delta_wav_24db"])
                print(f"  Auditioning excerpt '{exc_id}' (including +24dB difference residual)...")
                rev_res = reviewer.review_excerpt_pair(ref_wav, cand_wav, entry, delta_wav_path=delta_wav)
                rev_res["excerpt_id"] = exc_id
                reviewer_results.append(rev_res)
                if rev_res.get("status") == "success":
                    print(f"    Verdict: {rev_res['overall_verdict']} (Confidence: {rev_res['confidence']:.2f}) | Diff: {rev_res['magnitude_of_difference']}")
                    print(f"    Observation: {rev_res['supporting_observation']}")
                    if rev_res.get("residual_analysis"):
                        res_info = rev_res["residual_analysis"]
                        print(f"    Residual Delta (+24dB): [{res_info.get('residual_nature')}] {res_info.get('audible_content_description')}")
        else:
            print("\n[Step 4/5] Multimodal Reviewer skipped (OpenRouter API key not configured).")

    # 6. Synthesize Multidimensional Grading Framework
    print("\n[Step 5/5] Synthesizing multidimensional grading report...")
    report_dict = {
        "reference_file": str(ref_path.name),
        "candidate_file": str(cand_path.name),
        "alignment": align_meta,
        "metrics_comparison": {
            "integrated_lufs": {
                "reference": ref_prof["loudness"]["integrated_lufs"],
                "candidate": cand_prof["loudness"]["integrated_lufs"],
                "delta_lufs": round(cand_prof["loudness"]["integrated_lufs"] - ref_prof["loudness"]["integrated_lufs"], 2)
            },
            "true_peak_dbtp": {
                "reference": ref_prof["dynamics"]["true_peak_dbtp"],
                "candidate": cand_prof["dynamics"]["true_peak_dbtp"],
                "delta_dbtp": round(cand_prof["dynamics"]["true_peak_dbtp"] - ref_prof["dynamics"]["true_peak_dbtp"], 2)
            },
            "dynamic_spread_db": {
                "reference": ref_prof["dynamics"]["dynamic_spread_db"],
                "candidate": cand_prof["dynamics"]["dynamic_spread_db"],
            },
            "spectral_slope_db_oct": {
                "reference": ref_prof["spectral"].get("spectral_slope_db_oct"),
                "candidate": cand_prof["spectral"].get("spectral_slope_db_oct"),
            },
            "hf_grain_proxy": {
                "reference": ref_prof["spectral"].get("hf_grain_proxy"),
                "candidate": cand_prof["spectral"].get("hf_grain_proxy"),
            },
            "stereo_coherence": {
                "reference": ref_prof["spatial"].get("stereo_coherence"),
                "candidate": cand_prof["spatial"].get("stereo_coherence"),
            },
            "sub_side_rms_dbfs": {
                "reference": ref_prof["spatial"].get("sub_side_rms_dbfs"),
                "candidate": cand_prof["spatial"].get("sub_side_rms_dbfs"),
            },
            "mono_cancellation_cdi": {
                "reference": ref_prof["spatial"].get("cancellation_distortion_index"),
                "candidate": cand_prof["spatial"].get("cancellation_distortion_index"),
            }
        },
        "excerpts": manifest,
        "multimodal_review": reviewer_results,
    }

    # Write evaluation.json
    json_path = out_dir / "evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # Write evaluation_report.md
    md_path = out_dir / "evaluation_report.md"
    generate_markdown_report(report_dict, md_path)

    print(f"\nEvaluation Complete! Artifacts written to:")
    print(f"  - Machine Report: {json_path}")
    print(f"  - Markdown Report: {md_path}")
    print(f"  - Excerpt WAVs:    {excerpts_dir}")

    return report_dict


def generate_markdown_report(data: Dict[str, Any], out_path: Path) -> None:
    comp = data["metrics_comparison"]
    align = data["alignment"]
    rev = data.get("multimodal_review", [])

    lines = [
        f"# 🦞 ROACH EARS — Comparative Acoustic Evaluation Report",
        f"",
        f"**Reference Master**: `{data['reference_file']}`  ",
        f"**Candidate Master**: `{data['candidate_file']}`  ",
        f"**Alignment**: {align['subsample_lag_samples']} samples ({align['lag_time_ms']} ms) | **Peak Cross-Correlation**: {align['peak_cross_correlation']:.5f}  ",
        f"**Signal-to-Difference (SDR)**: `{align['sdr_db']} dB` | **Delta RMS**: `{align['delta_rms_dbfs']} dBFS`  ",
        f"",
        f"---",
        f"",
        f"## 1. Multidimensional Metric Grade Matrix",
        f"",
        f"| Dimension | Reference | Candidate | Delta / Impact | Acoustic Meaning |",
        f"| :--- | :--- | :--- | :--- | :--- |",
        f"| **Integrated Loudness** | {comp['integrated_lufs']['reference']} LUFS | {comp['integrated_lufs']['candidate']} LUFS | {comp['integrated_lufs']['delta_lufs']:+.2f} LU | BS.1770 integrated level matching |",
        f"| **True Peak Ceiling** | {comp['true_peak_dbtp']['reference']} dBTP | {comp['true_peak_dbtp']['candidate']} dBTP | {comp['true_peak_dbtp']['delta_dbtp']:+.2f} dB | 4x sinc oversampled ceiling safety |",
        f"| **Dynamic Spread ($L_{{10}}-L_{{90}}$)** | {comp['dynamic_spread_db']['reference']} dB | {comp['dynamic_spread_db']['candidate']} dB | {comp['dynamic_spread_db']['candidate'] - comp['dynamic_spread_db']['reference']:+.2f} dB | Short-term macro-dynamic breathing |",
        f"| **Spectral Tilt** | {comp['spectral_slope_db_oct']['reference']} dB/oct | {comp['spectral_slope_db_oct']['candidate']} dB/oct | {comp['spectral_slope_db_oct']['candidate'] - comp['spectral_slope_db_oct']['reference']:+.2f} dB/oct | Overall warm vs bright balance |",
        f"| **HF Grain Proxy** | {comp['hf_grain_proxy']['reference']:.5f} | {comp['hf_grain_proxy']['candidate']:.5f} | {comp['hf_grain_proxy']['candidate'] - comp['hf_grain_proxy']['reference']:+.5f} | High-frequency flux flutter / metallic harshness |",
        f"| **Stereo Coherence** | {comp['stereo_coherence']['reference']} | {comp['stereo_coherence']['candidate']} | {comp['stereo_coherence']['candidate'] - comp['stereo_coherence']['reference']:+.4f} | Left/Right correlation (>0.5 healthy) |",
        f"| **Sub-Bass Side Power** | {comp['sub_side_rms_dbfs']['reference']} dBFS | {comp['sub_side_rms_dbfs']['candidate']} dBFS | {comp['sub_side_rms_dbfs']['candidate'] - comp['sub_side_rms_dbfs']['reference']:+.2f} dB | Out-of-phase low-end rumble (<60 Hz) |",
        f"| **Mono Cancellation (CDI)** | {comp['mono_cancellation_cdi']['reference']} | {comp['mono_cancellation_cdi']['candidate']} | {comp['mono_cancellation_cdi']['candidate'] - comp['mono_cancellation_cdi']['reference']:+.4f} | Summed-to-mono energy preservation |",
        f"",
        f"---",
        f"",
        f"## 2. Autonomous Informative Excerpts",
        f"",
        f"The following 10-second regions were mined as the most informative listening checkpoints:",
        f"",
    ]

    for exc in data["excerpts"]:
        lines.append(f"- **`{exc['excerpt_id']}`** ({exc['start_sec']}s - {exc['end_sec']}s): {exc['selection_reason']}")
        lines.append(f"  * Reference Clip: `{Path(exc['files']['reference_wav']).name}`")
        lines.append(f"  * Candidate Clip: `{Path(exc['files']['candidate_wav']).name}`")
        lines.append(f"  * Delta +24dB Clip: `{Path(exc['files']['delta_wav_24db']).name}`")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Multimodal Audio Reviewer Findings (Layer D)",
        f"",
    ])

    if rev:
        for r in rev:
            if r.get("status") == "success":
                lines.append(f"### Excerpt `{r.get('excerpt_id')}` Audition Verdict")
                lines.append(f"- **Overall Preference**: **{r.get('overall_verdict').upper()}** (Confidence: {r.get('confidence')})")
                lines.append(f"- **Audible Difference Magnitude**: `{r.get('magnitude_of_difference')}`")
                lines.append(f"- **Acoustic Observations**: {r.get('supporting_observation')}")
                lines.append(f"- **Timestamp Evidence**: `{r.get('timestamp_evidence')}`")
                lines.append(f"")
                lines.append(f"| Acoustic Attribute | Favored Version | Detailed Reviewer Note |")
                lines.append(f"| :--- | :--- | :--- |")
                for attr, info in r.get("dimensions", {}).items():
                    lines.append(f"| **{attr.replace('_', ' ').title()}** | `{info['winner']}` | {info['observation']} |")
                lines.append(f"")
                if r.get("residual_analysis"):
                    res = r["residual_analysis"]
                    lines.append(f"- **Isolated Difference Residual (+24dB Delta Analysis)**:")
                    lines.append(f"  * **Residual Classification**: `{res.get('residual_nature', 'Unknown')}`")
                    lines.append(f"  * **Audible Content Details**: {res.get('audible_content_description', '')}")
                    lines.append(f"")
    else:
        lines.append(f"> *Multimodal listening review was bypassed or unavailable for this run.*")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="ROACH EARS: Autonomous Multimodal Audio Evaluator")
    subparsers = parser.add_subparsers(dest="command")

    # Compare
    p_comp = subparsers.add_parser("compare", help="Compare two audio versions")
    p_comp.add_argument("--reference", "-r", type=Path, required=True, help="Reference audio file")
    p_comp.add_argument("--candidate", "-c", type=Path, required=True, help="Candidate audio file")
    p_comp.add_argument("--out", "-o", type=Path, default=Path("runs/roach_ears_eval"), help="Output directory")
    p_comp.add_argument("--multimodal", action="store_true", default=True, help="Enable multimodal listening reviewer")
    p_comp.add_argument("--no-multimodal", dest="multimodal", action="store_false", help="Disable multimodal listening reviewer")
    p_comp.add_argument("--excerpt-duration", type=float, default=10.0, help="Duration of mined excerpts in seconds")
    p_comp.add_argument("--model", type=str, default="google/gemini-3.8-flash", help="Multimodal audio reviewer model ID")

    # Grade
    p_grade = subparsers.add_parser("grade", help="Extract single-file acoustic profile")
    p_grade.add_argument("--input", "-i", type=Path, required=True, help="Input audio file")
    p_grade.add_argument("--out-json", type=Path, default=None, help="Optional output JSON path")

    # Preference
    p_pref = subparsers.add_parser("preference", help="Manage human preference records and active learning")
    p_pref.add_argument("--list-ratings", action="store_true", help="Display Bradley-Terry skill ratings")
    p_pref.add_argument("--record", action="store_true", help="Record a pairwise preference")
    p_pref.add_argument("--cand-a", type=str, help="Candidate A ID")
    p_pref.add_argument("--cand-b", type=str, help="Candidate B ID")
    p_pref.add_argument("--verdict", choices=["A_preferred", "B_preferred", "indistinguishable", "different_no_preference"], help="Listening verdict")
    p_pref.add_argument("--notes", type=str, default=None, help="Listener comments")

    args = parser.parse_args()

    if args.command == "compare":
        compare_audio_pair(
            ref_path=args.reference,
            cand_path=args.candidate,
            out_dir=args.out,
            run_multimodal=args.multimodal,
            excerpt_duration_sec=args.excerpt_duration,
            model_name=args.model
        )
    elif args.command == "grade":
        prof = grade_single_audio(args.input)
        print(json.dumps(prof, indent=2))
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump(prof, f, indent=2)
    elif args.command == "preference":
        engine = PreferenceEngine()
        if args.record:
            if not args.cand_a or not args.cand_b or not args.verdict:
                print("Error: --cand-a, --cand-b, and --verdict required to record preference.")
                sys.exit(1)
            engine.add_preference(
                candidate_a=args.cand_a,
                candidate_b=args.cand_b,
                verdict=args.verdict,
                listener_notes=args.notes
            )
            print(f"Recorded preference: {args.cand_a} vs {args.cand_b} -> {args.verdict}")
        ratings = engine.fit_bradley_terry()
        print("\n=== ROACH EARS: Bradley-Terry Latent Skill Ratings ===")
        print(f"{'Candidate':<25} | {'Skill (mu)':<12} | {'Uncertainty (sigma)':<20} | {'Comparisons':<12}")
        print("-" * 75)
        for cand_id, r in sorted(ratings.items(), key=lambda x: x[1]["skill_mu"], reverse=True):
            print(f"{cand_id:<25} | {r['skill_mu']:<12.3f} | {r['skill_sigma']:<20.3f} | {r['comparisons_count']:<12}")
    else:
        parser.print_help()


if __name__ == "__main__":
    import numpy as np
    main()
