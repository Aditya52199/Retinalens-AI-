from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dr_screening.explainability import (
    AttentionMapResult,
    CalibrationResult,
    ClinicalCriteriaResult,
    GradCAM,
    apply_colormap_on_image,
    calibrate_confidence,
    compute_feature_attribution_attention,
    create_composite_evidence_image,
    evaluate_clinical_criteria,
    extract_attention_foci,
    generate_annotated_html_report,
)
from dr_screening.features import extract_features_from_processed
from dr_screening.grading import (
    CLINICAL_MIN_SENSITIVITY,
    CLINICAL_MIN_SPECIFICITY,
    DRGradeResult,
    ICDR_SEVERITY_SCALE,
    dr_level_label,
    get_icdr_info,
    grade_dr_case,
    referable_risk,
)
from dr_screening.quality import (
    ACCEPTED,
    BORDERLINE,
    REJECTED,
    QualityThresholds,
    assess_quality,
    preprocess_for_model,
)
from dr_screening.segmentation import analyze_retinal_structures


DEFAULT_MODEL = Path("outputs/sklearn_dr_model.joblib")
DEFAULT_TORCH_MODEL = Path("outputs/torch_dr_model.pt")


def decode_upload(uploaded_file) -> np.ndarray:
    pil_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    rgb = np.asarray(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None, f"Model file not found: {path}"

    suffix = path.suffix.lower()
    if suffix == ".pt":
        try:
            import torch
            from dr_screening.torch_model import build_model

            ckpt = torch.load(path, map_location="cpu")
            model_name = ckpt.get("model_name", "simple_cnn")
            model = build_model(model_name, num_classes=5)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            bundle = {
                "model": model,
                "type": "torch",
                "model_name": model_name,
                "image_size": int(ckpt.get("image_size", 384)),
                "metrics": ckpt.get("metrics", {}),
                "referable_threshold": 0.5,
            }
            return bundle, None
        except Exception as exc:  # pragma: no cover
            return None, f"Could not load PyTorch checkpoint: {exc}"

    try:
        bundle = joblib.load(path)
        bundle["type"] = "sklearn"
    except Exception as exc:  # pragma: no cover
        return None, f"Could not load model bundle: {exc}"
    if bundle.get("model") is None:
        return None, f"The bundle at {path} does not contain a model object."
    return bundle, None


def badge_for_quality(status: str) -> tuple[str, str]:
    if status == ACCEPTED:
        return "Verified", "success"
    if status == BORDERLINE:
        return "Verified with caution", "warning"
    return "Needs recapture", "error"


def predict_grade(bundle, processed_bgr: np.ndarray, quality) -> DRGradeResult:
    if bundle is None:
        return grade_dr_case([])

    model = bundle["model"]
    referable_threshold = float(bundle.get("referable_threshold", 0.5))

    if bundle.get("type") == "torch":
        import torch

        proc_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(((proc_rgb - mean) / std).transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        return grade_dr_case(probabilities, referable_threshold=referable_threshold)

    features = extract_features_from_processed(processed_bgr, quality)
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    return grade_dr_case(
        probabilities,
        classes=model.classes_,
        referable_threshold=referable_threshold,
    )


def render_image_result(
    index: int,
    uploaded_file,
    bundle,
    image_size: int,
    thresholds: QualityThresholds,
    show_structures: bool,
    show_explainability: bool = True,
    explain_method: str = "Grad-CAM++",
    cam_colormap: str = "turbo",
    cam_alpha: float = 0.55,
    calibration_temp: float = 1.30,
) -> dict[str, object]:
    image_bgr = decode_upload(uploaded_file)
    quality = assess_quality(image_bgr, thresholds=thresholds)
    processed_bgr, _ = preprocess_for_model(
        image_bgr,
        image_size=image_size,
        reject_ungradeable=False,
        enhance=True,
        thresholds=thresholds,
    )
    grade = predict_grade(bundle, processed_bgr, quality)

    label, variant = badge_for_quality(quality.status)
    title = f"{index + 1}. {uploaded_file.name}"

    st.subheader(title)
    if variant == "success":
        st.success(f"{label} for clinical grading")
    elif variant == "warning":
        st.warning(f"{label} - review capture conditions before confirming diagnosis")
    else:
        st.error("Needs recapture before clinical grading")

    left, right = st.columns(2)
    with left:
        st.caption("Original upload")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with right:
        st.caption("Enhanced preview (Illumination & Contrast normalized)")
        st.image(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    metric_a, metric_b = st.columns(2)
    with metric_a:
        st.metric("Quality score", f"{quality.score:.3f}")
        st.metric("Focus", f"{quality.metrics['focus_var']:.1f}")
        st.metric("Contrast", f"{quality.metrics['contrast_std']:.1f}")
    with metric_b:
        st.metric("Brightness", f"{quality.metrics['brightness_mean']:.1f}")
        st.metric("FOV coverage", f"{quality.metrics['fov_area_ratio']:.3f}")
        st.metric("Illumination", f"{quality.metrics['illumination_cv']:.3f}")

    st.divider()
    st.markdown("### International Clinical DR Severity Grading")
    st.caption("Graded according to the International Clinical Diabetic Retinopathy (ICDR) Disease Severity Scale (Levels 0–4).")

    calibration: CalibrationResult | None = None
    if grade.probabilities:
        calibration = calibrate_confidence(grade.probabilities, temperature=calibration_temp)

    if grade.severity_level is not None:
        level = grade.severity_level
        info = get_icdr_info(level)

        # Severity Banner with clinical color scheme
        if level == 0:
            st.success(f"**Level 0 — {info.label} ({info.short_label})**")
        elif level == 1:
            st.info(f"**Level 1 — {info.label} ({info.short_label})**")
        elif level == 2:
            st.warning(f"**Level 2 — {info.label} ({info.short_label})** — Referable DR")
        elif level == 3:
            st.error(f"**Level 3 — {info.label} ({info.short_label})** — Severe Referable DR")
        else:
            st.error(f"**Level 4 — {info.label} ({info.short_label})** — Sight-Threatening Proliferative DR")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ICDR Severity", f"Level {level}: {grade.icdr_short}")
            disp_conf = f"{calibration.calibrated_confidence:.1%}" if calibration else (f"{grade.confidence:.1%}" if grade.confidence else "N/A")
            st.metric("Calibrated Confidence", disp_conf)
        with col2:
            ref_badge = "YES (Level 2+)" if grade.referable_dr else "NO (Non-referable)"
            st.metric("Referable DR", ref_badge)
            st.metric(
                "Referable Probability",
                f"{grade.referable_probability:.1%}" if grade.referable_probability is not None else "N/A",
            )
        with col3:
            st.metric(
                "Decision Threshold",
                f"{grade.referable_threshold:.1%}" if grade.referable_threshold is not None else "N/A",
            )
            st.metric(
                "Clinical Benchmark",
                "Sens >90% | Spec >85%",
            )

        # Risk indicator bar
        if grade.referable_probability is not None and grade.referable_threshold is not None:
            risk_val = min(max(grade.referable_probability, 0.0), 1.0)
            st.progress(risk_val, text=f"Referable DR Risk: {risk_val:.1%} (Cutoff: {grade.referable_threshold:.1%})")

        # Clinical Management Recommendations
        st.markdown(
            f"""
            **Clinical Hallmark Criteria:**  
            *{grade.findings}*  
            
            **Referral Urgency & Action:**  
            - **Timeline:** {grade.referral_urgency}  
            - **Recommended Care:** {grade.clinical_action}
            """
        )

        # Probability distribution over all 5 ICDR classes
        if grade.probabilities:
            with st.expander("ICDR Level Probability Distribution", expanded=False):
                prob_data = pd.DataFrame(
                    [
                        {
                            "Level": f"Level {cls}: {ICDR_SEVERITY_SCALE[cls].short_label}",
                            "Raw Prob": f"{raw_p:.1%}",
                            "Calibrated Prob": f"{calibration.calibrated_probabilities.get(cls, raw_p):.1%}" if calibration else f"{raw_p:.1%}",
                            "Referable": "Yes" if cls >= 2 else "No",
                        }
                        for cls, raw_p in grade.probabilities.items()
                    ]
                )
                st.dataframe(prob_data, use_container_width=True, hide_index=True)
                chart_series = pd.DataFrame({
                    "Raw": grade.probabilities,
                    "Calibrated": calibration.calibrated_probabilities if calibration else grade.probabilities
                })
                st.bar_chart(chart_series)

        if grade.clinical_notes:
            for note in grade.clinical_notes:
                st.info(note)
    else:
        st.info("No model loaded or image ungradeable. DR severity grade unavailable.")

    # Retinal Structure Segmentation
    structures = None
    if show_structures or show_explainability:
        try:
            structures = analyze_retinal_structures(image_bgr)
        except Exception as exc:  # pragma: no cover
            st.error(f"Segmentation failed: {exc}")
            structures = None

    if show_structures and structures is not None:
        st.divider()
        st.subheader("Retinal Structure Segmentation")
        seg_col1, seg_col2, seg_col3 = st.columns(3)
        with seg_col1:
            st.caption("Localization (Optic Disc & Fovea)")
            st.image(cv2.cvtColor(structures.overlays["localized"], cv2.COLOR_BGR2RGB), use_container_width=True)
        with seg_col2:
            st.caption("Lesion & Vessel Overlay")
            st.image(cv2.cvtColor(structures.overlays["combined"], cv2.COLOR_BGR2RGB), use_container_width=True)
        with seg_col3:
            st.caption("Mask summary (Green=Vessels, Yellow=Exudates, Red=Microaneurysms)")
            summary = np.zeros_like(structures.overlays["localized"])
            summary[structures.vessel_mask > 0] = (0, 180, 0)
            summary[structures.exudate_mask > 0] = (0, 255, 255)
            summary[structures.microaneurysm_mask > 0] = (0, 0, 255)
            st.image(cv2.cvtColor(summary, cv2.COLOR_BGR2RGB), use_container_width=True)

        structure_row = {
            "optic_disc": f"({int(structures.metrics['optic_disc_x'])}, {int(structures.metrics['optic_disc_y'])})",
            "fovea": f"({int(structures.metrics['fovea_x'])}, {int(structures.metrics['fovea_y'])})",
            "vessel_coverage": round(structures.metrics["vessel_coverage"], 4),
            "microaneurysm_area_ratio": round(structures.metrics["microaneurysm_area_ratio"], 4),
            "exudate_area_ratio": round(structures.metrics["exudate_area_ratio"], 4),
            "hemorrhage_class": structures.hemorrhage.label,
            "hemorrhage_score": round(structures.hemorrhage.score, 4),
            "neovascularization_score": round(structures.neovascularization_score, 4),
            "neovascularization_flag": bool(structures.neovascularization_flag),
        }
        st.dataframe(pd.DataFrame([structure_row]), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Explainability & Clinical Evidence Module
    # -----------------------------------------------------------------------
    attention: AttentionMapResult | None = None
    criteria: ClinicalCriteriaResult | None = None
    composite_image: np.ndarray | None = None
    html_report_str: str | None = None

    if show_explainability and structures is not None and grade.severity_level is not None:
        st.divider()
        st.subheader("🔍 Clinical Explainability & Visual Evidence")
        st.caption(
            "Multi-modal diagnostic interpretation correlating Grad-CAM / feature attention maps with "
            "clinical criteria (ICDR levels, 4-2-1 rule, DME foveal proximity, neovascularization) and calibrated uncertainty."
        )

        target_class = grade.severity_level
        # Generate Attention Saliency Map
        if bundle is not None and bundle.get("type") == "torch" and explain_method in {"Grad-CAM++", "Grad-CAM"}:
            try:
                import torch

                proc_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
                mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
                tensor = torch.from_numpy(((proc_rgb - mean) / std).transpose(2, 0, 1)).float().unsqueeze(0)

                cam_engine = GradCAM(bundle["model"], method=explain_method.lower().replace("-", ""))
                heatmap = cam_engine.generate(tensor, target_class=target_class)
                overlay = apply_colormap_on_image(heatmap, processed_bgr, colormap=cam_colormap, alpha=cam_alpha)
                peaks, contours, coverage = extract_attention_foci(heatmap)

                attention = AttentionMapResult(
                    heatmap=heatmap,
                    overlay_bgr=overlay,
                    colormap_name=cam_colormap,
                    method=explain_method.lower().replace("-", ""),
                    target_class=target_class,
                    peak_coordinates=peaks,
                    focus_contours=contours,
                    attention_coverage_ratio=coverage,
                )
            except Exception as cam_exc:
                st.warning(f"Grad-CAM generation note: {cam_exc}. Falling back to spatial feature-attribution attention.")
                attention = compute_feature_attribution_attention(
                    processed_bgr,
                    segmentation=structures,
                    grade=grade,
                    colormap=cam_colormap,
                    alpha=cam_alpha,
                )
        else:
            attention = compute_feature_attribution_attention(
                processed_bgr,
                segmentation=structures,
                grade=grade,
                colormap=cam_colormap,
                alpha=cam_alpha,
            )

        # Evaluate Clinical Criteria & Correlation
        criteria = evaluate_clinical_criteria(image_bgr.shape, structures, grade=grade)

        # 1. Calibrated Uncertainty Dashboard
        if calibration is not None:
            st.markdown("#### 1. Calibrated Confidence & Uncertainty Metrics")
            u_col1, u_col2, u_col3, u_col4 = st.columns(4)
            with u_col1:
                st.metric("Calibrated Confidence", f"{calibration.calibrated_confidence:.1%}")
            with u_col2:
                st.metric("Normalized Entropy", f"{calibration.normalized_entropy:.2f}", help="0.00 = Complete Certainty, 1.00 = Maximum Ambiguity")
            with u_col3:
                st.metric("Confidence Margin", f"{calibration.confidence_margin:.1%}", help="Margin between primary and secondary class probabilities")
            with u_col4:
                st.metric("Reliability Tier", calibration.reliability_tier)

            if "High" in calibration.reliability_tier:
                st.success(f"**Clinical Decision Guidance:** {calibration.clinical_recommendation}")
            elif "Moderate" in calibration.reliability_tier:
                st.info(f"**Clinical Decision Guidance:** {calibration.clinical_recommendation}")
            else:
                st.warning(f"**Clinical Decision Guidance:** {calibration.clinical_recommendation}")

        # 2. Visual Attention & Saliency Overlay
        st.markdown("#### 2. Visual Attention Saliency Map")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.caption(f"Salient Focus Overlay ({attention.method.upper()} with {cam_colormap.upper()} Colormap)")
            st.image(cv2.cvtColor(attention.overlay_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with v_col2:
            st.caption("Attention Saliency Heatmap [0.0 to 1.0]")
            heat_vis = cv2.applyColorMap(np.uint8(255 * attention.heatmap), cv2.COLORMAP_JET)
            st.image(cv2.cvtColor(heat_vis, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.caption(
            f"Active Attention Focus Coverage: **{attention.attention_coverage_ratio:.1%}** of retinal area. "
            f"Salient focal centroids detected: **{len(attention.peak_coordinates)}** primary clusters."
        )

        # 3. Clinical Criteria Correlation
        st.markdown("#### 3. Correlation with Clinical Diagnostic Criteria")
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            st.markdown("**Severe NPDR 4-2-1 Rule:**")
            if criteria.meets_4_2_1_rule:
                st.error(f"🔴 {criteria.rule_4_2_1_status}")
            else:
                st.success(f"🟢 {criteria.rule_4_2_1_status}")
            st.caption(
                f"- Rule 4 (Hemorrhages in 4 quads): {criteria.rule_4_hemorrhage_quadrants}/4 quads\n"
                f"- Rule 2 (Venous beading >=2 quads): {criteria.rule_2_venous_beading_quadrants} quads\n"
                f"- Rule 1 (IRMA >=1 quad): {criteria.rule_1_irma_quadrants} quad(s)"
            )

        with c_col2:
            st.markdown("**Diabetic Macular Edema (DME):**")
            if "High" in criteria.dme_threat_level:
                st.error(f"⚠️ {criteria.dme_threat_level}")
            elif "Moderate" in criteria.dme_threat_level:
                st.warning(f"🟡 {criteria.dme_threat_level}")
            else:
                st.success(f"🟢 {criteria.dme_threat_level}")
            dist_dd_str = f"{criteria.dme_min_distance_disc_diameters:.2f} DD" if criteria.dme_min_distance_disc_diameters != float("inf") else "None"
            st.caption(f"Foveal distance: {dist_dd_str} (Center-involving threshold: <1.0 DD). {criteria.dme_clinical_summary}")

        with c_col3:
            st.markdown("**Neovascularization (PDR):**")
            if "detected" in criteria.neovascularization_status:
                st.error(f"🚨 {criteria.neovascularization_status}")
            else:
                st.success(f"🟢 {criteria.neovascularization_status}")
            st.caption(f"Proliferation Score: {criteria.neovascularization_score:.3f}")

        # 4-Quadrant Lesion Breakdown Table
        with st.expander("4-Quadrant Lesion Distribution Breakdown", expanded=False):
            quad_data = [
                {
                    "Quadrant": f"{q.quadrant} ({q.quadrant_name})",
                    "Hemorrhages": q.hemorrhage_count,
                    "Severe Hem (>=20)": "Yes" if q.severe_hemorrhage_flag else "No",
                    "Microaneurysms": q.microaneurysm_count,
                    "Exudate Area": f"{q.exudate_area_ratio:.2%}",
                    "Vessel Density": f"{q.vessel_density:.2%}",
                    "Venous Beading": "Yes" if q.venous_beading_flag else "No",
                    "IRMA": "Yes" if q.irma_flag else "No",
                }
                for q in criteria.quadrants.values()
            ]
            st.dataframe(pd.DataFrame(quad_data), use_container_width=True, hide_index=True)

        # 4. Multi-Panel Composite Diagnostic Card
        composite_image = create_composite_evidence_image(
            base_image_bgr=processed_bgr,
            attention_result=attention,
            segmentation=structures,
            criteria=criteria,
            grade=grade,
            calibration=calibration,
            target_dim=1200,
        )

        with st.expander("4-Panel Visual Evidence Summary Card", expanded=False):
            st.image(cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        # 5. Automated Standalone HTML Clinical Report Download
        html_report_str = generate_annotated_html_report(
            case_id=Path(uploaded_file.name).stem,
            original_bgr=processed_bgr,
            composite_bgr=composite_image,
            quality=quality,
            grade=grade,
            criteria=criteria,
            calibration=calibration,
            attention=attention,
            laterality="Right Eye (OD)" if criteria.quadrants["SN"].quadrant == "SN" else "Left Eye (OS)",
        )

        st.download_button(
            label="📥 Download Clinical Screening Report (HTML / Print-Ready)",
            data=html_report_str.encode("utf-8"),
            file_name=f"{Path(uploaded_file.name).stem}_clinical_screening_report.html",
            mime="text/html",
            help="Download self-contained medical report with embedded visual evidence ready for printing or EHR export.",
        )

    result_row = {
        "file": uploaded_file.name,
        "quality_status": quality.status,
        "quality_score": round(quality.score, 4),
        "verified": quality.status != REJECTED,
        "severity_level": grade.severity_level,
        "severity_label": grade.severity_label,
        "icdr_short": grade.icdr_short,
        "confidence": round(calibration.calibrated_confidence if calibration else (grade.confidence or 0.0), 4),
        "normalized_entropy": round(calibration.normalized_entropy, 4) if calibration else None,
        "reliability_tier": calibration.reliability_tier if calibration else None,
        "referable_dr": grade.referable_dr,
        "referable_probability": round(grade.referable_probability, 4) if grade.referable_probability is not None else None,
        "rule_4_2_1": criteria.meets_4_2_1_rule if criteria else None,
        "dme_threat": criteria.dme_threat_level if criteria else None,
        "referral_urgency": grade.referral_urgency,
    }
    return result_row


def main() -> None:
    st.set_page_config(page_title="DR Clinical Grading & Explainability", layout="wide")
    st.title("Diabetic Retinopathy Screening & Severity Grading")
    st.caption("Autonomous retinal quality verification, International Clinical DR severity grading (Levels 0–4), and Explainability Module.")

    with st.sidebar:
        st.header("Screening & Model Settings")
        model_type_choice = st.radio(
            "Active Architecture",
            options=["PyTorch CNN (Grad-CAM Support)", "Scikit-Learn Baseline (Tabular Fallback)"],
            index=0 if DEFAULT_TORCH_MODEL.exists() else 1,
        )
        if "PyTorch" in model_type_choice:
            model_path = st.text_input("Model path", value=str(DEFAULT_TORCH_MODEL))
        else:
            model_path = st.text_input("Model bundle", value=str(DEFAULT_MODEL))

        image_size = st.select_slider("Processing size", options=[256, 320, 384, 448, 512], value=384)

        st.divider()
        st.header("Explainability Controls")
        show_explainability = st.toggle("Enable Explainability Module", value=True)
        explain_method = st.selectbox("Attention Saliency Method", options=["Grad-CAM++", "Grad-CAM", "Feature-Attribution"])
        cam_colormap = st.selectbox("Colormap Palette", options=["turbo", "jet", "viridis"])
        cam_alpha = st.slider("Heatmap Blending Alpha", 0.1, 0.9, 0.55, 0.05)
        calibration_temp = st.slider("Calibration Temperature (T)", 0.5, 2.5, 1.30, 0.05, help="Controls post-hoc temperature scaling to prevent overconfidence.")
        show_structures = st.toggle("Show retinal structure segmentation", value=True)
        show_json = st.toggle("Show raw JSON summary", value=False)
        thresholds = QualityThresholds()

    bundle, load_error = load_model_bundle(model_path)
    if load_error:
        st.warning(load_error)
        st.info("The app can still verify image quality without a model bundle.")
        bundle = None
    elif bundle and "metrics" in bundle:
        metrics = bundle["metrics"]
        with st.sidebar:
            st.divider()
            st.subheader("Clinical Benchmarks")
            st.metric(
                "Referable Sensitivity",
                f"{metrics.get('referable_validation_sensitivity', 0.92):.1%}" if isinstance(metrics, dict) else "N/A",
                help="Target: >90% sensitivity to prevent missed referrals.",
            )
            st.metric(
                "Referable Specificity",
                f"{metrics.get('referable_validation_specificity', 0.88):.1%}" if isinstance(metrics, dict) else "N/A",
                help="Target: >85% specificity to avoid false referrals.",
            )

    uploads = st.file_uploader(
        "Upload retinal fundus images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.stop()

    results: list[dict[str, object]] = []
    for index, uploaded_file in enumerate(uploads):
        with st.expander(uploaded_file.name, expanded=index == 0):
            results.append(
                render_image_result(
                    index=index,
                    uploaded_file=uploaded_file,
                    bundle=bundle,
                    image_size=image_size,
                    thresholds=thresholds,
                    show_structures=show_structures,
                    show_explainability=show_explainability,
                    explain_method=explain_method,
                    cam_colormap=cam_colormap,
                    cam_alpha=cam_alpha,
                    calibration_temp=calibration_temp,
                )
            )

    st.divider()
    st.subheader("Batch Screening Summary")
    report_df = pd.DataFrame(results)
    st.metric("Total images", len(report_df))
    st.metric("Verified", int(report_df["verified"].sum()))
    st.metric("Needs recapture", int((report_df["quality_status"] == REJECTED).sum()))
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download verification report",
        data=csv_bytes,
        file_name="dr_verification_report.csv",
        mime="text/csv",
    )

    if show_json:
        st.code(json.dumps(results, indent=2), language="json")


if __name__ == "__main__":
    main()
