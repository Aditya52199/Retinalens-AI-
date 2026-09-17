from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dr_screening.explainability import (
    AttentionMapResult,
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
from dr_screening.grading import DRGradeResult, grade_dr_case
from dr_screening.quality import QualityThresholds, assess_quality, preprocess_for_model
from dr_screening.segmentation import analyze_retinal_structures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clinical explainability analysis (Grad-CAM, 4-2-1 criteria, calibrated uncertainty, HTML report)."
    )
    parser.add_argument("--image", default="", help="Single image file path to explain.")
    parser.add_argument("--image-dir", default="test_images", help="Directory containing images for batch mode.")
    parser.add_argument("--csv", default="", help="Optional CSV file with id_code for batch mode.")
    parser.add_argument("--torch-model", default="outputs/torch_dr_model.pt", help="PyTorch checkpoint path.")
    parser.add_argument("--sklearn-model", default="outputs/sklearn_dr_model.joblib", help="Scikit-learn bundle path.")
    parser.add_argument("--output-dir", default="outputs/explainability", help="Directory for generated reports.")
    parser.add_argument("--method", default="gradcam++", choices=["gradcam++", "gradcam", "feature_attribution"])
    parser.add_argument("--colormap", default="turbo", choices=["turbo", "jet", "viridis"])
    parser.add_argument("--alpha", type=float, default=0.55, help="Colormap blending alpha (0.0 to 1.0).")
    parser.add_argument("--temperature", type=float, default=1.30, help="Confidence calibration temperature.")
    parser.add_argument("--image-size", type=int, default=384, help="Model input dimension.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of processed images in batch mode.")
    return parser.parse_args()


def load_torch_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        import torch
        from dr_screening.torch_model import build_model
    except ImportError:
        return None

    ckpt = torch.load(path, map_location="cpu")
    model_name = ckpt.get("model_name", "simple_cnn")
    model = build_model(model_name, num_classes=5)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def explain_single_image(
    image_path: Path,
    output_dir: Path,
    torch_model,
    sklearn_bundle,
    method: str = "gradcam++",
    colormap: str = "turbo",
    alpha: float = 0.55,
    temperature: float = 1.30,
    image_size: int = 384,
) -> dict[str, object]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    thresholds = QualityThresholds()
    quality = assess_quality(image_bgr, thresholds=thresholds)
    processed_bgr, _ = preprocess_for_model(
        image_bgr,
        image_size=image_size,
        reject_ungradeable=False,
        enhance=True,
        thresholds=thresholds,
    )

    # Segment retinal structures (disc, fovea, vessels, lesions)
    segmentation = analyze_retinal_structures(image_bgr)

    # Grade image
    grade: DRGradeResult | None = None
    if sklearn_bundle is not None and "model" in sklearn_bundle:
        features = extract_features_from_processed(processed_bgr, quality)
        probs = sklearn_bundle["model"].predict_proba(features.reshape(1, -1))[0]
        referable_threshold = float(sklearn_bundle.get("referable_threshold", 0.5))
        grade = grade_dr_case(probs, classes=sklearn_bundle["model"].classes_, referable_threshold=referable_threshold)
    elif torch_model is not None:
        import torch

        proc_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (proc_rgb - mean) / std
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            logits = torch_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        grade = grade_dr_case(probs)
    else:
        probs = [0.80, 0.10, 0.05, 0.03, 0.02]
        grade = grade_dr_case(probs)

    target_class = grade.severity_level if (grade and grade.severity_level is not None) else 0

    # Generate Attention Map
    if torch_model is not None and method in {"gradcam", "gradcam++"}:
        import torch

        proc_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (proc_rgb - mean) / std
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).float().unsqueeze(0)

        gradcam = GradCAM(torch_model, method=method)
        heatmap = gradcam.generate(tensor, target_class=target_class)
        overlay = apply_colormap_on_image(heatmap, processed_bgr, colormap=colormap, alpha=alpha)
        peaks, contours, coverage = extract_attention_foci(heatmap)

        attention = AttentionMapResult(
            heatmap=heatmap,
            overlay_bgr=overlay,
            colormap_name=colormap,
            method=method,
            target_class=target_class,
            peak_coordinates=peaks,
            focus_contours=contours,
            attention_coverage_ratio=coverage,
        )
    else:
        attention = compute_feature_attribution_attention(
            processed_bgr,
            segmentation=segmentation,
            grade=grade,
            colormap=colormap,
            alpha=alpha,
        )

    # Clinical Criteria Evaluation
    criteria = evaluate_clinical_criteria(image_bgr.shape, segmentation, grade=grade)

    # Confidence Calibration
    calibration = calibrate_confidence(grade.probabilities, temperature=temperature)

    # 4-Panel Composite Visual Graphic
    composite = create_composite_evidence_image(
        base_image_bgr=processed_bgr,
        attention_result=attention,
        segmentation=segmentation,
        criteria=criteria,
        grade=grade,
        calibration=calibration,
        target_dim=1200,
    )

    # HTML Clinical Screening Report
    html_report = generate_annotated_html_report(
        case_id=image_path.stem,
        original_bgr=processed_bgr,
        composite_bgr=composite,
        quality=quality,
        grade=grade,
        criteria=criteria,
        calibration=calibration,
        attention=attention,
        laterality="Right Eye (OD)" if criteria.quadrants["SN"].quadrant == "SN" else "Left Eye (OS)",
    )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    comp_file = output_dir / f"{image_path.stem}_composite.jpg"
    html_file = output_dir / f"{image_path.stem}_report.html"
    cam_file = output_dir / f"{image_path.stem}_attention_{colormap}.jpg"

    cv2.imwrite(str(comp_file), composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(cam_file), attention.overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    html_file.write_text(html_report, encoding="utf-8")

    return {
        "case_id": image_path.stem,
        "severity_level": grade.severity_level,
        "severity_label": grade.severity_label,
        "referable_dr": grade.referable_dr,
        "calibrated_confidence": round(calibration.calibrated_confidence, 4),
        "normalized_entropy": round(calibration.normalized_entropy, 4),
        "reliability_tier": calibration.reliability_tier,
        "rule_4_2_1_status": criteria.rule_4_2_1_status,
        "dme_threat_level": criteria.dme_threat_level,
        "attention_method": attention.method,
        "attention_coverage": round(attention.attention_coverage_ratio, 4),
        "composite_path": str(comp_file),
        "html_report_path": str(html_file),
    }


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_model = load_torch_model(args.torch_model)
    sklearn_bundle = None
    if Path(args.sklearn_model).exists():
        try:
            sklearn_bundle = joblib.load(args.sklearn_model)
        except Exception:
            sklearn_bundle = None

    images_to_process: list[Path] = []
    if args.image:
        images_to_process = [Path(args.image)]
    elif args.csv:
        df = pd.read_csv(args.csv)
        if "id_code" in df.columns:
            for id_code in df["id_code"]:
                images_to_process.append(Path(args.image_dir) / f"{id_code}.png")
        if args.limit > 0:
            images_to_process = images_to_process[: args.limit]
    else:
        images_to_process = list(Path(args.image_dir).glob("*.png"))
        if args.limit > 0:
            images_to_process = images_to_process[: args.limit]

    if not images_to_process:
        print("No images found to explain.")
        return

    print(f"Running Explainability on {len(images_to_process)} image(s)...")
    records: list[dict[str, object]] = []

    for img_path in images_to_process:
        print(f"Explaining: {img_path.name}")
        record = explain_single_image(
            image_path=img_path,
            output_dir=output_dir,
            torch_model=torch_model,
            sklearn_bundle=sklearn_bundle,
            method=args.method,
            colormap=args.colormap,
            alpha=args.alpha,
            temperature=args.temperature,
            image_size=args.image_size,
        )
        records.append(record)
        print(
            f"  Grade: Level {record['severity_level']} ({record['severity_label']}) | "
            f"Referable: {record['referable_dr']} | "
            f"Conf: {record['calibrated_confidence']:.1%} | "
            f"Tier: {record['reliability_tier']}"
        )
        print(f"  Report saved: {record['html_report_path']}")
        print(f"  Composite saved: {record['composite_path']}")

    summary_df = pd.DataFrame(records)
    summary_path = output_dir / "explainability_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved batch summary: {summary_path}")


if __name__ == "__main__":
    main(parse_args())
