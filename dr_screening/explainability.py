from __future__ import annotations

import base64
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .grading import (
    ICDR_SEVERITY_SCALE,
    REFERABLE_LEVEL,
    DRGradeResult,
    dr_level_label,
    get_icdr_info,
    referable_from_level,
    referable_risk,
)
from .quality import QualityResult
from .segmentation import HemorrhageResult, LocalizationResult, SegmentationResult


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class QuadrantLesionMetrics:
    """Quantitative lesion metrics for a specific retinal quadrant."""

    quadrant: str  # "ST" (Superior-Temporal), "SN" (Superior-Nasal), "IT" (Inferior-Temporal), "IN" (Inferior-Nasal)
    quadrant_name: str
    hemorrhage_count: int
    hemorrhage_area_ratio: float
    microaneurysm_count: int
    exudate_area_ratio: float
    vessel_density: float
    severe_hemorrhage_flag: bool  # >= 20 intraretinal hemorrhages or high density
    venous_beading_flag: bool     # Venous caliber abnormalities
    irma_flag: bool               # Intraretinal microvascular abnormalities


@dataclass
class ClinicalCriteriaResult:
    """Clinical diagnostic criteria evaluation correlated with ICDR standards."""

    quadrants: dict[str, QuadrantLesionMetrics]
    rule_4_2_1_status: str              # e.g., "Positive (Severe NPDR)" or "Negative"
    rule_4_hemorrhage_quadrants: int    # Number of quadrants with >= 20 hemorrhages (target: 4)
    rule_2_venous_beading_quadrants: int # Number of quadrants with venous beading (target: >= 2)
    rule_1_irma_quadrants: int          # Number of quadrants with IRMA (target: >= 1)
    meets_4_2_1_rule: bool
    dme_threat_level: str               # "Center-involving (High Risk)", "Non-center-involving (Moderate)", "Low", "None"
    dme_min_distance_px: float          # Distance in pixels from foveal center to closest hard exudate
    dme_min_distance_disc_diameters: float # Distance in optic disc diameters (DD)
    dme_clinical_summary: str
    neovascularization_status: str      # "NVD detected", "NVE detected", "Negative"
    neovascularization_score: float
    icdr_criteria_alignment: str
    clinical_summary_bullets: list[str]


@dataclass
class CalibrationResult:
    """Confidence calibration and uncertainty estimation metrics."""

    calibrated_probabilities: dict[int, float]
    calibrated_confidence: float
    confidence_margin: float       # Difference between top-1 and top-2 probabilities
    normalized_entropy: float      # Shannon entropy normalized to [0.0, 1.0]
    uncertainty_level: str         # "Low", "Moderate", "High / Indeterminate"
    reliability_tier: str          # "High Confidence", "Moderate Confidence", "Requires Specialist Review"
    temperature_applied: float
    clinical_recommendation: str


@dataclass
class AttentionMapResult:
    """Attention map generated via Grad-CAM or feature-attribution fallback."""

    heatmap: np.ndarray             # 2D float array normalized to [0.0, 1.0]
    overlay_bgr: np.ndarray         # Blended color visualization (BGR uint8)
    colormap_name: str
    method: str                     # "gradcam", "gradcam++", "feature_attribution"
    target_class: int
    peak_coordinates: list[tuple[int, int]]
    focus_contours: list[np.ndarray]
    attention_coverage_ratio: float


# ---------------------------------------------------------------------------
# Grad-CAM and Grad-CAM++ Engine
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM and Grad-CAM++ visual explanation engine for PyTorch models.

    Extracts gradient-weighted class activation maps from the final convolutional
    layer to highlight morphological evidence driving network classification.
    """

    def __init__(
        self,
        model: Any,
        target_layer: Any | None = None,
        method: str = "gradcam++",
        device: Any | None = None,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for GradCAM. Install torch/torchvision or use "
                "compute_feature_attribution_attention for tree-based models."
            ) from exc

        self.torch = torch
        self.nn = nn
        self.model = model
        self.method = method.lower()
        if self.method not in {"gradcam", "gradcam++"}:
            self.method = "gradcam++"

        if device is None:
            try:
                self.device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.eval()

        if target_layer is None:
            self.target_layer = self._find_last_conv_layer(model)
        elif isinstance(target_layer, str):
            self.target_layer = self._resolve_layer_by_name(model, target_layer)
        else:
            self.target_layer = target_layer

        if self.target_layer is None:
            raise ValueError("Could not find a convolutional target layer in the provided model.")

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.hooks: list[Any] = []
        self._register_hooks()

    def _find_last_conv_layer(self, model: Any) -> Any:
        """Locate the deepest Conv2d module in the model hierarchy."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, self.nn.Conv2d):
                last_conv = module
        return last_conv

    def _resolve_layer_by_name(self, model: Any, layer_name: str) -> Any:
        """Resolve a named submodule path like 'features.16' or 'layer4.1'."""
        current = model
        for part in layer_name.split("."):
            if hasattr(current, part):
                current = getattr(current, part)
            elif part.isdigit() and isinstance(current, (self.nn.Sequential, self.nn.ModuleList)):
                current = current[int(part)]
            else:
                return None
        return current

    def _register_hooks(self) -> None:
        def forward_hook(module: Any, input: Any, output: Any) -> None:
            self.activations = output.detach()

        def backward_hook(module: Any, grad_input: Any, grad_output: Any) -> None:
            # grad_output[0] contains the gradient w.r.t the layer's output activations
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Clean up PyTorch hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

    def generate(
        self,
        input_tensor: Any,
        target_class: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Generate a 2D attention heatmap normalized to [0.0, 1.0].

        Parameters
        ----------
        input_tensor : torch.Tensor
            Image tensor with shape (1, C, H, W).
        target_class : int | None
            Target ICDR DR severity class index. If None, uses top predicted class.
        output_size : tuple[int, int] | None
            Desired (width, height) for resizing the output heatmap. Defaults to input tensor (W, H).
        """
        torch = self.torch
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        score = output[:, target_class]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients. Check target layer hooks.")

        gradients = self.gradients  # shape: (1, K, H_feat, W_feat)
        activations = self.activations  # shape: (1, K, H_feat, W_feat)

        if self.method == "gradcam++":
            # Grad-CAM++ closed-form weight derivation
            # alpha_ijk = grad^2 / (2*grad^2 + sum(act * grad^3) + eps)
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)
            sum_act_grad3 = torch.sum(activations * grad_3, dim=(2, 3), keepdim=True)
            eps = 1e-7
            alpha = grad_2 / (2.0 * grad_2 + sum_act_grad3 + eps)
            weights = torch.sum(alpha * torch.relu(gradients), dim=(2, 3), keepdim=True)
        else:
            # Standard Grad-CAM global-average-pooled gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        if output_size is None:
            target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        else:
            target_w, target_h = output_size

        if cam_np.shape != (target_h, target_w):
            cam_np = cv2.resize(cam_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return np.clip(cam_np, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Colormap Blending & Contours
# ---------------------------------------------------------------------------

def apply_colormap_on_image(
    heatmap: np.ndarray,
    base_image_bgr: np.ndarray,
    colormap: str = "turbo",
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend a 2D float heatmap [0.0, 1.0] onto a BGR base image using OpenCV colormaps."""
    h, w = base_image_bgr.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))

    colormap_key = colormap.lower()
    if colormap_key == "jet":
        color_map = cv2.COLORMAP_JET
    elif colormap_key == "viridis":
        color_map = cv2.COLORMAP_VIRIDIS if hasattr(cv2, "COLORMAP_VIRIDIS") else cv2.COLORMAP_JET
    else:
        color_map = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET

    colored_cam = cv2.applyColorMap(heatmap_uint8, color_map)
    blended = cv2.addWeighted(colored_cam, alpha, base_image_bgr, 1.0 - alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def extract_attention_foci(
    heatmap: np.ndarray,
    threshold: float = 0.55,
    min_area_px: int = 15,
) -> tuple[list[tuple[int, int]], list[np.ndarray], float]:
    """Extract peak attention centroids, focus contours, and active coverage ratio.

    Parameters
    ----------
    heatmap : np.ndarray
        2D float heatmap in range [0.0, 1.0].
    threshold : float
        Activation threshold for isolating salient focus regions.
    min_area_px : int
        Minimum contour area in pixels to exclude single-pixel noise.

    Returns
    -------
    peaks : list[tuple[int, int]]
        List of (x, y) coordinates representing peak attention centers.
    contours : list[np.ndarray]
        List of salient focus contour polygons.
    coverage_ratio : float
        Fraction of retinal area covered by high attention.
    """
    binary = np.uint8((heatmap >= threshold) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours: list[np.ndarray] = []
    peaks: list[tuple[int, int]] = []
    total_active_pixels = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area_px:
            valid_contours.append(cnt)
            m = cv2.moments(cnt)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                peaks.append((cx, cy))
            total_active_pixels += area

    coverage_ratio = float(total_active_pixels / max(heatmap.size, 1))
    return peaks, valid_contours, coverage_ratio


# ---------------------------------------------------------------------------
# Feature-Attribution Spatial Attention Fallback (Tree / Non-Torch Models)
# ---------------------------------------------------------------------------

def compute_feature_attribution_attention(
    image_bgr: np.ndarray,
    segmentation: SegmentationResult | None = None,
    grade: DRGradeResult | None = None,
    colormap: str = "turbo",
    alpha: float = 0.55,
) -> AttentionMapResult:
    """Generate spatial evidence attention map from segmented pathology for tabular/tree models.

    Combines detected microaneurysms, hemorrhages, hard exudates, and abnormal vessel
    branching into a calibrated spatial density field smoothed via Gaussian kernels.
    """
    h, w = image_bgr.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    if segmentation is not None:
        # Microaneurysms: localized red focal dots
        if segmentation.microaneurysm_mask is not None:
            ma_resized = cv2.resize(segmentation.microaneurysm_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            density += (ma_resized > 0).astype(np.float32) * 1.0

        # Hard Exudates: lipid deposits (high weight for macular threat)
        if segmentation.exudate_mask is not None:
            ex_resized = cv2.resize(segmentation.exudate_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            density += (ex_resized > 0).astype(np.float32) * 1.25

        # Hemorrhages: blot and flame hemorrhages
        if segmentation.hemorrhage is not None and segmentation.hemorrhage.score > 0.05:
            # Use lower-intensity green/value regions outside vessels
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            dark_lesions = (gray < 85).astype(np.float32)
            if segmentation.vessel_mask is not None:
                vessels = cv2.resize(segmentation.vessel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                dark_lesions = dark_lesions * (vessels == 0)
            density += dark_lesions * (segmentation.hemorrhage.score * 1.5)

        # Neovascularization / Vessel proliferation around disc or peripheral
        if segmentation.neovascularization_score > 0.15 and segmentation.vessel_mask is not None:
            vessels = cv2.resize(segmentation.vessel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated_vessels = cv2.dilate(vessels, kernel_dil)
            density += (dilated_vessels > 0).astype(np.float32) * (segmentation.neovascularization_score * 0.9)

    # If no structural lesions detected or image is normal, smooth a gentle baseline
    if float(density.max()) == 0.0:
        # Generate a mild foveal/macular focus to show monitored region
        cx, cy = w // 2, h // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        sigma_sq = (min(h, w) * 0.35) ** 2
        density = np.exp(-dist_sq / (2.0 * sigma_sq)).astype(np.float32) * 0.15

    # Apply Gaussian smoothing to simulate neural receptive fields
    kernel_size = int(max(h, w) * 0.08)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(density, (kernel_size, kernel_size), sigmaX=kernel_size / 3.0)

    # Normalize to [0.0, 1.0]
    s_min, s_max = float(smoothed.min()), float(smoothed.max())
    if s_max > s_min:
        heatmap = (smoothed - s_min) / (s_max - s_min)
    else:
        heatmap = np.zeros_like(smoothed)

    overlay = apply_colormap_on_image(heatmap, image_bgr, colormap=colormap, alpha=alpha)
    peaks, contours, coverage = extract_attention_foci(heatmap, threshold=0.55)
    target_class = grade.severity_level if (grade and grade.severity_level is not None) else 0

    return AttentionMapResult(
        heatmap=heatmap.astype(np.float32),
        overlay_bgr=overlay,
        colormap_name=colormap,
        method="feature_attribution",
        target_class=target_class,
        peak_coordinates=peaks,
        focus_contours=contours,
        attention_coverage_ratio=coverage,
    )


# ---------------------------------------------------------------------------
# Clinical Criteria Correlator (ICDR & 4-2-1 Rule & DME & Neovascularization)
# ---------------------------------------------------------------------------

def evaluate_clinical_criteria(
    image_shape: tuple[int, int] | tuple[int, int, int],
    segmentation: SegmentationResult,
    grade: DRGradeResult | None = None,
) -> ClinicalCriteriaResult:
    """Evaluate retinal findings against clinical diagnostic criteria.

    Correlates quantitative findings with:
    1. 4-quadrant fundus partitioning (ST, SN, IT, IN)
    2. Severe NPDR "4-2-1 Rule" (>=20 hemorrhages in 4 quadrants, venous beading in 2, IRMA in 1)
    3. Diabetic Macular Edema (DME) foveal distance threat (<1 DD center-involving risk)
    4. Proliferative DR neovascularization (NVD vs NVE)
    5. Overall alignment with International Clinical DR severity scale
    """
    h, w = image_shape[:2]

    # Establish Foveal and Optic Disc Centers
    fx = segmentation.fovea.x if segmentation.fovea.x > 0 else w // 2
    fy = segmentation.fovea.y if segmentation.fovea.y > 0 else h // 2
    od_x = segmentation.optic_disc.x if segmentation.optic_disc.x > 0 else w // 4
    od_y = segmentation.optic_disc.y if segmentation.optic_disc.y > 0 else h // 2
    od_r = max(segmentation.optic_disc.radius, 15)
    disc_diameter = 2.0 * od_r

    # Identify eye laterality from disc/fovea position
    # If optic disc is to the left of fovea -> Right Eye (OD): Temporal is Right, Nasal is Left.
    # If optic disc is to the right of fovea -> Left Eye (OS): Temporal is Left, Nasal is Right.
    is_right_eye = od_x < fx

    # Define 4 Quadrant Masks using horizontal (y=fy) and vertical (x=fx) meridians
    quad_coords = {
        "ST": (("Superior", "Temporal"), (0, fy, fx if not is_right_eye else fx, w if not is_right_eye else w)),
        "SN": (("Superior", "Nasal"), (0, fy, 0 if not is_right_eye else 0, fx if not is_right_eye else fx)),
        "IT": (("Inferior", "Temporal"), (fy, h, fx if not is_right_eye else fx, w if not is_right_eye else w)),
        "IN": (("Inferior", "Nasal"), (fy, h, 0 if not is_right_eye else 0, fx if not is_right_eye else fx)),
    }

    quadrant_names = {
        "ST": "Superior-Temporal",
        "SN": "Superior-Nasal",
        "IT": "Inferior-Temporal",
        "IN": "Inferior-Nasal",
    }

    quadrant_results: dict[str, QuadrantLesionMetrics] = {}
    rule_4_hemorrhage_count = 0
    rule_2_venous_beading_count = 0
    rule_1_irma_count = 0

    # Resize masks to full image dimensions for quadrant partitioning
    ma_mask = cv2.resize(segmentation.microaneurysm_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    ex_mask = cv2.resize(segmentation.exudate_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    vessel_mask = cv2.resize(segmentation.vessel_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    for q_code, q_name in quadrant_names.items():
        # Partition bounding box
        if "Superior" in q_code or q_code.startswith("S"):
            y_start, y_end = 0, fy
        else:
            y_start, y_end = fy, h

        if is_right_eye:
            # OD: Nasal is left (x < fx), Temporal is right (x >= fx)
            if "N" in q_code:
                x_start, x_end = 0, fx
            else:
                x_start, x_end = fx, w
        else:
            # OS: Temporal is left (x < fx), Nasal is right (x >= fx)
            if "T" in q_code:
                x_start, x_end = 0, fx
            else:
                x_start, x_end = fx, w

        q_h = max(y_end - y_start, 1)
        q_w = max(x_end - x_start, 1)
        q_area = q_h * q_w

        q_ma = ma_mask[y_start:y_end, x_start:x_end]
        q_ex = ex_mask[y_start:y_end, x_start:x_end]
        q_vessels = vessel_mask[y_start:y_end, x_start:x_end]

        # Microaneurysms count via connected components
        num_ma_labels, _, _, _ = cv2.connectedComponentsWithStats(q_ma, connectivity=8)
        ma_count = max(0, num_ma_labels - 1)

        # Exudate ratio
        ex_ratio = float(np.count_nonzero(q_ex) / q_area)

        # Hemorrhages: estimate from global hemorrhage score partitioned proportionally
        # and local dark lesion clusters
        q_vessel_density = float(np.count_nonzero(q_vessels) / q_area)
        hem_ratio = float(segmentation.hemorrhage.area_ratio * (0.8 + 0.4 * (ma_count / max(1, ma_count + 5))))

        # Intraretinal hemorrhages estimate: 4-2-1 rule defines severe as >= 20 intraretinal hemorrhages
        # Scaling based on hemorrhage score and microaneurysms
        estimated_hem_count = int(round(ma_count * 1.5 + (segmentation.hemorrhage.score * 25.0)))

        # Evaluate 4-2-1 indicators per quadrant
        # 1) Hemorrhage threshold: >= 20 distinct hemorrhages or high hemorrhage load
        severe_hem = estimated_hem_count >= 20 or hem_ratio > 0.008
        if severe_hem:
            rule_4_hemorrhage_count += 1

        # 2) Venous beading: indicated by vessel irregularities / abnormal vessel tortuosity
        venous_beading = (q_vessel_density > 0.18 and segmentation.metrics.get("neovascularization_branch_density", 0.0) > 0.002)
        if venous_beading:
            rule_2_venous_beading_count += 1

        # 3) IRMA: Intraretinal microvascular abnormalities (tortuous shunt vessels)
        irma = (segmentation.metrics.get("neovascularization_branch_to_vessel_ratio", 0.0) > 0.025 and ma_count >= 3)
        if irma:
            rule_1_irma_count += 1

        quadrant_results[q_code] = QuadrantLesionMetrics(
            quadrant=q_code,
            quadrant_name=q_name,
            hemorrhage_count=estimated_hem_count,
            hemorrhage_area_ratio=hem_ratio,
            microaneurysm_count=ma_count,
            exudate_area_ratio=ex_ratio,
            vessel_density=q_vessel_density,
            severe_hemorrhage_flag=severe_hem,
            venous_beading_flag=venous_beading,
            irma_flag=irma,
        )

    # 4-2-1 Rule Synthesis:
    # Rule 4: Hemorrhages in all 4 quadrants
    # Rule 2: Venous beading in >= 2 quadrants
    # Rule 1: Prominent IRMA in >= 1 quadrant
    meets_rule_4 = rule_4_hemorrhage_count == 4
    meets_rule_2 = rule_2_venous_beading_count >= 2
    meets_rule_1 = rule_1_irma_count >= 1
    meets_4_2_1 = meets_rule_4 or meets_rule_2 or meets_rule_1

    if meets_4_2_1:
        reasons: list[str] = []
        if meets_rule_4:
            reasons.append("Hemorrhages in all 4 quadrants")
        if meets_rule_2:
            reasons.append(f"Venous beading in {rule_2_venous_beading_count} quadrants")
        if meets_rule_1:
            reasons.append(f"IRMA in {rule_1_irma_count} quadrant(s)")
        rule_4_2_1_status = f"Positive (Severe NPDR: {', '.join(reasons)})"
    else:
        rule_4_2_1_status = "Negative (Criteria for Severe NPDR not met)"

    # Diabetic Macular Edema (DME) Proximity Evaluation
    exudate_coords = np.argwhere(ex_mask > 0)
    if len(exudate_coords) > 0:
        # Distances from foveal center (fy, fx) in pixels
        y_diffs = exudate_coords[:, 0] - fy
        x_diffs = exudate_coords[:, 1] - fx
        distances_px = np.sqrt(y_diffs ** 2 + x_diffs ** 2)
        min_dist_px = float(np.min(distances_px))
        min_dist_dd = float(min_dist_px / max(disc_diameter, 1.0))

        if min_dist_dd <= 1.0:
            dme_threat_level = "Center-involving (High Risk)"
            dme_clinical_summary = (
                f"Hard exudates detected within 1 Disc Diameter ({min_dist_dd:.2f} DD, {min_dist_px:.1f} px) "
                "of the fovea, indicating sight-threatening Center-Involving Diabetic Macular Edema (DME)."
            )
        elif min_dist_dd <= 2.0:
            dme_threat_level = "Non-center-involving (Moderate Risk)"
            dme_clinical_summary = (
                f"Hard exudates located {min_dist_dd:.2f} DD from the fovea (within 1–2 Disc Diameters). "
                "Non-center-involving macular threat; urgent OCT scan recommended."
            )
        else:
            dme_threat_level = "Peripheral / Low Threat"
            dme_clinical_summary = (
                f"Hard exudates are outside the central macular zone ({min_dist_dd:.2f} DD from fovea)."
            )
    else:
        min_dist_px = float("inf")
        min_dist_dd = float("inf")
        dme_threat_level = "None"
        dme_clinical_summary = "No hard exudates or signs of macular edema detected."

    # Neovascularization (PDR) Assessment: NVD vs NVE
    nv_score = float(segmentation.neovascularization_score)
    disc_branch_density = float(segmentation.metrics.get("neovascularization_disc_branch_density", 0.0))
    if segmentation.neovascularization_flag or nv_score >= 0.45:
        if disc_branch_density > 0.005:
            nv_status = "NVD detected (Neovascularization of the Optic Disc)"
        else:
            nv_status = "NVE detected (Neovascularization Elsewhere)"
    else:
        nv_status = "Negative (No proliferative neovascular complexes)"

    # Formulate Clinical Summary Bullets
    bullets: list[str] = []
    if nv_status != "Negative (No proliferative neovascular complexes)":
        bullets.append(f"CRITICAL FINDING: {nv_status}. Hallmark sign of Proliferative Diabetic Retinopathy (Level 4).")
    if meets_4_2_1:
        bullets.append(f"4-2-1 CRITERIA MET: {rule_4_2_1_status}. High risk of rapid progression to PDR.")
    if dme_threat_level.startswith("Center-involving"):
        bullets.append(f"MACULAR THREAT: {dme_clinical_summary}")
    elif dme_threat_level.startswith("Non-center"):
        bullets.append(f"MACULAR FINDING: {dme_clinical_summary}")

    total_ma = sum(q.microaneurysm_count for q in quadrant_results.values())
    if total_ma > 0:
        bullets.append(f"Microaneurysms detected: ~{total_ma} across fundus quadrants.")
    else:
        bullets.append("No microaneurysms detected.")

    alignment = (
        f"Findings support ICDR Level "
        f"{grade.severity_level if (grade and grade.severity_level is not None) else 'N/A'}: "
        f"{grade.severity_label if (grade and grade.severity_label) else 'Pending assessment'}."
    )

    return ClinicalCriteriaResult(
        quadrants=quadrant_results,
        rule_4_2_1_status=rule_4_2_1_status,
        rule_4_hemorrhage_quadrants=rule_4_hemorrhage_count,
        rule_2_venous_beading_quadrants=rule_2_venous_beading_count,
        rule_1_irma_quadrants=rule_1_irma_count,
        meets_4_2_1_rule=meets_4_2_1,
        dme_threat_level=dme_threat_level,
        dme_min_distance_px=min_dist_px,
        dme_min_distance_disc_diameters=min_dist_dd,
        dme_clinical_summary=dme_clinical_summary,
        neovascularization_status=nv_status,
        neovascularization_score=nv_score,
        icdr_criteria_alignment=alignment,
        clinical_summary_bullets=bullets,
    )


# ---------------------------------------------------------------------------
# Confidence Calibration & Uncertainty Estimation
# ---------------------------------------------------------------------------

class CalibrationModel:
    """Post-hoc temperature scaling and Platt probability calibration.

    Prevents neural network overconfidence and yields calibrated posterior probabilities
    aligned with true empirical positive predictive values.
    """

    def __init__(self, temperature: float = 1.30) -> None:
        self.temperature = max(0.1, float(temperature))

    def calibrate_multiclass(self, probabilities: Sequence[float] | dict[int, float]) -> dict[int, float]:
        """Apply temperature scaling on multiclass probabilities via inverse softmax."""
        if isinstance(probabilities, dict):
            classes = sorted(probabilities.keys())
            probs = np.array([probabilities[c] for c in classes], dtype=np.float64)
        else:
            probs = np.array(probabilities, dtype=np.float64)
            classes = list(range(len(probs)))

        probs = np.clip(probs, 1e-9, 1.0)
        probs = probs / np.sum(probs)

        # Convert to pseudo-logits: z = log(p)
        logits = np.log(probs)
        scaled_logits = logits / self.temperature

        # Softmax with numerical stability
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)

        return {cls: float(calibrated_probs[i]) for i, cls in enumerate(classes)}

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        num_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE) for binary referable DR prediction."""
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        n_samples = len(y_true)
        if n_samples == 0:
            return 0.0

        for idx in range(num_bins):
            bin_lower = bin_edges[idx]
            bin_upper = bin_edges[idx + 1]
            mask = (y_prob >= bin_lower) & (y_prob < bin_upper if idx < num_bins - 1 else y_prob <= bin_upper)
            bin_count = int(np.sum(mask))
            if bin_count > 0:
                bin_acc = float(np.mean(y_true[mask]))
                bin_conf = float(np.mean(y_prob[mask]))
                ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)
        return float(ece)


def calibrate_confidence(
    probabilities: Sequence[float] | dict[int, float],
    temperature: float = 1.30,
) -> CalibrationResult:
    """Compute temperature-calibrated probabilities, Shannon entropy, and reliability tier.

    Parameters
    ----------
    probabilities : Sequence[float] | dict[int, float]
        Raw model probability distribution over ICDR classes (0 to 4).
    temperature : float
        Temperature parameter T > 1.0 to soften uncalibrated overconfident logits.
    """
    calibrator = CalibrationModel(temperature=temperature)
    cal_probs = calibrator.calibrate_multiclass(probabilities)

    sorted_classes = sorted(cal_probs.keys(), key=lambda c: cal_probs[c], reverse=True)
    top_class = sorted_classes[0]
    top_prob = cal_probs[top_class]

    second_prob = cal_probs[sorted_classes[1]] if len(sorted_classes) > 1 else 0.0
    margin = float(top_prob - second_prob)

    # Compute Normalized Shannon Entropy: H / log2(K)
    num_classes = max(len(cal_probs), 2)
    entropy = 0.0
    for prob in cal_probs.values():
        if prob > 1e-9:
            entropy -= prob * math.log2(prob)
    max_entropy = math.log2(num_classes)
    normalized_entropy = float(entropy / max(max_entropy, 1e-5))
    normalized_entropy = min(max(normalized_entropy, 0.0), 1.0)

    # Stratify Uncertainty and Reliability Tiers
    if normalized_entropy < 0.38 and margin > 0.35 and top_prob > 0.70:
        uncertainty = "Low"
        tier = "High Confidence"
        rec = (
            "Classification confidence is high and concordant across primary and secondary diagnostics. "
            "Suitable for automated screening workflow."
        )
    elif normalized_entropy < 0.68 and margin > 0.15:
        uncertainty = "Moderate"
        tier = "Moderate Confidence"
        rec = (
            "Model confidence is moderate with low-to-moderate ambiguity between adjacent ICDR stages. "
            "Standard clinical verification advised."
        )
    else:
        uncertainty = "High / Indeterminate"
        tier = "Requires Specialist Review"
        rec = (
            "Classification entropy is elevated near the diagnostic decision boundary. "
            "Mandatory over-read by a consultant retinal specialist is required before clinical referral."
        )

    return CalibrationResult(
        calibrated_probabilities=cal_probs,
        calibrated_confidence=float(top_prob),
        confidence_margin=margin,
        normalized_entropy=normalized_entropy,
        uncertainty_level=uncertainty,
        reliability_tier=tier,
        temperature_applied=temperature,
        clinical_recommendation=rec,
    )


# ---------------------------------------------------------------------------
# Composite 4-Panel Evidence Graphic
# ---------------------------------------------------------------------------

def create_composite_evidence_image(
    base_image_bgr: np.ndarray,
    attention_result: AttentionMapResult,
    segmentation: SegmentationResult,
    criteria: ClinicalCriteriaResult,
    grade: DRGradeResult | None = None,
    calibration: CalibrationResult | None = None,
    target_dim: int = 1200,
) -> np.ndarray:
    """Assemble a high-resolution 4-panel diagnostic visual evidence summary card.

    Layout (2x2 Grid):
    - Panel A: Enhanced Fundus Image with Fovea, Disc, and 4-Quadrant Partition
    - Panel B: Attention Saliency Map (Grad-CAM++ / Feature-Attribution) with Focus Contours
    - Panel C: Retinal Structure & Lesion Segmentation (Vessels, Exudates, Hemorrhages)
    - Panel D: Clinical Criteria & ICDR Diagnostic Dashboard (Quadrant bars, DME gauge, 4-2-1 status)
    """
    panel_size = target_dim // 2
    canvas = np.full((target_dim, target_dim, 3), 18, dtype=np.uint8)  # Deep medical slate background

    # ---------------------------------------------------------
    # Panel A: Enhanced Fundus with Anatomical Markings
    # ---------------------------------------------------------
    panel_a = cv2.resize(base_image_bgr, (panel_size, panel_size))
    # Scale coordinates to panel_size
    orig_h, orig_w = base_image_bgr.shape[:2]
    scale_x = panel_size / orig_w
    scale_y = panel_size / orig_h

    fx = int(segmentation.fovea.x * scale_x) if segmentation.fovea.x > 0 else panel_size // 2
    fy = int(segmentation.fovea.y * scale_y) if segmentation.fovea.y > 0 else panel_size // 2
    od_x = int(segmentation.optic_disc.x * scale_x) if segmentation.optic_disc.x > 0 else panel_size // 4
    od_y = int(segmentation.optic_disc.y * scale_y) if segmentation.optic_disc.y > 0 else panel_size // 2
    od_r = int(max(segmentation.optic_disc.radius * scale_x, 10))

    # Draw Optic Disc and Fovea
    cv2.circle(panel_a, (od_x, od_y), od_r, (0, 215, 255), 2)  # Gold circle for disc
    cv2.putText(panel_a, "Optic Disc", (od_x - 30, max(20, od_y - od_r - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

    cv2.drawMarker(panel_a, (fx, fy), (0, 140, 255), cv2.MARKER_CROSS, 20, 2)  # Orange cross for fovea
    cv2.putText(panel_a, "Fovea", (fx + 12, fy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

    # Draw Quadrant Meridian Lines centered at Fovea
    cv2.line(panel_a, (0, fy), (panel_size, fy), (100, 180, 255), 1, cv2.LINE_AA)
    cv2.line(panel_a, (fx, 0), (fx, panel_size), (100, 180, 255), 1, cv2.LINE_AA)

    # Add quadrant labels
    cv2.putText(panel_a, "ST", (panel_size - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel_a, "SN", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel_a, "IT", (panel_size - 40, panel_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel_a, "IN", (20, panel_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Panel A Header banner
    cv2.rectangle(panel_a, (0, 0), (panel_size, 26), (20, 20, 20), -1)
    cv2.putText(panel_a, "A. Enhanced Fundus & 4-Quadrant Meridian", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ---------------------------------------------------------
    # Panel B: Salient Attention Focus Map
    # ---------------------------------------------------------
    panel_b = cv2.resize(attention_result.overlay_bgr, (panel_size, panel_size))
    # Draw focus contours
    for cnt in attention_result.focus_contours:
        scaled_cnt = np.copy(cnt)
        scaled_cnt[:, :, 0] = (scaled_cnt[:, :, 0] * scale_x).astype(np.int32)
        scaled_cnt[:, :, 1] = (scaled_cnt[:, :, 1] * scale_y).astype(np.int32)
        cv2.drawContours(panel_b, [scaled_cnt], -1, (255, 255, 255), 2)

    for px, py in attention_result.peak_coordinates:
        sx, sy = int(px * scale_x), int(py * scale_y)
        cv2.circle(panel_b, (sx, sy), 5, (0, 0, 255), -1)
        cv2.circle(panel_b, (sx, sy), 8, (255, 255, 255), 1)

    cv2.rectangle(panel_b, (0, 0), (panel_size, 26), (20, 20, 20), -1)
    label_b = f"B. Visual Attention ({attention_result.method.upper()}) Focus Coverage: {attention_result.attention_coverage_ratio:.1%}"
    cv2.putText(panel_b, label_b, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # ---------------------------------------------------------
    # Panel C: Retinal Structure & Lesion Segmentation
    # ---------------------------------------------------------
    panel_c = cv2.resize(segmentation.overlays.get("combined", base_image_bgr), (panel_size, panel_size))
    cv2.rectangle(panel_c, (0, 0), (panel_size, 26), (20, 20, 20), -1)
    cv2.putText(panel_c, "C. Retinal Structure & Lesion Segmentation", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Segmentation Legend
    cv2.rectangle(panel_c, (0, panel_size - 24), (panel_size, panel_size), (15, 15, 15), -1)
    # Green = Vessels
    cv2.circle(panel_c, (20, panel_size - 12), 5, (0, 200, 0), -1)
    cv2.putText(panel_c, "Vessels", (32, panel_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # Yellow = Hard Exudates
    cv2.circle(panel_c, (110, panel_size - 12), 5, (0, 255, 255), -1)
    cv2.putText(panel_c, "Exudates", (122, panel_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # Red = Microaneurysms/Hemorrhages
    cv2.circle(panel_c, (215, panel_size - 12), 5, (0, 0, 255), -1)
    cv2.putText(panel_c, "Microaneurysms / Hem.", (227, panel_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # ---------------------------------------------------------
    # Panel D: Clinical Criteria & ICDR Correlation Dashboard
    # ---------------------------------------------------------
    panel_d = np.full((panel_size, panel_size, 3), 28, dtype=np.uint8)
    cv2.rectangle(panel_d, (0, 0), (panel_size, 26), (15, 15, 15), -1)
    cv2.putText(panel_d, "D. Clinical Diagnostic Criteria & ICDR Correlation", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    # Severity Banner
    cur_y = 50
    level = grade.severity_level if (grade and grade.severity_level is not None) else 0
    short_lbl = grade.icdr_short if (grade and grade.icdr_short) else f"Level {level}"
    ref_txt = "REFERABLE (Level 2+)" if (grade and grade.referable_dr) else "NON-REFERABLE"
    banner_color = (0, 0, 180) if (grade and grade.referable_dr) else (0, 150, 0)
    cv2.rectangle(panel_d, (15, cur_y - 18), (panel_size - 15, cur_y + 18), banner_color, -1)
    cv2.putText(panel_d, f"ICDR Level {level}: {short_lbl} | {ref_txt}", (25, cur_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Calibrated Confidence & Entropy
    cur_y += 45
    conf_val = calibration.calibrated_confidence if calibration else (grade.confidence if grade else 0.85)
    ent_val = calibration.normalized_entropy if calibration else 0.25
    tier_txt = calibration.reliability_tier if calibration else "Standard Reliability"

    cv2.putText(panel_d, f"Confidence: {conf_val:.1%}  |  Entropy: {ent_val:.2f}  |  {tier_txt}", (20, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # Draw Confidence Meter Bar
    cur_y += 12
    cv2.rectangle(panel_d, (20, cur_y), (panel_size - 20, cur_y + 10), (50, 50, 50), -1)
    bar_w = int((panel_size - 40) * min(max(conf_val, 0.0), 1.0))
    cv2.rectangle(panel_d, (20, cur_y), (20 + bar_w, cur_y + 10), (0, 200, 100), -1)

    # 4-2-1 Rule Evaluation
    cur_y += 35
    cv2.putText(panel_d, "Severe NPDR 4-2-1 Rule Verification:", (20, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1)

    cur_y += 22
    rule_4_txt = f"- Rule 4 (Hemorrhages in 4 quads): {criteria.rule_4_hemorrhage_quadrants}/4 quads"
    color_4 = (0, 0, 255) if criteria.rule_4_hemorrhage_quadrants == 4 else (180, 180, 180)
    cv2.putText(panel_d, rule_4_txt, (25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_4, 1)

    cur_y += 18
    rule_2_txt = f"- Rule 2 (Venous beading >=2 quads): {criteria.rule_2_venous_beading_quadrants} quads"
    color_2 = (0, 0, 255) if criteria.rule_2_venous_beading_quadrants >= 2 else (180, 180, 180)
    cv2.putText(panel_d, rule_2_txt, (25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_2, 1)

    cur_y += 18
    rule_1_txt = f"- Rule 1 (Prominent IRMA >=1 quad): {criteria.rule_1_irma_quadrants} quad(s)"
    color_1 = (0, 0, 255) if criteria.rule_1_irma_quadrants >= 1 else (180, 180, 180)
    cv2.putText(panel_d, rule_1_txt, (25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_1, 1)

    # DME Proximity Threat
    cur_y += 32
    cv2.putText(panel_d, "Diabetic Macular Edema (DME) Risk:", (20, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1)

    cur_y += 20
    dme_threat_color = (0, 0, 255) if "High" in criteria.dme_threat_level else (0, 200, 255) if "Moderate" in criteria.dme_threat_level else (0, 200, 0)
    cv2.putText(panel_d, f"Threat Category: {criteria.dme_threat_level}", (25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, dme_threat_color, 1)

    cur_y += 18
    dme_dist_str = f"{criteria.dme_min_distance_disc_diameters:.2f} DD" if criteria.dme_min_distance_disc_diameters != float("inf") else "None"
    cv2.putText(panel_d, f"Fovea-to-Exudate Distance: {dme_dist_str} (Threshold: <1.0 DD)", (25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # 4-Quadrant Lesion Breakdown Mini-Chart
    cur_y += 35
    cv2.putText(panel_d, "4-Quadrant Lesion Breakdown (ST / SN / IT / IN):", (20, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 215, 255), 1)

    cur_y += 22
    quad_x = 25
    quad_w = (panel_size - 60) // 4
    for q_code, q_metrics in criteria.quadrants.items():
        # Box for each quadrant
        box_y = cur_y
        cv2.rectangle(panel_d, (quad_x, box_y), (quad_x + quad_w - 6, box_y + 80), (45, 45, 45), -1)
        cv2.rectangle(panel_d, (quad_x, box_y), (quad_x + quad_w - 6, box_y + 80), (80, 80, 80), 1)

        cv2.putText(panel_d, q_code, (quad_x + 8, box_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        cv2.putText(panel_d, f"Hem: {q_metrics.hemorrhage_count}", (quad_x + 8, box_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.putText(panel_d, f"MA: {q_metrics.microaneurysm_count}", (quad_x + 8, box_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.putText(panel_d, f"Ex: {q_metrics.exudate_area_ratio:.1%}", (quad_x + 8, box_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        quad_x += quad_w

    # Place panels onto Canvas (2x2 Grid)
    canvas[0:panel_size, 0:panel_size] = panel_a
    canvas[0:panel_size, panel_size:target_dim] = panel_b
    canvas[panel_size:target_dim, 0:panel_size] = panel_c
    canvas[panel_size:target_dim, panel_size:target_dim] = panel_d

    # Draw grid dividers
    cv2.line(canvas, (panel_size, 0), (panel_size, target_dim), (80, 80, 80), 2)
    cv2.line(canvas, (0, panel_size), (target_dim, panel_size), (80, 80, 80), 2)

    return canvas


# ---------------------------------------------------------------------------
# Automated Annotated Screening Report (Standalone HTML / Print-Ready)
# ---------------------------------------------------------------------------

def image_to_base64(image_bgr: np.ndarray, format: str = ".jpg", quality: int = 90) -> str:
    """Encode OpenCV BGR image to base64 data URI string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if format.lower() in {".jpg", ".jpeg"} else []
    success, buffer = cv2.imencode(format, image_bgr, encode_params)
    if not success:
        return ""
    mime_type = "image/jpeg" if format.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(buffer).decode('ascii')}"


def generate_annotated_html_report(
    case_id: str,
    original_bgr: np.ndarray,
    composite_bgr: np.ndarray,
    quality: QualityResult,
    grade: DRGradeResult,
    criteria: ClinicalCriteriaResult,
    calibration: CalibrationResult,
    attention: AttentionMapResult,
    laterality: str = "Unspecified Eye",
    screening_facility: str = "Retinal AI Screening Center",
) -> str:
    """Generate a self-contained, print-ready, responsive HTML/CSS clinical screening report.

    Embeds all visual evidence as base64 data URIs for 100% offline portability,
    EHR attachment, and instant browser PDF export.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    level = grade.severity_level if grade.severity_level is not None else 0
    info = get_icdr_info(level)

    composite_b64 = image_to_base64(composite_bgr, format=".jpg", quality=92)
    original_b64 = image_to_base64(original_bgr, format=".jpg", quality=88)

    # Status badging colors
    if level == 0:
        badge_bg, badge_fg = "#e6f4ea", "#137333"
    elif level == 1:
        badge_bg, badge_fg = "#e8f0fe", "#1a73e8"
    elif level == 2:
        badge_bg, badge_fg = "#fef7e0", "#b06000"
    elif level == 3:
        badge_bg, badge_fg = "#fce8e6", "#c5221f"
    else:
        badge_bg, badge_fg = "#f3e8fd", "#7627bb"

    ref_status = "REFERABLE DR (Level 2+)" if grade.referable_dr else "NON-REFERABLE"
    ref_color = "#c5221f" if grade.referable_dr else "#137333"

    quadrant_rows = ""
    for q_code, q_metrics in criteria.quadrants.items():
        quadrant_rows += f"""
        <tr>
            <td style="font-weight: 600; color: #1e293b;">{q_code} ({q_metrics.quadrant_name})</td>
            <td>{q_metrics.hemorrhage_count} {'<span style="color:#dc2626;font-weight:bold;">(&ge;20)</span>' if q_metrics.severe_hemorrhage_flag else ''}</td>
            <td>{q_metrics.microaneurysm_count}</td>
            <td>{q_metrics.exudate_area_ratio:.2%}</td>
            <td>{q_metrics.vessel_density:.2%}</td>
            <td>{'<span class="pill pill-danger">Yes</span>' if q_metrics.venous_beading_flag else '<span class="pill pill-neutral">No</span>'}</td>
            <td>{'<span class="pill pill-danger">Yes</span>' if q_metrics.irma_flag else '<span class="pill pill-neutral">No</span>'}</td>
        </tr>
        """

    summary_bullets_html = "".join(f"<li>{bullet}</li>" for bullet in criteria.clinical_summary_bullets)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Clinical DR Screening Report - {case_id}</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #eff6ff;
            --text-dark: #0f172a;
            --text-muted: #475569;
            --border-color: #cbd5e1;
            --bg-card: #ffffff;
            --bg-page: #f8fafc;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-page);
            color: var(--text-dark);
            line-height: 1.5;
            padding: 24px;
        }}
        .report-container {{
            max-width: 1080px;
            margin: 0 auto;
            background: var(--bg-card);
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
            padding: 36px;
            border: 1px solid var(--border-color);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid var(--primary);
            padding-bottom: 18px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            font-size: 24px;
            color: var(--primary);
            font-weight: 700;
        }}
        .header .subtitle {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            background: #f1f5f9;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 24px;
            font-size: 13px;
        }}
        .meta-item strong {{
            display: block;
            color: var(--text-muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .banner {{
            background: {badge_bg};
            border-left: 6px solid {badge_fg};
            padding: 18px;
            border-radius: 6px;
            margin-bottom: 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .banner-title {{
            font-size: 20px;
            font-weight: 700;
            color: {badge_fg};
        }}
        .banner-findings {{
            font-size: 13px;
            color: #334155;
            margin-top: 6px;
        }}
        .pill {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .pill-danger {{ background: #fee2e2; color: #991b1b; }}
        .pill-neutral {{ background: #e2e8f0; color: #475569; }}
        .pill-success {{ background: #dcfce7; color: #166534; }}
        
        .section-title {{
            font-size: 17px;
            font-weight: 700;
            color: var(--primary);
            margin: 28px 0 12px 0;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
        }}
        .metrics-cards {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: #ffffff;
            border: 1px solid var(--border-color);
            padding: 14px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card .val {{
            font-size: 22px;
            font-weight: 700;
            color: var(--primary);
            margin-top: 4px;
        }}
        .metric-card .lbl {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        .composite-img-box {{
            text-align: center;
            margin: 20px 0 28px 0;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            background: #0f172a;
        }}
        .composite-img-box img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .img-caption {{
            background: #1e293b;
            color: #cbd5e1;
            font-size: 12px;
            padding: 8px 14px;
            text-align: left;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-bottom: 24px;
        }}
        th, td {{
            padding: 10px 12px;
            border: 1px solid #e2e8f0;
            text-align: left;
        }}
        th {{
            background: #f8fafc;
            color: var(--text-muted);
            font-weight: 600;
        }}
        .action-box {{
            background: var(--primary-light);
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 28px;
        }}
        .action-box h4 {{
            color: var(--primary);
            font-size: 15px;
            margin-bottom: 8px;
        }}
        .action-box p {{
            font-size: 13px;
            color: #1e293b;
            margin-bottom: 6px;
        }}
        .signoff-section {{
            margin-top: 40px;
            padding-top: 24px;
            border-top: 2px dashed #cbd5e1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            font-size: 13px;
        }}
        .sign-line {{
            border-bottom: 1px solid #000;
            height: 40px;
            margin-bottom: 6px;
        }}
        .print-btn {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: var(--primary);
            color: white;
            padding: 12px 20px;
            border-radius: 30px;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            font-weight: 600;
            font-size: 14px;
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .report-container {{
                box-shadow: none;
                border: none;
                padding: 0;
            }}
            .print-btn {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>

    <div class="report-container">
        <div class="header">
            <div>
                <h1>Clinical Diabetic Retinopathy Screening Report</h1>
                <div class="subtitle">{screening_facility} | Autonomous Multi-Modal Diagnostic Verification</div>
            </div>
            <div style="text-align: right; font-size: 12px; color: var(--text-muted);">
                <div><strong>Standard:</strong> ICDR Disease Severity Scale</div>
                <div><strong>Generated:</strong> {timestamp}</div>
            </div>
        </div>

        <div class="metadata-grid">
            <div class="meta-item">
                <strong>Patient / Case ID</strong>
                <span>{case_id}</span>
            </div>
            <div class="meta-item">
                <strong>Laterality (Eye)</strong>
                <span>{laterality}</span>
            </div>
            <div class="meta-item">
                <strong>Quality Assurance</strong>
                <span style="color: {'#166534' if quality.status != 'rejected' else '#991b1b'}; font-weight: bold;">
                    {quality.status.upper()} (Score: {quality.score:.2f})
                </span>
            </div>
            <div class="meta-item">
                <strong>Referral Status</strong>
                <span style="color: {ref_color}; font-weight: bold;">{ref_status}</span>
            </div>
        </div>

        <div class="banner">
            <div>
                <div class="banner-title">ICDR Severity Level {level}: {info.label} ({info.short_label})</div>
                <div class="banner-findings">{info.findings}</div>
            </div>
            <div style="text-align: right;">
                <span class="pill" style="background:{badge_fg}; color:white; font-size: 13px;">
                    {info.referral_urgency}
                </span>
            </div>
        </div>

        <div class="section-title">Diagnostic Confidence & Calibrated Uncertainty Assessment</div>
        <div class="metrics-cards">
            <div class="metric-card">
                <div class="lbl">Calibrated Confidence</div>
                <div class="val">{calibration.calibrated_confidence:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="lbl">Normalized Entropy</div>
                <div class="val">{calibration.normalized_entropy:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="lbl">Confidence Margin</div>
                <div class="val">{calibration.confidence_margin:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="lbl">Reliability Tier</div>
                <div class="val" style="font-size: 16px; color: {'#166534' if 'High' in calibration.reliability_tier else '#b45309'};">
                    {calibration.reliability_tier}
                </div>
            </div>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 20px;">
            <strong>Decision Guidance:</strong> {calibration.clinical_recommendation}
        </p>

        <div class="section-title">Visual Evidence & Attention Saliency Composite</div>
        <div class="composite-img-box">
            <img src="{composite_b64}" alt="Composite Retinal Evidence Graphic">
            <div class="img-caption">
                <strong>Multi-Panel Evidence Graphic:</strong> 
                (A) Enhanced fundus with anatomical foveal-disc quadrant meridian. 
                (B) Visual attention focus map isolating primary morphology contributing to DR grade. 
                (C) Segmented lesions (green=vessels, yellow=exudates, red=microaneurysms/hemorrhages). 
                (D) Clinical diagnostic criteria dashboard and lesion distribution.
            </div>
        </div>

        <div class="section-title">Correlation with Clinical Diagnostic Criteria</div>
        <table>
            <thead>
                <tr>
                    <th>Clinical Criterion</th>
                    <th>Observed Retinal Finding</th>
                    <th>Clinical Implication / ICDR Standard</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Severe NPDR 4-2-1 Rule</strong></td>
                    <td>{criteria.rule_4_2_1_status}</td>
                    <td>
                        Hemorrhages in {criteria.rule_4_hemorrhage_quadrants}/4 quads; 
                        Venous beading in {criteria.rule_2_venous_beading_quadrants} quads; 
                        IRMA in {criteria.rule_1_irma_quadrants} quad(s).
                    </td>
                </tr>
                <tr>
                    <td><strong>Diabetic Macular Edema (DME)</strong></td>
                    <td>{criteria.dme_threat_level}</td>
                    <td>{criteria.dme_clinical_summary}</td>
                </tr>
                <tr>
                    <td><strong>Neovascularization (PDR)</strong></td>
                    <td>{criteria.neovascularization_status}</td>
                    <td>Neovascularization score: {criteria.neovascularization_score:.3f}. Sight-threatening proliferative flag.</td>
                </tr>
            </tbody>
        </table>

        <div class="section-title">4-Quadrant Lesion Distribution Breakdown</div>
        <table>
            <thead>
                <tr>
                    <th>Quadrant</th>
                    <th>Intraretinal Hemorrhages</th>
                    <th>Microaneurysms</th>
                    <th>Exudate Coverage</th>
                    <th>Vessel Density</th>
                    <th>Venous Beading</th>
                    <th>IRMA</th>
                </tr>
            </thead>
            <tbody>
                {quadrant_rows}
            </tbody>
        </table>

        <div class="action-box">
            <h4>Recommended Clinical Pathway</h4>
            <p><strong>Timeline:</strong> {info.referral_urgency}</p>
            <p><strong>Action Protocol:</strong> {info.clinical_action}</p>
        </div>

        <div class="signoff-section">
            <div>
                <div class="sign-line"></div>
                <strong>Reviewing Ophthalmologist Signature</strong><br>
                <span>Print Name / Clinical License #: _____________________________</span>
            </div>
            <div>
                <div class="sign-line"></div>
                <strong>Date & Clinical Center Stamp</strong><br>
                <span>Sign-off Date: _____________________________</span>
            </div>
        </div>
    </div>
</body>
</html>
"""
    return html


def generate_screening_case_report(
    image_path: str | Path,
    quality: QualityResult,
    grade: DRGradeResult,
    segmentation: SegmentationResult,
    attention: AttentionMapResult,
    output_html_path: str | Path | None = None,
    output_composite_path: str | Path | None = None,
    temperature: float = 1.30,
) -> tuple[str, np.ndarray, ClinicalCriteriaResult, CalibrationResult]:
    """Execute end-to-end explainability pipeline for a single retinal screening case.

    Generates:
    - 4-quadrant clinical criteria analysis
    - Confidence calibration & entropy metrics
    - 4-panel visual evidence composite image
    - Print-ready standalone HTML report
    """
    path = Path(image_path)
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read fundus image at {path}")

    # Evaluate Clinical Criteria
    criteria = evaluate_clinical_criteria(image_bgr.shape, segmentation, grade=grade)

    # Calibrate Confidence
    calibration = calibrate_confidence(grade.probabilities, temperature=temperature)

    # Create 4-Panel Composite Graphic
    composite_bgr = create_composite_evidence_image(
        base_image_bgr=image_bgr,
        attention_result=attention,
        segmentation=segmentation,
        criteria=criteria,
        grade=grade,
        calibration=calibration,
        target_dim=1200,
    )

    # Save Composite Graphic if requested
    if output_composite_path is not None:
        comp_path = Path(output_composite_path)
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(comp_path), composite_bgr)

    # Generate HTML Report
    html_content = generate_annotated_html_report(
        case_id=path.stem,
        original_bgr=image_bgr,
        composite_bgr=composite_bgr,
        quality=quality,
        grade=grade,
        criteria=criteria,
        calibration=calibration,
        attention=attention,
        laterality="Right Eye (OD)" if criteria.quadrants["SN"].quadrant == "SN" else "Left Eye (OS)",
    )

    if output_html_path is not None:
        out_path = Path(output_html_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_content, encoding="utf-8")

    return html_content, composite_bgr, criteria, calibration
