from pathlib import Path
import inspect
import json
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agents.spatial_agent import SpatialAgent
from agents.spectral_agent import SpectralAgent
from agents.feature_agent import FeatureAgent
from agents.conflict_agent import ConflictAgent
from agents.uncertainty_agent import UncertaintyAgent
from agents.cognitive import CognitiveAgent, analyze_candidates, to_legacy_candidate

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "web_ui"
MODEL_PATH = PROJECT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

app = FastAPI(title="Camouflage Detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(OUTPUT_DIR)), name="results")


def call_with_supported_kwargs(func, **kwargs):
    """Pass only parameters accepted by a method, while preserving positional-only APIs."""
    signature = inspect.signature(func)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in signature.parameters.values())
    usable = kwargs if accepts_kwargs else {
        key: value for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(**usable)


def safe_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def score_from(mapping, keys):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = safe_float(mapping.get(key))
        if value is not None:
            return value
    return None


def top_region_candidates(cognitive_results, uncertainty_results, conflict_result,
                          image_width=None, image_height=None, iou_threshold=0.45):
    """
    Rank candidates by the available cognitive suspicion score, then suppress
    lower-scoring overlapping boxes (non-maximum suppression). This keeps the
    highest-valued candidate when two boxes overlap substantially.
    """
    candidates = cognitive_results or uncertainty_results or (
        conflict_result.get("candidates", []) if isinstance(conflict_result, dict) else []
    )
    ranked = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        posterior = candidate.get("bayes_posterior", {})
        score = score_from(candidate, ["bayes_suspicion", "suspicion_score", "confidence"])
        if score is None:
            score = score_from(posterior, ["H0"])
        if score is None:
            score = score_from(candidate, ["overall_evidence", "evidence_score", "score"])
        bbox = candidate.get("bbox")
        if bbox is None:
            bbox = candidate.get("box")
        if bbox is None:
            bbox = candidate.get("xyxy")
        if bbox is None:
            bbox = candidate.get("region_bbox")
        ranked.append({
            "candidate": candidate,
            "index": index + 1,
            "score": score,
            "bbox": bbox,
            "bbox_format": "xyxy" if candidate.get("xyxy") is not None else "xywh",
            "decision": candidate.get("bayes_decision", candidate.get("decision", "Candidate")),
            "hypothesis": candidate.get("bayes_hypothesis"),
            "uncertainty": score_from(candidate, ["uncertainty"]),
            "confidence": score_from(candidate, ["confidence"]),
            "posterior": posterior if isinstance(posterior, dict) else {},
        })

    ranked.sort(key=lambda item: item["score"] if item["score"] is not None else -1, reverse=True)

    # Convert coordinates before overlap filtering. Missing/invalid boxes are
    # kept in the score list for the table but cannot participate in NMS.
    if image_width and image_height:
        kept = []
        for item in ranked:
            raw = item["bbox"]
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            vals = [safe_float(v) for v in raw[:4]]
            if any(v is None for v in vals):
                continue
            x, y, a, b = vals
            if item["bbox_format"] == "xyxy":
                coords = (int(x), int(y), int(a), int(b))
            else:
                coords = (int(x), int(y), int(x + a), int(y + b))
            x1, y1, x2, y2 = coords
            x1 = max(0, min(image_width - 1, x1))
            y1 = max(0, min(image_height - 1, y1))
            x2 = max(0, min(image_width - 1, x2))
            y2 = max(0, min(image_height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            duplicate = False
            for prev in kept:
                px1, py1, px2, py2 = prev["_coords"]
                ix1, iy1 = max(x1, px1), max(y1, py1)
                ix2, iy2 = min(x2, px2), min(y2, py2)
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (px2 - px1) * (py2 - py1)
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou >= iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                item["_coords"] = (x1, y1, x2, y2)
                kept.append(item)

        for item in kept:
            item.pop("_coords", None)
        ranked = kept

    return ranked[:5]

def normalize_bbox(raw_bbox, image_width, image_height, bbox_format="xywh"):
    """Normalize xywh or xyxy coordinates and clamp them to image dimensions."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return None
    vals = [safe_float(v) for v in raw_bbox[:4]]
    if any(v is None for v in vals):
        return None
    x1, y1, a, b = vals
    if bbox_format == "xyxy":
        x2, y2 = a, b
    else:
        x2, y2 = x1 + a, y1 + b
    x1, y1 = int(round(x1)), int(round(y1))
    x2, y2 = int(round(x2)), int(round(y2))
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def make_final_image(rgb_path, ranked, output_path):
    image = cv2.imread(str(rgb_path))
    if image is None:
        raise ValueError("Could not read the uploaded RGB image.")
    height, width = image.shape[:2]
    drawable = 0
    for rank, item in enumerate(ranked, start=1):
        raw = item["bbox"]
        if raw is None:
            continue
        bbox = normalize_bbox(raw, width, height, item.get("bbox_format", "xywh"))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        # Use one consistent, high-contrast cyan color for every box and tag.
        # OpenCV uses BGR order: (255, 255, 0) is bright cyan.
        box_color = (255, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

        label = f"Point {rank}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

        # Place the tag just above the box where possible, otherwise inside it.
        label_y = y1 - 7
        if label_y - th - baseline < 0:
            label_y = min(height - baseline - 2, y1 + th + 7)
        label_top = max(0, label_y - th - 5)
        label_bottom = min(height - 1, label_y + baseline)
        label_right = min(width - 1, x1 + tw + 8)

        cv2.rectangle(
            image, (x1, label_top), (label_right, label_bottom), box_color, -1
        )
        # Dark text remains readable on the bright cyan tag.
        cv2.putText(
            image, label, (x1 + 4, label_y - 2), font, scale,
            (0, 0, 0), thickness, cv2.LINE_AA
        )
        drawable += 1
    cv2.imwrite(str(output_path), image)
    return drawable


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camouflage Detection</title>
<style>
:root{font-family:Arial,sans-serif;color:#172033;background:#f3f6fb}body{margin:0;padding:28px}
main{max-width:1050px;margin:auto;background:white;border:1px solid #dce4ef;border-radius:16px;padding:28px}
h1{margin-top:0}p{color:#52647c}.form{display:flex;flex-wrap:wrap;gap:14px;align-items:end}
label{display:block;font-weight:600;margin-bottom:7px}input{max-width:100%}
button{border:0;background:#174ea6;color:white;border-radius:8px;padding:12px 18px;font-weight:700;cursor:pointer}
button:disabled{opacity:.55}.result{margin-top:26px}img{max-width:100%;height:auto;border:1px solid #dce4ef;border-radius:10px}
table{width:100%;border-collapse:collapse;margin-top:14px}th,td{text-align:left;padding:11px;border-bottom:1px solid #e6ebf2}th{background:#f7f9fc}
.status{margin-top:16px;white-space:pre-wrap}.muted{font-size:13px;color:#66758a}.hidden{display:none}
</style></head><body><main>
<h1>Final Camouflage Detection</h1>
<p>Upload a matching RGB image and LWIR thermal image. The output highlights the five highest-valued candidate regions returned by the analysis pipeline.</p>
<form id="uploadForm"><div class="form">
<div><label for="rgb">RGB image</label><input id="rgb" name="rgb" type="file" accept="image/*" required></div>
<div><label for="lwir">LWIR image</label><input id="lwir" name="lwir" type="file" accept="image/*" required></div>
<button id="runBtn" type="submit">Run detection</button>
</div></form><div id="status" class="status"></div>
<section id="results" class="result hidden">
<h2>Final output</h2><img id="finalImage" alt="Final camouflage detection with up to five ranked candidate boxes">
<p id="boxNote" class="muted"></p>
<h2>Top 5 regions and layer-wise values</h2>
<table><thead><tr><th>Rank</th><th>Candidate / decision</th><th>Camouflage suspicion</th><th>Confidence</th><th>Uncertainty</th><th>Cognitive posterior H0</th></tr></thead><tbody id="rows"></tbody></table>
<p class="muted">Scores are model outputs, not measured detection precision. If no box is drawn, the agent candidate data did not contain a usable bounding box.</p>
</section>
</main><script>
const form=document.getElementById('uploadForm'),statusEl=document.getElementById('status'),btn=document.getElementById('runBtn');
form.addEventListener('submit',async e=>{e.preventDefault();statusEl.textContent='Running analysis…';btn.disabled=true;
try{const fd=new FormData();fd.append('rgb',document.getElementById('rgb').files[0]);fd.append('lwir',document.getElementById('lwir').files[0]);
const res=await fetch('/api/analyze',{method:'POST',body:fd});const data=await res.json();if(!res.ok)throw new Error(data.detail||'Analysis failed');
document.getElementById('results').classList.remove('hidden');document.getElementById('finalImage').src=data.final_image+'?t='+Date.now();
document.getElementById('boxNote').textContent=`Bounding boxes drawn: ${data.boxes_drawn} of ${data.top_regions.length}.`;
const rows=document.getElementById('rows');rows.innerHTML='';
for(const r of data.top_regions){const tr=document.createElement('tr');for(const val of [r.rank,r.decision,r.suspicion,r.confidence,r.uncertainty,r.posterior_h0]){const td=document.createElement('td');td.textContent=val??'—';tr.appendChild(td)}rows.appendChild(tr)}
statusEl.textContent='Analysis complete.';}catch(err){statusEl.textContent='Error: '+err.message;}finally{btn.disabled=false;}});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.post("/api/analyze")
async def analyze(rgb: UploadFile = File(...), lwir: UploadFile = File(...)):
    request_dir = OUTPUT_DIR / uuid.uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = request_dir / f"rgb_{Path(rgb.filename or 'image.png').name}"
    lwir_path = request_dir / f"lwir_{Path(lwir.filename or 'image.png').name}"
    final_path = request_dir / "final_detection.png"

    for upload, target in ((rgb, rgb_path), (lwir, lwir_path)):
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{upload.filename or 'Image'} is empty.")
        target.write_bytes(content)
        if cv2.imread(str(target), cv2.IMREAD_UNCHANGED) is None:
            raise HTTPException(status_code=400, detail=f"Could not decode image: {upload.filename}")

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail=f"YOLO model weights not found: {MODEL_PATH}")

    try:
        spatial = SpatialAgent(model_path=str(MODEL_PATH))
        spectral = SpectralAgent()
        feature = FeatureAgent()
        conflict = ConflictAgent()
        uncertainty = UncertaintyAgent()

        spatial_result = spatial.analyze(str(rgb_path))
        spectral_result = spectral.analyze(str(lwir_path))
        feature_result = feature.analyze(rgb_path=str(rgb_path), lwir_path=str(lwir_path))
        conflict_result = conflict.analyze(spatial_result, spectral_result, feature_result)
        uncertainty_results = uncertainty.analyze(conflict_result)

        cognitive_agent = CognitiveAgent()
        cognitive_results = analyze_candidates(cognitive_agent, uncertainty_results)
        for candidate in cognitive_results:
            try:
                baseline = cognitive_agent.reason(to_legacy_candidate(candidate))
                candidate["rule_decision"] = getattr(baseline, "decision", str(baseline))
            except Exception:
                candidate.setdefault("rule_decision", None)

        rgb_for_size = cv2.imread(str(rgb_path))
        if rgb_for_size is None:
            raise ValueError("Could not read uploaded RGB image.")
        img_height, img_width = rgb_for_size.shape[:2]
        ranked = top_region_candidates(
            cognitive_results,
            uncertainty_results,
            conflict_result,
            image_width=img_width,
            image_height=img_height,
            iou_threshold=0.45,
        )
        boxes_drawn = make_final_image(rgb_path, ranked, final_path)

        # Present actual values only. Avoid inventing a combined score.
        table_rows = []
        for rank, item in enumerate(ranked, start=1):
            score = item["score"]
            table_rows.append({
                "rank": rank,
                "decision": str(item["decision"]),
                "suspicion": (f"{score * 100:.1f}%" if score is not None and 0 <= score <= 1
                              else (f"{score:.1f}" if score is not None else None)),
                "confidence": (f"{item['confidence'] * 100:.1f}%" if item["confidence"] is not None
                               and 0 <= item["confidence"] <= 1 else item["confidence"]),
                "uncertainty": (f"{item['uncertainty'] * 100:.1f}%" if item["uncertainty"] is not None
                                and 0 <= item["uncertainty"] <= 1 else item["uncertainty"]),
                "posterior_h0": (f"{safe_float(item['posterior'].get('H0')) * 100:.1f}%"
                                 if safe_float(item["posterior"].get("H0")) is not None else None),
            })
        return {
            "final_image": f"/results/{request_dir.name}/final_detection.png",
            "boxes_drawn": boxes_drawn,
            "top_regions": table_rows,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
