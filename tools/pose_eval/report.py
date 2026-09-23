"""An offline report with a single source-video clock across model panels."""
import json
from pathlib import Path

from . import MODEL_IDS


def create_report(directory):
    directory = Path(directory)
    results = []
    for model_id in MODEL_IDS:
        path = directory / f"{model_id}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") != "complete" or result.get("schema_version") != 1:
                raise ValueError(f"Incomplete or unsupported result: {path}")
            if result["provenance"]["model_id"] != model_id:
                raise ValueError(f"Model ID does not match result filename: {path}")
            results.append(result)
    if not results:
        raise ValueError(f"No completed model results in {directory}")
    source = results[0]["source"]
    frame_sequence = [(f["frame_index"], f["timestamp_ms"], f["width"], f["height"]) for f in results[0]["frames"]]
    for result in results[1:]:
        if result.get("annotation_sha256") != results[0].get("annotation_sha256"):
            raise ValueError("Cannot compare results with different annotation files; rerun using the same labels")
        if any(result["source"][key] != source[key] for key in ("sha256", "clip_id", "start_s", "end_s")):
            raise ValueError("Cannot compare results with different source videos or clip ranges")
        if [(f["frame_index"], f["timestamp_ms"], f["width"], f["height"]) for f in result["frames"]] != frame_sequence:
            raise ValueError("Cannot compare results with different decoded frame sequences")
    payload = json.dumps(results, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = Path(__file__).with_name("report.html").read_text(encoding="utf-8")
    output = directory / "comparison.html"
    output.write_text(template.replace("__RESULTS_JSON__", payload), encoding="utf-8")
    return output
