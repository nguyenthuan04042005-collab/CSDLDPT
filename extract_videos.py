"""Extract Chapter 3 A-F for each MP4. Run with --help for pilot/full options."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from itertools import groupby
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import cv2
import numpy as np

from features_af import (GROUPS, PairData, appearance, changes, feature_names, reference_motion,
                         regional_motion, runs, summarize)
from segmentation import PersonSegmenter, refine_silhouette

ROOT = Path(__file__).resolve().parent
PILOT = ("boxing/person01_boxing_d1.mp4", "boxing/person16_boxing_d3.mp4",
         "handclapping/person01_handclapping_d2.mp4", "handclapping/person16_handclapping_d4.mp4",
         "handwaving/person01_handwaving_d3.mp4", "handwaving/person16_handwaving_d1.mp4",
         "jogging/person01_jogging_d4.mp4", "jogging/person16_jogging_d2.mp4",
         "running/person01_running_d1.mp4", "running/person16_running_d3.mp4",
         "walking/person01_walking_d2.mp4", "walking/person16_walking_d4.mp4")


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def atomic_write(path, content):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def save_npz(path, **arrays):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(cfg, model):
    global SETTINGS, SEGMENTER
    SETTINGS = cfg
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    SEGMENTER = PersonSegmenter(model, cfg["segmentation"])


def read_frames(path, cfg):
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError("cannot_open_video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or abs(fps-cfg["fps"]) > cfg["fps_tolerance"]:
            raise ValueError(f"unsupported_fps: {fps}")
        while True:
            ok, color = capture.read()
            if not ok:
                break
            if color.shape[:2] != (cfg["height"], cfg["width"]):
                raise ValueError(f"unexpected_frame_size: {color.shape[:2]}")
            frames.append(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    if declared > 0 and len(frames) != declared:
        raise ValueError(f"incomplete_decode: {len(frames)} of {declared}")
    if len(frames) < 2:
        raise ValueError(f"too_few_frames: {len(frames)}")
    return frames, fps


def save_png(path, array):
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise OSError(f"Cannot encode {path}")
    encoded.tofile(path)


def farneback(gray, following, cfg):
    return cv2.calcOpticalFlowFarneback(gray, following, None, **cfg["farneback"])


def debug_images(directory, frames, masks, boxes, data, cfg):
    """Three valid pairs distributed over the visible action, with exact frame indexes."""
    directory.mkdir(parents=True, exist_ok=True)
    indices = np.flatnonzero(data.good)
    if not indices.size:
        indices = np.arange(data.n)
    selected = np.unique(indices[np.rint((len(indices)-1)*np.array([.2, .5, .8])).astype(int)])
    rows = []
    for idx in selected:
        t = int(idx)
        gray, mask, box = frames[t], masks[t], boxes[t]
        flow = farneback(gray, frames[t+1], cfg)
        mag = np.hypot(flow[...,0], flow[...,1])
        # Fixed 4 px/frame visualization scale makes brightness comparable across images.
        hue = np.mod(np.arctan2(flow[...,1], flow[...,0]), 2*np.pi) * (180/(2*np.pi))
        hsv = np.dstack([hue, np.full(gray.shape, 255), np.minimum(255, mag*255/4)]).astype(np.uint8)
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        input_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay = input_bgr.copy()
        tint = overlay.copy()
        tint[mask != 0] = (40, 180, 35)
        overlay = cv2.addWeighted(tint, .48, overlay, .52, 0)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(overlay, (x,y), (x+w-1,y+h-1), (0,200,255), 1)
        prefix = f"t{t:04d}"
        save_png(directory / f"{prefix}_mask.png", mask*255)
        save_png(directory / f"{prefix}_flow.png", flow_bgr)
        row = np.hstack([input_bgr, overlay, cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR), flow_bgr])
        row = cv2.copyMakeBorder(row, 18, 0, 0, 0, cv2.BORDER_CONSTANT)
        for col, label in enumerate((f"t={t}  input", "mask + box", "binary mask", "flow: color=direction")):
            cv2.putText(row, label, (col*gray.shape[1]+3,13), cv2.FONT_HERSHEY_SIMPLEX, .38, (235,235,235), 1, cv2.LINE_AA)
        rows.append(row)
    save_png(directory / "preview.png", np.vstack(rows))
    # Remove diagnostics from earlier trials if a different frame was selected.
    keep = {f"t{int(t):04d}_{kind}.png" for t in selected for kind in ("mask", "flow")}
    for kind in ("mask", "flow"):
        for old in directory.glob(f"t*_{kind}.png"):
            if old.name.partition("_")[0][1:].isdigit() and old.name not in keep:
                old.unlink()
    return selected.tolist()


def reason_intervals(reasons):
    spans = []
    offset = 0
    for reason, entries in groupby(reasons):
        length = sum(1 for _ in entries)
        if reason:
            spans.append([offset, offset + length, reason])
        offset += length
    return spans


def dimension_reasons(valid, names, info, cfg):
    reasons = {}
    length = info["longest_valid_pairs"]
    for name, ok in zip(names, valid):
        if ok:
            continue
        group = name[0]
        if group == "F":
            reason = "no_observed_F_cell"
        elif not info["valid_pairs"]:
            reason = "no_valid_G_pairs"
        elif group == "A":
            reason = "no_adjacent_valid_pairs"
        elif group == "C":
            reason = "undefined_pearson"
        elif group == "D":
            _, row_name, measure = name.split("_", 2)
            row = int(row_name[3:])
            if measure.startswith("quarter"):
                reason = "quarter_has_no_valid_pairs"
            elif measure.startswith("lag") and length < max(cfg["temporal"]["min_run"], 2*int(measure[3:])):
                reason = "run_too_short_for_lag"
            elif measure.startswith("peak"):
                reason = info["peak_status"][row]
            elif info["peak_status"][row] == "low_detrended_variance":
                reason = "low_detrended_variance"
            else:
                reason = "undefined_pearson"
        else:
            raise ArithmeticError(f"Unexpected unavailable feature: {name}")
        reasons[name] = reason
    return reasons


def extract(path, relative_id, cfg, segmenter, debug_dir=None):
    frames, fps = read_frames(path, cfg["input"])
    refine = cfg["segmentation"]["refine"]
    sample = np.rint(np.linspace(0, len(frames)-1, min(len(frames), refine["background_samples"]))).astype(int)
    background = cv2.GaussianBlur(np.median(np.stack([frames[i] for i in sample]), axis=0).astype(np.uint8), (3,3), 0)
    data = PairData(len(frames)-1)
    previous = None
    boxes, masks, reasons, areas, added = [], [], [], [], []
    empty_resized = []
    for t in range(data.n):
        gray, following = frames[t:t+2]
        mask, box = segmenter.frame(gray, t, previous)
        previous = box
        seed_area = cv2.countNonZero(mask)
        mask, box = refine_silhouette(gray, background, mask, box, refine)
        boxes.append(box)
        if debug_dir:
            masks.append(mask)
        data.change[t] = changes(gray, following, box, cfg["difference"])
        if box is None:
            reasons.append("no_person_mask")
            continue
        areas.append(cv2.countNonZero(mask))
        added.append((areas[-1]-seed_area)/areas[-1])
        flow = farneback(gray, following, cfg)
        ref, reason = reference_motion(flow, mask, box, cfg["reference"])
        reasons.append(reason)
        if ref is None:
            continue
        x, y, w, h = box
        person = mask[y:y+h, x:x+w]
        data.a[t], relative, data.center[t], data.height[t] = ref
        data.good[t] = True
        data.aspect[t] = w/h
        regional_motion(relative, person, data, t, cfg["epsilon"])
        data.shape[t], has_tiny_mask = appearance(gray[y:y+h, x:x+w], person)
        if not has_tiny_mask:
            empty_resized.append(t)
    vector, valid, run, peak_status = summarize(data, cfg)
    ok_count = int(data.good.sum())
    finite_f = np.isfinite(data.change)
    indices_f = finite_f.any(axis=1)
    height, width = frames[0].shape
    at_border = sum(1 for r in boxes if r and (r[0]==0 or r[1]==0 or r[0]+r[2]==width or r[1]+r[3]==height))
    info = {
        "video_id": relative_id, "status": "complete" if valid.all() else ("partial" if valid.any() else "unavailable"),
        "frames": len(frames), "pairs": data.n, "fps": fps, "width": width, "height": height,
        "valid_pairs": ok_count, "valid_ratio": ok_count/data.n,
        "valid_group_pair_ratios": {g: (float(indices_f.mean()) if g=="F" else ok_count/data.n) for g in GROUPS},
        "longest_valid_run": list(run), "longest_valid_pairs": run[1]-run[0],
        "invalid_G_intervals": reason_intervals(reasons), "adjacent_valid_pairs": int((data.good[1:] & data.good[:-1]).sum()),
        "visible_pairs": len(areas), "border_pairs": at_border,
        "median_mask_pixels": float(np.median(areas)) if areas else None,
        "median_fraction_added_to_final_mask": float(np.median(added)) if added else None,
        "median_box_height": float(np.median([b[3] for b in boxes if b])) if areas else None,
        "peak_status": peak_status, "B_cell_support_counts": data.b_support.sum(axis=0).tolist(),
        "E_empty_resized_mask_frames": empty_resized, "F_cell_valid_counts": finite_f.sum(axis=0).tolist(),
        "F_invalid_intervals_by_cell": [runs(~v) for v in finite_f.T],
        "F_invalid_box_intervals": reason_intervals(
            "no_person_box" if box is None else ("box_below_min_height" if box[3]<cfg["difference"]["min_height"] else "")
            for box in boxes),
        "valid_dimensions": int(valid.sum())}
    names = feature_names(cfg["temporal"]["lags"])
    info["unavailable"] = dimension_reasons(valid, names, info, cfg)
    if debug_dir:
        info["preview_pair_indices"] = debug_images(debug_dir, frames, masks, boxes, data, cfg)
    return vector, valid, info


def process_one(task):
    path, relative_id, debug_dir = task
    try:
        vector, valid, info = extract(path, relative_id, SETTINGS, SEGMENTER, debug_dir)
    except (ValueError, cv2.error, OSError, ArithmeticError) as error:
        # Preserve a file-level failure; never mark the placeholder as measured.
        vector = np.zeros(324, np.float32)
        valid = np.zeros(324, bool)
        info = {"video_id": relative_id, "status": "error", "error": str(error),
                "valid_dimensions": 0, "unavailable": {"*": "file_error"}}
    return relative_id, vector, valid, info


def export(out, rows, cfg, record):
    names = feature_names(cfg["temporal"]["lags"])
    ids = sorted(rows)
    infos = [rows[key][2] for key in ids]
    vectors = np.stack([rows[key][0] for key in ids])
    valid = np.stack([rows[key][1] for key in ids])
    save_npz(out/"features.npz", video_ids=np.array(ids), feature_names=np.array(names),
             vectors=vectors, valid=valid)
    atomic_write(out/"config.json", json_text(record))
    atomic_write(out/"quality.jsonl", "".join(json.dumps(v, ensure_ascii=False, allow_nan=False)+"\n" for v in infos))
    with (out/"feature_schema.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index_0based", "group", "name"])
        writer.writerows((i, name[0], name) for i, name in enumerate(names))
    with (out/"manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["video_id", "status", "frames", "fps", "valid_pairs", "valid_ratio", "longest_valid_pairs", "valid_dimensions", "input_sha256"]
        writer = csv.DictWriter(stream, fieldnames=["row"] + fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dict(v, row=i) for i, v in enumerate(infos))
    with (out/"exceptions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id","feature","reason","detail"])
        for info in infos:
            for key, reason in info["unavailable"].items():
                writer.writerow([info["video_id"], key, reason, info.get("error", "")])
    summary = {"videos": len(rows), "shape": list(vectors.shape), "status": dict(Counter(i["status"] for i in infos)),
               "decoded_frames": sum(i.get("frames", 0) for i in infos),
               "valid_dimensions_total": int(valid.sum()),
               "invalid_dimensions_total": int(valid.size-valid.sum()),
               "invalid_by_group": {g: int((~valid[:,[s[0]==g for s in names]]).sum()) for g in GROUPS},
               "exception_reasons": dict(Counter(r for info in infos for r in info["unavailable"].values()))}
    atomic_write(out/"summary.json", json_text(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=ROOT/"Dataset", type=Path)
    parser.add_argument("--config", default=ROOT/"extraction_config.json", type=Path)
    parser.add_argument("--mode", choices=("pilot","full"), required=True)
    parser.add_argument("--out", type=Path, help="Default: results/{pilot,full}")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    model = (args.config.resolve().parent / cfg["segmentation"]["model"]).resolve()
    if sha(model) != cfg["segmentation"]["sha256"]:
        parser.error("Unexpected model checksum")
    dataset = args.dataset.resolve()
    out = (args.out or ROOT/"results"/args.mode).resolve()
    files = list(PILOT) if args.mode == "pilot" else sorted(p.relative_to(dataset).as_posix() for p in dataset.rglob("*") if p.is_file() and p.suffix.lower()==".mp4")
    if not files or args.workers < 1:
        parser.error("Need video files and at least one worker")
    record = {"extraction": cfg, "python": sys.version.split()[0], "platform": platform.platform(),
              "libraries": {p: version(p) for p in ("numpy","scipy","opencv-python-headless")},
              "code_sha256": {p: sha(ROOT/p) for p in ("features_af.py","segmentation.py","extract_videos.py")}}
    stamp = hashlib.sha256(json_text(record).encode()).hexdigest()[:16]
    cache = out/"checkpoint"/stamp
    cache.mkdir(parents=True, exist_ok=True)
    rows, todo, paths, hashes = {}, [], {}, {}
    for key in files:
        paths[key] = cache/(hashlib.sha256(key.encode()).hexdigest()[:24]+".npz")
        hashes[key] = sha(dataset/key)
        debug = out/"images"/Path(key).stem if args.mode=="pilot" else None
        if paths[key].exists() and (debug is None or (debug/"preview.png").exists()):
            with np.load(paths[key], allow_pickle=False) as saved:
                info = json.loads(str(saved["quality"]))
                if info["input_sha256"]==hashes[key] and info["status"]!="error":
                    rows[key] = saved["vector"], saved["valid"], info
                    continue
        todo.append((dataset/key,key,debug))
    print(f"{args.mode}: {len(files)} files, {len(todo)} to process, {args.workers} workers", flush=True)
    if todo:
        with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(cfg,model)) as pool:
            futures = [pool.submit(process_one, task) for task in todo]
            for future in as_completed(futures):
                key, vector, valid, info = future.result()
                info["input_sha256"] = hashes[key]
                rows[key] = vector, valid, info
                save_npz(paths[key], vector=vector, valid=valid, quality=np.array(json_text(info)))
                if args.mode=="pilot" or len(rows)%25==0 or info["status"]=="error":
                    print(f"{len(rows)}/{len(files)} {key}  {info['status']}  {int(valid.sum())}/324", flush=True)
    print(json_text(export(out, rows, cfg, record)), flush=True)


if __name__ == "__main__":
    main()
