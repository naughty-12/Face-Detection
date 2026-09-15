"""Probe: peak VRAM and throughput at several batch sizes (gap G17).

Background: this is a 6 GB card, but the training log recorded 11.1 GB peak -- the run was
over-subscribing into shared system memory, and throughput collapsed from 5.8 to 1.7 it/s
as it did. Before committing to a multi-hour retrain, measure what each batch size
actually costs: one epoch over a 5% slice of the training set, with CUDA peak memory
reset between runs so the numbers are per-configuration rather than process-wide.

Nothing is written into the repository tree: ultralytics output goes to artifacts/logs,
which is git-ignored.

Usage:
    python artifacts/logs/probe_vram.py            # batches 4, 6, 8
    python artifacts/logs/probe_vram.py --batches 2 4 8
"""
import argparse
import os
import sys
import time

import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, os.getcwd())
from src.paths import LOGS_DIR, MODEL_CONFIG_PATH, WIDER_YAML  # noqa: E402


def gib(nbytes):
    return nbytes / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="fraction of the training set to use (keeps each run short)")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    config = yaml.safe_load(open(MODEL_CONFIG_PATH, encoding="utf-8"))
    model_name = config["model_name"]
    cfg = config["training"]

    total_mem = gib(torch.cuda.get_device_properties(0).total_memory)
    print("=" * 78)
    print(f"VRAM probe — {torch.cuda.get_device_name(0)}")
    print(f"physical VRAM: {total_mem:.2f} GiB   imgsz={args.imgsz}   "
          f"fraction={args.fraction}   workers=0")
    print("=" * 78)

    rows = []
    for batch in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = YOLO(model_name)
        t0 = time.perf_counter()
        model.train(
            data=WIDER_YAML,
            epochs=1,
            batch=batch,
            imgsz=args.imgsz,
            workers=0,               # single process: no multiprocessing pipes
            fraction=args.fraction,
            optimizer=cfg["optimizer"],
            lr0=cfg["lr0"],
            lrf=cfg["lrf"],
            cos_lr=cfg["cos_lr"],
            amp=cfg["amp"],
            mosaic=cfg["mosaic"],
            fliplr=cfg["fliplr"],
            val=False,
            plots=False,
            verbose=False,
            project=LOGS_DIR,
            name=f"vram_probe_b{batch}",
            exist_ok=True,
        )
        elapsed = time.perf_counter() - t0

        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        over = peak_reserved > torch.cuda.get_device_properties(0).total_memory

        rows.append({
            "batch": batch,
            "peak_alloc": peak_alloc,
            "peak_reserved": peak_reserved,
            "seconds": elapsed,
            "over": over,
        })
        print(f"\n  batch={batch:<3} peak allocated {gib(peak_alloc):5.2f} GiB   "
              f"peak reserved {gib(peak_reserved):5.2f} GiB   "
              f"{'OVER-SUBSCRIBED' if over else 'fits in VRAM'}   "
              f"elapsed {elapsed:.1f}s")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"  {'batch':>6}{'peak alloc GiB':>18}{'peak reserved GiB':>20}{'elapsed s':>12}{'':>18}")
    for r in rows:
        flag = " > physical VRAM" if r["over"] else ""
        print(f"  {r['batch']:>6}{gib(r['peak_alloc']):>18.2f}"
              f"{gib(r['peak_reserved']):>20.2f}{r['seconds']:>12.1f}{flag:>18}")

    fits = [r for r in rows if not r["over"]]
    if fits:
        best = min(fits, key=lambda r: r["batch"])
        print(f"\n  Largest batch that stays within {total_mem:.1f} GiB of physical VRAM: "
              f"{max(r['batch'] for r in fits)}")
    else:
        print("\n  Every tested batch over-subscribed physical VRAM.")
    print("  Note: one epoch on a 5% slice is a memory probe, not a throughput benchmark;")
    print("        per-run timing here is noisy (see the spread note in src/deploy/benchmark.py).")


if __name__ == "__main__":
    main()
