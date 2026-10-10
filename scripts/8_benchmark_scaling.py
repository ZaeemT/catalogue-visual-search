"""Measure search latency against catalogue size, to test the O(n) claim.

Retrieval is brute force: every query is compared against every catalogue
vector, so the theory says latency should grow linearly with n. This script
checks that instead of asserting it.

Method: load the full index once, then search prefixes of it — 500, 1,000,
2,000 ... up to the whole catalogue — with the same query vectors each time.
Only the scoring step is timed. The query image is embedded once up front,
because embedding cost does not depend on catalogue size and including it
would flatten the very curve being measured.

Prefixes rather than separately built indexes: the vectors are identical
either way, re-embedding 200k images per size would take hours, and slicing
keeps every point comparable because the data is the same.

Each prefix is copied into fresh contiguous memory. A NumPy slice shares its
parent's buffer, so without the copy a "500-item" search would stride through
a 200k-item array and measure cache behaviour rather than catalogue size.

Usage:
    python scripts/8_benchmark_scaling.py
    python scripts/8_benchmark_scaling.py --model marqo-fashionsiglip --repeats 50
    python scripts/8_benchmark_scaling.py --sizes 1000,10000,100000,200000
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.embedders import MODELS, get_embedder  # noqa: E402
from app.search.index import INDEX_ROOT, VectorIndex, index_dir  # noqa: E402

QUERY_DIR = Path("data/queries/images")
REPORT_DIR = Path("metrics/scaling")

# Roughly logarithmic, so the curve is sampled evenly on a log axis and a
# linear trend is still visible on a linear one.
DEFAULT_SIZES = [500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def query_vectors(model: str, count: int, dim: int, device: str | None) -> np.ndarray:
    """Embed real query images, falling back to random unit vectors.

    Real images matter less here than for recall — timing a dot product does
    not depend on what the vector means — but using them keeps this benchmark
    measuring the same operation the endpoint performs.
    """
    images = sorted(QUERY_DIR.glob("*.jpg"))[:count]
    if not images:
        print(f"no query images in {QUERY_DIR}; using random unit vectors")
        random = np.random.default_rng(0).standard_normal((count, dim)).astype(np.float32)
        return random / np.linalg.norm(random, axis=1, keepdims=True)

    from PIL import Image

    embedder = get_embedder(model, device=device)
    return embedder.embed_images([Image.open(p) for p in images])


def time_search(index: VectorIndex, queries: np.ndarray, top_k: int, repeats: int) -> list[float]:
    """Milliseconds per query, one sample per query per repeat."""
    samples = []
    for _ in range(repeats):
        for query in queries:
            started = time.perf_counter()
            index.search(query, top_k=top_k)
            samples.append((time.perf_counter() - started) * 1000)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="defaults to the only built index")
    parser.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    parser.add_argument("--sizes", default=None, help="comma-separated catalogue sizes")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    model = args.model or next(
        (n for n, s in MODELS.items() if index_dir(s["model_id"], args.index_root).exists()), None
    )
    if model is None:
        raise SystemExit(f"no index under {args.index_root} — run scripts/5_build_index.py")

    full = VectorIndex.load(index_dir(MODELS[model]["model_id"], root=args.index_root))
    print(f"model     : {model}, dim {full.dim}")
    print(f"index     : {full.size:,} items ({full.vectors.nbytes / 1e6:.0f} MB)")

    sizes = (
        [int(s) for s in args.sizes.split(",")]
        if args.sizes
        else [s for s in DEFAULT_SIZES if s <= full.size]
    )
    if full.size not in sizes:
        sizes.append(full.size)
    sizes = sorted(set(sizes))

    queries = query_vectors(model, args.queries, full.dim, args.device)
    print(f"workload  : {len(queries)} queries x {args.repeats} repeats, top_k={args.top_k}")
    print(f"sizes     : {', '.join(f'{s:,}' for s in sizes)}\n")

    rows = []
    for size in sizes:
        # Copy, so each prefix is its own contiguous block rather than a view
        # into the full matrix.
        subset = VectorIndex(
            vectors=np.ascontiguousarray(full.vectors[:size]),
            ids=full.ids[:size],
            manifest=full.manifest,
        )

        time_search(subset, queries[:3], args.top_k, 2)  # warm up, discarded
        samples = time_search(subset, queries, args.top_k, args.repeats)

        median = statistics.median(samples)
        rows.append(
            {
                "catalogue_size": size,
                "median_ms": round(median, 4),
                "p95_ms": round(percentile(samples, 0.95), 4),
                "ms_per_1k_items": round(median / size * 1000, 5),
                "memory_mb": round(subset.vectors.nbytes / 1e6, 1),
                "samples": len(samples),
            }
        )
        print(
            f"  {size:>8,} items  {median:>8.3f} ms  "
            f"({rows[-1]['ms_per_1k_items']:.4f} ms per 1k)  "
            f"{rows[-1]['memory_mb']:>7.1f} MB"
        )
        del subset

    # Least-squares fit of latency against n. A brute-force scan should be
    # linear; R^2 says how well the measurements actually agree with that.
    x = np.array([r["catalogue_size"] for r in rows], dtype=np.float64)
    y = np.array([r["median_ms"] for r in rows], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    r_squared = 1 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    print(f"\nlinear fit: {slope * 1e6:.3f} ms per million items, intercept {intercept:.3f} ms")
    print(f"R^2 = {r_squared:.4f}  ({'linear, as predicted' if r_squared > 0.98 else 'not linear — see the plot'})")
    for target in (1_000_000, 2_000_000):
        print(f"extrapolated to {target / 1e6:.0f}M items: {slope * target + intercept:.1f} ms/query")

    args.out.mkdir(parents=True, exist_ok=True)

    csv_path = args.out / "latency_vs_size.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.out / "latency_vs_size.json"
    json_path.write_text(
        json.dumps(
            {
                "model": model,
                "dim": full.dim,
                "top_k": args.top_k,
                "queries": len(queries),
                "repeats": args.repeats,
                "fit": {
                    "ms_per_million_items": round(slope * 1e6, 4),
                    "intercept_ms": round(intercept, 4),
                    "r_squared": round(float(r_squared), 5),
                },
                "extrapolation": {
                    "1M_items_ms": round(slope * 1_000_000 + intercept, 2),
                    "2M_items_ms": round(slope * 2_000_000 + intercept, 2),
                },
                "measurements": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_path = plot(rows, slope, intercept, r_squared, model, args.out)
    print(f"\nwritten   : {csv_path}, {json_path}" + (f", {plot_path}" if plot_path else ""))


def plot(rows, slope, intercept, r_squared, model, out_dir: Path) -> Path | None:
    """Latency against catalogue size, linear and log-log side by side.

    The log-log panel is the more informative of the two: on it, any power law
    is a straight line whose slope is the exponent, so O(n) shows up as a
    gradient of 1 and can be read off rather than inferred.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the plot (CSV and JSON still written)")
        return None

    sizes = np.array([r["catalogue_size"] for r in rows], dtype=float)
    medians = np.array([r["median_ms"] for r in rows], dtype=float)
    p95s = np.array([r["p95_ms"] for r in rows], dtype=float)

    fig, (linear, loglog) = plt.subplots(1, 2, figsize=(12, 4.8))

    linear.plot(sizes, medians, "o-", label="median", color="#2563eb", zorder=3)
    linear.plot(sizes, p95s, "o--", label="p95", color="#93c5fd", zorder=2)
    fit_x = np.linspace(0, sizes.max(), 100)
    linear.plot(
        fit_x, slope * fit_x + intercept, ":", color="#64748b",
        label=f"linear fit (R²={r_squared:.4f})",
    )
    linear.set_xlabel("catalogue size (items)")
    linear.set_ylabel("search latency (ms)")
    linear.set_title("Brute-force search latency")
    linear.legend()
    linear.grid(alpha=0.3)

    loglog.loglog(sizes, medians, "o-", color="#2563eb", label="measured median")
    reference = medians[0] * (sizes / sizes[0])
    loglog.loglog(sizes, reference, ":", color="#64748b", label="exact O(n) reference")
    loglog.set_xlabel("catalogue size (items, log)")
    loglog.set_ylabel("search latency (ms, log)")
    loglog.set_title("Same data, log-log: O(n) is a slope of 1")
    loglog.legend()
    loglog.grid(alpha=0.3, which="both")

    fig.suptitle(f"{model} — {rows[-1]['catalogue_size']:,} item catalogue", fontsize=11)
    fig.tight_layout()

    path = out_dir / "latency_vs_size.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
