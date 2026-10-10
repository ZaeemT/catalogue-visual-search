# Catalogue Visual Search

Upload a photo of a clothing item, get back visually similar products from a
384,022-image catalogue, each with a plain-English reason for why it matched.

```
curl -X POST http://127.0.0.1:8000/search/ \
  -F "image=@data/queries/images/000009.jpg" \
  -F "top_k=3"
```

```json
{
  "query": { "top_k": 3, "explain": true },
  "model": "Marqo/marqo-fashionSigLIP",
  "catalogue_size": 384022,
  "results": [
    { "rank": 1, "item_id": "000025", "image_url": "/search/images/000025",
      "category": "Сари", "colors": ["розовый", "коричневый"], "similarity": 0.8876,
      "explanation": "A very close match for your photo, in the same pink and brown tones: long sari in silk." }
  ],
  "timings_ms": { "preprocess": 0.53, "embed": 24.21, "search": 12.72, "explain": 0.05, "total": 37.51 }
}
```

Design decisions, benchmarks and the scaling analysis are in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- ~25 GB free disk (parquet shards, extracted images, model weights, indexes)
- Apple silicon (MPS), CUDA, or CPU — the device is detected automatically
- Optional: [Ollama](https://ollama.com) for English category names in explanations

## 1. Setup

```bash
uv sync
cp .env.example .env
```

Only `MONGO_URI`, `DB_NAME` and `JWT_SECRET` are required (they belong to the
API boilerplate this is built on and are unused by search). Everything else has
a working default.

Model weights download on first use into `data/hf/` — about 3 GB for all three
models, ~800 MB if you only use the default. This is set by `HF_HOME` in
`app/core/hf_cache.py`, so nothing lands in `~/.cache/huggingface`.

Optional, for English category names in explanations:

```bash
ollama serve &          # if not already running
ollama pull llama3.2:1b
```

Without it the pipeline still works — category names stay in the dataset's
original Russian. Set `EXPLAIN_WITH_LLM=false` to skip it deliberately.

## 2. Get the data

List the available subsets and shard sizes without downloading anything
(this reads parquet footers only, a few KB):

```bash
uv run python scripts/1_download_shard.py --check
```

Download one ~1 GB shard (resumable; a repeat run is a no-op):

```bash
uv run python scripts/1_download_shard.py --config myntra --shard 0
```

Look at what is inside it — prints the schema and saves sample images to
`data/samples/` so you can see what you are indexing:

```bash
uv run python scripts/2_explore_dataset.py --rows 5
```

> **Tip:** streaming this dataset over HTTP times out on a slow link. These
> scripts always download a whole shard and read it locally instead.

## 3. Build the catalogue

Extracts the JPEG bytes to files and writes one metadata row per image:

```bash
uv run python scripts/3_build_catalogue.py --limit 4000
```

For the full catalogue, pass several shards — `--limit` is split evenly across
them, and a shard that comes up short passes its leftover to the next:

```bash
uv run python scripts/3_build_catalogue.py \
  --parquet "data/raw/myntra-train-*.parquet" --limit 384022
```

Build a held-out query set from the **test** split, so queries are never items
already in the catalogue:

```bash
uv run python scripts/1_download_shard.py --config myntra --split test
uv run python scripts/3_build_catalogue.py \
  --parquet data/raw/myntra-test-00001-of-00001.parquet \
  --out data/queries --limit 20
```

Produces:

```
data/catalogue/
├── images/000000.jpg …          original bytes, never re-encoded
├── metadata.parquet             id, file, name, colors, sex, styles,
│                                materials, length, fit, caption, …
└── catalogue_manifest.json      which shards, how many, how selected
```

## 4. Build the index

```bash
uv run python scripts/5_build_index.py --model marqo-fashionsiglip
```

```
data/index/marqo-fashionSigLIP/
├── embeddings.npy    (n, 768) float32, L2-normalised
├── ids.npy           row -> item id
└── manifest.json     model, commit hash, preprocessing, timings, provenance
```

Each model gets its own directory, so indexes coexist and the server cannot
load a matrix built by a different encoder. Throughput on an M2 Pro is ~94
img/s for SigLIP and ~235 img/s for CLIP B/32, so 4,000 images takes about a
minute and the full 384k takes just over an hour.

Options: `--model clip-vit-b32 | marqo-fashionclip | marqo-fashionsiglip`,
`--limit N`, `--batch-size N`, `--device mps|cuda|cpu`.

## 5. Run the API

```bash
uv run uvicorn app.main:app --reload
```

Startup loads the model onto the GPU, reads the index into memory, and runs two
warm-up queries — about 10 seconds. Watch for:

```
loaded marqo-fashionsiglip on mps: 384022 items, dim 768 (9.8s)
explanations: ollama llama3.2:1b (category translation)
warm-up 1/2: 34.62 ms
warm-up 2/2: 20.37 ms
```

Interactive docs: <http://127.0.0.1:8000/docs>

### Endpoints

| | |
|---|---|
| `POST /search/` | multipart upload, returns ranked matches |
| `GET /search/images/{item_id}` | a catalogue image, as referenced by `image_url` |
| `GET /search/info` | loaded model, revision, preprocessing, device, catalogue size |

### `POST /search/` parameters

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | file | required | jpeg / png / webp |
| `top_k` | int | 10 | clamped to 1–50 |
| `explain` | bool | true | false skips explanations entirely |

### curl examples

```bash
# What is loaded
curl -s http://127.0.0.1:8000/search/info | python -m json.tool

# Basic search
curl -X POST http://127.0.0.1:8000/search/ \
  -F "image=@data/queries/images/000009.jpg"

# Ten results, no explanations
curl -X POST http://127.0.0.1:8000/search/ \
  -F "image=@data/queries/images/000009.jpg" \
  -F "top_k=10" -F "explain=false"

# Save the top match's image
curl -s http://127.0.0.1:8000/search/images/000025 -o match.jpg
```

> The trailing slash on `/search/` matters — `/search` returns a 307 redirect.
> curl follows it; some clients do not.

### Searching without the server

```bash
uv run python scripts/6_search.py data/queries/images/000009.jpg --top-k 5
```

## 6. Reproduce the benchmarks

Stage-by-stage latency against the running server (20 queries × 5 repeats):

```bash
uv run python scripts/7_benchmark_latency.py --json metrics/bench/latency.json
```

Latency vs catalogue size — the O(n) check. Runs in-process against prefixes of
the index, so it needs no server:

```bash
uv run python scripts/8_benchmark_scaling.py --model marqo-fashionsiglip --repeats 50
```

Writes `metrics/scaling/latency_vs_size.{csv,json,png}`.

Compare embedding models side by side, as an HTML contact sheet:

```bash
uv run python scripts/4_bakeoff.py --models clip-vit-b32,marqo-fashionsiglip
```

## Configuration

Everything is in `.env` (see `.env.example`). The ones that matter:

| Variable | Default | What it does |
|---|---|---|
| `EMBEDDING_MODEL` | `marqo-fashionsiglip` | which index and weights to serve |
| `EMBEDDING_DEVICE` | auto | `mps` / `cuda` / `cpu` |
| `SEARCH_TOP_K_DEFAULT` | 10 | when the request omits `top_k` |
| `SEARCH_TOP_K_MAX` | 50 | ceiling that larger requests clamp to |
| `EXPLAIN_WITH_LLM` | true | false keeps category names in Russian |
| `OLLAMA_MODEL` | `llama3.2:1b` | must be pulled first |

Switching models is one line — but build that model's index first, or startup
refuses with an index/model mismatch.

## Layout

```
app/
├── core/config.py           settings
├── core/hf_cache.py         keeps model downloads inside the project
├── search/embedders.py      one interface, two backends, three models
├── search/index.py          the matrix, ids, manifest; brute-force search
├── search/retriever.py      query -> ranked results; image cleanup
├── search/signals.py        measured facts an explanation may use
├── search/explain.py        sentence assembly + LLM translation
└── modules/search/          FastAPI controller, service, DTOs
scripts/                     1-8, in the order you run them
metrics/                     benchmark output
data/                        gitignored: shards, images, indexes, weights
```

## Troubleshooting

**`no index at data/index/… — build it first`**
Run `scripts/5_build_index.py` for the model named in `EMBEDDING_MODEL`.

**`index/model mismatch on 'model_id'`**
`EMBEDDING_MODEL` does not match the index being loaded. Build that model's
index, or switch back. The refusal is deliberate — mixing encoders produces
plausible nonsense rather than an error.

**`The read operation timed out` while downloading**
A slow link. `HF_HUB_DOWNLOAD_TIMEOUT` is already raised to 120 s in
`app/core/hf_cache.py`; downloads are resumable, so rerun the script.

**Explanations have Russian category names**
Ollama is not reachable. Start it with `ollama serve` and confirm the model is
pulled. The API logs which mode it chose at startup.

**First request is slow**
Expected if warm-up is disabled (`SEARCH_WARMUP_QUERIES=0`). The first search
costs ~15× steady state; the default two warm-up queries absorb it.
