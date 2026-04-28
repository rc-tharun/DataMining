# CiteSense 🎓
### *Citations, Communities, and Categories*

> **What does the citation graph alone tell us about a paper's topic — and how much does a graph neural network add on top of plain text?**
> A data-mining final project on the **OGBN-Arxiv** citation network (169,343 papers, 1,166,243 citations, 40 arXiv CS subject areas).

This repo is the final deliverable for **CSCE 676 — Data Mining** at Texas A&M. It pulls together a curated end-to-end notebook and the two earlier checkpoint notebooks in one place. ✨

---

## 🎥 Project video

**👉 2-minute walkthrough on YouTube: https://youtu.be/t7-Fm36iWl8**

---

## 👉 Start here: [`main_notebook.ipynb`](./main_notebook.ipynb)

`main_notebook.ipynb` is the **single curated notebook** that tells the whole story end-to-end — motivation, dataset, EDA, three models, results, ethics, and conclusions. If you only open one file in this repo, open that one.

---

## 📌 The big idea

Citations are a tiny signal — one paper pointing to another. **Does that signal alone, with no text whatsoever, already know what a paper is about?** If it does, how much extra lift does combining it with text features give us via a graph neural network? And when the GNN does win — *why* does it win where it does?

We built three models, evaluated them on the official OGB time-aware split (train ≤ 2017 / val = 2018 / test ≥ 2019), and dissected the results by class, by year, and — for RQ-C — by per-class homophily.

---

## ❓ Research questions

| | Question | Method | Primary metric |
|---|---|---|---|
| **RQ-A** *(course material)* | Can **Louvain community detection**, applied to the undirected citation graph with no access to text or labels, recover the 40 arXiv subject categories? | Unsupervised modularity-maximisation + majority-label classifier | **Modularity, NMI, accuracy** |
| **RQ-B** *(beyond course)* | Does a **GraphSAGE GNN** that combines structure with text features beat a text-only MLP baseline on held-out future papers? | Supervised 2-layer MLP vs. 3-layer GraphSAGE on identical 128-d features | **Macro-F1** on the official OGB time split |
| **RQ-C** *(mechanism)* | Is GraphSAGE's per-class lift over the MLP **mechanically driven by per-class graph homophily**? In other words: do classes whose citations stay within-class get the biggest GNN boost? | Compute per-class homophily H(c) and per-class lift Δ F1(c), then test the rank correlation across all 40 classes | **Spearman ρ(H(c), Δ F1(c))** |

---

## 📊 Results summary

| Model | Inputs | Test Accuracy | Test Macro-F1 |
|---|---|---:|---:|
| **Random baseline** (1 / 40) | — | 2.50% | 0.025 |
| **Louvain + majority label** | graph only | **47.21%** | 0.1875 |
| **2-layer MLP** | text only (128-d) | 56.12% | 0.3301 |
| **GraphSAGE (3 layers)** | **graph + text** | **70.93%** | **0.5183** |

**The headline:** adding the citation graph to the same 128-d text features lifts accuracy by **+14.8 pp** and Macro-F1 by **+18.8 pts** over a text-only MLP. The lift is largest on small, hard classes — exactly where text is sparse — and concentrated on classes whose citation neighborhoods stay within-class (see RQ-C below).

Other findings worth knowing about:

- **Citation homophily = 65.5%** — 26.2× the random baseline of 2.5%. This is the structural premise that makes the GNN work.
- **Modularity = 0.7020, NMI = 0.3853** — Louvain communities align meaningfully with the 40-way arXiv taxonomy without seeing a single label.
- **RQ-C — homophily explains the lift.** Per-class graph homophily H(c) is positively rank-correlated with per-class GraphSAGE lift Δ F1(c) across the 40 classes: **Spearman ρ = +0.424 (p = 0.0065)**, **Pearson r = +0.437 (p = 0.0048)** — both significant at the 1% level. The +14.8 pp aggregate boost is concentrated where citation neighborhoods are locally label-coherent, mechanically tying RQ-A's structural finding to RQ-B's modeling finding.
- **Spearman ρ(PageRank, in-degree) = 0.9254** — but both are heavily age-biased. Top-500-by-PageRank papers have median year **2011**, six years older than the graph-wide median (2017). Any downstream ranking system should treat citation-based "importance" as exposure-over-time, not quality.
- **Drift is mild** — GraphSAGE accuracy is stable from 2019 → 2020 (71.2% → 69.9%), and so is the MLP (56.2% → 55.6%), so the +14.8 pp gap is not a single-year artefact.

The full analysis with figures, ablations, per-class breakdowns, and ethics discussion lives in [`main_notebook.ipynb`](./main_notebook.ipynb). ✨

---

## 🗂️ Data

**Dataset:** [OGBN-Arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) — Open Graph Benchmark, node property prediction.

| Property | Value |
|---|---|
| Nodes (papers) | 169,343 |
| Directed edges (citations) | 1,166,243 |
| Node features | 128-d skip-gram text embedding |
| Labels | 40 arXiv CS subject categories |
| Years | 1971 – 2020 |
| Split | **Time-aware** — train ≤ 2017 (90,941) · val = 2018 (29,799) · test ≥ 2019 (48,603) |
| License | ODC-BY |

**How the data is loaded.** The notebook downloads the dataset on first run via the `ogb` Python package — no manual download is required:

```python
from ogb.nodeproppred import PygNodePropPredDataset
dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="dataset/")
```

**Preprocessing.** Two passes only:
1. The directed citation graph is **symmetrised** (`to_undirected`) for both Louvain and GraphSAGE — Louvain because it requires undirected input, GraphSAGE so messages can flow in both directions along citations.
2. Train / val / test splits are **time-based** as provided by OGB; no shuffling, no leakage.

The downloaded dataset folder is git-ignored — the `ogb` package recreates it on demand.

---

## ▶️ How to reproduce

This project was built in **Google Colab** with a free **T4 GPU**.

1. Open [`main_notebook.ipynb`](./main_notebook.ipynb) in Colab (`File → Open notebook → GitHub` and paste this repo URL).
2. `Runtime → Change runtime type → GPU` (any T4 / L4 / V100 works).
3. **`Runtime → Run all`** — that's it. The notebook installs everything it needs, downloads the OGBN-Arxiv dataset on first run (~80 MB), and runs end-to-end in roughly **15–25 minutes** on a Colab T4 (most of which is the 250-epoch GraphSAGE training; Louvain and the EDA take only a few minutes each).

To pin the exact environment locally:

```bash
pip install -r requirements.txt
```

**Run order** (if you want to walk through the project's progression):

| Step | File | What it does |
|---|---|---|
| 1 | [`checkpoints/checkpoint_1.ipynb`](./checkpoints/checkpoint_1.ipynb) | Dataset selection, comparison of 3 candidate datasets, EDA on OGBN-Arxiv |
| 2 | [`checkpoints/checkpoint_2.ipynb`](./checkpoints/checkpoint_2.ipynb) | First end-to-end pass: Louvain + MLP + GraphSAGE prototype |
| 3 | [`main_notebook.ipynb`](./main_notebook.ipynb) | **Final curated story — open this for grading** |

The two checkpoint notebooks are kept verbatim so you can see the project's progression across the semester.

---

## 🔧 Key dependencies and versions

Built and tested on **Colab Python 3.11**.

| Package | Version | Why it's here |
|---|---|---|
| `python` | **3.11** | Colab default |
| `numpy` | 1.26.4 | numerics |
| `pandas` | 2.2.2 | data wrangling |
| `scipy` | 1.13.1 | sparse matrices |
| `matplotlib` | 3.7.1 | plots |
| `networkx` | 3.3 | graph utilities |
| `python-louvain` | 0.16 | Louvain community detection |
| `scikit-learn` | 1.4.2 | NMI, F1, classification metrics |
| `torch` | 2.3.0 | GraphSAGE |
| `torch-geometric` | 2.5.3 | `SAGEConv`, `to_undirected` |
| `ogb` | 1.3.6 | OGBN-Arxiv loader + official splits |

Full pinned list lives in [`requirements.txt`](./requirements.txt). 🌱

---

## 🌳 Repo structure

```
.
├── README.md                                    ← you are here
├── main_notebook.ipynb                          ← 👉 the final curated deliverable
├── requirements.txt                             ← Colab-frozen dependency versions
├── .gitignore
├── checkpoints/
│   ├── checkpoint_1.ipynb                       ← dataset selection + EDA
│   └── checkpoint_2.ipynb                       ← first end-to-end pass
└── deck/
    ├── 137009909_Final_Project.pptx             ← 17-slide presentation deck
    └── 137009909_Final_Project.pdf              ← PDF export of the same deck
```

---

## 🙋 Author

**Tharun Reddy Challabotla** · NetID `137009909` · CSCE 676 — Data Mining · Texas A&M University
