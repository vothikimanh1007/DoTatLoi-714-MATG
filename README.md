# HyperG-TCM: A Manifold-Aware Hypergraph Framework for Synergy Prediction

**HyperG-TCM** is an open-source library and benchmarking framework designed for high-order synergy prediction in heterogeneous networks. It was developed to bridge the gap between traditional herbal medicine clinical wisdom and modern systems pharmacology using geometric deep learning.

This repository contains the official implementation of the **Manifold-Aware Transformer Gating (MATG)** network and the **DoTatLoi-714 Benchmark**.

## 🌟 Core Contributions (v82 Final)

1. **The DoTatLoi-714 Benchmark**: A curated hypergraph dataset derived from authoritative Vietnamese Traditional Medicine (VTM) pharmacopoeias, representing multi-herb synergy as node-hyperedge incidence.
2. **PMEA Alignment**: The *Probabilistic Multi-Evidence Alignment* pipeline used to anchor local nomenclature to global molecular registries (TCMID, CTMID, and VietHerb).
3. **MATG Architecture**: A geometric deep learning model (v82) that utilizes Poincaré hyperbolic distances to resolve "Dimensional Congestion" in hierarchical herbal interaction spaces.
4. **NeuMapper TDA**: A multi-layer Topological Data Analysis (TDA) toolkit for visualizing and validating the non-Euclidean nature of herbal manifolds.

## 🧠 Problem Formulation

We formulate herbal synergy prediction as a **Node-Hyperedge Incidence Prediction** task on a heterogeneous hypergraph ![](data:image/png;base64...).

* **Nodes (![](data:image/png;base64...))**: 714 unique medicinal herbs (ViThuoc).
* **Hyperedges (![](data:image/png;base64...))**: 150 multi-herb clinical formulations (BaiThuoc).
* **Target Variable**: Predict the probability ![](data:image/png;base64...) that an herb ![](data:image/png;base64...) participates in a synergistic formulation ![](data:image/png;base64...).
* **Class Balance**: The framework utilizes a 1:5 negative sampling ratio, calibrated with a **Graph Focal Loss** (![](data:image/png;base64...)) to handle extreme sparsity.

## 📦 Installation

# Clone the repository
git clone [[https://github.com//vothikimanh1007/DoTatLoi-714-MATG/hypersynergy.git](https://github.com/vothikimanh1007/DoTatLoi-714-MATG.git)] cd hypersynergy

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as a package
pip install -e .

## 🚀 Quick Start

### 1. Load the Benchmark Data

The framework includes a dedicated loader for the aligned DoTatLoi-714 dataset.

from hypersynergy.data\_loader import DoTatLoiBenchmark

# Loads CSVs and constructs the hypergraph with 1:5 negative sampling
dataset, vtm\_feats, tcm\_feats, form\_feats, \*\_ = DoTatLoiBenchmark.load\_and\_build\_graph(k\_negative=5)

### 2. Synergy Prediction with MATG

Initialize the Manifold-Aware Transformer Gating network.

from hypersynergy.models import MATG\_Model, SynergyPredictor

# Initialize the v82 MATG Model (embed\_dim=12)
model = MATG\_Model(
 num\_nodes=714,
 num\_hyperedges=150,
 vtm\_feats=vtm\_feats,
 tcm\_feats=tcm\_feats,
 form\_feats=form\_feats,
 mode='proposed',
 embed\_dim=12
)

predictor = SynergyPredictor(model)
prob = predictor.predict(herb\_indices=[16], formula\_indices=[5])
print(f"Predicted Synergy Probability: {prob[0]:.4f}")

## 📊 Experimental Results (Benchmark v82)

Evaluated under 5-fold cross-validation with an aggressive weight decay (![](data:image/png;base64...)) to test manifold stability:

| **Architecture** | **Accuracy** | **F1-Score** | **ROC-AUC** |
| --- | --- | --- | --- |
| GCN Baseline | 0.8538 ± 0.024 | 0.4909 | 0.8504 |
| GAT Attentive | 0.8821 ± 0.022 | 0.5468 | 0.8533 |
| **MATG (Proposed)** | **0.9051 ± 0.024** | **0.6224** | **0.8329** |

*Note: MATG sacrifices global ROC-AUC smoothness to establish superior hard decision boundaries, resulting in a ~26.7% improvement in F1-score over the GCN baseline.*

## 🔍 Explainable AI (XAI)

The library includes the **NeuMapperExplainer** and attention gating analysis for 15 core structural hubs, including:

* **Core Hubs**: Đương Quy (Index 16), Bạch Thược (Index 22)
* **Synergy Bridges**: Hương Phụ, Liên kiều, Xuyên khung, Đại táo, Ích Mẫu, Ngải Cứu, etc.

# Extract the Topological (α) vs Semantic (1-α) gating weights
weights = predictor.get\_explainability\_weights(herb\_indices=[16, 22], formula\_indices=[0, 0])
# Result reflects how the model trusts hierarchical wisdom (manifold) for hub nodes.

## 📂 Repository Structure

* hypersynergy/: Core package containing models.py, data\_loader.py, and tda\_utils.py.
* data/raw/: Place CongThuc\_updated.csv, ViThuoc\_final.csv, DoTatLoi\_714\_Enriched.csv, and Harmonized\_Global\_Herbal\_Dataset.csv here.
* notebooks/: Comprehensive v82\_tutorial.ipynb for replication.
* requirements.txt: Minimal dependencies (PyTorch, PyG, Scikit-Learn, NetworkX).

## 📖 Citation

If you use this framework or the DoTatLoi-714 benchmark, please cite our work:

@article{anh2026topological,
 title={Topological Knowledge Bridging: The DoTatLoi-714 Benchmark and a Heterogeneous Hypergraph Framework for Cross-Cultural Herbal Synergy Prediction},
 author={Vo Thi Kim Anh and Václav Snášel},
 journal={Under Review},
 year={2026}
}

## 📧 Contact

For dataset access or collaboration: vothikimanh@tdtu.edu.vn
## 图片内容
[检测到图片但LLM不可用，无法识别内容]
