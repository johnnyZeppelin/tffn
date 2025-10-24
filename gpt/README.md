
TFFN reproduction (SOLID dataset)
================================

This repository contains an implementation to reproduce the TFFN model (Three-branch Feature Fusion Network)
for the SOLID dataset as described in the supplied paper (file: TFFN_TMM_CAMERA_READY_0829.pdf). Key design choices follow that paper:

- Viewport-based TPF block (binocular difference BD + binocular summation BS using ResNet50 feature maps).
- PDIE block: uses *restored viewports* (already provided) to compute difference maps and extracts monocular features with VMamba.
- FF block: asymmetric fusion and score regression with Euclidean loss.
- Training settings: SGD momentum=0.9, weight_decay=1e-4, batch_size=32, lr=1e-3, train/test split 8:2. (From paper.)

What is included
-----------------
- data/: dataset loader scaffold (expects your `SOLID/`, `viewports/`, `restored_viewports/`, and `vmamba/` to be placed by you)
- models/: PyTorch model implementations (ResNet50 backbone usage, simplified SW-MSA approximation, VMamba loader fallback)
- train.py: training loop for SOLID (uses entire train split)
- test.py: evaluation on test split with PLCC/SRCC/RMSE metrics
- utils.py: helper functions (dataset parsing, metrics, checkpointing)
- requirements.txt: minimal python package list
- example_run.sh: example commands to train/test
- LICENSE / README with notes and citations to the paper.

Important notes
---------------
1. This code **assumes** you already extracted viewports and restored viewports (InstructIR results) as described in your message.
   PDIE will load difference maps from `restored_viewports/` (files with "_res.png").
2. A pretrained VMamba is expected in `vmamba/`. If not found, the code falls back to a small CNN for MF extraction (to allow running).
3. SW-MSA in the paper is approximated here by a windowed multi-head self-attention implemented on flattened spatial tokens.
4. The code is designed to run on a single GPU but will run on CPU for small-scale tests.
5. Cite the paper if you publish results from this code.

References:
- Paper: TFFN_TMM_CAMERA_READY_0829 (provided).
