I read the paper (TFFN) you uploaded and prepared a runnable reproduction project tailored to the SOLID dataset and your folder layout. I put everything into a zip at:

[Download the project zip — tffn_solid_repo.zip](sandbox:/mnt/data/tffn_solid_repo.zip)

Summary of what I did (quick):

* Read the paper you provided and used the paper's design / implementation details (TPF/BD/BS, PDIE using InstructIR pseudo-reference, VMamba for monocular features, FF fusion, Euclidean loss, and the experimental hyperparameters). See the paper snippets I used for design reference:   .
* Created a self-contained PyTorch project that:

  * Loads SOLID viewports and restored viewports (expects directories `SOLID/viewports/` and `SOLID/restored_viewports/`, plus `SOLID/BPGmos.xlsx` & `SOLID/JPEGmos.xlsx` for MOS).
  * Implements ResNet50 backbone stage extraction (stage2/3/4), simplified SW-MSA (windowed MHSA approximation), BD/BS modules, VMamba wrapper (loads your `vmamba/` if present, otherwise a small CNN fallback), FF fusion and regression head.
  * Training loop (SGD momentum=0.9, weight_decay=1e-4, lr=1e-3 by default), evaluation computing PLCC/SRCC/RMSE on the test split (paper used 8:2 split). See Implementation Details in paper. 
  * Uses the *restored viewports* you already computed (files ending with `_res.png`) for PDIE difference map (right - restored_right).
  * Contains fallback behavior so you can run small-scale tests without the actual VMamba implementation (useful for debugging).

What’s in the zip (high level)

* `README.md` — how to use and notes.
* `train.py`, `test.py` — training & testing entrypoints.
* `utils.py` — dataset loader for the SOLID viewport groups, MOS loading, metrics (PLCC/SRCC/RMSE).
* `models/`:

  * `backbones.py` — ResNet50 stage extractor.
  * `sw_msa.py` — simplified windowed self-attention approximation.
  * `vmamba_wrapper.py` — tries to import your `vmamba/` package under provided root, else uses a small CNN fallback.
  * `tffn.py` — main TFFN model wiring BD/BS/PDIE/FF & regression head.
* `requirements.txt`, `example_run.sh`.
* `checkpoints/` will be created by training; default location can be changed via CLI.

Important implementation notes & assumptions (I implemented reasonable defaults where the paper left implementation freedom):

1. **Viewports & PDIE**: The paper uses InstructIR to restore viewports. You told me those restored viewports already exist — the code reads them from `SOLID/restored_viewports/` with names matching the original viewport plus `_res.png`. PDIE computes element-wise difference `MDi = IDi ⊖ IPRi` (paper eq.5) and feeds concatenated diffs into VMamba (paper eq.6). In code we average viewports per side into a single left/right/restored tensor per SOI (practical simplification for memory and to match the paper's use of aggregated viewport features). See loader comments in `utils.py`.

2. **VMamba**: The model tries to import a `vmamba` package from the path you pass in `--root_vmamba` (or `.`). If a `vmamba.VMamba` class is found it will be used. Otherwise a small CNN fallback is used so you can run tests. (Paper references VMamba for monocular features: .)

3. **SW-MSA**: The paper uses Shifted-Window MSA (Swin). For reproducibility I implemented a simple windowed MultiheadAttention approximation (`models/sw_msa.py`). It captures the windowed attention idea without requiring the entire Swin dependency.

4. **Training recipes**: I adopted the paper's recommended settings: SGD momentum 0.9, weight decay 1e-4, batch size defaults (you can override), lr 1e-3, dataset split 8:2. See `train.py` and the Implementation Details in the paper. 

5. **Evaluation metrics**: implemented PLCC/SRCC/RMSE per paper (eqs 12–14). See `utils.py`.

6. **Simplifications**: To keep the code practical and runnable on one GPU (or CPU for debug), some practical simplifications were made:

   * The viewport-level hierarchical fusion in the paper is approximated by averaging viewports per side then extracting multi-scale features from ResNet stages 2/3/4.
   * The BD/BS modules in the paper are implemented following the equations but in a pragmatic vectorized manner (see `models/tffn.py`).
   * The FF block approximates SW-MSA usage for BD strengthening.

Citations (paper locations that guided the code design):

* PDIE and VMamba usage: .
* BD/BS/TPF design and equations:  .
* Implementation details & hyperparameters: .
* VMamba reference in paper: .

Next steps I recommend (pick any, I can do any for you):

* I can run a smoke test (small subset) on your machine if you want me to create a tiny synthetic dataset and run one epoch — **I cannot run it on your machine** but I can prepare commands and minor scripts for you.
* If you want the repository adjusted (e.g., true Swin implementation, exact multi-viewport hierarchical fusion, or the InstructIR pipeline integrated), tell me which parts to upgrade and I’ll modify the code and repackage.
* If you want, I can (now) add a script that converts the full `8192x8192` SOLID images into the viewport crop files expected by the loader (but you said viewports are already available).

If anything is unclear or you want me to change defaults (viewport size, batch size, how viewports are aggregated — average vs per-viewport feature stack), tell me which change you want and I’ll update the repository and re-zip it.

Finally — the code I produced is a careful engineering reproduction, but some implementation choices were necessarily inferred from the paper's description (e.g., exact VMamba API, exact SW-MSA implementation). I've documented these choices in `README.md` so you can replace components (like plugging in your exact VMamba) without changing the rest of the pipeline.

Would you like me to:
A) Add a small script that validates presence and paths of `SOLID/`, `viewports/`, `restored_viewports/`, and `vmamba/` and prints a short readiness report?
B) Integrate the official SwinTransformer (instead of the simplified windowed attention)?
C) Update the dataset loader to keep per-viewport features (more faithful to the paper but will increase memory)?

Pick any and I’ll modify the repo and provide a new zip.
