# TFFN.

## Project Structure

```
TFFN_SOLID/
├── config.py
├── data/
│   ├── SOLID/ (dataset folder as described)
│   ├── viewports/ (extracted viewports)
│   └── restored_viewports/ (InstructIR restored viewports)
├── models/
│   ├── __init__.py
│   ├── tffn.py
│   ├── tpf.py
│   ├── pdienet.py
│   └── fusion.py
├── utils/
│   ├── __init__.py
│   ├── dataset.py
│   ├── metrics.py
│   └── helpers.py
├── train.py
├── test.py
├── vmamba/ (VMamba model code)
└── checkpoints/ (for saving trained models)
```

## Usage Instructions

1. **Setup the environment:**
```bash
pip install torch torchvision pandas openpyxl scipy tqdm matplotlib
```

2. **Organize the data:**
   - Place SOLID dataset in `data/SOLID/`
   - Place viewports in `data/viewports/`
   - Place restored viewports in `data/restored_viewports/`

3. **Train the model:**
```bash
python train.py
```

4. **Test the model:**
```bash
python test.py
```

