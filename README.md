# Temporal Fusion Transformers

## Setup
To install the required packages, run:
```bash
conda create -n tft python=3.10 -y
conda activate tft
pip install -e .
```

## Train
To train the model, modify the parameters in `config.yaml` as needed and run:
```bash
python train.py
```

## TensorBoard
To visualize training metrics, start TensorBoard with:
```bash
tensorboard --logdir runs
```
Open the printed URL (usually http://localhost:6006).
