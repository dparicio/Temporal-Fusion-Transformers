# Temporal Fusion Transformers

## Datasets
The datasets used in this project can be found at:
- Electricity: https://huggingface.co/datasets/danipaez/electricity_tft
- Velib https://huggingface.co/datasets/danipaez/velib_tft

## Environment Setup
To install the required packages, run:
```bash
conda create -n tft python=3.10 -y
conda activate tft
pip install -e .
```

## Train
To train the model, modify the parameters in `config.yaml` (it contains both dataset structure and model hyperparameters) as needed and run:
```bash
python train.py
```

## TensorBoard
To visualize training metrics, start TensorBoard with:
```bash
tensorboard --logdir runs
```