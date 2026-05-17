# march

# setup

## Main Dependencies
```bash
conda create -n march python=3.10
conda activate march

conda install nvidia::cuda-toolkit==12.8.2

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

## Examples Dependencies
```bash
pip install jupyter
```