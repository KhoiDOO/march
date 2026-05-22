# march

# setup

## Main Dependencies
```bash
## Cloning
git clone https://github.com/KhoiDOO/march.git
cd march

conda create -c conda-forge -n march python=3.10 gxx_linux-64=13 gcc_linux-64=13 -y
conda activate march

conda install nvidia::cuda-toolkit==12.8.2 -y

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

## Examples Dependencies

### Pytorch Example

```bash
# Install march package
pip isntall -e . --no-build-isolation

# Install march package with logging
rm -rf output.log && pip install -e . --no-build-isolation --log output.log

# Install package for torch-based (non-differentiable) examples
pip install jupyter plotly
pip install imageio trimesh open3d tqdm matplotlib ninja

# Differentiable Rendering
## Torch Scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

## Nvdiffrast
pip install setuptools wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

## Kaolin
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

## Point Cloud Utils
pip install point-cloud-utils==0.34.0
```

### Cuda Example
```bash
mkdir build
cd build
cmake ..
make -j8

./bin/mc_forward # Other examples are shown at ./examples/cuda/
```