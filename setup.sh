#!/bin/bash
source ~/.bashrc

### SETUP FOR GROUNDING DINO ###
# make sure the install is made at the system level
conda deactivate

# move into the GroundingDINO directory
cd ../GroundingDINO/

# set the CUDA_HOME variable and check it
export CUDA_HOME=/usr/local/cuda
echo $CUDA_HOME

# remove the previously built GroundingDINO package
rm -rf build dist *.egg-info

# install pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# build GroundingDINO
pip install -e .

### SETUP FOR OPEN GROUNDING DINO ###
# move into the Open GroundingDINO directory
cd ../Open-Grounding-DINO/

# create a conda environment and activate it
conda create --name open_grounding_dino python=3.12.3 --yes
conda activate open_grounding_dino

# install pytorch and the dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# build deformable attention
cd models/GroundingDINO/ops
python setup.py build install

# test deformable attention 
python test.py
