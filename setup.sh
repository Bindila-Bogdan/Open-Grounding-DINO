# create the environment
conda create --name open_grounding_dino python=3.12.3 --yes
conda activate open_grounding_dino

# install the dependencies
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt

# build deformable attention
cd models/GroundingDINO/ops
python setup.py build install

# test deformable attention 
python test.py
