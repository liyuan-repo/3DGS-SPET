pip install torch-2.0.1+cu117-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.15.2+cu117-cp38-cp38-linux_x86_64.whl
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install xformers==0.0.22

cd submodules/Connected_components_PyTorch
python setup.py install

cd ../diff-gaussian-rasterization
python setup.py install

cd ../simple-knn
python setup.py install

cd ../../model/curope
python setup.py install
cd ../..


