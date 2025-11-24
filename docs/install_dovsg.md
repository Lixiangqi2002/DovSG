# Environment Setup

You need to get anygrasp [license and checkpoint](https://github.com/graspnet/anygrasp_sdk/blob/main/README.md#license-registration). It usually reply in 2 work days. If you do not receive the reply in 2 days, **please check the spam folder.**
You need to organize the folders like `DovSG/license`.


Then, you should install cuda-12.1 for this project
> create conda env, `python=3.9 is indeed when using anygrasp api`
```bash
git clone --recursive git@github.com:BJHYZJ/DovSG.git
cd DovSG
```



please fellow the above step to setup environment

> Use cuda-12.1 and install `torch==2.3.1`
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121  # -i https://pypi.tuna.tsinghua.edu.cn/simple

```

> Install sam2
```bash
# Edit sam2's set.py
cd third_party/segment-anything-2
# we use '7e1596c' branch
# change setup.py
# line 27: "numpy>=1.24.4" ==> "numpy>=1.23.0",
# line 144: python_requires=">=3.10.0" ==> python_requires=">=3.9.0"

pip install -e .  # -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e ".[demo]"  # -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> Install groundingdino

```bash
cd ../../third_party/GroundingDINO
# we use `856dde2`
pip install -e . --no-build-isolation  # -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> Install RAM & Tag2Text:

```bash
cd ../../third_party
pip install -r ./recognize-anything/requirements.txt  # -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e ./recognize-anything/  # -i https://pypi.tuna.tsinghua.edu.cn/simple
```


> Install ACE

```bash
# We use ace as the relocalization model
# https://github.com/nianticlabs/ace.git
# For ease of use, we have modified some of the code, so please use the ace code provided in our repo directly
conda install opencv
cd ../ace/dsacstar
python setup.py install
```


> Install `lightglue` for searching similar image 

```bash
cd ../../third_party/LightGlue
# we use `edb2b83`
python -m pip install -e .
```



> Install `Faiss`
```bash
# Install the Faiss library (CPU version should be fine)
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
```


> Install `pytorch3d`

```bash
cd ../../third_party/pytorch3d
# we use `05cbea1`
python setup.py install
```


> Install other package

```bash
cd ../../
pip install ipython cmake pybind11 ninja scipy==1.10.1 scikit-learn==1.4.0 pandas==2.0.3 hydra-core opencv-python openai-clip timm matplotlib==3.7.2 imageio timm open3d numpy-quaternion more-itertools pyliblzfse einops transformers pytorch-lightning wget gdown tqdm zmq torch_geometric numpy==1.23.0  # -i https://pypi.tuna.tsinghua.edu.cn/simple
```


> Install `protobuf==3.19.0, MinkowskiEngine graspnet api`. this package is a little difficult to instll
```bash
pip install protobuf==3.19.0  # -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install git+https://github.com/pccws/MinkowskiEngine --no-build-isolation # -i https://pypi.tuna.tsinghua.edu.cn/simple
# if you meet ` No module named 'distutils.msvccompiler` error, use: conda install "setuptools <65" 
pip install graspnetAPI  # -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> Install `torch-cluster`

```bash
# download whl file from https://pytorch-geometric.com/whl/torch-2.3.0%2Bcu121.html
# wget https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_cluster-1.6.3%2Bpt23cu121-cp39-cp39-linux_x86_64.whl
pip install "/path/to/torch_cluster*.whl"
# For example: pip install torch_cluster-1.6.3+pt23cu121-cp39-cp39-linux_x86_64.whl
```

```bash
pip install numpy==1.23.0 supervision==0.14.0 shapely alphashape   # -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyrealsense2 open_clip_torch graphviz pyrender  # -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai==1.56.1
pip install transforms3d==0.3.1 scikit-image==0.19.3
```

> Install dovsg
```bash
pip install -e .
```

Then, install the [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) environment by following [install_droidslam.md](install_droidslam.md).