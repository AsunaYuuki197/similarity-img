# Image retrieval

In this project, I re-evaluated the results from two research papers: [Hypergraph Propagation and Community Selection for Objects Retrieval (NeurIPS 2021)](https://sgvr.kaist.ac.kr/~guoyuan/hypergraph_propagation/NeurIPS_2021_Final.pdf) and [Cluster-Aware Similarity Diffusion for Instance Retrieval (ICML 2025)](https://arxiv.org/pdf/2406.02343v3). I utilized the authors' provided codebases to reproduce and compare their experimental results in ROxford5k dataset. Additionally, I integrated both models into a Streamlit web application for interactive demonstration and analysis, enabling easier visualization and evaluation of their performance across dataset.

### Pre-Requisites:
1. Install Git Version Control
[ https://git-scm.com/downloads ]

2. Install Python (v3.12.1 recommended)
[ https://www.python.org/downloads/ ]

3. Install Pip (Package Manager)
[ https://pip.pypa.io/en/stable/installation/ ]


## Installation

**1. Create a Folder where you want to save the project**

**2. Create a Virtual Environment and Activate**

Install Virtual Environment First
```
pip install virtualenv
```

Create Virtual Environment

For Windows
```
python -m venv venv
```
For Mac
```
python3 -m venv venv
```
For Linux
```
virtualenv .
```

Activate Virtual Environment

For Windows
```
venv\Scripts\activate
```

For Mac
```
source venv/bin/activate
```

For Linux
```
source bin/activate
```

**3. Clone this project**

Open a terminal at a directory of your choice and enter these commands (change the folder name if you want to):
```
  git clone https://github.com/AsunaYuuki197/image-retrieval.git
  cd image-retrieval
```

Inside **image-retrieval** folder, you will see several subfolders.

### Setup
To setup the environment, you can install all the required dependencies listed in the `requirements.txt` file by running the following command:

```
pip install -r requirements.txt
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu121
```

### Dataset

1. Extracted [image features](https://drive.google.com/drive/folders/1u3ZN1ItC__IqIk-_hn8apkXCn7MnpoYs?usp=sharing), put in `dataset/roxford5k/features`.


2. Download and unzip the 'features' and 'graph' folders from the [hypergraph propagation project page](https://sgvr.kaist.ac.kr/~guoyuan/hypergraph_propagation/). They are the global DELG features and precomputed matching information.

The running directory structure in **hp_and_cs** should be:

├─data  
│  ├─roxford  
│  └─rparis  
├─features  
│  ├─distractor_np_delg_features   
│  ├─roxford_np_delg_features   
│  └─rparis_np_delg_features  
├─graph  
│  └─delg  
│ | | | ├─R1Moxford  
│ | | | ├─R1Mparis  
│ | | | ├─roxford  
│ | | | └─rparis  
├─utils  

3. To implement geometric verificatoin, you need to download the ROxford/RParis datasets and extract their local features. I already extracted, you can download it in this link [Local Features](https://drive.google.com/file/d/13oFdVlnEDC5miwiGIekLuSH-wpY35qOG/view?usp=sharing) and put all *.npy file to **hp_and_cs/features/roxford_np_delg_features** folder

4. Download Oxford5k dataset in this [link](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz)
, unzip and put in **dataset/roxford5k/jpg**


### CUDA code compilation (For GPU)

Navigate to the folder `reranker/matrix_utils` and `reranker/sparse_divergence` and `reranker/feature_operations` and `reranker/geodesic_distance`, run the following command in the terminal:

```
python setup.py build_ext --inplace
```

If the compilation completes without errors, you will see the complied files like `sparse_divergence.cpython-312-x86_64-linux-gnu.so` in the corresponding folders. These files can be imported as libraries in the project.



## Evaluation
### Cluster-Aware Similarity Diffusion for Instance Retrieval 

```
cd reranking
python3 eval.py
```

You can also change the feature name in eval.py to use a different feature extraction model.

### Hypergraph Propagation and Community Selection for Objects Retrieval

```
cd hp_and_cs
python3 eval.py
```

You can also change the COMMUNITY_SELECTION = 2 in eval.py to use community selection and geometric verification. (But you need to download local features)

## Application

In main folder **image-retrieval**, 

**If you want to evaluate dataset by interactive demonstration, run**

```
streamlit run eval_app.py
```

You can also change the feature name in eval_app.py to use a different feature extraction model, and COMMUNITY_SELECTION = 2 in eval_app.py to use community selection and geometric verification

**Or real-world application, run**

```
streamlit run app.py
```

**Note that:** app.py will use CLIP to extract global features of all images in your dataset, so it takes a lot of time depending on your dataset.

## Troubleshooting

1. ImportError: numpy.core.multiarray failed to import

```
pip uninstall numpy
pip install "numpy<2.0"
```



