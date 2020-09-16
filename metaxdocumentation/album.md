---
sort: 2
---

# Documentation
## metaX.dataset

### Meta-Learning

#### Omniglot
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/omniglot.jpg" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of omniglot dataset</strong></p>

Omniglot dataset is generally used for one-shot learning. It contains 1623 different handwritten characters from 50 different alphabets written by 20 different people. That means, it has 1623 classes with 20 examples each. Each image is of size 105x105. 


```python
from metaX.datasets import OmniglotDatabase

database = OmniglotDatabase(
    raw_data_address="dataset\raw_data\omniglot",
    random_seed=47,
    num_train_classes=1200,
    num_val_classes=100)

```

#### MiniImageNet
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/miniImagenet.PNG" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of MiniImageNet dataset</strong></p>

MiniImageNet dataset is generally used for few-shot learning. This dataset contains 100 different classes in total that are divided into training, validation and test class splits. 


```python
from metaX.datasets import MiniImagenetDatabase

database=MiniImagenetDatabase(
		    raw_data_address="\dataset\raw_data\mini_imagenet",
            random_seed=-1)

```

#### CropDisease
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/CropDisease.jpg" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of CropDisease dataset</strong></p>

CropDisease dataset is one of the CD-FSL(Cross-Dimain Few-Shot Learning). This dataset contains 38 different classes in total that have diseased and healthy palnt leaves.

```python
from metaX.datasets import CropDiseaseDatabase

database=CropDiseaseDatabase(
		    raw_data_address="\dataset\raw_data\cropdisease",
            random_seed=-1)

```
#### EuroSAT
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/EuroSAT.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of EuroSAT dataset</strong></p>

EuroSAT dataset is based on Sentinel-2 satellite imagery covering 13 spectral bands and consists of 10 classes containing 27000 labeled and georeferenced samples.


```python
from metaX.datasets import EuroSATDatabase

database=EuroSATDatabase(
		    raw_data_address="\dataset\raw_data\eurosat",
            random_seed=-1)

```
#### ISIC
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/ISIC.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of ISIC dataset</strong></p>

ISIC dataset is from the ISIC Machine Learning Callenges. There are 8010 samples in the training dataset for seven disease categories which are Melanoma (M), Melanocytic nevus (N), Basal cell carcinoma (BCC), Actinic keratosis / Bowen's disease-intraepithelial carcinoma (AK), Benign keratosis- solar lentigo / seborrheic keratosis / lichen planus-like keratosis (PBK), Dermatofibroma (D) and Vascular lesion (VL).

```python
from metaX.datasets import ISICDatabase

database=ISICDatabase(
		    raw_data_address="\dataset\raw_data\isic",
            random_seed=-1)

```
#### ChestX
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/ChestX.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of ChestX dataset</strong></p>

Chest X dataset contains 108,948 frontalview X-ray images of 32,717 unique patients with the textmined eight disease image labels (where each image can have multi-labels), from the associated radiological reports.


```python
from metaX.datasets import ChestXDatabase

database=ChestXDatabase(
		    raw_data_address="\dataset\raw_data\chestx",
            random_seed=-1)

```
### Multimodal Learning

#### Korean Tourist Spot(KTS)
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/KTS.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of KTS dataset</strong></p>

The KTS Dataset is consists of data on 10 labels related to tourist spots in Korea collected from the Instagram.

```python
from metaX.datasets import KTSDatabase

database=KTSDatabase(
		    raw_data_address="\dataset\raw_data\kts",
            random_seed=-1)

```


#### Oxford Flowers 102
Oxford Flowers 102 dataset is a consistent of 102 flower categories commonly occurring in the United Kingdom. Each class consists of between 40 and 258 images. The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.


<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/Oxford_flowers_102.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of Oxford Flowers 102 dataset</strong></p>

```python
from metaX.datasets import FlowersDatabase

database=FlowersDatabase(
		    raw_data_address="\dataset\raw_data\flowers",
            random_seed=-1)

```
#### K-MSCOCO
2세부 작성
#### K-VQA
2세부 작성

## metaX.model

### Metric-based

#### Siamese Network
#### Prototypical Network

### Model-based

#### MANN

### Optimization-based

#### MAML
#### Reptile

### Heterogeneous_data_analysis

#### Heterogeneous Emedding
#### Vis LSTM
#### Modified mCNN
