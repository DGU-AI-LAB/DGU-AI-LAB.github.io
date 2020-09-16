---
sort: 2
---

# Documentation
## metaX.dataset

### 1. Meta-Learning

#### 1. Omniglot
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/omniglot.jpg" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of omniglot dataset</strong></p>

**Description**
Omniglot dataset is generally used for one-shot learning. It contains 1623 different handwritten characters from 50 different alphabets written by 20 different people. That means, it has 1623 classes with 20 examples each. Each image is of size 105x105. 

**Arguments**
* **raw_data_address** (str) - Omniglot raw data path.
* **random_seed** (int, default=-1) - Random seed value.
* **num_train_classes** (int) - Number of train classes.
* **num_val_classes** (int) - Number of validation classes.

**Example**
```python
from metaX.datasets import OmniglotDatabase

database = OmniglotDatabase(
    raw_data_address="dataset\raw_data\omniglot",
    random_seed=47,
    num_train_classes=1200,
    num_val_classes=100)

```

#### 2. MiniImageNet
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/miniImagenet.PNG" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of MiniImageNet dataset</strong></p>

**Description**
MiniImageNet dataset is generally used for few-shot learning. This dataset contains 100 different classes in total that are divided into training, validation and test class splits. 

**Arguments**
* **raw_data_address** (str) - MiniImageNet raw data path.
* **random_seed** (int, default=-1) - Random seed value.


**Example**
```python
from metaX.datasets import MiniImagenetDatabase

database=MiniImagenetDatabase(
			raw_data_address="\dataset\raw_data\mini_imagenet",
			random_seed=-1)

```

#### 3. CropDisease
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/CropDisease.jpg" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of CropDisease dataset</strong></p>

**Description**
CropDisease dataset is one of the CD-FSL(Cross-Dimain Few-Shot Learning). This dataset contains 38 different classes in total that have diseased and healthy palnt leaves.

**Arguments**
* **raw_data_address** (str) - CropDisease raw data path.
* **random_seed** (int, default=-1) - Random seed value.

**Example**
```python
from metaX.datasets import CropDiseaseDatabase

database=CropDiseaseDatabase(
			raw_data_address="\dataset\raw_data\cropdisease",
			random_seed=-1)

```
#### 4. EuroSAT
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/EuroSAT.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of EuroSAT dataset</strong></p>

**Description**
EuroSAT dataset is based on Sentinel-2 satellite imagery covering 13 spectral bands and consists of 10 classes containing 27000 labeled and georeferenced samples.

**Arguments**
* **raw_data_address** (str) - EuroSAT raw data path.
* **random_seed** (int, default=-1) - Random seed value.

**Example**
```python
from metaX.datasets import EuroSATDatabase

database=EuroSATDatabase(
			raw_data_address="\dataset\raw_data\eurosat",
			random_seed=-1)

```
#### 5. ISIC
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/ISIC.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of ISIC dataset</strong></p>

**Description**
ISIC dataset is from the ISIC Machine Learning Callenges. There are 8010 samples in the training dataset for seven disease categories which are Melanoma (M), Melanocytic nevus (N), Basal cell carcinoma (BCC), Actinic keratosis / Bowen's disease-intraepithelial carcinoma (AK), Benign keratosis- solar lentigo / seborrheic keratosis / lichen planus-like keratosis (PBK), Dermatofibroma (D) and Vascular lesion (VL).

**Arguments**
* **raw_data_address** (str) - ISIC raw data path.
* **random_seed** (int, default=-1) - Random seed value.

**Example**
```python
from metaX.datasets import ISICDatabase

database=ISICDatabase(
			raw_data_address="\dataset\raw_data\isic",
			random_seed=-1)

```
#### 6. ChestX
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/ChestX.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of ChestX dataset</strong></p>

**Description**
Chest X dataset contains 108,948 frontalview X-ray images of 32,717 unique patients with the textmined eight disease image labels (where each image can have multi-labels), from the associated radiological reports.

**Arguments**
* **raw_data_address** (str) - ChestX raw data path.
* **random_seed** (int, default=-1) - Random seed value.

**Example**
```python
from metaX.datasets import ChestXDatabase

database=ChestXDatabase(
			raw_data_address="\dataset\raw_data\chestx",
			random_seed=-1)

```

---------

### 2. Multimodal Learning

#### 1. Korean Tourist Spot(KTS)
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/KTS.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of KTS dataset</strong></p>

**Description**
The KTS Dataset is consists of data on 10 labels related to tourist spots in Korea collected from the Instagram.

**Arguments**
* **raw_data_address** (str) - KTS raw data path.
* **random_seed** (int, default=-1) - Random seed value.

**Example**
```python
from metaX.datasets import KTSDatabase

database=KTSDatabase(
			raw_data_address="\dataset\raw_data\kts",
			random_seed=-1)

```


#### 2. Oxford Flowers 102

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/Oxford_flowers_102.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of Oxford Flowers 102 dataset</strong></p>

**Description**
Oxford Flowers 102 dataset is a consistent of 102 flower categories commonly occurring in the United Kingdom. Each class consists of between 40 and 258 images. The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.

**Arguments**
* **raw_data_address** (str) - Oxford flowers 102 raw data path.
* **random_seed** (int, default=-1) - Random seed value.

**Example**
```python
from metaX.datasets import FlowersDatabase

database=FlowersDatabase(
			raw_data_address="\dataset\raw_data\flowers",
			random_seed=-1)

```
#### 3. K-MSCOCO
2세부 작성
#### 4. K-VQA
2세부 작성


---
---


## metaX.model

### 1. Metric-based

#### 1. Siamese Network

**Description**
High-level implementation of Siamese Network.

This class contains it with meta_train(), meta_train(), load_model() and predict_with_support() methods.

##### References
1. Koch et al.2015. "Siamese neural networks for one-shot image recognition."


Siamese_Network
```python
from metaX.model.metric_based.Siamese_Network import Siamese_Network
from metaX.model.metric_based.Siamese_Network import OmniglotModel

network_cls=OmniglotModel
siamese_network = Siamese_Network(args, database, network_cls)
```
**Description**
어쩌구저쩌구
**Arguments**
* **args** (str) - 어쩌구저쩌구
* **database** (int, default=-1) - 어쩌구저쩌구
* **network_cls** (int, default=-1) - 어쩌구저쩌구

```python
siamese_network.meta_train(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구


```python
siamese_network.meta_test(iterations = args.iterations)
```
**Description**
어쩌구저쩌구
**Arguments**
* **iterations** (str) - 어쩌구저쩌구



```python
siamese_network.load_model(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구



```python
print(siamese_network.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
**Description**
어쩌구저쩌구
**Arguments**
* **meta_test_path** (str) - 어쩌구저쩌구

#### 2. Prototypical Network


**Description**
High-level implementation of Prototypical Network.

This class contains it with meta_train(), meta_train(), load_model() and predict_with_support() methods.


##### References
1. Snell et al.2017. "Prototypical networks for few-shot learning."

Prototypical_Network
```python
from metaX.model.metric_based.Prototypical_Network import Prototypical_Network
from metaX.model.metric_based.Prototypical_Network import OmniglotModel

network_cls=OmniglotModel
prototypical_network = Prototypical_Network(args, database, network_cls)
```
**Description**
어쩌구저쩌구
**Arguments**
* **args** (str) - 어쩌구저쩌구
* **database** (int, default=-1) - 어쩌구저쩌구
* **network_cls** (int, default=-1) - 어쩌구저쩌구

```python
prototypical_network.meta_train(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구


```python
prototypical_network.meta_test(iterations = args.iterations)
```
**Description**
어쩌구저쩌구
**Arguments**
* **iterations** (str) - 어쩌구저쩌구

```python
prototypical_network.load_model(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구

```python
print(prototypical_network.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
**Description**
어쩌구저쩌구
**Arguments**
* **meta_test_path** (str) - 어쩌구저쩌구

### 2. Model-based

#### 1. MANN

**Description**
High-level implementation of Memory-Augmented Neural Networks.

This class contains it with meta_train(), meta_train(), load_model() and predict_with_support() methods.

**Arguments**

##### References
1. Santoro et al.2016. "Meta-learning with memory-augmented neural networks."


MANN
```python
from metaX.model.model_based.MANN import MANN
from metaX.model.model_based.MANN import OmniglotModel

network_cls=OmniglotModel
mann = MANN(args, database, network_cls)
```
**Description**
어쩌구저쩌구
**Arguments**
* **args** (str) - 어쩌구저쩌구
* **database** (int, default=-1) - 어쩌구저쩌구
* **network_cls** (int, default=-1) - 어쩌구저쩌구


```python
mann.meta_train(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구


```python
mann.meta_test(iterations = args.iterations)
```
**Description**
어쩌구저쩌구
**Arguments**
* **iterations** (str) - 어쩌구저쩌구


```python
mann.load_model(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구


```python
print(mann.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
**Description**
어쩌구저쩌구
**Arguments**
* **meta_test_path** (str) - 어쩌구저쩌구



### 3. Optimization-based

#### 1. MAML

**Description**
High-level implementation of Model-Agnostic Meta-Learning.

This class contains it with meta_train(), meta_train(), load_model() and predict_with_support() methods.

**Arguments**

##### References
1. Finn et al.2017. "Model-agnostic meta-learning for fast adaptation of deep networks."

ModelAgnosticMetaLearning
```python
from metaX.model.optimization_based.MAML import ModelAgnosticMetaLearning
from metaX.model.optimization_based.MAML import OmniglotModel

network_cls=OmniglotModel
maml = ModelAgnosticMetaLearning(args, database, network_cls)
```
**Description**
어쩌구저쩌구
**Arguments**
* **args** (str) - 어쩌구저쩌구
* **database** (int, default=-1) - 어쩌구저쩌구
* **network_cls** (int, default=-1) - 어쩌구저쩌구


```python
maml.meta_train(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구


```python
maml.meta_test(iterations = args.iterations)
```
**Description**
어쩌구저쩌구
**Arguments**
* **iterations** (str) - 어쩌구저쩌구


```python
maml.load_model(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구


```python
print(maml.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
**Description**
어쩌구저쩌구
**Arguments**
* **meta_test_path** (str) - 어쩌구저쩌구


#### 2. Reptile

**Description**
High-level implementation of Reptile.

This class contains it with meta_train(), meta_train(), load_model() and predict_with_support() methods.

**Arguments**

##### References
1. Nichol et al.2018. "On first-order meta-learning algorithms."

Reptile
```python
from metaX.model.optimization_based.Reptile import Reptile
from metaX.model.optimization_based.Reptile import OmniglotModel

network_cls=OmniglotModel
reptile = Reptile(args, database, network_cls)
```
**Description**
어쩌구저쩌구
**Arguments**
* **args** (str) - 어쩌구저쩌구
* **database** (str) - 어쩌구저쩌구
* **network_cls** (str) - 어쩌구저쩌구

```python
reptile.meta_train(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구


```python
reptile.meta_test(iterations = args.iterations)
```
**Description**
어쩌구저쩌구
**Arguments**
* **iterations** (str) - 어쩌구저쩌구


```python
reptile.load_model(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구


```python
print(reptile.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
**Description**
어쩌구저쩌구
**Arguments**
* **meta_test_path** (str) - 어쩌구저쩌구


### 4. Heterogeneous_data_analysis

#### 1. Heterogeneous Emedding


**Description**
This class contains it with train(), train(), load_model() and predict() methods.

##### References


Heterogeneous_Emedding
```python
from metaX.model.heterogeneous.Heterogeneous_Emedding import Heterogeneous_Emedding
from metaX.model.heterogeneous.Heterogeneous_Emedding import Heterogeneous_Emedding_Model

network_cls=Heterogeneous_Emedding_Model
heterogeneous_emedding = Heterogeneous_Emedding(args, database, network_cls)
```
**Description**
어쩌구저쩌구
**Arguments**
* **args** (str) - 어쩌구저쩌구
* **database** (str) - 어쩌구저쩌구
* **network_cls** (str) - 어쩌구저쩌구


```python
heterogeneous_emedding.train(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구


```python
heterogeneous_emedding.test(iterations = args.iterations)
```
**Description**
어쩌구저쩌구
**Arguments**
* **iterations** (str) - 어쩌구저쩌구


```python
heterogeneous_emedding.load_model(epochs = args.epochs)
```
**Description**
어쩌구저쩌구
**Arguments**
* **epochs** (str) - 어쩌구저쩌구


```python
print(heterogeneous_emedding.predict(predict_path='/dataset/data/kts/test'))
```
**Description**
어쩌구저쩌구
**Arguments**
* **predict_path** (str) - 어쩌구저쩌구

#### Vis LSTM
2세부 작성
#### Modified mCNN
2세부 작성
