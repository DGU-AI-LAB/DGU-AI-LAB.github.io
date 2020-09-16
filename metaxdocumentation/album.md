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

##### Description
Omniglot dataset is generally used for one-shot learning. It contains 1623 different handwritten characters from 50 different alphabets written by 20 different people. That means, it has 1623 classes with 20 examples each. Each image is of size 105x105. 

##### Example
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

##### Description
MiniImageNet dataset is generally used for few-shot learning. This dataset contains 100 different classes in total that are divided into training, validation and test class splits. 

##### Example
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

##### Description
CropDisease dataset is one of the CD-FSL(Cross-Dimain Few-Shot Learning). This dataset contains 38 different classes in total that have diseased and healthy palnt leaves.

##### Example
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

##### Description
EuroSAT dataset is based on Sentinel-2 satellite imagery covering 13 spectral bands and consists of 10 classes containing 27000 labeled and georeferenced samples.

##### Example
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

##### Description
ISIC dataset is from the ISIC Machine Learning Callenges. There are 8010 samples in the training dataset for seven disease categories which are Melanoma (M), Melanocytic nevus (N), Basal cell carcinoma (BCC), Actinic keratosis / Bowen's disease-intraepithelial carcinoma (AK), Benign keratosis- solar lentigo / seborrheic keratosis / lichen planus-like keratosis (PBK), Dermatofibroma (D) and Vascular lesion (VL).

##### Example
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

##### Description
Chest X dataset contains 108,948 frontalview X-ray images of 32,717 unique patients with the textmined eight disease image labels (where each image can have multi-labels), from the associated radiological reports.

##### Example
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

##### Description
The KTS Dataset is consists of data on 10 labels related to tourist spots in Korea collected from the Instagram.

##### Example
```python
from metaX.datasets import KTSDatabase

database=KTSDatabase(
		    raw_data_address="\dataset\raw_data\kts",
            random_seed=-1)

```


#### Oxford Flowers 102

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/Oxford_flowers_102.png" width="70%" height="70%">
    <br>
</p>
<p align="center"><strong>Example of Oxford Flowers 102 dataset</strong></p>

##### Description
Oxford Flowers 102 dataset is a consistent of 102 flower categories commonly occurring in the United Kingdom. Each class consists of between 40 and 258 images. The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.

##### Example
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

##### Description
High-level implementation of Siamese Network.

This class contains it with meta_train(), meta_train(), load_model() and predict_with_support() methods.

##### Arguments

##### References
1. Koch et al.2015. "Siamese neural networks for one-shot image recognition."


##### Example
```python
from metaX.model.metric_based.Siamese_Network import Siamese_Network

if __ name__ == '__main__':
    siamese_network = Siamese_Network(args, database, network_cls)
    siamese_network.meta_train(epochs = args.epochs)
    siamese_network.meta_test(iterations = args.iterations)
    siamese_network.load_model(epochs = args.epochs)
    print(siamese_network.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
#### Prototypical Network


##### Description
High-level implementation of Prototypical Network.

This class contains it with meta_train(), meta_train(), load_model() and predict_with_support() methods.

##### Arguments

##### References
1. Snell et al.2017. "Prototypical networks for few-shot learning."

##### Example
```python
from metaX.model.metric_based.Prototypical_Network import Prototypical_Network

if __ name__ == '__main__':
    prototypical_network = Prototypical_Network(args, database, network_cls)
    prototypical_network.meta_train(epochs = args.epochs)
    prototypical_network.meta_test(iterations = args.iterations)
    prototypical_network.load_model(epochs = args.epochs)
    print(prototypical_network.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```

### Model-based

#### MANN


##### Description
High-level implementation of Memory-Augmented Neural Networks.

This class contains it with meta_train(), meta_train(), load_model() and predict_with_support() methods.

##### Arguments

##### References
1. Santoro et al.2016. "Meta-learning with memory-augmented neural networks."


##### Example
```python
from metaX.model.model_based.MANN import MANN

if __ name__ == '__main__':
    mann = MANN(args, database, network_cls)
    mann.meta_train(epochs = args.epochs)
    mann.meta_test(iterations = args.iterations)
    mann.load_model(epochs = args.epochs)
    print(mann.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```

### Optimization-based

#### MAML

##### Description
High-level implementation of Model-Agnostic Meta-Learning.

This class contains it with meta_train(), meta_train(), load_model() and predict_with_support() methods.

##### Arguments

##### References
1. Finn et al.2017. "Model-agnostic meta-learning for fast adaptation of deep networks."

##### Example
```python
from metaX.model.optimization_based.MANN import MAML

if __ name__ == '__main__':
    maml = MAML(args, database, network_cls)
    maml.meta_train(epochs = args.epochs)
    maml.meta_test(iterations = args.iterations)
    maml.load_model(epochs = args.epochs)
    print(maml.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
#### Reptile

##### Description
High-level implementation of Reptile.

This class contains it with meta_train(), meta_train(), load_model() and predict_with_support() methods.

##### Arguments

##### References
1. Nichol et al.2018. "On first-order meta-learning algorithms."

##### Example
```python
from metaX.model.optimization_based.MANN import Reptile

if __ name__ == '__main__':
    reptile = Reptile(args, database, network_cls)
    reptile.meta_train(epochs = args.epochs)
    reptile.meta_test(iterations = args.iterations)
    reptile.load_model(epochs = args.epochs)
    print(reptile.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
### Heterogeneous_data_analysis

#### Heterogeneous Emedding


##### Description
This class contains it with train(), train(), load_model() and predict() methods.

##### Arguments

##### References


##### Example
```python
from metaX.model.heterogeneous.Heterogeneous_Emedding import Heterogeneous_Emedding

if __ name__ == '__main__':
    heterogeneous_emedding = Heterogeneous_Emedding(args, database, network_cls)
    heterogeneous_emedding.train(epochs = args.epochs)
    heterogeneous_emedding.test(iterations = args.iterations)
    heterogeneous_emedding.load_model(epochs = args.epochs)
    print(heterogeneous_emedding.predict(predict_path='/dataset/data/kts/test'))
```
#### Vis LSTM
2세부 작성
#### Modified mCNN
2세부 작성
