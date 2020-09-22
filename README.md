# metaX_Library
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/logo_transparent.png" width="200">
    <br>
</p>
<p align="center"><strong>Meta-Learning & Multimodal Learning Library for TensorFlow 2.0</strong></p>

-----------------------------------------  
metaX library is a python library with deep neural networks and datasets for meta learning and multi-view learning base on Tensorflow 2.0.

We provide...
- Deep neural networks for meta learning that can solve one-shot or few-shot problems.
- Deep neural networks for multi-view learning
- Various types of experimental datasets that can be used for experiments using the provided model 
- Load codes for provided dataset with few-shot learning settings

## Overview
- metaX.dataset
- metaX.model

## Directory
```
dataset/
	data_generator.py (Omniglot, mini-ImageNet) (Completed)
        KTS_data_generator.py                       (Completed)
	FLOWER_data_generator.py                    (In progress)
	KMSCOCO_data_generator.py                   (In progress)
	KVQA_data_generator.py                      (In progress)
	CropDisease.py                              (Completed)
	EuroSAT.py                                  (Completed)
	ISIC.py                                     (Completed)
 	ChestX.py                                   (Completed)
  data/
  raw_data/
  
model/
	LearningType.py 
	metric_based/
		Relation_network.py                 (In progress)
		Prototypical_network.py             (In progress)
		Siamese_network.py                  (Completed)
	model_based/
		MANN.py                             (Completed)
		SNAIL.py
	optimization_based/
		MAML.py                             (Completed)
		MetaSGD.py
		Reptile.py                          (In progress)
	heterogeneous_data_analysis/
		image_text_embeding.py              (In progress)
		Vis_LSTM.py                         (In progress)
                Modified_mCNN.py                    (In progress)
		
train.py
utils.py (accuracy, mse)
```

## Resources
- [Website](https://dgu-ai-lab.github.io/)
- [Documentation](documentation)
- [Tutorials](https://dgu-ai-lab.github.io/)
- [Examples](https://dgu-ai-lab.github.io/)
- [GitHub](https://github.com/DGU-AI-LAB/metaX_dev)

## Installation
```bash
pip install metax
```

## Snippets & Examples
### MAML
**Results**
Among various optimization based meta-learning algorithms for few-shot learning, MAML(model-agnostic meta-learning) has been highly popular due to its great performance on several benchmaks. This idea is to establish a meta-learner that seeks an initialization useful for fast learning of different tasks, then adapt to specific tasks quickly and efficiently.

**Usage**
```python
from metaX.model.optimization_based.MAML import ModelAgnosticMetaLearning
from metaX.model.optimization_based.MAML import OmniglotModel
from metaX.datasets import OmniglotDatabase

# 1. Preprocess the Dataset
database = OmniglotDatabase(
    raw_data_address="dataset\raw_data\omniglot",
    random_seed=47,
    num_train_classes=1200,
    num_val_classes=100)

# 2. Create the learner model
network_cls=OmniglotModel

# 3. Wrap the meta-learning method(MAML) on the learner model and dataset
maml = ModelAgnosticMetaLearning(args, database, network_cls)

# 4. Meta-Train
maml.meta_train(epochs = args.epochs)

# 5. Meta-Test
maml.meta_test(iterations = args.iterations)

# 6. Load the trained model
maml.load_model(epochs = args.epochs)

# 7. Predict with support set
print(maml.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
## Lisences

## Acknowledgements
- This library was supported by Next-Generation Information Computing Development Program through the National Research Foundation of Korea (NRF)
funded by the Ministry of Science, ICT (NRF-2017M3C4A7083279).

## Contributor
- Department of Computer Engineering, Dongguk University.
- Department of Computer Engineering, Duksung Women's University.
- Department of Statistics, Chungang University.
- Department of Statistics, Dongguk University.
- BI MATRIX CO., LTD. 



<p>
