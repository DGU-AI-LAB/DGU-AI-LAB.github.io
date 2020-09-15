# metaX_Library


<p align="center">
    <img src="https://raw.githubusercontent.com/DGU-AI-LAB/DGU-AI-LAB.github.io/master/images/logo_transparent.png" width="200"/>
</p>

<h3 align="center">
<p>Meta-Learnign & Multimodal Learning Library for TensorFlow 2.0
</h3>

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
- Website : [https://dgu-ai-lab.github.io/](https://dgu-ai-lab.github.io/)
- Documentation : [https://dgu-ai-lab.github.io/](https://dgu-ai-lab.github.io/)
- Tutorials : [https://dgu-ai-lab.github.io/](https://dgu-ai-lab.github.io/)
- Examples : [https://dgu-ai-lab.github.io/](https://dgu-ai-lab.github.io/)
- GitHub : [https://github.com/DGU-AI-LAB/metaX_dev](https://github.com/DGU-AI-LAB/metaX_dev)

## Installation

## Snippets & Examples

## Lisences

## Acknowledgements
- Next Generation Computing...

<p>
