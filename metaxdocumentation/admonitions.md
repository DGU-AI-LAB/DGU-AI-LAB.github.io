---
sort: 3
---

# Examples
## Meta-Learning

### MAML

**Results**
Among various optimization based meta-learning algorithms for few-shot learning, MAML(model-agnostic meta-learning) has been highly popular due to its great performance on several benchmaks. This idea is to establish a meta-learner that seeks an initialization useful for fast learning of different tasks, then adapt to specific tasks quickly and efficiently.

**Usage**
```python
from metaX.model.optimization_based.MAML import ModelAgnosticMetaLearning
from metaX.model.optimization_based.MAML import OmniglotModel

network_cls=OmniglotModel
maml = ModelAgnosticMetaLearning(args, database, network_cls)
maml.meta_train(epochs = args.epochs)
maml.meta_test(iterations = args.iterations)
maml.load_model(epochs = args.epochs)
print(maml.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
### Prototypical Networks

**Results**
Among various metric based meta-learning algorithms for few-shot learning, prototypical networks has been highly popular due to its great performance on several benchmaks. 
This idea is to learn a metric space in which classification can be performed by computing distances to prototype representations of each class.

**Usage**
```python
from metaX.model.metric_based.Prototypical_Network import Prototypical_Network
from metaX.model.metric_based.Prototypical_Network import OmniglotModel

network_cls=OmniglotModel
prototypical_network = Prototypical_Network(args, database, network_cls)
prototypical_network.meta_train(epochs = args.epochs)
prototypical_network.meta_test(iterations = args.iterations)
prototypical_network.load_model(epochs = args.epochs)
print(prototypical_network.predict_with_support(meta_test_path='/dataset/data/omniglot/test'))
```
## Multimodal Learning

### Vis LSTM

### Modified mCNN
