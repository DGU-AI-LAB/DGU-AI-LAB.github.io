---
sort: 3
---

# Examples
## Meta-Learning

### MAML

**Results**

**Usage**
```python
from metaX.model.optimization_based.MAML import ModelAgnosticMetaLearning
from metaX.model.optimization_based.MAML import OmniglotModel

network_cls=OmniglotModel
maml = ModelAgnosticMetaLearning(args, database, network_cls)
```
### Prototypical Networks

**Results**

**Usage**
```python
from metaX.model.metric_based.Prototypical_Network import Prototypical_Network
from metaX.model.metric_based.Prototypical_Network import OmniglotModel

network_cls=OmniglotModel
prototypical_network = Prototypical_Network(args, database, network_cls)
```
## Multimodal Learning

### Vis LSTM

### Modified mCNN
