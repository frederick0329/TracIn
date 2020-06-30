# ImageNet
The notebooks are meant to be read to gain insights and adapted to your dataset. Running the notebooks require work such as preparing ImageNet data and checkpoints during training. To read the notebook, it is easier to read from colab by clicking the **Open In Colab** badge due to file size.   

## If you really want to run the colab/notebook
* The notebook is prepared with Colab + TPU. To convert it for GPUs, please replace TPU strategy with GPU strategy. For details, please refer to [tf.distribute.Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy).
* Running with Colab can be easier since it provides free TPU quota.
* Download [ImageNet](http://www.image-net.org/) and load with [TensorFlow Datasets](https://www.tensorflow.org/datasets). Please refer to [tfds imagenet2012](https://www.tensorflow.org/datasets/catalog/imagenet2012) for instructions. 
* [Train ResNet-50](https://github.com/tensorflow/models/tree/master/official/vision/image_classification) and save the checkpoints of each epoch either in [checkpoint](https://www.tensorflow.org/guide/checkpoint)/[saved_model](https://www.tensorflow.org/guide/saved_model).

## Application of TrackIn
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frederick0329/TrackIn/blob/master/imagenet/resnet50_imagenet_self_influence.ipynb) Inspecting Training data with Self-Influence
* [Coming Soon] Identifying Influential Training Points of a Test point
