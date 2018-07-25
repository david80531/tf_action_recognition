# tf-action-recogntion

Using tf-openpose as preprocessing, and adding deep learning model to do action recognition.

### Download Tensorflow Graph File(pb file)

Before running demo, you should download graph files.

- cmu (trained in 656x368)
- mobilenet_thin (trained in 432x368)

CMU's model graphs are too large for git, so I uploaded them on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder.

```
$ cd models/graph/cmu
$ bash download.sh
```

### Download DataSet

Download labeled video dataset to train the action recognizing model

```
$ bash download_dataset.sh
```