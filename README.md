# The Manifold Tangent Classifier (MTC) reproduction.

[Paper](http://papers.neurips.cc/paper/4409-the-manifold-tangent-classifier.pdf)

```
mkdir saved_weights
mkdir B_saved_weights
mkdir runs
```

## Train MTC&KNN using CAE+H
```
CUDA_VISIBLE_DEVICES=0 python main.py --CAEH True  --MTC True --MTC_epochs 40  --pretrained_autoencoder_path saved_weights/caeh_120_60_1e-05_1e-07.pth
```

## Train MTC&KNN using AS
```
CUDA_VISIBLE_DEVICES=0 python Manifold_tensorflow.py --ALTER True  --MTC True --MTC_epochs 40  --pretrained_autoencoder_path saved_weights/alter_120_60_10.0_1.0.pth
```
