# DW2TF: Darknet Weights to TensorFlow

This is a simple convector which converts Darknet weights file to Tensorflow weights file.

## Use it

```
python3 main.py \
    --cfg data/yolo.cfg \
    --weights data/yolo.weights \
    --output data \
    --prefix yolo \
    --gpu 0
```

## Todo

- More layer types
