# DW2TF: Darknet to TensorFlow

This is a simple converter which converts:
- Darknet weights (`.weights`) to TensorFlow weights (`.ckpt`)
- Darknet model (`.cfg`) to TensorFlow graph (`.pb`, `.meta`)


## Use it

#### Yolo-v2
```
python3 main.py \
    --cfg 'data/yolov2.cfg' \
    --weights 'data/yolov2.weights' \
    --output 'data/' \
    --prefix 'yolov2/' \
    --gpu 0
```

#### Yolo-v3
```
python3 main.py \
    --cfg 'data/yolov3.cfg' \
    --weights 'data/yolov3.weights' \
    --output 'data/' \
    --prefix 'yolov3/' \
    --gpu 0
```

Provide optional argument `--training` to generate training graph (uses training version of batch norm).

For a full list of options:
```
python3 main.py -h
```


## Todo

- More layer types

## Thanks

- [darkflow](https://github.com/thtrieu/darkflow)
