# DW2TF: Darknet to TensorFlow

This is a simple converter which converts:
- Darknet weights (`.weights`) to TensorFlow weights (`.ckpt`)
- Darknet model (`.cfg`) to TensorFlow graph (`.pb`, `.meta`)

## Requirements
- Ubuntu
- Python 3.6 (known issues with Python 3.7)

## Use it

For a full list of options:
```
python3 main.py -h
```

Provide optional argument `--training` to generate training graph (uses batch norm in training mode).

### Object Detection Networks

#### yolov2
```
python3 main.py \
    --cfg 'data/yolov2.cfg' \
    --weights 'data/yolov2.weights' \
    --output 'data/' \
    --prefix 'yolov2/' \
    --gpu 0
```

#### yolov2-tiny
```
python3 main.py \
    --cfg 'data/yolov2-tiny.cfg' \
    --weights 'data/yolov2-tiny.weights' \
    --output 'data/' \
    --prefix 'yolov2-tiny/' \
    --gpu 0
```

#### yolov3
```
python3 main.py \
    --cfg 'data/yolov3.cfg' \
    --weights 'data/yolov3.weights' \
    --output 'data/' \
    --prefix 'yolov3/' \
    --gpu 0
```

#### yolov3-tiny
```
python3 main.py \
    --cfg 'data/yolov3-tiny.cfg' \
    --weights 'data/yolov3-tiny.weights' \
    --output 'data/' \
    --prefix 'yolov3-tiny/' \
    --gpu 0
```

### Image Classification Networks

#### darknet19
```
python3 main.py \
    --cfg 'data/darknet19.cfg' \
    --weights 'data/darknet19.weights' \
    --output 'data/' \
    --prefix 'darknet19/' \
    --gpu 0
```

#### darknet19_448
```
python3 main.py \
    --cfg 'data/darknet19_448.cfg' \
    --weights 'data/darknet19_448.weights' \
    --output 'data/' \
    --prefix 'darknet19_448/' \
    --gpu 0
```

## Todo

- More layer types

## Thanks

- [darkflow](https://github.com/thtrieu/darkflow)
