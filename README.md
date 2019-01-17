# DW2TF: Darknet to TensorFlow

This is a simple converter which converts:
- Darknet weights (`.weights`) to TensorFlow weights (`.ckpt`)
- Darknet model (`.cfg`) to TensorFlow graph (`.pb`, `.meta`)


## Use it

```
python3 main.py \
    --cfg 'data/yolov2.cfg' \
    --weights 'data/yolov2.weights' \
    --output 'data/' \
    --prefix 'yolov2/' \
    --gpu 0
```

## Todo

- More layer types

## Thanks

- [darkflow](https://github.com/thtrieu/darkflow)
