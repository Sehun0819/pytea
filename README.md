# Torch2H: PyTorch Graph -> H Program
## Requirements

- `node.js >= 12.x`
- `python >= 3.8`

## How to build and use

```bash
# install dependencies
npm run install:all

# build
npm run build

# running examples
python torch2h.py examples/mnist/main.py
```

## TODO

* Make realistic weights by learning network. Current weights are just randomly generated floats.
* Add more benchmarks and functions accordingly.

## Etc.

This program has built on [PyTea](https://github.com/ropas/pytea).
