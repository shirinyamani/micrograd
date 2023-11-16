# micrograd


![cat](https://github.com/shirinyamani/micrograd/assets/75791599/cc6852a5-a84e-4d0a-9a70-27d577090b44)

implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. (inspired by [Andrej Karapthy](https://github.com/karpathy)) Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.