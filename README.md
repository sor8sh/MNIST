# MNIST

> This repository is made for the Computational Intelligence course project - Feb 2018.

**Dependencies:**

- [Keras](https://keras.io/)
- [Matplotlib](https://matplotlib.org/)

**Dataset:**

- [MNIST](http://yann.lecun.com/exdb/mnist/)

---

A DNN model for MNIST classification problem, implemented in python with Keras.

Model's architecture:
- Sequential: Dense (512, relu) + Dropout + Dense (512, relu) + Dropout + Dense (10, softmax)
- Loss function: Categorical Crossentropy
- Optimizer: Adam
