##Variational Auto-encoder

This is an improved implementation of the paper [Stochastic Gradient VB and the Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by D. Kingma and Prof. Dr. M. Welling. This code uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster.

I also created a [Theano](https://github.com/y0ast/Variational-Autoencoder) and a [Torch](https://github.com/y0ast/VAE-Torch) version.

To run the MNIST experiment:

`python main.py`

###NB:
This code is not as nicely polished as the Torch7 and Theano version. It is mainly for playing around with TensorFlow, which is why I tried to add as many of its bells and whistles as possible. PRs to make it more "TensorFlowy" are welcomed! Specifically if I made a mistake that causes a slow down.

There is no continuous version for now, but there will probably be one in the near future.


The code is MIT licensed.
