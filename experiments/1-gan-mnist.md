# Model

Generative adversarial networks were introduced in

> Generative Adversarial Networks
> Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
> https://arxiv.org/abs/1406.2661
> https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf

This experiment follows the paper, using a basic GAN with a fully connected
generator and discriminator. We use MNIST as dataset

> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner
> Gradient-based learning applied to document recognition
> Proceedings of the IEEE, 86(11):2278-2324, November 1998
> http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

# Parameters

The paper does not mention explicit parameters, but says

> The generator nets used a mixture of rectifier linear activations
> and sigmoid activations, while the discriminator net used maxout activations.
> Dropout was applied in training the discriminator net.

Fortunately, the precise hyperparameters are available from
https://github.com/goodfeli/adversarial, more precisely
https://github.com/goodfeli/adversarial/blob/master/mnist.yaml


This motivates the following choice of parameters:

```
  --loss gan \
  --generator constant-fc \
  --generator-layers 3 \
  --generator-layer-size 1200 \
  --generator-lr 0.1 \
  --discriminator fc-maxout \
  --discriminator-layers 3 \
  --discriminator-layer-size 240 \
  --discriminator-maxout-size 5 \
  --discriminator-iterations 1 \
  --discriminator-dropout 0.3 \
  --discriminator-lr 0.1
```