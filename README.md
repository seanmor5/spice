# Spice

> He who controls the Spice, controls the universe!

This is just random stuff I save so I don't have to rewrite it later.

## Layers

Implementations from: [U-Net](https://arxiv.org/pdf/1505.04597.pdf), see also: [Pix2Pix](https://www.tensorflow.org/tutorials/generative/pix2pix)

* `UNetEncoderBlock`
* `UNetDecoderBlock`

## GAN

Implementations from: [LOGAN](https://arxiv.org/abs/1912.00953)

* `latent_optimization_gd`
* `latent_optimization_ngd`

## Losses

Implementations from: [Deviation Networks](https://arxiv.org/abs/1911.08623)

* `deviation`
* `z_score_loss`

Implementations from: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

* `wasserstein_loss`

## Metrics

* `z_score_accuracy`

## Utils

Implementations from: [Faster Training with Data Echoing](https://arxiv.org/pdf/1907.05550.pdf)

* `echo`

Just random helper stuff

* `prepare_dataset`
* `decode_image`
* `use_tpu`
