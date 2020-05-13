# Random GAN stuff

# latent space optimization with gradient descent: https://arxiv.org/abs/1912.00953
# takes latent which is generated from a gaussian prior NOT uniform
# also needs generator, discriminator, batch size and learning rate
def latent_optimization_gd(G, D, z, batch_size, alpha=0.9):

  with tf.GradientTape() as tape:
    tape.watch(z)
    x_hat = G(z)
    f_z = D(x_hat)

    fz_dz = tape.gradient(f_z, z, output_gradients=tf.ones_like(f_z))
    delta_z = tf.ones_like(fz_dz)
    delta_z = alpha * delta_z

    z_prime = tf.clip_by_value(z + delta_z, clip_value_min=-1.0, clip_value_max=1.0)

    return z_prime

# latent space optimization with natural gradient descent: https://arxiv.org/abs/1912.00953
def latent_optimization_ngd(G, D, z, batch_size, alpha=0.9, beta=0.5, norm=1000):

  with tf.GradientTape() as tape:
    tape.watch(z)
    x_hat = G(z)
    f_z = D(x_hat)

  fz_dz = tape.gradient(f_z, z, output_gradients=tf.ones_like(f_z))
  delta_z = tf.ones_like(fz_dz)
  delta_z = (alpha * fz_dz) / (beta + tf.norm(delta_z, ord=2, axis=0) / norm)

  z_prime = tf.clip_by_value(z + delta_z, clip_value_min=-1.0, clip_value_max=1.0)

  return z_prime