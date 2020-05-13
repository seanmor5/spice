# Random loss functions and utilities from papers

# Deviation function from: https://arxiv.org/abs/1911.08623
# TODO: support for other priors (cauchy, binomial, exp, etc.)
# defaults from the paper, l has to be large for the CLT to take effect
def deviation(x, l=5000, mean=0.0, stddev=1.0):
  # calculate the scores
  scores = tf.random.uniform((l, 1), mean=mean, stddev=stddev)
  # mean, std
  mu = tf.math.reduce_mean(scores)
  sigma = tf.math.reduce_std(scores)
  # z-score
  return (x - mu) / sigma

# Z-Score loss from: https://arxiv.org/abs/1911.08623
# Forces a model to produce scores that are normally distributed
# Used in "Deviation Networks" for anomaly detection
# I don't think they work very well, but we'll see
def z_loss(confidence_margin=5.)
  # have to do some partial application for additional parameters
  # usage: model.compile(..., loss=z_loss(confidence_margin=4.5))
  def _loss(y_true, y_pred):
    return (
        tf.math.reduce_mean(
          (1 - y_true) * tf.abs(deviation(y_pred)) +
          y_true * tf.abs(tf.maximum(confidence_margin - deviation(y_pred), 0.))
        )
    )

  return _loss

def wasserstein_loss():
  return