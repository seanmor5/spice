def z_score_loss(confidence_margin=5.):
  # have to do some partial application for additional parameters
  # usage: model.compile(..., loss=z_loss(confidence_margin=4.5))

  def _deviation(x, l=5000, mean=0.0, stddev=1.0):
    # calculate the scores
    scores = tf.random.normal((l, 1), mean=mean, stddev=stddev)
    # mean, std
    mu = tf.math.reduce_mean(scores)
    sigma = tf.math.reduce_std(scores)
    # z-score
    return (x - mu) / sigma

  def _loss(y_true, y_pred):
    return (
        tf.math.reduce_mean(
          (1 - y_true) * tf.abs(_deviation(y_pred)) +
          y_true * tf.abs(tf.maximum(confidence_margin - deviation(y_pred), 0.))
        )
    )

  return _loss