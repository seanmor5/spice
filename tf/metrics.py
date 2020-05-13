# Custom metrics for different things

# For use with z_score_loss
# default confidence is 95%, need to change this to auto calculate threshold
def z_score_accuracy(confidence=1.96):
  # more partial application fun
  def _accuracy(y_true, y_pred):
    return tf.reduce_mean(
        tf.where(
          tf.equal(y_true - tf.where(y_pred, confidence, x=tf.ones_like(y_pred), y=tf.zeros_like(y_pred)), 0),
          x= tf.ones_like(y_pred),
          y=tf.zeros_like(y_pred)
        )
      )

  return _accuracy