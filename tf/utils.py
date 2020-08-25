def prepare_dataset(ds, cache=True, shuffle_buffer_size=10000, repeat=True, batch_size=32, prefetch=True):
  assert ds is not None
  if cache:
    ds = ds.cache()
  ds = ds.shuffle(shuffle_buffer_size)
  if repeat:
    ds = ds.repeat()
  ds = ds.batch(batch_size)
  if prefetch:
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds

def decode_image(path, img_height=512, img_width=512, channels=3):
  assert path is not None

  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=channels) # TODO: support different images
  img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  return tf.image.resize(img, [img_height, img_width])

def use_tpu(tpu_addr):
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + tpu_addr)
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)
  return strategy

def echo(ds, e=1):
  return ds.flat_map(lambda t: tf.data.Dataset.from_tensors(t).repeat(e))
