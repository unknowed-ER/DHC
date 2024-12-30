import tensorflow as tf
import tensorflow_probability as tfp


def gumbel_softmax(temperature, probs=None, hard=True):
    """
    logits: `[batch_size, num_features]` unnormalized log probabilities; here use probs
    tau: non-negative scalar temperature; tau --> 0; sample will be one-hot; tau--> inf, sample will be 1/k
    hard: if ``True``, the returned samples will be discretized as one-hot vectors,
        but will be differentiated as if it is the soft sample in autograd
    """
    num_classes = tf.shape(probs)[-1]

    sampler = tfp.distributions.RelaxedOneHotCategorical(temperature, probs=probs)
    sample = sampler.sample()
    if hard:
        sample_hard = tf.one_hot(tf.argmax(sample, axis=-1),
                                 num_classes, dtype=tf.float32)
        sample_onehot = tf.stop_gradient(sample_hard - sample) + sample
        sample_idx = tf.cast(tf.argmax(sample_onehot, axis=-1), dtype=tf.int32)
        return sample_onehot, sample_idx
    else:
        return sample
