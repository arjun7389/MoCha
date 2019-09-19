from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import RepeatVector, Permute
from keras import regularizers
from keras import constraints
from keras.legacy import interfaces
from keras.utils.generic_utils import to_list
from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def mixture_prior_params(sigma_1, sigma_2, pi, return_sigma=False):
    params = K.variable([sigma_1, sigma_2, pi], name='mixture_prior_params')
    sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
    return params, sigma

def log_mixture_prior_prob(w):
    comp_1_dist = tfp.distributions.Normal(0.0, prior_params[0])
    comp_2_dist = tfp.distributions.Normal(0.0, prior_params[1])
    comp_1_weight = prior_params[2]    
    return K.log(comp_1_weight * comp_1_dist.prob(w) + (1 - comp_1_weight) * comp_2_dist.prob(w))    

# Mixture prior parameters shared  layer instances
prior_params, prior_sigma = mixture_prior_params(sigma_1=1.0, sigma_2=0.1, pi=0.2)
class EmbeddingVariation(Layer):

    def __init__(self, input_dim, output_dim, kl_loss_weight, mask_zero=False, input_length=None, embeddings_initializer=initializers.normal(stddev=prior_sigma), **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kl_loss_weight = kl_loss_weight
        self.embeddings_initializer = embeddings_initializer
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(EmbeddingVariation, self).__init__(**kwargs)

    def build(self, input_shape):  
        self._trainable_weights.append(prior_params) 

        self.kernel_mu = self.add_weight(name='kernel_mu', 
                                         shape=(self.input_dim, self.output_dim),
                                         initializer=initializers.normal(stddev=prior_sigma),
                                         trainable=True)
        #self.bias_mu = self.add_weight(name='bias_mu', 
        #                               shape=(self.output_dim,),
        #                               initializer=initializers.normal(stddev=prior_sigma),
        #                               trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                          shape=(self.input_dim, self.output_dim),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        #self.bias_rho = self.add_weight(name='bias_rho', 
        #                                shape=(self.output_dim,),
        #                                initializer=initializers.constant(0.0),
        #                                trainable=True)
        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        output_mask = K.not_equal(inputs, 0)
        return output_mask

    def compute_output_shape(self, input_shape):
        if self.input_length is None:
            return input_shape + (self.output_dim,)
        else:
            # input_length can be tuple if input is 3D or higher
            in_lens = to_list(self.input_length, allow_tuple=True)
            if len(in_lens) != len(input_shape) - 1:
                raise ValueError(
                    '"input_length" is %s, but received input has shape %s' %
                    (str(self.input_length), str(input_shape)))
            else:
                for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
                    if s1 is not None and s2 is not None and s1 != s2:
                        raise ValueError(
                            '"input_length" is %s, but received input has shape %s' %
                            (str(self.input_length), str(input_shape)))
                    elif s1 is None:
                        in_lens[i] = s2
            return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)

    def call(self, inputs):
        #if K.dtype(inputs) != 'int32':
        #    inputs = K.cast(inputs, 'int32')
        #out = K.gather(self.embeddings, inputs)
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        #bias_sigma = tf.math.softplus(self.bias_rho)
        #bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
                
        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma))# + 
        #              self.kl_loss(bias, self.bias_mu, bias_sigma))
        x=inputs*np.ones(self.input_dim)
        #Kernel=Permute((2,1))(kernel)
        return (K.dot(x, kernel))# + bias)
    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_loss_weight * K.sum(variational_dist.log_prob(w) - log_mixture_prior_prob(w))
    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'embeddings_initializer':
                      initializers.serialize(self.embeddings_initializer),

                  'mask_zero': self.mask_zero,
                  'input_length': self.input_length}
        base_config = super(EmbeddingVariation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))