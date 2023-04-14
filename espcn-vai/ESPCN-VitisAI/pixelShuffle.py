import numpy as np

class Shuffle_layer(object):

  def __init__(self, scale):
    '''
    Shuffler Layer, performs a Pixel Shuffle.
      input shape (batch, w, h, c) -> (batch, w * scale, h * scale, c // scale**2) out shape
    Paramenters:
      scale : int, scale of the shuffler.
    '''
    self.scale = scale
    self.scale_step = scale * scale

    self.batch, self.w, self.h, self.c = (0, 0, 0, 0)

    self.output, self.delta = (None, None)

  def __str__(self):
    batch, out_width, out_height, out_channels = self.out_shape()
    return 'Shuffler x {:3d}            {:>4d} x{:>4d} x{:>4d} x{:>4d}   ->  {:>4d} x{:>4d} x{:>4d} x{:>4d}'.format(
           self.scale,
           batch, self.w, self.h, self.c,
           batch, out_width, out_height, out_channels)

  def out_shape(self):
    return (self.batch, self.w * self.scale, self.h * self.scale, self.c // (self.scale_step))

  def _phase_shift(self, inpt, scale):
    '''
    Shuffles of the pixel in a given input
    Parameters:
      inpt : the input of this function is not the entire batch of images, but only
        a N channels at a time taken from every image, where N = out_c // scale**2
      scale : int, scale factor of the layer
    '''
    b, w, h, c = inpt.shape
    X = inpt.transpose(1, 2, 3, 0).reshape(w, h, scale, scale, b)
    X = np.concatenate(X, axis=1)
    X = np.concatenate(X, axis=1)
    X = X.transpose(2, 0, 1)
    return np.reshape(X, (b, w * scale, h * scale, 1))

  def _reverse(self, delta, scale):
    '''
    Reverse function of _phase_shift
    Parameters:
      delta : input batch of deltas with shape (batch, out_w, out_h, 1)
      scale : int ,scale factor of the layer
    '''
    # This function apply numpy.split as a reverse function to numpy.concatenate
    # along the same axis also

    delta = delta.transpose(1, 2, 0)

    delta = np.asarray(np.split(delta, self.h, axis=1))
    delta = np.asarray(np.split(delta, self.w, axis=1))
    delta = delta.reshape(self.w, self.h, scale*scale, self.batch)

    # It returns an output of the correct shape (batch, in_w, in_h, scale**2)
    # for the concatenate in the backward function
    return delta.transpose(3, 0, 1, 2)

  def forward(self, inpt):
    '''
    Forward function of the shuffler layer: it recieves as input an image in
    the format ('batch' not yet , in_w, in_h, in_c) and it produce an output
    with shape ('batch', in_w * scale, in_h * scale, in_c // scale**2)
    Parameters:
      inpt : input batch of images to be reorganized, with format (batch, in_w, in_h, in_c)
    '''

    self.batch, self.w, self.h, self.c = inpt.shape

    channel_output = self.c // self.scale_step # out_C


    # The function phase shift receives only in_c // out_c channels at a time
    # the concatenate stitches toghether every output of the function.

    self.output = np.concatenate([self._phase_shift(inpt[:, :, :, range(i, self.c, channel_output)], self.scale)
                                  for i in range(channel_output)], axis=3)

    # output shape = (batch, in_w * scale, in_h * scale, in_c // scale**2)

  def backward(self, delta):
    '''
    Backward function of the shuffler layer: it reorganize the delta to match the
    input shape, the operation is the exact inverse of the forward pass.
    Parameters:
      delta : global delta to be backpropagated with shape (batch, out_w, out_h, out_c)
    '''

    channel_out = self.c // self.scale_step  #out_c

    # I apply the reverse function  only for a single channel
    X = np.concatenate([self._reverse(self.delta[:, :, :, i],self.scale)
                                      for i in range(channel_out)], axis=3)


    # The 'reverse' concatenate actually put the correct channels toghether but in a
    #  weird order, so this part sorts the 'layers' correctly
    idx = sum([list(range(i, self.c, channel_out)) for i in range(channel_out)], [])
    idx = np.argsort(idx)

    delta[:] = X[:, :, :, idx]

