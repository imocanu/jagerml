"""
Shape: The length (number of elements) of each of the axes of a tensor.
Rank: Number of tensor axes ( scalar=0, vector=1, matrix=2 )
Axis or Dimension: A particular dimension of a tensor.
Size: The total number of items in the tensor, the product shape vector.
"""

import tensorflow as tf

t = tf.constant(4)
print(t)
t = tf.zeros([3, 2, 4, 5])
print(t._rank())
print(t.ndim)