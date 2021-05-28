# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['cat', 'prodpair', 'append_embedding', 'relation']

# Cell
try:
    import torch
except ImportError:
    pass
try:
    import numpy as np
except ImportError:
    pass
from einops import rearrange, repeat, reduce

def cat(xs, axis):
    try:
        return torch.cat(xs, axis)
    except TypeError:
        return np.concatenate(xs, axis=axis)

# Cell
def prodpair(x):
    """Creates cartesian pairwise matrix for each example in the minibatch,
    pairing vectors on the trailing dimension."""
    b, o, c = x.shape
    return cat([repeat(x, 'b o c -> b (o m) c', m=o),
                repeat(x, 'b o c -> b (m o) c', m=o)], 2)

# Cell
def append_embedding(pairs, embedding):
    """Add an embedding to every paired token."""
    b, osq, c2 = pairs.shape
    return cat([pairs, repeat(embedding, 'b c -> b osq c', osq=osq)], 2)

# Cell
def relation(input, g, embedding=None, max_pairwise=None, reduction='sum'):
    r"""Applies an all-to-all pairwise relation function to a set of objects.
    See :class:`~torch.nn.Relation` for details.
    """
    # Batch size, number of objects, feature size
    b, o, c = input.shape
    # Create pairwise matrix
    pairs = prodpair(input)
    # Append embedding if provided
    if embedding is not None:
        pairs = append_embedding(pairs, embedding)
    # Calculate new feature size
    c = pairs.shape[2]
    # Pack into batches
    pairs = rearrange(pairs, 'b om c -> (b om) c')
    # Pass through g
    if max_pairwise is None:
        output = g(pairs)
    else:
        outputs = []
        for batch in range(0, b * o ** 2, max_pairwise):
            outputs.append(g(pairs[batch:batch + max_pairwise]))
        output = torch.cat(outputs, 0)
    # Unpack
    output = reduce(output, '(b o2) c -> b c', reduction, o2=o**2)
    return output