import numpy as np
import pdb

def tensorToComplex(tensor):
    [batch, t, f] = tensor.shape
    reshape_tensor = np.reshape(tensor, [batch, t, int(f/2), 2])
    real_tensor = reshape_tensor[..., 0]
    im_tensor = reshape_tensor[..., 1]
    out_tensor = real_tensor + 1j * im_tensor
    return out_tensor
