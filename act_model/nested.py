"""Minimal tensor container for the optional learned position encoding interface."""
class NestedTensor:
    def __init__(self, tensors, mask=None):
        self.tensors, self.mask = tensors, mask

    def to(self, device):
        return NestedTensor(self.tensors.to(device),
                            self.mask.to(device) if self.mask is not None else None)

    def decompose(self):
        return self.tensors, self.mask
