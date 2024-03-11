"""
FCI Algorithm
An extension of PC algorithm. FCI differs from PC in the way
it represents relationships and in consideration of latent variables
since it does not assume causal sufficiency.

References
---------

"""
from cosmolib.algorithms.pc import PC


class FCI(PC):
    """
    Fast Causal Inference Algorithm
    """
    def __init__(self):
        super(FCI, self).__init__()


    def

