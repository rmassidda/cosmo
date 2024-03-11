from .generic import CausalDiscoveryAlgorithm
from .notears import NOTEARS
from .dagma import DAGMA
from .cosmo import COSMO
from .golem import GOLEM
from .nocurl import NOCURL
from .cyclic import Cyclic
from .daggnn import DAGGNN
# from .pc import PC
# from .ges import GES

_algorithms = {
    'notears': NOTEARS,
    'golem': GOLEM,
    'dagma': DAGMA,
    'cosmo': COSMO,
    'nocurl': NOCURL,
    'cyclic': Cyclic,
    'daggnn': DAGGNN,
    # 'pc': PC,
    # 'ges': GES,
}


def get_algorithm(alg_name: str, *alg_args, **alg_kwargs) \
        -> CausalDiscoveryAlgorithm:
    """
    Wrapper to get a Continuous Causal Discovery
    Algorithm by its name.
    """
    return _algorithms[alg_name](*alg_args, **alg_kwargs)


__all__ = [
    'NOTEARS',
    'DAGMA',
    'COSMO',
    'GOLEM',
    'NOCURL',
    'Cyclic',
    'DAGGNN',
    # 'PC',
    # 'GES'
]
