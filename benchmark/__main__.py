import fire

from benchmark.grid import grid_search as _grid_search


def grid(*args, **kwargs):
    _grid_search(*args, **kwargs)
    return


if __name__ == "__main__":
    fire.Fire()
