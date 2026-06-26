def test_gnn_package_imports():
    import orbscreen.gnn  # noqa: F401


def test_torch_geometric_available():
    import torch_geometric  # noqa: F401
    from torch_geometric.data import Data  # noqa: F401
