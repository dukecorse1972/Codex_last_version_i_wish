import pytest

torch = pytest.importorskip("torch")

from lse_tcn.models.tcn import LSETCN


def test_tcn_forward_shape() -> None:
    model = LSETCN(input_dim=144, num_classes=51)
    x = torch.randn(4, 60, 144)
    y = model(x)
    assert y.shape == (4, 51)
