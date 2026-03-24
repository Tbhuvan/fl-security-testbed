"""Tests for model architecture and parameter utilities."""
import numpy as np
import pytest
import torch

from clients.model import MNISTNet, SimpleMLP, get_model, get_parameters, set_parameters


class TestMNISTNet:
    def test_forward_shape(self):
        model = MNISTNet()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_no_nan_in_output(self):
        model = MNISTNet()
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert not torch.isnan(out).any()


class TestSimpleMLP:
    def test_forward_shape(self):
        model = SimpleMLP()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_custom_dims(self):
        model = SimpleMLP(input_dim=100, hidden=64, num_classes=5)
        x = torch.randn(2, 100)
        out = model(x)
        assert out.shape == (2, 5)


class TestGetModel:
    def test_cnn(self):
        model = get_model("cnn")
        assert isinstance(model, MNISTNet)

    def test_mlp(self):
        model = get_model("mlp")
        assert isinstance(model, SimpleMLP)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_model("resnet999")


class TestParameterRoundtrip:
    """get_parameters + set_parameters must be a lossless roundtrip."""

    def test_extract_and_reload(self):
        model_a = get_model("cnn")
        params = get_parameters(model_a)

        model_b = get_model("cnn")
        set_parameters(model_b, params)

        params_b = get_parameters(model_b)
        for a, b in zip(params, params_b):
            np.testing.assert_array_almost_equal(a, b)

    def test_params_are_numpy(self):
        model = get_model("cnn")
        params = get_parameters(model)
        for p in params:
            assert isinstance(p, np.ndarray)
