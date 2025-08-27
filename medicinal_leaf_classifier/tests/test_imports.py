from medicinal_leaf_classifier import some_module
import pytest

def test_imports():
    assert some_module is not None

if __name__ == "__main__":
    pytest.main()