"""
Tests for the main module.
"""
import pytest
from src.main import hello_world


def test_hello_world():
    """Test the hello_world function."""
    assert hello_world() == "Hello, World!"
    assert isinstance(hello_world(), str) 