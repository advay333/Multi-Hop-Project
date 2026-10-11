import pytest

from mhtable import registry
from mhtable.config import Config


@registry.register("reader", "_test_echo")
class Echo:
    def __init__(self, prefix="", env=None):
        self.prefix, self.env = prefix, env


def test_build_from_config_section_with_extras():
    obj = registry.build("reader", Config({"name": "_test_echo", "prefix": ">"}), env="E")
    assert isinstance(obj, Echo) and obj.prefix == ">" and obj.env == "E"
    assert isinstance(registry.build("reader", "_test_echo"), Echo)
    assert "_test_echo" in registry.available("reader")


def test_errors():
    with pytest.raises(KeyError, match="registered"):
        registry.get("reader", "nope")
    with pytest.raises(KeyError, match="unknown component kind"):
        registry.register("widget", "x")
    with pytest.raises(KeyError, match="already registered"):
        registry.register("reader", "_test_echo")(type("Other", (), {}))
    with pytest.raises(KeyError, match="name"):
        registry.build("reader", {"prefix": "x"})
