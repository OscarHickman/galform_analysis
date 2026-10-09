import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import pytest  # noqa: E402

from galform_analysis.utils.matplotlib_config import (  # noqa: E402
    DEFAULT_CONFIG_DICT,
    RuntimeConfig,
    register_matplotlib_setconfig,
    setconfig,
)


@pytest.fixture(autouse=True)
def _restore_rcparams():
    with mpl.rc_context():
        yield


def test_update_merges_sections_without_touching_defaults():
    cfg = RuntimeConfig({"font": {"size": 11}, "savefig": {"dpi": 150}})
    assert cfg.config_dict["font"] == {
        "family": "serif",
        "weight": "normal",
        "size": 11,
    }
    assert cfg.config_dict["savefig"] == {"dpi": 150}
    assert DEFAULT_CONFIG_DICT["font"]["size"] == 20
    cfg.update(None)
    assert cfg.config_dict["font"]["size"] == 11


def test_setconfig_applies_rcparams():
    setconfig({"lines": {"linewidth": 1.25}})
    assert mpl.rcParams["lines.linewidth"] == 1.25
    assert mpl.rcParams["font.size"] == 20
    assert mpl.rcParams["xtick.direction"] == "in"


def test_set_global_uses_given_module():
    calls = []

    class FakeMpl:
        @staticmethod
        def rc(group, **kwargs):
            calls.append((group, kwargs))

    RuntimeConfig({"font": {"size": 9}}).set_global(FakeMpl)
    assert ("font", {"family": "serif", "weight": "normal", "size": 9}) in calls
    assert len(calls) == len(DEFAULT_CONFIG_DICT)


def test_register_adds_setconfig_to_module():
    class FakeMpl:
        def __init__(self):
            self.calls = []

        def rc(self, group, **kwargs):
            self.calls.append(group)

    fake = FakeMpl()
    fn = register_matplotlib_setconfig(fake)
    assert fake.setconfig is fn
    fake.setconfig()
    assert fake.calls == list(DEFAULT_CONFIG_DICT)
    assert callable(mpl.setconfig)  # registered on import
