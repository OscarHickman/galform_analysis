"""A fake GALFORM checkout whose dust driver is tests/fake_dust_driver.py."""

import stat
import sys
from pathlib import Path

import pytest

import galform_analysis.dust.galaxy as galaxy
from tests.fake_dust_driver import make_fake_galform

FAKE_DRIVER = Path(__file__).parents[2] / "fake_dust_driver.py"


@pytest.fixture
def fake_galform(tmp_path, monkeypatch):
    """GALFORM_DIR pointing at a fake checkout, with the stand-in driver."""
    root = make_fake_galform(tmp_path / "galform")
    driver = tmp_path / "galform_dust_driver"
    driver.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_DRIVER}" "$@"\n')
    driver.chmod(driver.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("GALFORM_DIR", str(root))
    monkeypatch.delenv("GALFORM_BUILD_DIR", raising=False)
    monkeypatch.setattr(galaxy, "dust_driver", lambda galform: driver)
    return root
