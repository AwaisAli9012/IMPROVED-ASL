import shutil
import subprocess

import pytest

from signbridge.config import ROOT

WEB = ROOT / "web" / "app"


@pytest.mark.skipif(shutil.which("npm") is None or not (WEB / "node_modules").exists(), reason="run npm install in web/app first")
def test_browser_logic_in_node():
    r = subprocess.run(["npm", "--prefix", str(WEB), "test"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
