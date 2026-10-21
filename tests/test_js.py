import shutil
import subprocess

import pytest

from signbridge.config import ROOT


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_browser_logic_in_node():
    r = subprocess.run(["node", "--experimental-default-type=module", "--test", "tests/js/"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
