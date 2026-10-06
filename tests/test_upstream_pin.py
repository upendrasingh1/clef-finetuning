import hashlib

from clef_finetuning.config import ROOT

UPSTREAM = ROOT / "src/clef_finetuning/models/clef_upstream/joint_schema_model.py"
PINNED_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"


def test_upstream_file_unmodified():
    assert hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() == PINNED_SHA256
