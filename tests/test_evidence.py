from pathlib import Path

from closedloop.evidence import verify_evidence_bundle, write_evidence_bundle


def test_evidence_bundle_hashes_and_verifies(tmp_path: Path):
    out = write_evidence_bundle(
        tmp_path / "run",
        config={"a": 1},
        summary={"q": 2.0},
        timeseries=[{"t": 0.0}],
        gates={"pass": True},
        repo_root=tmp_path,
    )
    ok, errors = verify_evidence_bundle(out)
    assert ok
    assert errors == []


def test_evidence_tamper_is_detected(tmp_path: Path):
    out = write_evidence_bundle(
        tmp_path / "run",
        config={"a": 1},
        summary={"q": 2.0},
        timeseries=[],
        gates={},
        repo_root=tmp_path,
    )
    (out / "summary.json").write_text("tampered\n", encoding="utf-8")
    ok, errors = verify_evidence_bundle(out)
    assert not ok
    assert any("hash mismatch" in e for e in errors)


def test_evidence_preserves_json_booleans(tmp_path: Path):
    out = write_evidence_bundle(
        tmp_path / "run",
        config={"enabled": True},
        summary={"experimental_validation": False},
        timeseries=[],
        gates={"pass": True},
        repo_root=tmp_path,
    )
    text = (out / "summary.json").read_text(encoding="utf-8")
    assert '"experimental_validation": false' in text
