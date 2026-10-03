"""Exercise the harness against real pytest processes, not invented exit statuses."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import mutation_check as harness


@pytest.mark.parametrize("body,outcome", [
    ("def test_ok():\n    assert 2 + 2 == 4\n", "passed"),
    ("def test_bad():\n    assert 2 + 2 == 5\n", "failed"),
    ("def test_bad(:\n    pass\n", "error"),
    ("import pytest\n@pytest.fixture\ndef broken():\n    raise RuntimeError('setup')\n"
     "def test_setup(broken):\n    pass\n", "error"),
    ("import pytest\n@pytest.fixture\ndef broken():\n    yield\n    raise RuntimeError('teardown')\n"
     "def test_teardown(broken):\n    assert True\n", "error"),
    ("import pytest\n@pytest.mark.skip(reason='no execution')\ndef test_skip():\n    pass\n", "error"),
    ("# no test cases\n", "error"),
])
def test_real_runner_classification(tmp_path, body, outcome):
    (tmp_path / "test_case.py").write_text(body, encoding="utf-8")
    evidence = tmp_path / "evidence"
    result = harness.run_tests("test_case.py", tmp_path, evidence, 30)
    assert result["outcome"] == outcome
    assert json.loads((evidence / "result.json").read_text())["outcome"] == outcome
    assert (evidence / "stdout.log").exists()
    assert (evidence / "stderr.log").exists()


def test_timeout_is_error_and_keeps_output(tmp_path):
    (tmp_path / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(20)\n", encoding="utf-8")
    result = harness.run_tests("test_slow.py", tmp_path, tmp_path / "logs", .5)
    assert result["outcome"] == "error"
    assert result["reason"] == "timeout"


def test_timeout_stops_pytest_descendants_before_return(tmp_path):
    marker = tmp_path / "child-survived.txt"
    child = ("import time; from pathlib import Path; time.sleep(3); "
             f"Path({str(marker)!r}).write_text('child was left running')")
    (tmp_path / "test_descendant.py").write_text(
        "import subprocess,sys,time\n"
        "def test_descendant():\n"
        f"    subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "    print('child-started', flush=True)\n"
        "    time.sleep(30)\n", encoding="utf-8")
    result = harness.run_tests("-s test_descendant.py", tmp_path, tmp_path / "logs", 2)
    assert result["outcome"] == "error" and result["reason"] == "timeout"
    assert b"child-started" in (tmp_path / "logs" / "stdout.log").read_bytes()
    time.sleep(3.5)
    assert not marker.exists()


@pytest.mark.parametrize("xml", [None, "<broken", "<testsuites/>", "<unrelated/>"])
def test_missing_or_empty_report_cannot_kill(tmp_path, xml):
    report = tmp_path / "report.xml"
    if xml is not None:
        report.write_text(xml, encoding="utf-8")
    assert harness.classify_report(1, report)["outcome"] == "error"


@pytest.mark.parametrize("returncode", [0, 2, 3, 4, 5, -9])
def test_failure_xml_cannot_override_runner_exit(tmp_path, returncode):
    report = tmp_path / "report.xml"
    report.write_text('<testsuites><testsuite><testcase><failure/></testcase></testsuite></testsuites>')
    assert harness.classify_report(returncode, report)["outcome"] == "error"


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_isolated_mutations_count_only_behavioral_failure(tmp_path, line_ending):
    source = tmp_path / "source"
    source.mkdir()
    code = b'# comment\nVALUE = 7\n'.replace(b'\n', line_ending)
    (source / "subject.py").write_bytes(code)
    (source / "test_subject.py").write_text(
        "from subject import VALUE\ndef test_value():\n    assert VALUE == 7\n", encoding="utf-8")
    (source / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent))\n",
        encoding="utf-8")
    mutations = [
        ("M1", "subject.py", "VALUE = 7\n", "VALUE = 8\n", "wrong value", "test_subject.py"),
        ("M2", "subject.py", "VALUE = 7", "VALUE = (", "syntax error", "test_subject.py"),
        ("M3", "subject.py", "VALUE = 7", "raise RuntimeError('import')", "import error", "test_subject.py"),
        ("M4", "subject.py", "missing target", "anything", "missing", "test_subject.py"),
    ]
    equivalent = ("M0", "subject.py", "# comment", "# a different comment", "control", "test_subject.py")
    output = source / "custom-evidence"
    summary = harness.check_mutations(source, mutations, equivalent, output, 30)
    assert summary["status"] == "failed"
    assert summary["equivalent"]["outcome"] == "survived"
    assert [x["outcome"] for x in summary["results"]] == ["killed", "error", "error", "error"]
    assert summary["killed"] == 1
    assert summary["source_sha256"]["subject.py"] == hashlib.sha256(code).hexdigest()
    assert (source / "subject.py").read_bytes() == code
    assert not (source / "__pycache__").exists()
    assert len(list(output.glob("run-*/manifest.json"))) == 1


def test_cli_unknown_id_is_rejected():
    result = subprocess.run([sys.executable, str(Path(harness.__file__)), "--only", "M404"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
