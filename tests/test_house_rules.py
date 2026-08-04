"""Tests for the house-rule checker, seeded with real violations.

A checker nobody has seen fail is a checker that does not work. Every rule here
gets a fixture that breaks it, because the failure mode this whole file exists
to prevent is a green CI job that greps for the wrong thing and reports nothing
for a year. The 2026-07-28 rewrite of 23 of 25 repositories happened with CI
green the entire time.

The offending characters and strings below are built with ``chr()`` and with
string concatenation rather than written out. If they were written out this test
file would itself violate the rules it tests, and the checker would then need an
exemption for the file that proves the checker works, which is a hole in exactly
the wrong place.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER_PATH = os.path.join(REPO, "scripts", "check_house_rules.py")

# scripts/ is not a package, on purpose: the checker has to be runnable as a
# bare file by CI and by a git hook, neither of which installs anything. The
# sys.modules registration is not optional. Under "from __future__ import
# annotations" a frozen dataclass resolves its field types through
# sys.modules[cls.__module__], so a module loaded by spec and never registered
# raises AttributeError at class creation time.
_spec = importlib.util.spec_from_file_location("plr_lr_house_rules", CHECKER_PATH)
assert _spec is not None and _spec.loader is not None
house_rules = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = house_rules
_spec.loader.exec_module(house_rules)

EM_DASH = chr(0x2014)
RIGHT_ARROW = chr(0x2192)
LEFT_CURLY_QUOTE = chr(0x201C)
E_ACUTE = chr(0x00E9)

# Split so the literal never appears in this file. See the module docstring.
CO_AUTHOR_TRAILER = "Co-Authored" + "-By: Someone <someone@example.com>"
GENERATED_STRING = "Generated" + " with a tool"

CLEAN = "\n".join([
  "# a clean file",
  "",
  "def f(x):",
  "  return x + 1  # ascii only, arrows written as ->",
  "",
])


def _write(tmp_path, name, text, encoding="utf-8"):
  path = tmp_path / name
  path.write_bytes(text.encode(encoding))
  return str(path)


def _categories(violations):
  return sorted({v.category for v in violations})


def test_clean_fixture_reports_nothing(tmp_path):
  path = _write(tmp_path, "clean.py", CLEAN)
  found, skipped = house_rules.scan_paths(str(tmp_path), [path])
  assert found == []
  assert skipped == []


def test_non_ascii_character_is_caught(tmp_path):
  path = _write(tmp_path, "notes.md", "first line\nresum" + E_ACUTE + " here\n")
  found, _ = house_rules.scan_paths(str(tmp_path), [path])
  assert len(found) == 1
  assert found[0].category == "non-ascii"
  assert found[0].line == 2
  assert found[0].column == 6  # 1-based, after "resum"
  assert "U+00E9" in found[0].detail
  assert found[0].path == path


def test_em_dash_is_caught_and_named(tmp_path):
  path = _write(tmp_path, "readme.md", "a sentence " + EM_DASH + " and its clause\n")
  found, _ = house_rules.scan_paths(str(tmp_path), [path])
  assert len(found) == 1
  assert found[0].category == "typography"
  assert "em dash" in found[0].detail
  assert "--" in found[0].detail


def test_curly_quote_is_caught(tmp_path):
  path = _write(tmp_path, "doc.md", "he said " + LEFT_CURLY_QUOTE + "hello\n")
  found, _ = house_rules.scan_paths(str(tmp_path), [path])
  assert _categories(found) == ["typography"]
  assert "quote" in found[0].detail


def test_unicode_arrow_is_caught_with_the_ascii_spelling(tmp_path):
  path = _write(tmp_path, "flow.md", "capture " + RIGHT_ARROW + " screen\n")
  found, _ = house_rules.scan_paths(str(tmp_path), [path])
  assert _categories(found) == ["typography"]
  assert '"->"' in found[0].detail


def test_co_author_trailer_is_caught(tmp_path):
  path = _write(tmp_path, "MESSAGE.txt", "a change\n\n" + CO_AUTHOR_TRAILER + "\n")
  found, _ = house_rules.scan_paths(str(tmp_path), [path])
  assert len(found) == 1
  assert found[0].category == "attribution"
  assert found[0].line == 3


def test_attribution_phrase_is_caught_case_insensitively(tmp_path):
  path = _write(tmp_path, "notes.txt", GENERATED_STRING.lower() + "\n")
  found, _ = house_rules.scan_paths(str(tmp_path), [path])
  assert _categories(found) == ["attribution"]


def test_all_three_categories_fire_from_one_seeded_file(tmp_path):
  """The required seed: one violation of each content category, one fixture."""
  text = "\n".join([
    "line one is fine",
    "caf" + E_ACUTE + " is not ascii",
    "a clause " + EM_DASH + " a dash",
    CO_AUTHOR_TRAILER,
    "",
  ])
  path = _write(tmp_path, "seeded.md", text)
  found, _ = house_rules.scan_paths(str(tmp_path), [path])
  assert _categories(found) == ["attribution", "non-ascii", "typography"]
  by_line = {v.line: v.category for v in found}
  assert by_line == {2: "non-ascii", 3: "typography", 4: "attribution"}


def test_extensions_other_than_py_are_checked(tmp_path):
  """Rule 1 is over all text extensions. Curly quotes arrive in markdown, not code."""
  names = ["a.md", "b.yml", "c.toml", "d.txt", "e.cfg", "f.json", "g"]
  paths = [_write(tmp_path, name, "x " + EM_DASH + " y\n") for name in names]
  found, skipped = house_rules.scan_paths(str(tmp_path), paths)
  assert skipped == []
  assert len(found) == len(names)


def test_binary_file_is_skipped_and_reported_as_skipped(tmp_path):
  path = tmp_path / "frame.png"
  path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe" + EM_DASH.encode("utf-8"))
  found, skipped = house_rules.scan_paths(str(tmp_path), [str(path)])
  assert found == []
  assert skipped == [str(path)]


def test_undecodable_bytes_are_reported_not_dropped(tmp_path):
  path = _write(tmp_path, "latin.txt", "caf" + E_ACUTE + "\n", encoding="latin-1")
  found, _ = house_rules.scan_paths(str(tmp_path), [str(path)])
  assert _categories(found) == ["non-ascii"]
  assert "0xe9" in found[0].detail


def test_a_real_replacement_character_is_still_not_ascii(tmp_path):
  """U+FFFD is a codepoint a valid UTF-8 file can hold, not only a decode artifact.

  Suppressing it unconditionally would be an exemption granted to the one
  character that looks like an error, and it is the obvious place to hide a byte.
  """
  path = _write(tmp_path, "valid.md", "ok " + chr(0xFFFD) + " ok\n")
  found, _ = house_rules.scan_paths(str(tmp_path), [path])
  assert _categories(found) == ["non-ascii"]
  assert "U+FFFD" in found[0].detail


def test_findings_never_echo_the_offending_character(tmp_path):
  """A report that is not ASCII cannot be pasted anywhere without violating rule 1."""
  text = "caf" + E_ACUTE + " " + EM_DASH + " " + RIGHT_ARROW + "\n"
  path = _write(tmp_path, "mixed.md", text)
  found, _ = house_rules.scan_paths(str(tmp_path), [path])
  assert len(found) == 3
  for violation in found:
    violation.as_line().encode("ascii")


def test_history_attribution_is_caught_in_commit_bodies():
  records = [("a" * 40, "a real subject line\n\n" + CO_AUTHOR_TRAILER)]
  found = house_rules.history_violations(records)
  assert _categories(found) == ["attribution"]
  assert found[0].path.startswith("commit aaaaaaaaaaaa")


def test_history_typography_is_caught_in_commit_bodies():
  records = [("b" * 40, "subject " + EM_DASH + " with a dash")]
  found = house_rules.history_violations(records)
  assert _categories(found) == ["typography"]


def test_history_clean_bodies_report_nothing():
  records = [("c" * 40, "add the screen\n\nrefuses when the frame is stale.")]
  assert house_rules.history_violations(records) == []


def test_identity_single_expected_pair_is_clean():
  assert house_rules.identity_violations(["di-omics|di-omics"] * 12) == []


def test_identity_foreign_author_is_flagged():
  found = house_rules.identity_violations(["di-omics|di-omics", "Someone Else|di-omics"])
  assert _categories(found) == ["identity"]
  assert "Someone Else|di-omics" in found[0].detail


def test_identity_two_distinct_pairs_are_both_reported():
  found = house_rules.identity_violations(["a|a", "b|b", "di-omics|di-omics"])
  assert len(found) == 2


def test_identity_empty_history_is_a_failure_not_a_pass():
  """A shallow clone must not read as a clean history. See section 3.5's rule."""
  found = house_rules.identity_violations([])
  assert len(found) == 1
  assert "shallow" in found[0].detail


def test_main_exits_one_on_a_seeded_violation_and_zero_when_clean(tmp_path, capsys):
  dirty = _write(tmp_path, "dirty.md", "x " + EM_DASH + " y\n")
  clean = _write(tmp_path, "clean.md", "x -- y\n")
  assert house_rules.main(["--root", str(tmp_path), "--paths", clean]) == 0
  assert house_rules.main(["--root", str(tmp_path), "--paths", dirty]) == 1
  out = capsys.readouterr().out
  assert "typography" in out
  assert "1 violation(s)" in out


def test_checker_and_its_own_tests_pass_the_check():
  """No self-exemption, so the checker and this file have to satisfy the rules."""
  names = [
    os.path.join("scripts", "check_house_rules.py"),
    os.path.join("tests", "test_house_rules.py"),
    os.path.join(".github", "workflows", "house-rules.yml"),
    os.path.join(".github", "workflows", "tests.yml"),
  ]
  present = [n for n in names if os.path.isfile(os.path.join(REPO, n))]
  assert len(present) == len(names), "expected all four files to exist"
  found, _ = house_rules.scan_paths(REPO, present)
  assert found == [], "\n".join(v.as_line() for v in found)


@pytest.mark.skipif(
  subprocess.call(["git", "-C", REPO, "rev-parse", "--git-dir"],
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0,
  reason="not a git checkout")
def test_every_tracked_file_in_this_repo_is_clean():
  found, _ = house_rules.scan_paths(REPO, house_rules.tracked_files(REPO))
  assert found == [], "\n".join(v.as_line() for v in found)
