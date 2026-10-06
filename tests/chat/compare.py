#!/usr/bin/env python3
"""Compare two JSON Lines outputs produced by tools/chat_test.cpp and render_reference.py.

    python tests/chat/compare.py reference.jsonl cpp.jsonl

Exit code 0 only when every case matches.  A case matches when both sides agree on whether it
is an error and then either the error messages or the rendered outputs are byte-for-byte equal.
"""

import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8")
import difflib
import json
import sys


def load(path):
    records = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["name"]] = record
    return records


def show_diff(name, expected, actual):
    print("MISMATCH: {}".format(name))
    expected_lines = expected.splitlines(keepends=True)
    actual_lines = actual.splitlines(keepends=True)
    diff = difflib.unified_diff(expected_lines, actual_lines, fromfile="reference", tofile="cpp")
    for chunk in diff:
        sys.stdout.write("    " + chunk)
    if not expected_lines and not actual_lines:
        print("    (both empty)")


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: compare.py reference.jsonl cpp.jsonl\n")
        return 2

    reference = load(sys.argv[1])
    actual = load(sys.argv[2])

    failures = 0
    names = list(dict.fromkeys(list(reference.keys()) + list(actual.keys())))
    for name in names:
        expected = reference.get(name)
        got = actual.get(name)
        if expected is None or got is None:
            print("MISMATCH: {} (missing from {})".format(name, "reference" if expected is None else "cpp"))
            failures += 1
            continue

        expected_error = "error" in expected
        actual_error = "error" in got
        if expected_error != actual_error:
            print("MISMATCH: {} (one side errored)".format(name))
            print("    reference: {}".format(expected.get("error", expected.get("output", ""))))
            print("    cpp:       {}".format(got.get("error", got.get("output", ""))))
            failures += 1
        elif expected_error:
            if expected["error"] != got["error"]:
                print("MISMATCH: {} (different error message)".format(name))
                print("    reference: {!r}".format(expected["error"]))
                print("    cpp:       {!r}".format(got["error"]))
                failures += 1
        elif expected["output"] != got["output"]:
            show_diff(name, expected["output"], got["output"])
            failures += 1

    if failures:
        print("{} of {} case(s) mismatched".format(failures, len(names)))
        return 1
    print("all {} case(s) match".format(len(names)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
