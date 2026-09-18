"""Audit constructor JSONL logs against their declared instance splits.

This checks evidence identity and completeness, not metric correctness or independent
authentication of the corpus. A header's test role and a checkpoint's init_summary are
not evaluation evidence. Historical quality logs may contain NaN metric values; metrics
are deliberately outside this audit's scope.

Usage: python probes/constructor_evidence_audit.py --require-heldout-test run.log
"""

import argparse
from collections import Counter
import json
from pathlib import Path


SCHEMA = "constructor-evidence-audit-v1"
PROTOCOLS = {"constructor-curriculum-gate2-v1", "constructor-quality-v2"}
SPLITS = ("train", "heldout")
COMPLETION_FOOTER = "CURRICULUM GATE 2 DONE"


class DuplicateKeyError(ValueError):
    """JSON object keys must not silently overwrite instance identities."""


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateKeyError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _metadata(value):
    """Keep only canonical metadata scalars, even when the input is malformed."""
    return value if type(value) in (str, int, bool, type(None)) else None


def audit_log(path, *, require_final=True, require_heldout_test=False):
    """Return a JSON-serializable report; invalid or unreadable logs fail closed.

    Final rows are required for each configured evaluation split by default. Setting
    require_final=False permits inspecting partial logs, but does not relax an explicit
    require_heldout_test request, which always needs a final held-out test evaluation.
    """
    path = Path(path)
    errors, evaluations, headers = [], [], []
    report = {
        "schema": SCHEMA, "path": str(path), "ok": False,
        "require_final": require_final, "require_heldout_test": require_heldout_test,
        "header": None, "evaluations": [], "heldout_test_evidence": False,
        "errors": errors,
    }

    def error(code, message, line=None, **details):
        errors.append({"code": code, "message": message,
                       **({"line": line} if line is not None else {}), **details})

    try:
        with path.open(encoding="utf-8") as stream:
            for number, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                # constructor_curriculum.run emits this after its JSON summary.
                # It is a log marker only, never a substitute for evaluation rows.
                if text.strip() == COMPLETION_FOOTER:
                    continue
                try:
                    row = json.loads(text, object_pairs_hook=_unique_object)
                except DuplicateKeyError as exc:
                    error("duplicate_json_key", str(exc), number)
                    continue
                except (ValueError, RecursionError) as exc:
                    error("malformed_json", str(exc), number)
                    continue
                if not isinstance(row, dict):
                    error("malformed_record", "each log row must be a JSON object", number)
                elif "protocol" in row:
                    headers.append((number, row))
                elif "evaluation" in row:
                    evaluations.append((number, row))
    except (OSError, UnicodeError) as exc:
        error("unreadable_log", str(exc))
        return report

    if not headers:
        error("missing_header", "no constructor protocol header was found")
        return report
    if len(headers) != 1:
        error("duplicate_header", "audit one run per log; multiple protocol headers found",
              lines=[line for line, _ in headers])
    header_line, header = headers[0]
    if not isinstance(header.get("protocol"), str) or header["protocol"] not in PROTOCOLS:
        error("unsupported_protocol", "unrecognized constructor curriculum protocol", header_line)

    def identities(value, label, line):
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            error("malformed_identities", f"{label} must be a list of nonempty string IDs", line)
            return None
        duplicate = sorted(item for item, count in Counter(value).items() if count > 1)
        if duplicate:
            error("duplicate_identity", f"{label} contains repeated IDs", line, ids=duplicate)
        return set(value)

    declared = {name: identities(header.get(name), f"header.{name}", header_line)
                for name in SPLITS}
    if all(ids is not None for ids in declared.values()):
        overlap = sorted(declared["train"] & declared["heldout"])
        if overlap:
            error("split_overlap", "train and heldout declarations overlap", header_line,
                  ids=overlap)
    role = header.get("heldout_role")
    if role not in (None, "validation", "test"):
        error("invalid_heldout_role", "heldout_role must be validation or test", header_line)
    eval_sets = header.get("eval_sets", "both")
    if eval_sets not in ("both", "heldout"):
        error("invalid_eval_sets", "eval_sets must be both or heldout", header_line)
    required = (["heldout"] if role == "test" or eval_sets == "heldout"
                or header.get("objective") == "deployment"
                or header.get("evaluation_objective") == "deployment" else list(SPLITS))
    report["header"] = {
        "line": header_line, "protocol": _metadata(header.get("protocol")),
        "heldout_role": _metadata(role), "manifest_split": _metadata(header.get("manifest_split")),
        "eval_sets": _metadata(eval_sets),
        "declared_instances": {name: len(ids) if ids is not None else None
                               for name, ids in declared.items()},
        "required_final_sets": required if require_final else [],
    }
    valid_final_sets, seen = set(), set()
    header_valid = not errors
    for line, row in evaluations:
        before = len(errors)
        tag, split = row.get("evaluation"), row.get("set")
        if line < header_line:
            error("evaluation_before_header", "evaluation precedes its run header", line)
        if not isinstance(tag, str) or not tag.strip():
            error("invalid_evaluation", "evaluation must be a nonempty string tag", line)
        if not isinstance(split, str) or split not in SPLITS:
            error("invalid_set", "evaluation set must be train or heldout", line)
        if isinstance(tag, str) and isinstance(split, str):
            key = (tag, split)
            if key in seen:
                error("duplicate_evaluation", "evaluation tag/set pair occurs more than once", line,
                      evaluation=tag, set=split)
            seen.add(key)

        count = row.get("instances")
        if type(count) is not int or count <= 0:
            error("invalid_count", "instances must be a positive integer", line)
        evidence = []
        if "per_instance" in row:
            values = row["per_instance"]
            if not isinstance(values, dict):
                error("malformed_per_instance", "per_instance must map instance IDs to values", line)
            else:
                evidence.append(("per_instance", list(values)))
        if "rows" in row:
            rows = row["rows"]
            if not isinstance(rows, list) or any(not isinstance(item, dict) for item in rows):
                error("malformed_rows", "rows must be a list of instance records", line)
            else:
                evidence.append(("rows", [item.get("instance") for item in rows]))
        if "per_instance" not in row and "rows" not in row:
            error("missing_instance_evidence", "evaluation needs per_instance or rows evidence", line)
        observed_count = None
        for label, values in evidence:
            ids = identities(values, label, line)
            if ids is None:
                continue
            observed_count = len(values)
            if type(count) is int and count != len(values):
                error("count_mismatch", f"instances disagrees with {label} length", line,
                      reported=count, observed=len(values))
            expected = declared.get(split) if isinstance(split, str) else None
            if expected is not None:
                if ids != expected:
                    error("split_mismatch", f"{label} IDs do not match declared {split} split", line,
                          set=split, missing=sorted(expected - ids), unexpected=sorted(ids - expected))
                if type(count) is int and count != len(expected):
                    error("declared_count_mismatch", "instances disagrees with declared split size",
                          line, reported=count, declared=len(expected))
        valid = len(errors) == before and header_valid
        report["evaluations"].append({
            "line": line, "evaluation": _metadata(tag), "set": _metadata(split),
            "reported_instances": _metadata(count), "observed_instances": observed_count,
            "valid": valid,
        })
        if valid and tag == "final":
            valid_final_sets.add(split)

    if not evaluations:
        error("missing_evaluation", "no evaluation rows were found")
    if require_final:
        for split in required:
            if split not in valid_final_sets:
                error("missing_final_evaluation", "no valid final evaluation for required split",
                      set=split)
    report["heldout_test_evidence"] = (
        role == "test" and header.get("manifest_split") is True
        and "heldout" in valid_final_sets and not errors
    )
    if require_heldout_test and not report["heldout_test_evidence"]:
        error("missing_heldout_test_evidence",
              "requires manifest_split=true, heldout_role=test and a valid final heldout evaluation")
    report["ok"] = not errors
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="constructor JSONL log files")
    parser.add_argument("--require-heldout-test", action="store_true",
                        help="require final evidence on the declared manifest test split")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="inspect partial logs without requiring final rows; test proof stays final")
    args = parser.parse_args(argv)
    audits = [audit_log(path, require_final=not args.allow_incomplete,
                        require_heldout_test=args.require_heldout_test) for path in args.logs]
    ok = all(audit["ok"] for audit in audits)
    print(json.dumps({"schema": SCHEMA, "ok": ok, "logs": audits}, sort_keys=True, allow_nan=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
