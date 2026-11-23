"""Compare atomic operations and their *effective* memory orders (explicit and implicit)
between the base commit (HEAD) and the working tree.

The previous version compared only the explicitly written memory_order_* tokens, so an
implicit default (seq_cst) that became an explicit release was invisible. This version
resolves every call to (success_order, failure_order) using the C++ defaults, so such a
change is reported as a difference.
"""

import collections
import re
import subprocess
import sys

FILES = [
    "src/audioengine/ISRRetireLatency.h",
    "src/audioengine/WorldLifetimeRecorder.h",
    "src/core/ReaderMeasureRegistry.h",
    "src/core/ScopedEpochReader.h",
    "src/audioengine/ISRDebugRuntime.cpp",
    "src/core/EpochDomain.h",
    "src/core/SnapshotCoordinator.h",
]

MEMBER = re.compile(
    r"\.(load|store|exchange|compare_exchange_weak|compare_exchange_strong"
    r"|fetch_add|fetch_sub|fetch_or|fetch_and)\s*\("
)
HELPER = re.compile(
    r"convo::(consumeAtomic|publishAtomic|exchangeAtomic|compareExchangeAtomic"
    r"|compareExchangeWeakAtomic|fetchAddAtomic|fetchSubAtomic|fetchOrAtomic|fetchAndAtomic)\s*\("
)

HELPER_OP = {
    "consumeAtomic": "load",
    "publishAtomic": "store",
    "exchangeAtomic": "exchange",
    "compareExchangeAtomic": "compare_exchange_strong",
    "compareExchangeWeakAtomic": "compare_exchange_weak",
    "fetchAddAtomic": "fetch_add",
    "fetchSubAtomic": "fetch_sub",
    "fetchOrAtomic": "fetch_or",
    "fetchAndAtomic": "fetch_and",
}

# Defaults for the standard std::atomic member functions.
MEMBER_DEFAULT = {
    "load": ("seq_cst", None),
    "store": ("seq_cst", None),
    "exchange": ("seq_cst", None),
    "fetch_add": ("seq_cst", None),
    "fetch_sub": ("seq_cst", None),
    "fetch_or": ("seq_cst", None),
    "fetch_and": ("seq_cst", None),
    "compare_exchange_strong": ("seq_cst", "seq_cst"),
    "compare_exchange_weak": ("seq_cst", "seq_cst"),
}

# Defaults for the project helpers (see AtomicAccess.h).
HELPER_DEFAULT = {
    "load": ("acquire", None),
    "store": ("release", None),
    "exchange": ("acq_rel", None),
    "fetch_add": ("acq_rel", None),
    "fetch_sub": ("acq_rel", None),
    "fetch_or": ("acq_rel", None),
    "fetch_and": ("acq_rel", None),
    "compare_exchange_strong": ("acq_rel", "acquire"),
    "compare_exchange_weak": ("acq_rel", "acquire"),
}


def split_args(args_text):
    """Split a top-level comma separated argument list (respecting nested parens/brackets)."""
    parts, depth, cur = [], 0, []
    for ch in args_text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return [p.strip() for p in parts]


def orders_in(text):
    return [o.replace("memory_order_", "") for o in re.findall(r"memory_order_(\w+)", text)]


def effective_orders(op, args, is_helper):
    """Resolve the effective (success, failure) memory orders of one call."""
    given = []
    for a in args:
        given.extend(orders_in(a))
    default = HELPER_DEFAULT if is_helper else MEMBER_DEFAULT
    succ_default, fail_default = default[op]

    if op in ("load", "store", "exchange", "fetch_add", "fetch_sub", "fetch_or", "fetch_and"):
        return (given[0] if given else succ_default, None)

    # compare-exchange: 0 orders -> both defaults; 1 order -> success=given, failure=default
    if len(given) >= 2:
        return (given[0], given[1])
    if len(given) == 1:
        return (given[0], fail_default)
    return (succ_default, fail_default)


def find_calls(text, pattern, is_helper):
    calls = []
    for m in pattern.finditer(text):
        open_idx = m.end() - 1
        depth = 0
        i = open_idx
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        args = split_args(text[open_idx + 1:i])
        op = HELPER_OP[m.group(1)] if is_helper else m.group(1)
        calls.append({
            "pos": m.start(),
            "line": text.count("\n", 0, m.start()) + 1,
            "op": op,
            "orders": effective_orders(op, args, is_helper),
            "snippet": text[m.start():i + 1].split("\n")[0].strip()[:110],
        })
    return calls


def collect(text):
    calls = find_calls(text, MEMBER, False) + find_calls(text, HELPER, True)
    calls.sort(key=lambda c: c["pos"])
    return calls


def main():
    total_ops = 0
    total_diffs = 0
    all_ok = True
    print("Effective memory-order comparison (implicit defaults included)")
    print("before = git HEAD (173e6995), after = working tree")
    print()

    for path in FILES:
        before_text = subprocess.run(
            ["git", "show", "HEAD:" + path], capture_output=True, text=True, encoding="utf-8"
        ).stdout
        after_text = open(path, encoding="utf-8").read()

        before = collect(before_text)
        after = collect(after_text)
        total_ops += len(before)

        ops_before = [c["op"] for c in before]
        ops_after = [c["op"] for c in after]
        if ops_before != ops_after:
            print(f"{path}: OPERATION SEQUENCE MISMATCH "
                  f"({len(ops_before)} vs {len(ops_after)})")
            all_ok = False
            continue

        diffs = [(b, a) for b, a in zip(before, after) if b["orders"] != a["orders"]]
        total_diffs += len(diffs)
        print(f"{path}: ops={len(before)} order_changes={len(diffs)}")
        for b, a in diffs:
            print(f"    before line {b['line']:5}  op={b['op']:24} "
                  f"{b['orders'][0]}/{b['orders'][1]} -> {a['orders'][0]}/{a['orders'][1]}")
            print(f"        {b['snippet']}")

    print()
    print(f"TOTAL_OPS={total_ops} TOTAL_ORDER_CHANGES={total_diffs}")
    print("NOTE: every reported change must be reviewed; a change outside reset() would")
    print("      mean the reset-exclusivity argument does not cover it.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
