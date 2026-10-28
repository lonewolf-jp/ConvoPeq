#!/usr/bin/env python3
"""output_sourcecode_markdown.py のスナップショット版管理 回帰テスト。

Owner 承認 2026-10-28（P2 案 (b)）で追加した以下を検証する:

  1. 現行 ConvoPeq.md の固定名と参照互換性
  2. スナップショット名（ディレクトリ名）と stamp の一致
  3. アーカイブ先への保存動作
  4. --check による指定スナップショットの検証
  5. 既存スナップショットがある場合の衝突・再実行動作

保存規約はリポジトリ既存の doc/ConvoPeqMD/<YYYYmmddHHMMSS>/ConvoPeq.md に従う。
すべて一時ディレクトリ内で完結し、正本・保全対象には触れない。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPT = HERE.parents[2] / "output_sourcecode_markdown.py"

passed = 0
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed
    if condition:
        passed += 1
    else:
        failures.append(f"{name} {detail}".rstrip())


def load_module():
    spec = importlib.util.spec_from_file_location("osm_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_project(root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "alpha.cpp").write_text("int alpha() { return 1; }\n", encoding="utf-8")
    (root / "src" / "beta.h").write_text("#pragma once\n", encoding="utf-8")
    (root / "build.bat").write_text("@echo off\n", encoding="utf-8")
    (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
    # --check は mtime の strict '>' で判定する（stamp は秒精度）。生成より前に
    # 更新された状態を作るため、明示的に古くしておく。
    past = time.time() - 120
    for path in list((root / "src").glob("*")) + [root / "build.bat", root / "CMakeLists.txt"]:
        os.utime(path, (past, past))


def run_cli(cwd: Path, args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace(os.sep, "/")


def main() -> int:
    module = load_module()
    work = Path(tempfile.mkdtemp(prefix="osm-snapshot-tests-"))
    try:
        # =====================================================================
        # 1) 固定名 ConvoPeq.md の生成と参照互換性
        # =====================================================================
        project = work / "project"
        project.mkdir()
        make_project(project)

        code, out = run_cli(project, [])
        check("1/CLI 生成 exit", code == 0, f"actual={code} out={out[-300:]}")
        canonical = project / "ConvoPeq.md"
        check("1/ConvoPeq.md が固定名で生成される", canonical.is_file())
        if not canonical.is_file():
            raise SystemExit("canonical not generated; aborting")

        canonical_text = canonical.read_text(encoding="utf-8")
        check("1/ヘッダに Generated がある", "\n> Generated: " in canonical_text)

        header_stamp = module.parse_generated_stamp(canonical)
        check("1/ヘッダ stamp が解釈できる", header_stamp is not None)

        code_check, out_check = run_cli(project, ["--check"])
        check("1/--check 既定は ConvoPeq.md を見る", code_check == 0, f"actual={code_check} out={out_check[-300:]}")
        check("1/--check 出力に ConvoPeq.md が出る", "ConvoPeq.md" in out_check)

        # =====================================================================
        # 2) スナップショット名（ディレクトリ名）と stamp の一致
        # =====================================================================
        archive_dir = project / "doc" / "ConvoPeqMD"
        snapshots = sorted(archive_dir.glob("*/ConvoPeq.md")) if archive_dir.is_dir() else []
        check("2/アーカイブ先が作成される", archive_dir.is_dir())
        check("2/スナップショットが 1 件生成される", len(snapshots) == 1, f"actual={len(snapshots)}")

        snapshot = snapshots[0]
        expected_dir = header_stamp.strftime("%Y%m%d%H%M%S")
        check("2/ディレクトリ名が <YYYYmmddHHMMSS>", snapshot.parent.name == expected_dir,
              f"expected={expected_dir} actual={snapshot.parent.name}")
        check("2/ファイル名は固定の ConvoPeq.md", snapshot.name == "ConvoPeq.md", f"actual={snapshot.name}")

        snap_header = module.parse_generated_stamp(snapshot)
        check("2/スナップショットのヘッダ stamp がディレクトリ名と一致", snap_header == header_stamp,
              f"dir={header_stamp} header={snap_header}")

        ok, report = module.verify_snapshot_identity(project, rel(snapshot.parent, project))
        check("2/verify_snapshot_identity(ディレクトリ指定) が OK", ok, report)

        ok_file, report_file = module.verify_snapshot_identity(project, rel(snapshot, project))
        check("2/verify_snapshot_identity(ファイル指定) も OK", ok_file, report_file)

        code_verify, out_verify = run_cli(project, ["--verify-snapshot", rel(snapshot.parent, project)])
        check("2/CLI --verify-snapshot exit", code_verify == 0, f"actual={code_verify} out={out_verify[-300:]}")

        # =====================================================================
        # 3) アーカイブ先への保存動作
        # =====================================================================
        check("3/スナップショット内容が ConvoPeq.md とバイト一致",
              snapshot.read_bytes() == canonical.read_bytes())

        # =====================================================================
        # 4) --check による指定スナップショットの検証
        # =====================================================================
        rel_snapshot = rel(snapshot.parent, project)
        code_snap, out_snap = run_cli(project, ["--check", "--snapshot", rel_snapshot])
        check("4/--check --snapshot は FRESH で exit 0", code_snap == 0, f"actual={code_snap} out={out_snap[-300:]}")
        check("4/--check --snapshot 出力に対象が出る", snapshot.parent.name in out_snap)

        # ソースを新しくすると STALE になる（mtime ベースの契約）
        time.sleep(1.1)
        (project / "src" / "alpha.cpp").write_text("int alpha() { return 2; }\n", encoding="utf-8")
        future = time.time() + 5
        os.utime(project / "src" / "alpha.cpp", (future, future))

        code_stale, out_stale = run_cli(project, ["--check"])
        check("4/ソース更新後は既定 --check が STALE(exit 1)", code_stale == 1, f"actual={code_stale}")
        check("4/STALE 出力に STATUS 行がある", "STALE" in out_stale)

        code_snap2, _ = run_cli(project, ["--check", "--snapshot", rel_snapshot])
        check("4/ソース更新後は指定スナップショットも STALE(exit 1)", code_snap2 == 1, f"actual={code_snap2}")

        code_escape, _ = run_cli(project, ["--check", "--snapshot", "../outside.md"])
        check("4/ルート外のスナップショット指定は拒否(exit 1)", code_escape == 1, f"actual={code_escape}")

        # =====================================================================
        # 5) 衝突・再実行の動作（決定論のためモジュール関数を直接使う）
        # =====================================================================
        fixed = datetime(2026, 10, 28, 12, 0, 0)
        module.combine_source_codes(str(project), "ConvoPeq.md", fixed)
        saved1, notes1 = module.archive_snapshot(project, "ConvoPeq.md", fixed)
        check("5/同一 stamp で 1 件目が保存される", saved1 is not None, f"notes={notes1}")
        check("5/1 件目のディレクトリ名が <stamp>",
              saved1 is not None and saved1.parent.name == "20261028120000",
              f"actual={saved1.parent.name if saved1 else None}")
        bytes_first = saved1.read_bytes() if saved1 else b""

        # 同内容の再実行 -> 何も増えない（冪等）
        saved2, notes2 = module.archive_snapshot(project, "ConvoPeq.md", fixed)
        check("5/同内容の再実行は追加保存しない", saved2 is None, f"actual={saved2}")
        check("5/再実行で既存ファイルは上書きされない",
              saved1 is not None and saved1.read_bytes() == bytes_first)

        # 同 stamp で内容を変える -> <stamp>-2 が作られ、既存は不変
        (project / "src" / "gamma.cpp").write_text("int gamma() { return 3; }\n", encoding="utf-8")
        module.combine_source_codes(str(project), "ConvoPeq.md", fixed)
        saved3, notes3 = module.archive_snapshot(project, "ConvoPeq.md", fixed)
        check("5/内容が異なる場合は別ディレクトリで保存される", saved3 is not None, f"notes={notes3}")
        check("5/別名は <stamp>-2",
              saved3 is not None and saved3.parent.name == "20261028120000-2",
              f"actual={saved3.parent.name if saved3 else None}")
        check("5/既存スナップショットは変更されない",
              saved1 is not None and saved1.read_bytes() == bytes_first)

        # --no-archive は保存しない
        before = sorted(archive_dir.glob("*/ConvoPeq.md"))
        code_noarch, out_noarch = run_cli(project, ["--no-archive"])
        after = sorted(archive_dir.glob("*/ConvoPeq.md"))
        check("5/--no-archive でスナップショットが増えない", len(after) == len(before),
              f"before={len(before)} after={len(after)}")
        check("5/--no-archive のメッセージ", "スキップ" in out_noarch, out_noarch[-200:])

        # ディレクトリ名 stamp とヘッダ stamp の不一致を検出する
        bad_dir = archive_dir / "20200101000000"
        bad_dir.mkdir(parents=True, exist_ok=True)
        (bad_dir / "ConvoPeq.md").write_bytes(bytes_first)
        ok_bad, report_bad = module.verify_snapshot_identity(project, rel(bad_dir, project))
        check("5/ディレクトリ名とヘッダ stamp の不一致を検出する", not ok_bad, report_bad)

        # 内容だけ改変した場合は DIFFERS で失敗する
        body = bytes_first.decode("utf-8")
        broken_dir = archive_dir / "20261028130000"
        broken_dir.mkdir(parents=True, exist_ok=True)
        (broken_dir / "ConvoPeq.md").write_text(
            body.replace("int alpha()", "int alphaX(", 1), encoding="utf-8"
        )
        ok_body, report_body = module.verify_snapshot_identity(project, rel(broken_dir, project))
        check("5/内容改変を DIFFERS として検出する", (not ok_body) and ("DIFFERS" in report_body), report_body)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    total = passed + len(failures)
    print(f"snapshot tests: checks={total} passed={passed} failed={len(failures)}")
    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        return 1
    print("[PASS] snapshot versioning regression tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
