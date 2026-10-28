import os
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path

# --- 設定項目 ---
# 結合したいルート直下の「フォルダ名」や「ファイル名」を記述してください
# ※ このフォルダの配下にあるサブフォルダやファイルも自動ですべて処理されます
TARGET_ITEMS = {'src', 'build.bat', 'CMakeLists.txt'}

# 安全のために常に除外するバイナリ拡張子
IGNORE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.pyc', '.exe', '.dll', '.so'}

# コードブロックのシンタックスハイライト用マッピング
EXT_TO_LANG = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
    '.tsx': 'tsx', '.jsx': 'jsx', '.html': 'html', '.css': 'css',
    '.json': 'json', '.md': 'markdown', '.sh': 'bash',
    '.yml': 'yaml', '.yaml': 'yaml',
}

def ensure_within_root(path: Path, root_path: Path) -> Path:
    """path が root_path 配下に収まることを検証して返す（path traversal 防護の単一実装）。

    外部由来パス（CLI 引数・iterdir/rglob の列挙結果）は使用前に必ずこの関数を通す。
    逸脱した場合は ValueError で即座に中断する。

    比較は「解決後 vs 解決後」で行う。Windows の 8.3 短縮名（RUNNER~1 など）では
    path.resolve() が長い名前を返し、未解決の root と一致しないため。
    """
    resolved = path.resolve()
    root_resolved = Path(root_path).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError("path escapes the project root: %s" % path)
    return resolved

def resolve_safe_output_path(root_path: Path, output_file) -> Path:
    """出力先パスを安全に解決する。

    - 相対パスのみ許可（絶対パス・上位ディレクトリ参照は拒否）
    - 解決後のパスがプロジェクトルート直下に収まることを保証する
    """
    out_path = Path(output_file)
    resolved = ensure_within_root(root_path / out_path, root_path)
    if out_path.is_absolute() or resolved.parent != Path(root_path).resolve():
        raise ValueError("output file must be a relative path directly under the project root: %r" % (output_file,))
    return resolved

def is_subpath_of_target(path: Path, root_path: Path) -> bool:
    """指定されたパスが、TARGET_ITEMSで指定されたいずれかのフォルダ配下、またはファイル自身であるか判定"""
    try:
        relative = path.relative_to(root_path)
        parts = relative.parts
        if not parts:
            return False
        # 最上位の親要素（またはファイル自身）が TARGET_ITEMS に含まれているか
        return parts[0] in TARGET_ITEMS
    except ValueError:
        return False

def iter_tree_items(dir_path: Path, root_path: Path):
    """dir_path 直下のうち、ターゲット配下で除外拡張子以外の項目を Path で返す"""
    for entry in sorted(dir_path.iterdir()):
        try:
            item_path = ensure_within_root(entry, root_path)
        except ValueError:
            continue
        if item_path.suffix in IGNORE_EXTS:
            continue
        if is_subpath_of_target(item_path, root_path):
            yield item_path

def generate_tree(dir_path, root_path, prefix="", is_last=True):
    """ターゲットに合致するフォルダ・ファイルのみでツリー図を生成（再帰関数）"""
    dir_path = ensure_within_root(Path(dir_path), root_path)
    dir_name = dir_path.name

    # ルート自身以外のとき、ターゲット配下でなければツリーに含めない
    if dir_path != root_path and not is_subpath_of_target(dir_path, root_path):
        return ""

    tree_str = prefix + ("└── " if is_last else "├── ") + dir_name + "/\n"
    prefix += "    " if is_last else "│   "

    try:
        valid_items = list(iter_tree_items(dir_path, root_path))
        for idx, item_path in enumerate(valid_items):
            item_is_last = (idx == len(valid_items) - 1)

            if item_path.is_dir():
                tree_str += generate_tree(item_path, root_path, prefix, item_is_last)
            else:
                tree_str += prefix + ("└── " if item_is_last else "├── ") + item_path.name + "\n"
    except PermissionError:
        pass

    return tree_str

def iter_source_files(root_path: Path, out_path: Path):
    """結合対象ファイルを列挙する（ensure_within_root で全境界を検証済み）。"""
    for path in sorted(root_path.rglob("*")):
        try:
            file_path = ensure_within_root(path, root_path)
        except ValueError:
            continue
        if not file_path.is_file():
            continue
        if not is_subpath_of_target(file_path, root_path):
            continue
        if file_path.suffix in IGNORE_EXTS or file_path == out_path:
            continue
        yield file_path

def build_markdown_text(root_path: Path, out_path: Path, generated_at: datetime) -> str:
    """結合 Markdown の本文を組み立てる（ファイルへは書かない）。

    out_path は「生成対象から除外するファイル」を決めるためだけに使う。
    スナップショット検証でも同じ本文を再現できるよう、書き出しと分離している。
    """
    # 8.3 短縮名などで「解決済み」と「未解決」が混ざるとツリー生成が空になるため、
    # ここで必ず正規化する（root と path の比較を同一表現に揃える）。
    root_path = Path(root_path).resolve()
    out_path = Path(out_path).resolve()
    stamp = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    parts = []
    parts.append(f"# Project Extract & Source Code: {root_path.name}\n\n")
    parts.append(f"> Generated: {stamp}\n\n")
    parts.append("## 📁 Directory Tree (Selected Targets Only)\n\n")
    parts.append("```text\n")
    parts.append(generate_tree(root_path, root_path, is_last=True))
    parts.append("```\n\n")
    parts.append("## 📄 Source Code Contents\n\n")

    for file_path in iter_source_files(root_path, out_path):
        relative_path = file_path.relative_to(root_path)
        lang = EXT_TO_LANG.get(file_path.suffix, "")
        parts.append(f"### 📄 `{relative_path}`\n\n")
        parts.append(f"```{lang}\n")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as sf:
                parts.append(sf.read())
        except Exception as e:
            parts.append(f"/* [Error] ファイルの読み込みに失敗しました: {e} */")
        parts.append("\n```\n\n")

    return "".join(parts)


def combine_source_codes(root_dir, output_file, generated_at=None):
    """指定されたフォルダ配下の全ファイルを再帰的に結合してMarkdownを出力

    generated_at を渡すと、ヘッダの stamp とスナップショット名を同じ時刻に揃えられる。
    """
    root_path = Path(root_dir).resolve()
    out_path = resolve_safe_output_path(root_path, output_file)
    when = generated_at or datetime.now()
    text = build_markdown_text(root_path, out_path, when)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return when


def iter_target_files(root_path: Path):
    """--check 用: TARGET_ITEMS 配下（または指定ファイル自身）の検査対象ファイルを列挙する。

    combine_source_codes と同一の対象集合（is_subpath_of_target + IGNORE_EXTS）を
    ここからのみ導出するため、生成対象が増えても check の取りこぼしが発生しない。
    出力 Markdown 自身は検査対象から除外する。
    """
    output_file = resolve_safe_output_path(root_path, "ConvoPeq.md")
    for path in sorted(root_path.rglob("*")):
        try:
            file_path = ensure_within_root(path, root_path)
        except ValueError:
            continue
        if not file_path.is_file():
            continue
        if not is_subpath_of_target(file_path, root_path):
            continue
        if file_path.suffix in IGNORE_EXTS or file_path == output_file:
            continue
        yield file_path


def parse_generated_stamp(snapshot_path: Path):
    """ConvoPeq.md の先頭付近から Generated: YYYY-MM-DD HH:MM:SS を抽出する。"""
    with open(snapshot_path, 'r', encoding='utf-8', errors='replace') as f:
        for _ in range(20):
            line = f.readline()
            if not line:
                break
            stripped = line.strip().lstrip('>').strip()
            if stripped.startswith('Generated:'):
                stamp_str = stripped[len('Generated:'):].strip()
                return datetime.strptime(stamp_str, "%Y-%m-%d %H:%M:%S")
    return None


# =============================================================================
# スナップショット版管理（Owner 承認 2026-10-28 / P2 案 (b)）
#   * 正本は従来どおり固定名 ConvoPeq.md のまま維持する。
#   * 生成のたびに doc/ConvoPeqMD/<YYYYmmddHHMMSS>/ConvoPeq.md へ複製を残す
#     （リポジトリ既存の規約。doc/ConvoPeqMD/ は .gitignore 済み）。
#   * 既存スナップショットは上書きしない。同一内容なら何もしない（冪等）。
#   * 生成物の自動削除は行わない。
# =============================================================================
SNAPSHOT_STAMP_FORMAT = "%Y%m%d%H%M%S"
SNAPSHOT_FILE_NAME = "ConvoPeq.md"
DEFAULT_ARCHIVE_DIR = Path("doc") / "ConvoPeqMD"
SNAPSHOT_DIR_RE = re.compile(r"^(\d{14})(?:-(\d+))?$")


def build_snapshot_dir_name(generated_at: datetime) -> str:
    """スナップショットのディレクトリ名を生成する（ヘッダ stamp と同じ時刻書式）。"""
    return generated_at.strftime(SNAPSHOT_STAMP_FORMAT)


def resolve_archive_dir(root_path: Path, archive_dir) -> Path:
    """アーカイブ先を解決する（相対パス限定・ルート配下を強制）。"""
    rel = Path(archive_dir)
    if rel.is_absolute():
        raise ValueError("archive dir must be a relative path under the project root: %r" % (archive_dir,))
    return ensure_within_root(root_path / rel, root_path)


def archive_snapshot(root_path: Path, source_name: str, generated_at: datetime,
                     archive_dir=DEFAULT_ARCHIVE_DIR):
    """生成済み ConvoPeq.md を doc/ConvoPeqMD/<stamp>/ConvoPeq.md へ保存する。

    戻り値: (保存先ファイル Path または None, 警告メッセージのリスト)
    """
    root_path = Path(root_path).resolve()
    src = resolve_safe_output_path(root_path, source_name)
    notes = []
    if not src.exists():
        return None, ["snapshot source not found: %s" % (src,)]

    data = src.read_bytes()
    dest_root = resolve_archive_dir(root_path, archive_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    base = build_snapshot_dir_name(generated_at)
    candidate_dir = dest_root / base
    index = 2
    while candidate_dir.exists():
        candidate_file = candidate_dir / SNAPSHOT_FILE_NAME
        try:
            if candidate_file.is_file() and candidate_file.read_bytes() == data:
                # 同一内容が既にある: 何もせず終了（上書きも複製もしない）
                return None, []
        except OSError as exc:
            notes.append("could not read existing snapshot %s: %s" % (candidate_file, exc))
        candidate_dir = dest_root / ("%s-%d" % (base, index))
        index += 1

    candidate_dir.mkdir(parents=True, exist_ok=False)
    target = candidate_dir / SNAPSHOT_FILE_NAME
    target.write_bytes(data)
    return target, notes


def _normalize_stamp_lines(text: str) -> str:
    return "\n".join(
        "> Generated: <stamp>" if line.startswith("> Generated:") else line
        for line in text.splitlines()
    )


def _resolve_snapshot_pair(root_path: Path, snapshot_rel: str):
    """スナップショット指定（ディレクトリでもファイルでも可）を (dir, file) に解決する。"""
    target = ensure_within_root(root_path / Path(snapshot_rel), root_path)
    if target.is_dir():
        return target, target / SNAPSHOT_FILE_NAME
    return target.parent, target


def verify_snapshot_identity(root_path: Path, snapshot_rel: str):
    """スナップショットのディレクトリ名 stamp・ヘッダ stamp・内容の一致を検証する。

    戻り値: (ok: bool, report: str)
    """
    root_path = Path(root_path).resolve()
    try:
        snap_dir, snap_file = _resolve_snapshot_pair(root_path, snapshot_rel)
    except ValueError as exc:
        return False, "ERROR: %s\n" % exc

    if not snap_file.is_file():
        return False, "ERROR: snapshot not found: %s\n" % snap_file

    match = SNAPSHOT_DIR_RE.match(snap_dir.name)
    if not match:
        return False, ("ERROR: snapshot directory does not match <YYYYmmddHHMMSS>: %s\n" % snap_dir.name)

    name_stamp = datetime.strptime(match.group(1), SNAPSHOT_STAMP_FORMAT)
    header_stamp = parse_generated_stamp(snap_file)
    if header_stamp is None:
        return False, "ERROR: 'Generated:' header not found in %s\n" % snap_file.name

    canonical = resolve_safe_output_path(root_path, "ConvoPeq.md")
    fresh = build_markdown_text(root_path, canonical, header_stamp)

    actual = snap_file.read_text(encoding="utf-8", errors="replace")
    name_matches = (name_stamp == header_stamp)
    content_matches = (_normalize_stamp_lines(actual) == _normalize_stamp_lines(fresh))

    lines = [
        "snapshot           : %s/%s" % (snap_dir.name, snap_file.name),
        "name stamp         : %s" % name_stamp.strftime("%Y-%m-%d %H:%M:%S"),
        "header stamp       : %s" % header_stamp.strftime("%Y-%m-%d %H:%M:%S"),
        "name == header     : %s" % ("OK" if name_matches else "MISMATCH"),
        "content == current : %s" % ("OK" if content_matches else "DIFFERS"),
    ]
    ok = name_matches and content_matches
    lines.append("STATUS             : %s" % ("OK" if ok else "FAILED"))
    return ok, "\n".join(lines) + "\n"


def check_freshness(root_dir, snapshot_file):
    """鮮度検査: snapshot の Generated より新しい対象ファイルがあるかを報告する。

    判定は strict '>': mtime == Generated → OK / mtime > Generated → stale。
    戻り値: (exit_code, printed_report)
    """
    root_path = Path(root_dir).resolve()
    # ディレクトリ指定（doc/ConvoPeqMD/<stamp>）でもファイル指定でも受け付ける
    try:
        snapshot_dir, snapshot_path = _resolve_snapshot_pair(root_path, snapshot_file)
    except ValueError as exc:
        return 1, f"ERROR: {exc}\n"

    if not snapshot_path.exists():
        return 1, f"ERROR: {snapshot_path} が存在しません。先に生成を実行してください。\n"

    baseline = parse_generated_stamp(snapshot_path)
    if baseline is None:
        return 1, f"ERROR: {snapshot_path} に 'Generated: YYYY-MM-DD HH:MM:SS' ヘッダが見つかりません。\n"

    stale_files = []
    for file_path in iter_target_files(root_path):
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        if mtime > baseline:
            stale_files.append((file_path.relative_to(root_path), mtime))

    lines = []
    lines.append(f"baseline Generated : {baseline.strftime('%Y-%m-%d %H:%M:%S')}  ({snapshot_path})")
    lines.append(f"NEWER_SRC_COUNT    : {len(stale_files)}")
    if stale_files:
        lines.append("STATUS             : STALE — snapshot が現行ソースを反映していません")
        for rel, mtime in stale_files:
            lines.append(f"  newer: {rel}  (mtime {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        lines.append("推奨対処           : python output_sourcecode_markdown.py を実行して再生成してください。")
        lines.append("（監査・scope 確認の baseline として旧 stamp を引用しないこと）")
        return 1, "\n".join(lines) + "\n"

    lines.append("STATUS             : FRESH — snapshot は現行ソースを反映しています")
    return 0, "\n".join(lines) + "\n"


if __name__ == "__main__":
    # Owner 承認 2026-10-28 (P2): CI（例: cp1252 のコンソール）では日本語メッセージの
    # print が UnicodeEncodeError となり、生成前に異常終了していた。
    # 既存のエンコーディングは変えず、表現できない文字だけ置換して継続する。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="ソースコード結合 Markdown の生成 / 鮮度検査 / スナップショット版管理")
    parser.add_argument("--check", action="store_true",
                        help="生成を行わず、対象スナップショットが現行ソースより古いか（NEWER_SRC_COUNT）を検査する")
    parser.add_argument("--snapshot", default=None,
                        help="--check の対象を明示指定する（ルート配下の相対パス。既定は ConvoPeq.md）")
    parser.add_argument("--no-archive", action="store_true",
                        help="doc/ConvoPeqMD へのスナップショット保存を行わない")
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR),
                        help="スナップショット保存先（ルート配下の相対パス。既定 doc/ConvoPeqMD）")
    parser.add_argument("--verify-snapshot", default=None,
                        help="指定スナップショットのファイル名 stamp・ヘッダ stamp・内容を検証する")
    args = parser.parse_args()

    target_directory = "."
    output_markdown = "ConvoPeq.md"

    if args.check:
        snapshot_name = args.snapshot or output_markdown
        exit_code, report = check_freshness(target_directory, snapshot_name)
        print(report, end="")
        sys.exit(exit_code)

    if args.verify_snapshot:
        root_path = Path(target_directory).resolve()
        ok, report = verify_snapshot_identity(root_path, args.verify_snapshot)
        print(report, end="")
        sys.exit(0 if ok else 1)

    generated_at = datetime.now()
    print(f"[{generated_at.strftime('%H:%M:%S')}] 処理開始: {os.path.abspath(target_directory)}")
    combine_source_codes(target_directory, output_markdown, generated_at)
    print(f"[{generated_at.strftime('%H:%M:%S')}] 完了しました！ 出力先: {output_markdown}")

    if args.no_archive:
        print("スナップショット保存はスキップされました（--no-archive）。")
    else:
        root_path = Path(target_directory).resolve()
        saved, notes = archive_snapshot(root_path, output_markdown, generated_at, args.archive_dir)
        for note in notes:
            print(f"[WARN] {note}")
        if saved is not None:
            print(f"スナップショット保存: {saved}")
        else:
            print("スナップショット: 同一内容が既に存在するため追加保存なし。")
