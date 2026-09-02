import os
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

def generate_tree(dir_path, root_path, prefix="", is_last=True):
    """ターゲットに合致するフォルダ・ファイルのみでツリー図を生成（再帰関数）"""
    dir_name = Path(dir_path).name

    # ルート自身以外のとき、ターゲット配下でなければツリーに含めない
    if dir_path != root_path and not is_subpath_of_target(Path(dir_path), root_path):
        return ""

    tree_str = prefix + ("└── " if is_last else "├── ") + dir_name + "/\n"
    prefix += "    " if is_last else "│   "

    try:
        items = sorted(os.listdir(dir_path))
        valid_items = []

        for item in items:
            item_path = Path(dir_path) / item
            if item_path.suffix in IGNORE_EXTS:
                continue
            # そのアイテム自体がターゲット配下ならツリーの候補に入れる
            if is_subpath_of_target(item_path, root_path):
                valid_items.append(item)

        for idx, item in enumerate(valid_items):
            item_path = os.path.join(dir_path, item)
            item_is_last = (idx == len(valid_items) - 1)

            if os.path.isdir(item_path):
                tree_str += generate_tree(item_path, root_path, prefix, item_is_last)
            else:
                tree_str += prefix + ("└── " if item_is_last else "├── ") + item + "\n"
    except PermissionError:
        pass

    return tree_str

def combine_source_codes(root_dir, output_file):
    """指定されたフォルダ配下の全ファイルを再帰的に結合してMarkdownを出力"""
    root_path = Path(root_dir).resolve()

    with open(output_file, 'w', encoding='utf-8') as f:
        # 1. タイトルとフォルダ構造の出力
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"# Project Extract & Source Code: {root_path.name}\n\n")
        f.write(f"> Generated: {now_str}\n\n")
        f.write("## 📁 Directory Tree (Selected Targets Only)\n\n")
        f.write("```text\n")
        f.write(generate_tree(root_path, root_path, is_last=True))
        f.write("```\n\n")
        f.write("## 📄 Source Code Contents\n\n")

        # 2. 各ファイルのコンテンツを出力
        for current_dir, _, files in os.walk(root_path):
            current_dir_path = Path(current_dir)

            # 現在のディレクトリ自体がターゲット配下でない場合は、その中のファイルもすべてスキップ
            if current_dir_path != root_path and not is_subpath_of_target(current_dir_path, root_path):
                # ファイル単体で指定されているケースを考慮し、ファイルループ側でも二重チェックします
                pass

            for file in sorted(files):
                file_path = current_dir_path / file

                # 指定ターゲット配下（または指定ファイルそのもの）でなければスキップ
                if not is_subpath_of_target(file_path, root_path):
                    continue
                # 除外拡張子、または出力ファイル自身ならスキップ
                if file_path.suffix in IGNORE_EXTS or file_path == Path(output_file).resolve():
                    continue

                relative_path = file_path.relative_to(root_path)
                lang = EXT_TO_LANG.get(file_path.suffix, "")

                f.write(f"### 📄 `{relative_path}`\n\n")
                f.write(f"```{lang}\n")

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as sf:
                        f.write(sf.read())
                except Exception as e:
                    f.write(f"/* [Error] ファイルの読み込みに失敗しました: {e} */")

                f.write("\n```\n\n")

def iter_target_files(root_path: Path):
    """--check 用: TARGET_ITEMS 配下（または指定ファイル自身）の検査対象ファイルを列挙する。

    combine_source_codes と同一の対象集合（is_subpath_of_target + IGNORE_EXTS）を
    ここからのみ導出するため、生成対象が増えても check の取りこぼしが発生しない。
    出力 Markdown 自身は検査対象から除外する。
    """
    output_file = (root_path / "ConvoPeq.md").resolve()
    for current_dir, _, files in os.walk(root_path):
        current_dir_path = Path(current_dir)
        if current_dir_path != root_path and not is_subpath_of_target(current_dir_path, root_path):
            continue
        for file in sorted(files):
            file_path = current_dir_path / file
            if not is_subpath_of_target(file_path, root_path):
                continue
            if file_path.suffix in IGNORE_EXTS or file_path.resolve() == output_file:
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


def check_freshness(root_dir, snapshot_file):
    """鮮度検査: snapshot の Generated より新しい対象ファイルがあるかを報告する。

    判定は strict '>': mtime == Generated → OK / mtime > Generated → stale。
    戻り値: (exit_code, printed_report)
    """
    root_path = Path(root_dir).resolve()
    snapshot_path = root_path / snapshot_file

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
    lines.append(f"baseline Generated : {baseline.strftime('%Y-%m-%d %H:%M:%S')}  ({snapshot_file})")
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
    parser = argparse.ArgumentParser(description="ソースコード結合 Markdown の生成 / 鮮度検査")
    parser.add_argument("--check", action="store_true",
                        help="生成を行わず、ConvoPeq.md が現行ソースより古いか（NEWER_SRC_COUNT）を検査する")
    args = parser.parse_args()

    target_directory = "."
    output_markdown = "ConvoPeq.md"

    if args.check:
        exit_code, report = check_freshness(target_directory, output_markdown)
        print(report, end="")
        sys.exit(exit_code)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 処理開始: {os.path.abspath(target_directory)}")
    combine_source_codes(target_directory, output_markdown)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 完了しました！ 出力先: {output_markdown}")
