import os
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
        f.write(f"# Project Extract & Source Code: {root_path.name}\n\n")
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

if __name__ == "__main__":
    target_directory = "."
    output_markdown = "ConvoPeq.md"
    
    print(f"指定フォルダ・ファイルの再帰処理を開始します: {os.path.abspath(target_directory)}")
    combine_source_codes(target_directory, output_markdown)
    print(f"完了しました！ 出力先: {output_markdown}")
