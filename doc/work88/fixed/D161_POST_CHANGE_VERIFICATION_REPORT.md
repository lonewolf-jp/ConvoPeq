# D161 — Snapshot Staleness Guard Post-Change Verification（read-only）

```text
Production source 変更: 0 / Test source 変更: 0 / CMake 変更: 0 / Build: NOT RUN / CTest: NOT RUN / stress: 0
D1〜D6 再開: 0 / --deep 実装: 0 / CLOSED 領域再監査: 0 / 新規監査体系追加: 0（完全 read-only）
対象: D160 CR-2026-09-01-01 の変更実体（output_sourcecode_markdown.py + ConvoPeq.md 派生更新）
性格: D160 受入れを最終確定するための **一度だけの軽量 verification**（以後、通常開発変更に毎回 D 番号監査を付けない — D159 原則の維持）
```

## 総合判定

> ## **D161 PASS — D160 受入れ確定。以後、通常開発へ戻る。**

---

## ① 通常生成経路のコード確認（非改変）

- `git diff --numstat output_sourcecode_markdown.py` = **84 追加 / 0 削除**。diff が純追加であるため、既存関数（`combine_source_codes` / `generate_tree` / `is_subpath_of_target` / `TARGET_ITEMS` / `IGNORE_EXTS` 定義）への変更は構造的に不可能（削除行 0 件を実測で確認）。
- `__main__` の引数なし実行は従来どおり `combine_source_codes(".") → ConvoPeq.md` を呼ぶ（後方互換維持）。`--help` で両モードが正しく表示されることを実測。
- T3（D160）で引数なし生成を実際に実行済み → 動作としても正常（`Generated: 15:30:13` を出力）。

## ② `--check` 分岐の scope containment

- `__main__` 構造: `if args.check: → check_freshness() → sys.exit()` であり、**生成呼び出しに到達しない**（コード確認）。
- `--check` 経路の関数（`iter_target_files` / `parse_generated_stamp` / `check_freshness`）の `open()` は **read のみ**（`grep open(` 実測: `'w'` は既存 `combine_source_codes` 内の 1 箇所のみ）。`--check` はファイルを書き換えない。
- `--deep` は実装されていない（`grep -c deep` = **0 件**実測）。scope lock 遵守。

## ③ 変更ファイル境界

| ファイル | 実測 diff | 分類 |
|---|---|---|
| `output_sourcecode_markdown.py` | +84 / −0 | D160 の本変更（Python ツールのみ） |
| `ConvoPeq.md` | +1 / −1（stamp 行のみ） | **D160 T3 の派生成果物** — production source change として扱わない |
| `src/**` / `CMakeLists.txt` / `build.bat` | **diff 0 件** | 変更なし |

## ④ D160 確定仕様（固定）

```text
TARGET_ITEMS / target traversal（生成器と完全共有: is_subpath_of_target + IGNORE_EXTS・出力自身を除外）
        ↓
baseline = ConvoPeq.md の Generated: タイムスタンプ
        ↓
判定は strict '>'（mtime == Generated → OK / mtime > Generated → stale）
        ↓
NEWER_SRC_COUNT 出力（stale 一覧 = 相対パス + mtime）
        ↓
exit 0 = FRESH
exit 1 = STALE / malformed snapshot（stamp 不在含む）
--deep = 未実装（将来別 CR で要求する）
```

## 派生 snapshot の現行基準（固定）

D160 T3 の再生成により、**現行派生 snapshot = `ConvoPeq.md` `Generated: 2026-09-01 15:30:13`**。以後のソース監査・scope 確認はこの生成物を現行派生 snapshot として扱い、旧 stamp（08:11:53 等）を引用しない。再生成が行われた場合はその都度新しい stamp に更新する。`--check`（exit 0）が鮮度の機械判定としてこれを代替する。

付記: 再生成後の `ConvoPeq.md` diff が stamp 行 1 行のみであったことは、ソースツリーが 08:11:53 世代から変化していないことの副次的証明でもある。

## 実測チェック一覧（⑩項目・全 PASS）

| # | 項目 | 実測 |
|---|---|---|
| 1 | diff stat が 2 ファイルのみ | PASS |
| 2 | src/CMakeLists.txt/build.bat diff = 空 | PASS |
| 3 | .py diff が定義部に触れない | PASS（削除 0 行で構造的に保証） |
| 4 | `--deep` 不在 | PASS（0 件） |
| 5 | `__main__` 分離（check → exit、生成に到達しない） | PASS |
| 6 | .py diff = 84/0（純追加） | PASS |
| 7 | ConvoPeq.md 差分 = stamp 行のみ | PASS |
| 8 | snapshot に .py 自身は未収録 | PASS（0 件） |
| 9 | 新規関数の open は read のみ | PASS |
| 10 | read-only 再実行: `--check` = FRESH / exit 0 | PASS |

## 遷移

```text
D159 Project Closure
   ↓
D160 CR-2026-09-01-01（Snapshot Staleness Guard 実装・T1〜T3 PASS）
   ↓
D161 Post-Change Verification PASS   ← 本報告・D160 受入れ最終確定
   ↓
Normal Development（新しい Change Request が発生したときだけ評価・実装）
```
