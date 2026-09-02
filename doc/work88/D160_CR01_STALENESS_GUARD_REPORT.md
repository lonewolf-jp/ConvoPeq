# D160 — CR-2026-09-01-01 Snapshot Staleness Guard（`--check` 実装報告）

```text
Production source 変更: 0 / Test source 変更: 0 / CMake・build.bat 変更: 0 / Build: NOT RUN / CTest: NOT RUN / stress: 0
変更ファイル: output_sourcecode_markdown.py のみ（+84 行 / 0 削除）＋派生スナップショット ConvoPeq.md 再生成（15:30:13・T3 の副産物）
scope lock 遵守: --deep は実装しない（別 CR 候補として保留・D160 ゲート判定どおり）
```

## 総合判定

> ## **D160 完了 — CR-2026-09-01-01（Snapshot Staleness Guard）実装 + targeted test T1〜T3 全 PASS**

通常開発移行後の最初の Change Request。鮮度検査（`--check`）のみを実装し、semantic-consistency 検査（`--deep`）は scope lock に従い除外した。

## 実装内容（output_sourcecode_markdown.py +84 行）

1. **`iter_target_files(root_path)`** — 検査対象ファイルの列挙。既存 `combine_source_codes` と同一の判定関数（`is_subpath_of_target` + `IGNORE_EXTS`）を共用し、出力 Markdown 自身を除外。**対象集合の single source を維持**（承認条件「生成器と完全共有」）。
2. **`parse_generated_stamp(snapshot_path)`** — 先頭 20 行から `> Generated: YYYY-MM-DD HH:MM:SS` を抽出。stamp 不在時は `None`。
3. **`check_freshness(root_dir, snapshot_file)`** — baseline = `ConvoPeq.md` の `Generated:`、判定は **strict `>`**（`mtime > baseline` → stale candidate）。これは既存 governance convention（D158 の `NEWER_SRC_COUNT=0`・`find -newermt` 実測）と意味的に一致。出力: baseline stamp / NEWER_SRC_COUNT / STALE or FRESH / stale ファイル一覧（相対パス + mtime）/ 推奨対処（再生成コマンド）。戻り値 `(exit_code, report)`。
4. **CLI** — `argparse` で `--check` を追加。`--check` 時は生成を行わず検査のみ（exit 0 = FRESH / exit 1 = STALE または snapshot 異常）。引数なしは従来どおりの生成動作（後方互換）。

**`--deep`（semantic marker validation）は本 CR には含めない** — D160 ゲート判定「保留 → 初回は必須化しない」に従い、将来別 CR として要求する。

## 実装中に検出・修正したバグ

- `lstrip('>')` 後の先頭スペース残存で `startswith('Generated:')` が常に失敗（T1 初回実行で exit 1）→ `lstrip('>').strip()` に修正。エラーメッセージのタイポ（`HH:MM:S` → `HH:MM:SS`）も同時修正。**T1 が直ちにこのバグを検出した**（targeted test の有効性の実証）。

## targeted test 結果

| Test | 手順 | 期待 | 実測 | 判定 |
|---|---|---|---|---|
| **T1** | 現行ツリーで `--check` | exit 0・NEWER_SRC_COUNT=0 | `baseline 08:11:53 / NEWER_SRC_COUNT=0 / FRESH / exit=0` | **PASS** |
| **T2a** | `touch src/audioengine/AudioEngine.h`（元 mtime を事前保存）→ `--check` | exit 1・該当ファイル列挙 | `NEWER_SRC_COUNT=1 / newer: src\audioengine\AudioEngine.h (15:29:56) / STALE / exit=1` | **PASS** |
| **T2b（復元）** | 保存済み mtime へ `os.utime` 復元 → `--check` | exit 0 復帰 | `NEWER_SRC_COUNT=0 / FRESH / exit=0`（ツリーは元の mtime 状態） | **PASS** |
| **T3** | 再生成実行 → `--check` | exit 0（stamp 更新） | `Generated: 15:30:13 / NEWER_SRC_COUNT=0 / FRESH / exit=0` | **PASS** |

T2 は承認条件どおり **touch → check → restore → check を 1 セット**として実施し、作業ツリーの mtime 状態を元に復元済み。T3 の再生成は targeted test の統合確認であり、同時に `ConvoPeq.md` を本日 15:30:13 の現行世代へ更新した（監査 baseline 引用時は以後この stamp を使用）。

## 境界確認（実装後の最終確認）

- CLOSED boundary 8 領域 / closed exception 2 行: **不接触**（変更は Python ツール 1 ファイルのみ）
- DEFER trigger D1〜D6: **不発生**
- STALE 復活 / dash2 current task 化: **なし**
- Architecture Invariant: **影響なし**
- git status 実測: `M ConvoPeq.md`（派生スナップショット）+ `M output_sourcecode_markdown.py` の 2 件のみ。production source / tests / CMake / build.bat は無変更

## 遷移

```text
D159 Project Closure
   ↓
D160 CR-2026-09-01-01 APPROVED WITH SCOPE CONDITION（--check GO / --deep 保留）
   ↓
実装 + T1〜T3 PASS   ← 本報告（D160 完了）
   ↓
通常開発継続（次の変更要求を待機）
```
