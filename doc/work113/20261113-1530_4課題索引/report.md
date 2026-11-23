# 残留 4 課題の索引と状態

- 作業日時: 2026-11-13 15:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 目的: 4 課題の判定と報告書の所在を 1 枚で示す。

---

## 1. 判定

| # | 課題 | 判定 | 報告書 |
| --- | --- | --- | --- |
| 1 | BUG-2（起動・終了遅延） | **完了（原因確定）**。修正は作業ツリーに存在し、追加変更なし | `doc/work113/20261113-1530_BUG2原因確定/report.md` |
| 2 | XML 読込上限 8 MiB と書込ガード 10 MB の不整合 | **完了**。読込上限へ統一（実装・テスト・ビルド実施） | `doc/work113/20261113-1530_XML上限統一/report.md` |
| 3 | shutdown 関連の未着手事項 | **未着手**（実装承認が無いため、判断材料のみ） | `doc/work113/20261113-1530_shutdown棚卸/report.md` |
| 4 | U-3（W3・W4・W7 の停止要求 API） | **未着手**（未承認。前提の v2-a も未実装） | `doc/work113/20261113-1530_U-3棚卸/report.md` |

## 2. 今回変更したファイル

| ファイル | 変更 | 追跡状態 |
| --- | --- | --- |
| `src/InputInspection.h` | `isXmlTextWithinLimit()` を追加 | **未追跡**（B-05 の既存未コミット群） |
| `src/DeviceSettings.cpp` | 学習状態の保存ガードを読込上限へ統一 | 追跡済み（既存の未コミット変更あり） |
| `src/tests/InputInspectionTests.cpp` | X-7（上限の境界）を追加 | **未追跡** |

BUG-2・shutdown・U-3 はソース変更なし。

## 3. 基準情報

| 項目 | 値 |
| --- | --- |
| HEAD | `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`） |
| working tree | 154 件（変更 38 ／ 未追跡 116）。本作業前は 149 件で、増分は報告フォルダ 5 件 |
| `ConvoPeq.md` | 7,011,745 B / `1801bbbacf4d65484418a1cf978acf0c09fc728028b3c8c8105ddce95f25e14f` / Generated `2026-11-12 00:48:16` |
| スナップショットとの差 | **あり**。`ConvoPeq.md` は W-1・BUG-3・BUG-3b・今回の変更を反映していない。判定は working tree を直接読んで行った |

## 4. 課題間の独立性

- BUG-2 の結論（`computeMaxElementDepth` の O(n²)）は、XML 上限の仕様判断に流用していない。
- XML 上限の変更は、shutdown・U-3 の設計判断に流用していない。
- shutdown と U-3 は同じ設計系列（v2-a / v2-b）だが、U-3 の未承認を shutdown 側の承認と読み替えていない。

## 5. 対象外として触っていないもの

- BUG-2 の追加修正（不要と判断）、shutdown の実装、U-3 の実装、Git 操作（commit / push）、
  対象外のリファクタリング。
