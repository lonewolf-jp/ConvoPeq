# 課題: AC1 のミラー乖離（`allDrained` が 14 条件のまま）

- 起票日: 2026-11-13
- 種別: **独立修正課題**（C9-A の検証範囲に属する。改名課題とは独立）
- 状態: **起票のみ**。修正の実装は未承認（別途 Owner の GO が必要）
- 発見経緯: `retireBacklogCount_` 改名の実装中（2026-11-13）。報告は
  `doc/work113/20261113-2342_改名実装/report.md` §6。

---

## 1. 事象

| 対象 | 条件数 | 状態 |
| --- | ---: | --- |
| `ShutdownScheduler::isFullyDrained()` | 13 | C9-A（2026-11-13）で `retireAcceptedCount_ == 0` を除外済み |
| `testAC1_drainBreakdownMatchesPredicate` の `allDrained` ラムダ | 14 | `b.retireBacklog == 0` が残存 |

差分は 1 条件（`b.retireBacklog == 0`）だけ。

## 2. なぜ問題か

AC1 は「診断側の条件が実際の drain 述語と一致すること」を固定するためのテストである。
述語が 13 条件になったのにミラーが 14 条件のままだと、**テストが実装の述語を正しく表現していない**。

- 現状のシナリオ（fresh / publication intent 1 件）では `retireBacklog` が常に 0 のため PASS する。
  つまり **乖離を検出できない**（偽陰性）。
- 将来 `retireBacklog` が非 0 のシナリオを追加すると、AC1 は実装と無関係に FAIL する。

## 3. 修正案（1 行）

`src/tests/ISRSemanticValidationTests.cpp` の `allDrained` ラムダから次を削除する。

```cpp
&& b.retireBacklog == 0
```

`b.retireBacklog` フィールド自体（`DrainBreakdown`）と、その取得（`collectDrainBreakdown`）は
残す。診断値としては有用であり、C9-A が除外したのは**述語**であって**診断項目**ではない。

## 4. 検証計画（実装時）

| # | 検証 | 期待 |
| --- | --- | --- |
| 1 | `ISRSemanticValidationTests` | PASS（既存シナリオ） |
| 2 | **`retireBacklog` 非 0 のシナリオを追加** | `isFullyDrained()` が真（13 条件成立時）、`allDrained(b)` も真 → ミラー一致を直接確認 |
| 3 | 述語側の条件を 1 つ非 0 にしたシナリオ | 両者とも偽 |
| 4 | `ctest` 61 件 | PASS |
| 5 | ミューテーション | `allDrained` から別の条件を落とすと FAIL すること |

Owner 提案の「`retireBacklog` が非ゼロの場合に drain 判定がどうなるか」は #2 で確認する。

## 5. 制約

- `isFullyDrained()` は変更しない（13 条件を維持）。
- `collectDrainBreakdown()` の 14 フィールドは変更しない（診断は全条件を出す）。
- C3-D / C9-A の実装（本体）は変更しない。
- 改名（`retireAcceptedCount_`）は完了済み。本課題では触れない。

## 6. 承認状態

| 項目 | 状態 |
| --- | --- |
| 課題起票 | 完了（2026-11-13） |
| 修正の実装 | **未承認**（別途 GO） |
