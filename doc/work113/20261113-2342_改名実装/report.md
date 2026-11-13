# `retireBacklogCount_` 改名の実装

- 作業日時: 2026-11-13 23:42 JST
- 種別: **改名のみ**。意味論・更新経路・drain 判定・C3-D/C9-A の実装は変更していない。
- 根拠: 起票済み課題票 `doc/work113/20261113-2250_改名課題/issue.md` と Owner GO（2026-11-13）。

---

## 1. 課題票との照合（実行前の必須確認）

| 課題票の項目 | 現行コード | 一致 |
| --- | --- | --- |
| 改名候補 `retireAcceptedCount_` | 未使用（衝突なし） | OK |
| 変更対象「メンバ変数」 | `retireBacklogCount_`（宣言 1・初期化 1・参照 12） | OK |
| 変更対象「getter」 | `getRetireBacklogCount()`（宣言 1・定義 1・参照 1） | OK |
| 変更対象「TEST-ONLY setter」 | `setRetireBacklogCount()`（宣言 1・定義 1・呼出し 10） | OK |
| 関連テスト `testPressureStateNormalizationContract` | `setRetireBacklogCount` を 6 回使用 | OK（改名で追随） |
| 関連テスト `testC9A_*` | `setRetireBacklogCount(5)` を 2 回使用 | OK（改名で追随） |
| 移行時の注意「シリアライズ・外部公開名・ログ項目」 | いずれも該当なし（§4） | OK |

**不一致は無かった**ため、停止条件（課題票との不一致）には該当しない。

## 2. 変更内容

| 旧 | 新 | 変更箇所 |
| --- | --- | --- |
| `retireBacklogCount_` | `retireAcceptedCount_` | `.h` 2（宣言 1・コメント 1）、`.cpp` 12 |
| `getRetireBacklogCount` | `getRetireAcceptedCount` | `.h` 1、`.cpp` 1、テスト 1 |
| `setRetireBacklogCount` | `setRetireAcceptedCount` | `.h` 6（宣言 1・コメント 5）、`.cpp` 3、テスト 10 |

変更ファイル:

| ファイル | 内容 |
| --- | --- |
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | メンバ名・getter/setter 宣言・関連コメント、宣言部に意味と改名の注記を追加 |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | 初期化・更新・参照・getter/setter 定義・関連コメント |
| `src/tests/ISRSemanticValidationTests.cpp` | setter 呼出し 10 箇所、getter 参照 1 箇所 |

### 2.1 宣言部に追加した注記

```cpp
// ★ 改名 (2026-11-13, Owner 承認): retireBacklogCount_ → retireAcceptedCount_。
//   実体は「累積 accepted 数」で残存件数ではない（onRetireAccepted で +1、production の
//   減算経路は無い）。drain 判定からは C9-A で除外済みで、pressure の slope 検出にのみ使う。
//   注: 更新通知の private ヘルパ noteRetireBacklogChanged と previousRetireBacklogCount_ は
//   課題票の変更対象外のため旧名のまま（別途判断）。
```

## 3. 範囲外として残したもの（意図的）

| 対象 | 残した理由 |
| --- | --- |
| `noteRetireBacklogChanged()`（private ヘルパ） | 課題票の変更対象は「メンバ変数・getter・TEST-ONLY setter・関連コメント」で、メソッド名は含まれない |
| `previousRetireBacklogCount_`（private メンバ） | 同上。slope 計算用の前回値で、カウンター本体ではない |
| `DrainBreakdown::retireBacklog`（診断構造体のフィールド） | 診断専用。ログの key は `c9_rbl` で旧名を含まない |
| 他ファイルの歴史的コメント（`AudioEngine.Commit.cpp` / `AudioEngine.h` / `AudioEngine.Threading.cpp` / `invariant_INV3_INV5.cpp`） | いずれも「撤去した旧 API」を過去形で説明している。改名すると履歴の記述が変わるため触れていない |

**残存する旧名**: 実コードは 0。コメントは、上記の歴史的記述 7 箇所と、改名を記録する注記 1 箇所
（宣言部の `retireBacklogCount_ → retireAcceptedCount_`）のみ。

## 4. 移行時の注意（課題票）の確認結果

| 項目 | 確認 |
| --- | --- |
| シリアライズ | 該当なし（この counter を JSON へ書く経路は無い） |
| 外部公開名 | 該当なし（`RuntimeIntentCoordinator` 内部の getter/setter のみ。他ファイルからの参照は 0） |
| ログ項目 | 該当なし（`[DRAIN_DIAG]` の key は `c9_rbl`） |

## 5. 検証

| # | 検証 | 結果 |
| --- | --- | --- |
| 1 | 参照整合（旧名の残存） | 実コード 0。コメント 7 箇所は歴史的記述として残置 |
| 2 | ビルド | `cmake --build build --config Release` → 319 ターゲット・エラー 0 |
| 3 | `ISRSemanticValidationTests` | PASS（exit 0） |
| 4 | `ctest` | **61/61 成功**（152.20 s） |
| 5 | CLI 3 回 | exit 0、4.61 / 4.24 / 3.92 s、タイムアウト 0、`DRAIN_DIAG` 1 行（worldClear のみ） |
| 6 | 意味論の不変 | 実行結果は改名前と同水準（4.28〜4.61 s・タイムアウトなし） |

## 6. 発見した別件（今回は未修正）

**AC1 のミラーが 14 条件のまま**である。`testAC1_drainBreakdownMatchesPredicate` の
`allDrained` ラムダは `b.retireBacklog == 0` を含んでいるが、C9-A 以降の
`isFullyDrained()` は 13 条件（retireBacklog を除外）である。

- 現状の AC1 シナリオでは `retireBacklog` が常に 0 のため PASS する（検出漏れ）。
- 将来 `retireBacklog` が非 0 のシナリオを追加すると、AC1 は誤って FAIL する。
- 修正は C9-A の検証範囲に属するため、本改名では触っていない。1 行（`&& b.retireBacklog == 0` の削除）で解消する。

## 7. 残留リスク

- 改名のみで意味論は不変。`retireAcceptedCount_` は依然として単調増加し、pressure の slope 以外では使わない。
- 課題票の範囲外である `noteRetireBacklogChanged` / `previousRetireBacklogCount_` に旧名が残る。
  名前の一貫性だけを見ると中途半端だが、承認範囲を守るため意図的に残した（§3）。

## 8. ロールバック

3 ファイルの識別子を逆置換する（`retireAcceptedCount_` → `retireBacklogCount_` など）。意味論・
構造の変更が無いため、他の復元作業は不要。

## 9. 判定

- 課題票に定めた改名は**完了**。参照整合・ビルド・テスト（61/61）を確認した。
- 意味論・更新経路・drain 判定・C3-D/C9-A の実装は**変更していない**。
- AC1 のミラー乖離を別件として報告する（未修正）。
