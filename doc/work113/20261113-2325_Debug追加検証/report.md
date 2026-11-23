# C3-D / C9-A の Debug 構成 追加検証

- 作業日時: 2026-11-13 23:25 JST
- 種別: **追加検証のみ**。ソース変更・実装変更・改名は行っていない。
- 承認範囲: Debug ビルド、`ISRSemanticValidationTests` の実行、関連テストの実行、Debug の ctest 全件、
  結果と失敗内容の記録。
- 前提: Release 構成の完了判定（C3-D / C9-A 実装・検証完了）は変更しない。本書は独立した検証層である。

---

## 1. 実施内容

| # | 項目 | 実施 |
| --- | --- | --- |
| 1 | Debug 構成のビルド | 実施（776 ターゲット） |
| 2 | Debug の `ISRSemanticValidationTests.exe` | 実施（PASS） |
| 3 | C3-D / C9-A 関連テスト | 実施（全 PASS。§4） |
| 4 | Debug の ctest 全件 | 実施（59/61。失敗 2 件は §3） |
| 5 | 結果・失敗内容・条件の記録 | 本書 |

## 2. ビルド（Debug）

| 項目 | 結果 |
| --- | --- |
| コマンド | `cmake --build build --config Debug` |
| ターゲット | 776（全件） |
| エラー | **0** |
| 警告 | 13（C4324 ×8 = alignas パディング、C4834 ×2、C4458 / C4996 / C4244 各 1） |
| 実行ファイル | `build/Debug/ISRSemanticValidationTests.exe`（2026-11-13 23:15:38） |

## 3. テスト結果

### 3.1 `ISRSemanticValidationTests`（Debug・単体実行）

```
ISRSemanticValidationTests: PASS (D105-R5-9 completion authority: ...)
exit=0
```

AC2（head identity の 3 要素）と C9-A の 2 テスト（Bootstrapping 側 / Faulted 側）を含めて PASS。
Debug は `jassert` が有効なため、契約違反があればここで停止する。

### 3.2 Debug ctest 全件

```
97% tests passed, 2 tests failed out of 61
Total Test time (real) = 228.86 sec
The following tests FAILED:
    5 - D8_2_B_2_Tests (SEGFAULT)
   50 - TailLayerIntegrityTests (Failed)
```

### 3.3 失敗 1: `D8_2_B_2_Tests`（SEGFAULT）

| 項目 | 内容 |
| --- | --- |
| 終了コード | `-1073741571` = **0xC00000FD（STACK_OVERFLOW）** |
| 症状 | 出力が残らないまま異常終了（stdout がブロックバッファのため） |
| 原因の確認 | 同じ Debug バイナリの**複製**に `editbin /STACK:8388608` を適用して実行 → **T1〜T13 すべて PASS（exit 0）** |
| リポジトリへの影響 | なし（複製は `%TEMP%\opencode\D8_2_B_2_Tests_stack8m.exe`。ビルドツリーとソースは無変更） |

**原因**: スタック不足。テストは `convo::EpochDomain`（約 230 KB。AGENTS.md に既知の落とし穴として記載）を
スタック上に生成しており（`D8_2_B_2_Tests.cpp:1003, 1141, 1234, 1381`、および
`FullQueueEpochProvider`（EpochDomain 派生）:1079 ほか）、このターゲットには `/STACK` 指定が無い
（`/STACK:8388608` が付いているのは `BuildInputSemanticContractTests` のみ）。Debug はフレームが
大きいため 1 MB を超える。スタックを 8 MB にすると全ケースが通る。

**C3-D / C9-A との関係**: 無関係と判断する。根拠は 3 点。

1. このテストは `RuntimeIntentCoordinator` / `ShutdownScheduler` / `peekHead` / `inspectRetireHead` を
   参照していない（ソース検索で 0 件）。
2. 今回の変更はヘッダへの**追加**（`RetireHeadSnapshot` の 2 フィールド、`peekHead` の out 引数）で、
   `EpochDomain` のサイズやスタック使用量を増やしていない。
3. スタックを広げた同一バイナリで全ケースが通る＝ロジック不良ではない。

### 3.4 失敗 2: `TailLayerIntegrityTests`（Failed）

| 項目 | 内容 |
| --- | --- |
| 失敗した検査 | `RES-F` のみ（8〜9 件、実行ごとに 1 件変動） |
| 判定式 | `cpuMsPerAudioSec <= 10.0` かつ `memDeltaMB <= 32.0`（`TailLayerIntegrityTests.cpp:41-42, 259`） |
| 実測 | cpu = 11.65〜34.01 ms/s（予算超過）、mem = 1.98〜11.62 MB（予算内） |
| 意味 | **CPU 予算の超過**。Debug は最適化が無く 2〜3 倍遅いため、性能閾値に届かない |
| 機能面 | F-1〜F-4 / H-1〜H-7 は PASS（再構成誤差・テールエネルギー・窓比・形状） |

**C3-D / C9-A との関係**: 無関係。テール層 DSP（リサンプラ）の CPU/メモリ予算検査であり、
shutdown 経路・drain 判定・retire 計数を含まない。Release 構成では同じテストが PASS している。

## 4. 関連テストの Debug 結果

| テスト | 結果 | 所要 |
| --- | --- | --- |
| `ISRSemanticValidationRejects`（AC2・C9-A を含む） | **Passed** | 0.68 s |
| `DeferredDeletionQueueReclaimTests`（peekHead の対象クラス） | **Passed** | 3.23 s |
| `ShutdownRetireIntentDrain` | **Passed** | 0.42 s |
| `StuckReaderFallbackDrain` | **Passed** | 0.47 s |
| `RetireGraceSemantics` | **Passed** | 0.41 s |
| `STG11EQRetire` | **Passed** | 1.29 s |
| `STG11D2SnapshotRetire` | **Passed** | 0.61 s |
| `DomainTagTests` | **Passed** | 0.14 s |
| `ISRSoakTests` | **Passed** | 0.78 s |
| `NormalRetireDSPHandleCompare` | **Passed** | 0.13 s |
| `AudioEngineHarness`（shutdown 経路を含む） | **Passed** | 161.63 s |

## 5. 未実施

| # | 内容 | 理由 |
| --- | --- | --- |
| 1 | Debug での CLI 3 回計測 | 承認範囲は「ビルド・テスト実行」で、実機計測は含まれない |
| 2 | `D8_2_B_2_Tests` の修正（`/STACK` 追加やヒープ化） | 承認外（原因調査まで） |
| 3 | `TailLayerIntegrityTests` の Debug 用閾値調整 | 承認外（性能閾値の設計変更になる） |

## 6. Release 判定との関係

- Release 構成の結果（C3-D / C9-A 完了、`ctest` 61/61、2 秒タイムアウト解消）は**変更しない**。
- Debug の 2 件の失敗は、いずれも**構成固有**（スタック深さ・最適化の有無）で、今回の変更に起因しない。
- ただし `D8_2_B_2_Tests` は Debug で常時失敗する状態であり、Debug 回帰を回す際のノイズになる。
  修正するかは別途判断が必要（本書では提案に留める）。

## 7. 判定

| 項目 | 判定 |
| --- | --- |
| Debug ビルド | **PASS**（776 ターゲット・エラー 0） |
| `ISRSemanticValidationTests`（Debug） | **PASS** |
| C3-D / C9-A 関連テスト（11 件） | **全 PASS** |
| Debug ctest 全件 | **59/61**（失敗 2 件は構成固有・原因特定済み） |
| ソース変更 | なし |
| Release 判定への影響 | なし |
