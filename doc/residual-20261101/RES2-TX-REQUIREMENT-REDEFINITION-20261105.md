# RES2-13 TX-02／TX-10 の現行要件 再定義案

- 作成: 2026-11-05
- 前提: 過去の Waiver（2026-05）は現行要件の免除として自動適用しない（Owner 判断）。
- 位置づけ: 再定義案の提示まで。実機検証・免除の確定は別判断。

## 1. 原典の要求（2026-05 の nextgen 設計）

| ID | 手順 | 合格条件（当時） |
|---|---|---|
| TX-02 | IR finalized の状態で 1 回だけ IR を差し替える | Build → Validate → Warmup → Publish → Crossfade → Retire の順が崩れない。旧 engine が retire され reclaim される |
| TX-10 | 同一窓で 100 件規模のパラメータ更新を投入する | coalescing で rebuild 回数が圧縮される。最終値が一致する。CPU スパイクが許容範囲 |

当時の設計は `EngineRuntime` の単一 publish と crossfade を前提にしている。現行は `RuntimeWorld`／`worldAuthority` が公開モデルで、`EngineRuntime` は `[[deprecated]]` である。

## 2. 現行アーキテクチャとの対応

| 当時の観測点 | 現行の対応物 | 根拠 |
|---|---|---|
| Build → Validate → Warmup → Publish | 再構築の受付（admission）と公開（publication） | `IRLoadAdmissionTests`、`PublishPipelineIntegrationTests` |
| Crossfade | 旧 world から新 world への切り替え | `RuntimeWorldAuthorityProjectionTests`、`DeferredFlowIntegrationTests` |
| Retire と reclaim | world の退役と参照解放 | `V11EngineRetireProtectionTests`、`WorldRetirementMeasurementTests`、`RetireGraceSemanticsTests` |
| rebuild 回数の圧縮 | coalescing（同一窓の要求統合） | `ISRSemanticValidationTests`（coalescing 70 箇所）、`ConvolverStateRoundTripTests` |
| 最終値の一致 | 最終要求の反映 | `ISRSemanticValidationTests`、`DeferredPublishViewStateMachineTests` |

## 3. 選択肢

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| A | TX-02／TX-10 を現行要件として再定義し、観測点と数値基準を現行アーキテクチャに合わせる（推奨） | 要件の連続性を保てる。自動テストで大部分を代替できる | 再定義そのものの承認が必要 |
| B | 現行アーキテクチャでは別の検証（自動テスト）で代替できると明示し、実機手順を廃止する | 実機コストが無い | 「実機でしか見えない部分」の説明が必要 |
| C | 現行要件として不要と記録し、Waiver を改めて判断する | 最も軽い | 根拠文書が無いまま不要と断定することになる |

**推奨は A**。当時の合格条件のうち、順序・回数・最終値・retire は自動テストで数値として確認できる。実機でしか確認できないのは、ASIO の XRUN と可聴クリックである。

## 4. 再定義案（数値基準）

### TX-02（再定義）: IR 差し替え 1 回

| 観測項目 | 基準（案） | 測定方法 |
|---|---|---|
| publish 回数 | 差し替え 1 回につき 1 | `PublishPipelineIntegrationTests` の公開カウンタ |
| 中間 publish | 0（superseded の公開なし） | 同上 |
| 旧 world の退役 | 差し替え後、旧 world の参照が 0 になる | `WorldRetirementMeasurementTests` |
| overrun／redrive | `deferredOverwriteCount` = 0、`recoveryRetryRedriveCount` = 0 | 既存カウンタ |
| RT 側の新規確保 | 0 | `rtalloc` プローブ（テスト専用） |
| XRUN・可聴クリック | 実機のみ | ASIO 実機（未実施） |

### TX-10（再定義）: automation burst 100 件

| 観測項目 | 基準（案） | 測定方法 |
|---|---|---|
| 最終値 | 最後の更新と一致 | `ISRSemanticValidationTests` の coalescing ケース |
| rebuild 回数 | 同一パラメータの連続更新なら 1、異なる値の混在なら「最終値の種類数」以下 | 既存の再構築カウンタ |
| 中間 publish | 0（superseded を公開しない） | 公開カウンタ |
| ブロック処理時間 | 1.25 ms ブロックに対し、最大が 1.25 ms を超えない（候補。既存の RT 制約に合わせる） | `RtMeasureOverheadTests` と同様の計測 |
| CPU スパイク | 実機のみ | 実機（未実施） |

- ブロック処理時間の基準は候補である。既存仕様に数値の定めが無いため、確定には Owner の承認が要る。

## 5. 結論

- 現行要件としての再定義（案 A）を提案する。順序・回数・最終値・retire は自動テストで確認でき、実機は XRUN と可聴クリックに限定できる。
- 再定義を承認するまでは、TX-02／TX-10 を「未確定」として扱う。Waiver の確定・実機検証の実施は、再定義の後に行う。
- 実機検証を実施する場合は、Windows 実機＋ASIO と、計測項目（XRUN、publish 回数、最終値、ブロック処理時間）を事前に固定する。
