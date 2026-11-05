# RES2-13 TX-02 / TX-10 の要件照合資料（2026-11-05）

- 基準: `main` = `9dd57529`
- Owner 判断: 現行要件での必須性を再確認する。Waiver の確定には現行要件との整合性の根拠が必要。過去 waiver を自動免除として扱わない。

## 1. 事実

| 項目 | 内容 | 根拠 |
|---|---|---|
| Waiver | TX-02（IR Replace 単発）と TX-10（automation burst 100 件）は「ユーザー判断で実機試験スキップ」 | `doc/phase1_tx_validation_log_jp.md:23-25` |
| 当時の必須セット | TX-02・TX-07・TX-10 を必須合格とする | `doc/nextgen_runtime_transition_design_jp.md:508, 691` |
| 設計の位置づけ | 次世代アーキ移行（EngineRuntime 単一 publish 等）を前提とした手順 | 同 `:333-465` |
| 現行アーキテクチャ | `RuntimeWorld` / `worldAuthority` が公開モデル。`EngineRuntime` は `[[deprecated]]` | `AudioEngine.h:150, 3300-3308`、`RuntimeWorld` 428 件 |
| 後続の手順書 | TX-11〜TX-15（パラメータ網羅性）は別手順として存在。TX-02/10 への言及は無い | `doc/tx11_tx15_test_procedure_jp.md` |

## 2. 照合結果

- 旧設計（nextgen）は 2026-05 の RuntimeWorld 移行前に書かれた内容を含み、現行の公開モデルと一致しない箇所がある。
- **現行要件として TX-02/TX-10 が必須かどうかを示す文書は見つかっていない。** したがって「Waiver により免除済み」と断定できない。
- 過去 waiver は当時の判断であり、現行要件の免除として自動適用しない（Owner 判断どおり）。

## 3. 実機検証が必要な場合の計画（未実施）

| 項目 | TX-02（IR Replace 単発） | TX-10（automation burst 100 件） |
|---|---|---|
| 入力 | 固定 IR、再生中の IR 差替 | 同一窓で 100 件のパラメータ更新 |
| 観測 | rebuild 回数、publish 回数、XRUN、出力の連続性（クリック） | coalescing 後の rebuild 回数、最終値一致、CPU スパイク |
| 期待値 | rebuild 1 回、XRUN 0、クリックなし | 最終値が最後の更新と一致、rebuild 回数 ≤ 既定上限 |
| 計測 | ログのタイムスタンプ、`deferredOverwriteCount`、`recoveryRetryRedriveCount` | 同上 |
| 必要な環境 | Windows 実機＋ASIO デバイス | 同左 |
| 判定 | 上記がすべて成立 | 同左 |

## 4. 結論

- 現行要件の必須性は未確定。決定するには、現行アーキテクチャに対する TX 手順の再定義（対象・期待値）が必要。
- 実機検証を実施する場合は、実機環境の提供と Owner 承認を必要とする。
- 現時点では「免除」ではなく「未確定」として扱う。
