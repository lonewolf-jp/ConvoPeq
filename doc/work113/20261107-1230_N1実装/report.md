# N1 バイパス時のテール切断の解消（Issue #158）

- 実施: 2026-11-07
- 基準: `main` = `ae91385d`（最新 `ConvoPeq.md` は `7faf9fa9` のソースを反映）
- 承認: Owner 指示（N1 の改善実装）

## 1. 実装前の確認

- 該当箇所: `ConvolverProcessor::process` の bypass dispatch と `processBypassWithLatencyCompensation`（`ConvolverProcessor.Runtime.cpp`、`ConvoPeq.md` L54784/L54958）。
- 現行動作: bypass が真のブロックで即座に dry 経路へ切り替わり、wet の畳み込みを実行しない。ウェット状態は凍結され、テールは不連続に切れる。
- 既存の遷移機構: `activeMixSmoother` と equal-power の wet/dry ミックスが既にあり、IR 差し替えのクロスフェードにも使われている。
- 制約: RT スレッドで確保・ロック不可。レイテンシ補償とゲイン契約を変えない。

## 2. 実装内容

| ファイル | 変更 |
|---|---|
| `src/convolver/ConvolverProcessor.Runtime.cpp` | bypass dispatch を「ミックスが dry に収束した後」に限定。bypass 中はミックス目標を 0 に固定 |
| `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp` | 減衰 IR を使ったテール連続性テスト `checkBypassTailContinuity` を追加 |

動作:

1. bypass ON: 既存のミックス smoother で wet→dry を `smoothingTimeSec`（既定 20 ms）かけてクロスフェードする。この間は wet を処理し続けるため、テールは自然に減衰する。
2. 収束後: 従来の bypass 経路（dry リングのみ）に切り替える。以後は畳み込みを実行しない。
3. bypass OFF: ミックス目標をユーザー設定へ戻し、dry→wet を同じ時定数でクロスフェードする。

レイテンシ補償（H-01）と equal-power ゲインは既存の経路をそのまま使う。RT 側の追加確保・ロックはない。

### 状態遷移・終了条件・リソース寿命

| 状態 | 条件 | 音声処理 |
|---|---|---|
| A. 通常（wet） | `bypassed == false`、ミックス目標 = ユーザー設定 | wet + dry をミックス |
| B. フェードアウト | `bypassed == true`、ミックスが 0 へ遷移中 | wet を処理し続け、ミックスを 0 へ |
| C. バイパス定常 | `bypassed == true`、ミックス 0 かつ smoother 停止 | dry リングのみ |
| D. フェードイン | `bypassed == false`、ミックスがユーザー設定へ遷移中 | wet + dry をミックス |

- B → C の終了条件は「ミックス値が 1.0e-4 以下」かつ「smoother が非動作」。このときだけ従来のバイパス経路へ dispatch する。
- D の終了条件はミックスがユーザー設定値に到達し smoother が停止した時点。以降は A と同じ。
- フェード中にバイパスが反転した場合は、smoother の目標値が入れ替わり、同じ経路のまま逆方向へ戻る。dispatch は収束後のみ行うため、途中で経路が飛ばない。
- リソース寿命は不変。dry/wet/遅延リングとプロセススナップショットは従来どおり prepare 時に確保され、フェード中も同じバッファへ書き続ける。追加確保・解放はなく、フェード終了で破棄するものもない。

## 3. 検証結果

| 項目 | 結果 |
|---|---|
| ビルド | 成功（`CONVOPEQ_ENABLE_TEST_HOOKS=ON`） |
| 変更前コードでの回帰テスト | **FAIL**（`delta=6.0e-6` が `0.25 × pre=3.7e-6` を超える。切替境界で波形が飛ぶ） |
| テール連続性（bypass ON） | PASS（`pre=3.677e-06`、`delta=6.979e-09`） |
| フェード中のテール保持 | PASS（区間平均 `mean=2.157e-06` が切替前テールの 20% 以上） |
| 収束後の減衰 | PASS（無音時の経路残留の 1.25 倍以下） |
| bypass OFF の連続性 | PASS（先頭サンプルが同じ許容内） |
| ハーネス全体 | exit 0、`all publish pipeline tests PASS` |
| `ctest -C Release` | 54 / 54 PASS（326.4 秒、フック有効ビルド） |
| CI（PR #168） | Windows MSVC / Release、ASan、TSan、CodeQL（c-cpp）ほか（結果は PR を参照） |

### 基準（変更前）と回帰の示し方

- 基準は `main` = `ae91385d`。この時点の CI は全ジョブ成功で、既存テストは緑。
- 追加した `checkBypassTailContinuity` を、N1 の変更を外した状態（`main` と同じ即時切替）で実行すると FAIL する。実際の出力は
  `[N1] FAIL: discontinuity at bypass (pre=0.000004 first=0.000010 delta=0.000006)`。
- 同じテストを本変更後に行うと PASS（`delta=6.979e-09`）。境界の段差が 3 桁小さくなり、テールが切れていないことを示す。

### 測定の注意

- テスト環境の `ConvolverProcessor` では、減衰 IR のウェット出力振幅が約 4.6e-5 と小さい（IR 読み込み時の後段フェードによる）。そのため判定は絶対値ではなく、切替前テールと、同じ経路で無音を流したときの残留に対する相対値で行っている。
- バイパス定常経路には約 `1.9e-5`（-94 dBFS 相当）の残留がある。これは本変更前から観測される既存の挙動で、本変更では導入していない。収束判定はこの残留を基準にしている。
- 連続性の確認は波形のサンプル差とエネルギーで行い、聴感評価は行っていない。

## 4. 影響評価

- 遅延: 既存の dry アラインメントを使用。PDC の申告値は変えない。
- ゲイン: equal-power クロスフェード。定常状態のゲインは従来と同じ。
- RT: フェード中のみ畳み込みを継続。追加の確保・ロックはない。
- 消費: bypass 定常時は従来どおり畳み込みを実行しない。

## 5. 残存リスク・未検証

- フェード長は `smoothingTimeSec` に依存する。極端に短い設定ではクリック抑制が弱くなる。
- 実機での聴感評価は未実施。
- IR 差し替えクロスフェードとの同時発生は未検証。
- テスト環境の振幅が小さいため、絶対レベルの連続性は別途確認が必要。

## 6. 状態

| 項目 | 状態 |
|---|---|
| 実装 | 完了 |
| テスト | 追加・PASS |
| PR | 作成済み（#168、マージ未実施） |

## 7. Owner 判断が必要な事項

1. 本 PR のマージ可否。
2. フェード時間は既存の `smoothingTimeSec`（既定 20 ms）をそのまま使うか、バイパス専用の時定数を設けるか。
3. 実機での聴感確認をマージ前に行うか、マージ後の確認でよいか。
