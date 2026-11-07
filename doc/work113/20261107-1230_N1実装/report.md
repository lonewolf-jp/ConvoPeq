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

## 3. 検証結果

| 項目 | 結果 |
|---|---|
| ビルド | 成功（`CONVOPEQ_ENABLE_TEST_HOOKS=ON`） |
| テール連続性（bypass ON） | PASS（境界のサンプル差が切替前テールの 25% 以下） |
| フェード中のテール保持 | PASS（フェード区間の平均振幅が切替前テールの 20% 以上） |
| 収束後の減衰 | PASS（無音時の残留フロアの 1.25 倍以下） |
| bypass OFF の連続性 | PASS（先頭サンプルが同じ許容内） |
| ハーネス全体 | exit 0、`all publish pipeline tests PASS` |
| `ctest -C Release` | 54 / 54 PASS（353.5 秒、フック有効ビルド） |
| CI（PR #168） | Windows MSVC / Release、ASan、TSan、CodeQL（c-cpp）ほか（結果は PR を参照） |

### 測定の注意

- テスト環境の `ConvolverProcessor` では、減衰 IR のウェット出力振幅が約 4.6e-5 と小さい（IR 読み込みの正規化による）。そのため判定は絶対値ではなく、切替前テールと無音時の残留フロアに対する相対値で行っている。
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
