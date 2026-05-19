# 即着手タスクリスト（依存順）

本タスクリストは、plan.md / rule.md に基づく実装順をそのまま着手可能な依存順で固定したものです。

## 1. C3: publish済みruntime直接mutationの停止

対象:

- src/audioengine/AudioEngine.UIEvents.cpp

対象関数:

- convolverParamsChanged

作業:

- [x] activeDSP->convolverRt().syncParametersFrom(...) を削除
- [x] 差分検出後は requestRebuild のみ実行

完了条件:

- [x] activeDSP->convolverRt().syncParametersFrom の呼び出しが0件

## 2. C1-1: Crossfade準備をMessage Threadへ移管

対象:

- src/audioengine/AudioEngine.Commit.cpp
- src/audioengine/AudioEngine.h

対象関数:

- commitNewDSP
- publishRuntimeSnapshots 周辺

作業:

- [x] CrossfadePreparedState（仮称）導入
- [x] fadeSec / latencyDelay / useDryAsOld / startDelayBlocks を commit 側で確定
- [x] prepared state の publish 経路を追加

完了条件:

- [x] crossfade開始に必要な初期化値が Message Thread 側で準備完了する

## 3. C1-2: Audio Thread側をactivate専用化

対象:

- src/audioengine/AudioEngine.h
- src/audioengine/AudioEngine.Processing.BlockDouble.cpp
- src/audioengine/AudioEngine.Processing.AudioBlock.cpp

対象関数:

- armCrossfadeIfPending（必要なら activateCrossfadeIfPrepared に改名）

作業:

- [x] Audio Thread から dspCrossfadeGain.reset(...) を除去
- [x] Audio Thread から setCurrentAndTargetValue(0.0) を除去
- [x] 準備済み状態の activate のみ実行する構造へ変更

完了条件:

- [x] armCrossfadeIfPending 内に初期化APIが存在しない

## 4. C4-1: Publication helper API導入

対象:

- src/audioengine/AudioEngine.h

対象関数:

- publishEngineRuntimeState
- publishRuntimeSnapshots
- clearPublishedRuntimeSnapshotsNonRt
- publishCurrentDSPAndTakeOwnership
- replaceFadingOutDSPAndRetirePrevious

作業:

- [x] publishAtomicPtr / consumeAtomicPtr / exchangeAtomicPtr（命名は最終規約に合わせる）を導入
- [x] 対象関数の公開境界で helper 経由へ置換

完了条件:

- [x] 対象関数で memory_order 直書きが原則解消

## 5. C4-2: 公開ドメイン分裂の是正（最小版）

対象:

- src/audioengine/AudioEngine.h

対象関数:

- publishRuntimeSnapshots 系

作業:

- [x] engineRuntimeState / runtimeGraphState の更新順序を helper 契約で固定
- [x] 部分可視の窓を閉じる publish シーケンスに統一

完了条件:

- [x] 単一路の publish シーケンスに整理されている

## 6. C2-1: EQのSmoothedValue型置換

対象:

- src/eqprocessor/EQProcessor.h

対象関数:

- EQProcessor メンバ定義部

作業:

- [x] smoothTotalGain を convo::LinearRamp に置換
- [x] bypassFadeGain を convo::LinearRamp に置換

完了条件:

- [x] EQProcessor から juce::SmoothedValue 実体参照が消える

## 7. C2-2: EQ初期化・処理経路のLinearRamp整合

対象:

- src/eqprocessor/EQProcessor.Core.cpp
- src/eqprocessor/EQProcessor.Processing.cpp

対象関数:

- prepare
- process

作業:

- [x] reset / setTarget / getNext / skip の呼び出しを LinearRamp 前提へ統一
- [x] バイパス遷移完了時の挙動を現行互換で維持

完了条件:

- [x] EQの動作回帰がなく、RTパスにlibm混入がない

## 8. H1: RCUReaderのコピー/ムーブ禁止

対象:

- src/core/RCUReader.h

対象関数:

- class RCUReader

作業:

- [x] copy ctor / copy assign を delete
- [x] move ctor / move assign を delete

完了条件:

- [x] RCUReader 値コピーがコンパイルエラーになる

## 9. H3: commit通知合流フラグ導入

対象:

- src/audioengine/AudioEngine.Commit.cpp
- src/audioengine/AudioEngine.h

対象関数:

- commitNewDSP
- AsyncUpdater関連

作業:

- [x] pendingChangeNotification（仮称）導入
- [x] 通知フラッドを合流

完了条件:

- [x] 連続commit時の過剰通知が抑制される

## 10. H2: aligned_make_unique導入と適用開始

対象候補:

- src/eqprocessor/EQProcessor.Core.cpp
- src/audioengine/DSPExecutionState.h

対象関数:

- 非RT確保ルート

作業:

- [x] 64byteアライン・例外安全な aligned_make_unique を導入
- [x] 新規確保箇所から段階適用

完了条件:

- [x] 新規実装で手書き placement new が増えない

## 共通検証チェック

- [x] get_errors で変更ファイルに新規エラーなし
- [x] grep で禁止パターン再確認
- [x] Debug または Release ビルド成功
- [x] 変更内容が IR-1〜IR-7 のどれに寄与するか記録
