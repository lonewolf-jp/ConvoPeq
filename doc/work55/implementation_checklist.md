# 改修実装チェックリスト

**ベース計画書**: `doc/work55/fix_plan.md` v6.1
**作成日**: 2026-06-24

---

## Phase 1: F3 r8brain IRテイル切り捨て修正

**ファイル**: `src/IRDSP.cpp`

| # | タスク | 状態 | 確認日 |
|---|--------|------|--------|
| 1.1 | バッファサイズ計算を `getMaxOutLen()` に変更 | ✅ | 2026-06-24 |
| 1.2 | `maxOutLen <= 0` ガード追加 | ✅ | 2026-06-24 |
| 1.3 | `channelDone` + `anyChannelCancelled` 導入 | ✅ | 2026-06-24 |
| 1.4 | `futures[i].wait()` → `futures[i].get()` に変更 | ✅ | 2026-06-24 |
| 1.5 | try/catch で例外保護 | ✅ | 2026-06-24 |
| 1.6 | 条件付きtrim（`maxDone < maxOutLen` 時のみ縮小） | ✅ | 2026-06-24 |
| 1.7 | Debug ビルド成功確認 | ✅ | 2026-06-24 |
| 1.8 | Release ビルド成功確認 | ✅ | 2026-06-24 |

## Phase 2: F5 applyAllpassToIR 削除

**ファイル**: `src/AllpassDesigner.h`, `src/AllpassDesigner.cpp`

| # | タスク | 状態 | 確認日 |
|---|--------|------|--------|
| 2.1 | `AllpassDesigner.h` から `applyAllpassToIR` 宣言を削除 | ✅ | 2026-06-24 |
| 2.2 | `AllpassDesigner.cpp` から `applyAllpassToIR` 実装を削除 | ✅ | 2026-06-24 |
| 2.3 | `DftiHandle.h` include 削除 | ✅ | 2026-06-24 |
| 2.4 | Debug ビルド成功確認 | ✅ | 2026-06-24 |
| 2.5 | Release ビルド成功確認 | ✅ | 2026-06-24 |

## Phase 3: F2 mixedPreRingTau 削除

**対象17ファイル**

| # | ファイル | タスク | 状態 | 確認日 |
|---|----------|--------|------|--------|
| 3.1 | `ConvolverProcessor.h` | `MIXED_TAU_MIN/MAX/DEFAULT` 定数削除 | ✅ | 2026-06-24 |
| 3.2 | `ConvolverProcessor.h` | `BuildSnapshot::mixedPreRingTau` 削除 | ✅ | 2026-06-24 |
| 3.3 | `ConvolverProcessor.h` | `IRCacheKey::tau` + `operator<`該当行削除 | ✅ | 2026-06-24 |
| 3.4 | `ConvolverProcessor.h` | `setMixedPreRingTau/getMixedPreRingTau` 宣言削除 | ✅ | 2026-06-24 |
| 3.5 | `ConvolverProcessor.Runtime.cpp` | setter/getter実装削除 | ✅ | 2026-06-24 |
| 3.6 | `ConvolverProcessor.StateAndUI.cpp` | hash/serialize/deserialize 全参照削除 | ✅ | 2026-06-24 |
| 3.7 | `ConvolverProcessor.LoaderThreadInline.h` | `mixedPreRingTau` メンバ削除 | ✅ | 2026-06-24 |
| 3.8 | `ConvolverProcessor.LoaderThread.cpp` | コンストラクタ引数+メンバ初期化子削除 | ✅ | 2026-06-24 |
| 3.9 | `ConvolverProcessor.LoadPipeline.cpp` | `buildSnapshot.mixedPreRingTau` 2箇所削除 | ✅ | 2026-06-24 |
| 3.10 | `ConvolverProcessor.Rebuild.cpp` | `clampedMixedTau` 参照削除 | ✅ | 2026-06-24 |
| 3.11 | `ConvolverProcessor.MixedPhase.cpp` | `key.tau` 5箇所削除 | ✅ | 2026-06-24 |
| 3.12 | `ConvolverControlPanel.h` | `mixedTauSlider/label` + `pendingMixedTau*` 削除 | ✅ | 2026-06-24 |
| 3.13 | `ConvolverControlPanel.cpp` | 全 `mixedTau*` 参照削除 | ✅ | 2026-06-24 |
| 3.14 | `audioengine/AudioEngine.h` | `setConvolverMixedPreRingTau()` 宣言削除 | ✅ | 2026-06-24 |
| 3.15 | `audioengine/AudioEngine.Parameters.cpp` | setter実装 + `"mixedTau"` バリデーション削除 | ✅ | 2026-06-24 |
| 3.16 | `MainWindow.cpp` | `--cli-pre-ring-tau` 処理削除 | ✅ | 2026-06-24 |
| 3.17 | `MixedPhasePersistentCache.cpp` | `tau` 引数削除(6関数) + `kLastUsedTimeOffset`→`offsetof` | ✅ | 2026-06-24 |
| 3.18 | `MixedPhasePersistentCache.h` | 全宣言の`tau`引数削除 + `kVersion: 1→2` | ✅ | 2026-06-24 |
| 3.19 | Debug ビルド成功確認 | ✅ | 2026-06-24 |
| 3.20 | Release ビルド成功確認 | ✅ | 2026-06-24 |

## Phase 4: F1 Mixed Phase クロスオーバー方向修正

**ファイル**: `src/convolver/ConvolverProcessor.MixedPhase.cpp`

| # | タスク | 状態 | 確認日 |
|---|--------|------|--------|
| 4.1 | `convertToMixedPhaseAllpass()` 内 wLinear/wMinimum 反転 | ✅ | 2026-06-24 |
| 4.2 | `convertToMixedPhaseFallback()` 内 wLinear/wMinimum 反転 | ✅ | 2026-06-24 |
| 4.3 | Debug ビルド成功確認 | ✅ | 2026-06-24 |
| 4.4 | Release ビルド成功確認 | ✅ | 2026-06-24 |

## Phase 5: F4 computeMasteringSizing 削除

**削除範囲**: 関数・構造体削除 + init/呼び出し側の引数整理

| # | ファイル | タスク | 状態 | 確認日 |
|---|----------|--------|------|--------|
| 5.1 | `ConvolverProcessor.Internal.h` | `ConvolverSizing`構造体 + `computeMasteringSizing` 関数削除 | ✅ | 2026-06-24 |
| 5.2 | `ConvolverProcessor.h` | `storedMaxFFTSize`/`storedFirstPartition` メンバ削除 | ✅ | 2026-06-24 |
| 5.3 | `ConvolverProcessor.h` | `init()` から `maxFFTSize`/`firstPartition` 引数削除 | ✅ | 2026-06-24 |
| 5.4 | `ConvolverProcessor.h` | `clone()` 内 `init()` 呼び出し修正 | ✅ | 2026-06-24 |
| 5.5 | `ConvolverProcessor.h` | `finalizeNUCEngineOnMessageThread` 宣言修正 | ✅ | 2026-06-24 |
| 5.6 | `ConvolverProcessor.Lifecycle.cpp` | `computeMasteringSizing` 呼び出し削除 + `init()` 呼び出し修正 | ✅ | 2026-06-24 |
| 5.7 | `ConvolverProcessor.LoaderThread.cpp` | `performLoad()` 内計算削除 + 関数シグネチャ修正 | ✅ | 2026-06-24 |
| 5.8 | `ConvolverProcessor.LoaderThreadInline.h` | `initializeConvolverSynchronously`/`queueFinalizeOnMessageThread` 宣言修正 | ✅ | 2026-06-24 |
| 5.9 | `ConvolverProcessor.LoadPipeline.cpp` | `finalizeNUCEngineOnMessageThread` 実装シグネチャ+呼び出し修正 | ✅ | 2026-06-24 |
| 5.10 | Debug ビルド成功確認 | | ✅ | 2026-06-24 |
| 5.11 | Release ビルド成功確認 | | ✅ | 2026-06-24 |
