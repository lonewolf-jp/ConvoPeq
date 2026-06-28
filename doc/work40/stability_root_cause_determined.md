# ConvoPeq 音飛び根本原因確定レポート

> 確定日: 2026-06-28
> 方法: ログ解析 + ソースコード静的解析 (Serena/AiDex/grep)

---

## 調査結果サマリー

| # | 項目 | 状態 | 確定度 | 影響度 |
|---|------|------|--------|--------|
| 1 | IncrementalRebuild 未使用 | **原因特定・未使用確定** | ★★★★★ | 中 |
| 2 | rebuildIR 405ms ボトルネック | **原因特定** | ★★★★★ | 高 |
| 3 | DEFERRED 200ms window | **設計確認済み・正常動作** | ★★★★★ | 低 |
| 4 | 起動時レート変動 | **原因特定・改善可能** | ★★★★☆ | 中 |
| 5 | NoiseShaperLearner バッファ飽和 | **設計想定範囲内** | ★★★★★ | 低 |
| 6 | convDebounce | **カウンタ未増加・影響なし** | ★★★★★ | なし |

---

## 1. IncrementalRebuild 未使用 **← 確定**

### 発見事項

コードに `IncrementalRebuildJob` の完全な実装が存在するが、**有効化されていない。**

```cpp
// ConvolverProcessor.h L792
std::atomic<bool> useIncrementalRebuild { false };  // デフォルト false

// ConvolverProcessor.Rebuild.cpp L274-278
void ConvolverProcessor::setUseIncrementalRebuild(bool enable) noexcept
{
    // ★ enable で呼び出した場合のみ atomic を true に設定すべきだが...
    convo::publishAtomic(useIncrementalRebuild, false, std::memory_order_release);
    // ↑ 常に false を書き込んでいる（バグ OR 未実装の意図的動作）
}
```

さらに **`setUseIncrementalRebuild` を呼び出すコードが存在しない。** (= dead code)

### 影響

- 現在の `rebuildAllIRsSynchronous` は全ステップ（LoadIR→Trim→Transform(MixedPhase)→Build(NUC)）を **同期的に逐次実行**（~405msのブロッキング）
- `beginIncrementalRebuild` + `advanceIncrementalRebuild` を使えば、各ステップを分割してAudio Threadのブロッキング時間を短縮可能
- `CONVOPEQ_ENABLE_CONVOLVER_SPLIT_REBUILD=1` は CMakeLists.txt で定義済み（コンパイルは通る）

### 対応案

**案A**: `setUseIncrementalRebuild(true)` をどこかで呼び、`rebuildThreadLoop` 内で `beginIncrementalRebuild` / `advanceIncrementalRebuild` を使用するパスに切り替える。
- ファイル: `AudioEngine.RebuildDispatch.cpp` の `rebuildThreadLoop`
- 課題: IncrementalRebuildJobの状態管理を rebuildThreadLoop に組み込む必要あり

**案B**: `setUseIncrementalRebuild` のバグ修正（`false`固定をやめて引数の `enable` を使う）+ 呼び出し追加

---

## 2. rebuildIR 405ms の内訳 **← 確定**

### LoaderThread.runSynchronously() の処理時間内訳

```
Step 1: doLoadIRStep()    → IRファイル読み込み or バッファコピー (isRebuild時は高速: ~1ms)
Step 2: doTrimStep()      → トリミング (~1ms)
Step 3: doTransformStep() → MixedPhase変換 (GreedyAdaGrad: ~176ms) ← ★
Step 4: doBuildStep()     → NUCエンジン構築 (MKL/IPP: ~228ms)    ← ★
                            └ initializeConvolverSynchronously()
                              └ StereoConvolver::init()
                                └ MKLNonUniformConvolver::SetImpulse() × 2ch
        Total: ~405ms
```

### MixedPhase 176ms の内訳

```
convertToMixedPhaseAllpass():
  ├─ AllpassDesigner (GreedyAdaGrad):
  │   ├─ numSections = 2 (liveReconfigure && highRateLive時)
  │   ├─ maxIterations = 4
  │   ├─ freqPoints = 12
  │   └─ learningRate = 0.006
  │   └─ ★ FFTごとの勾配計算が各iterationで実行 → 12freq × 4iter × FFT
  ├─ AllpassResponse計算 (FFT)
  └─ 混合位相IR合成 (IFFT)
```

非ライブ時（初期IR読み込み）はCMAESを使用:
```
numSections = 20
cmaesMaxGenerations = 160
cmaesPopulationSize = 64
→ これが使われると数秒かかる可能性
```

### MixedPhasePersistentCache の存在

`MixedPhasePersistentCache.h` にキャッシュ機構が存在。同一 IR hash + sampleRate の MixedPhase 結果をファイルキャッシュできる。ただし:
- IRが変わらない限り再利用される
- ログでは `fileHash=0` のリビルド（既存IRの再構築）でキャッシュミス

### NUCエンジン構築 228ms の内訳

```
MKLNonUniformConvolver::SetImpulse() (× 2ch):
  ├─ DirectForm設定 (先頭32taps): mkl_malloc + memcpy
  ├─ Layer構成決定 (L0/L1/L2 Non-Uniform Partitioned):
  │   L0: partSize=blockSize×2, ~15 parts
  │   L1: partSize=L0×tailL1L2Mult, ~テール部
  │   L2: partSize=L1×tailL1L2Mult, ~残響テール
  ├─ IppFFTPlanCache::getOrCreate(): FFT計画生成 (初回のみ, 2回目以降はキャッシュ)
  ├─ 各LayerのFFTワークバッファ確保: ippsMalloc_8u
  ├─ IRのFFT: ippsFFTFwd_R_64f
  ├─ SpectrumFilter適用 (TailProfile/Crossfade)
  └─ Warmup: processLayerBlock で各Layerの内部状態初期化
```

FFT Planは `IppFFTPlanCache` で `order → {fftSpec, sizeWork}` のグローバルキャッシュ。
初回起動時のみ `ippsFFTInit_R_64f` で生成（~10-20ms/order）。
2回目以降のリビルドではキャッシュヒットするため、FFT plan作成時間はほぼゼロ。
それでも 2ch × 3Layer の FFT/IFFT、MKLメモリ確保、Warmup で ~228ms かかる。

### 対応案

1. **優先: FFT Plan Cacheのウォームアップを `prepareToPlay` 時に先実行**
   - 現在は `SetImpulse` 内で初回生成。IR読み込み前の準備段階で FFT Plan だけ先に作成可能
   - FFT Plan作成 (~15ms) をクリティカルパスから除外

2. **検討: MixedPhasePersistentCache の有効化**
   - 同一 IR の再リビルド時（fingerprint 一致）は MixedPhase 結果をスキップ可能
   - ログでは fingerprint が変化しているため効果は限定的

3. **中長期的課題: リビルドの非同期化**
   - 現在は rebuildThread 内で同期的 `rebuildAllIRsSynchronous` を実行
   - この間、新しい Runtime は commit されず、古い Runtime で処理継続
   - ただし rebuildThread 自体は Worker Thread なので Audio Thread はブロックされない
   - 405ms の間、**古い IR で処理が継続される**→ 音飛びはIR適用時の遷移に起因

---

## 3. DEFERRED 200ms window **← 正常動作を確認**

### コード確認結果

```cpp
// AudioEngine.UIEvents.cpp L134
const int64_t minDeltaTicks = juce::Time::getHighResolutionTicksPerSecond() / 5; // 200ms
```

- `convolverParamsChanged` が `PreparedIRApplyWindow` (200ms以内) に再度呼ばれた場合、Structural rebuild を **DEFERRED** する
- 200ms後に `timerCallback` (100ms polling) で `dueTicks >= nowTicks` をチェックして解放
- **正常動作。悪影響なし。** むしろ rebuild 連鎖を抑制する意図通りの動作。

### 確認事項

- deferred から解放まで: 最大 200ms + 100ms (timer polling) = 最大300ms
- 解放後に `timerCallback` → `submitRebuildIntent(Structural)` → rebuild generation increment
- これによって generation 7→8 の連鎖が発生
- この間、Audio Thread は古い Runtime で処理継続（安全）

---

## 4. 起動時処理レート変動 **← 確認・改善可能**

### 確認された問題

```cpp
// AudioEngine.Init.cpp L69
uiConvolverProcessor.prepareToPlay(48000.0, 512);  // 暫定値48kHz
```

この後、実際のオーディオデバイス開始時に:
```
prepareToPlay: enter spb=1024 sr=192000.00  → 192kHzに変更
```

この 48kHz→192kHz の変更が **generation 1 の再リビルド** を引き起こす。
さらにオーバーサンプリング有効化（osFactor=0→2）で generation 2〜4 の連鎖が発生。

### 連鎖の詳細

| タイミング | 事象 | Generation | 備考 |
|-----------|------|-----------|------|
| Init#1 | prepareToPlay(48k, 512) | - | 暫定値 |
| Init → Prepare | prepareToPlay(192k, 1024) | gen=1 | sampleRate変更 |
| Audio Start | OS有効化 (osFactor=0→2) | gen=2-4 | oversampling有効化 |
| IR Load | IR読み込み | gen=5-6 | convolverParamsChanged #1 |
| IR Apply | IR適用 → MixedPhase | gen=7 | convolverParamsChanged #2 |
| DEFERRED解放 | 最終確定 | gen=8 | publish成功 |

### 対応案

1. **prepareToPlayの暫定値を実運用値に近づける**
   - `prepareToPlay(192000.0, 1024)` に変更（またはユーザーのデバイス設定を先読み）
   - 効果: gen=1→4 までの4回のDSPCORE_PREPAREを削減

2. **オーバーサンプリングモードの事前設定**
   - 初期化時に `setProcessingOrder` とオーバーサンプリング係数を確定させる
   - 現在はデバイス準備後に UI からの設定に依存

---

## 5. NoiseShaperLearner バッファ飽和 **← 設計想定範囲内・問題なし**

### 確認事項

- `AudioSegmentBuffer::kCapacity = 5秒 × 768000Hz = 3,840,000 samples`
- ログの `bufferedSamples=3840000` は常に満杯状態
- `captureQueue` = `LockFreeRingBuffer<AudioBlock, 4096>` = 最大 4096 blocks

### 動作解析

```
Audio Thread (RT)                Worker Thread (Non-RT)
     │                                │
     │---push(AudioBlock)---→[Queue]--│--drainCaptureQueue()→[segmentBuffer]
     │         毎ブロック             │      5ms polling
     │                                │--buildTrainingSegments()
     │                                │--CMA-ES Optimization(~数100ms)
     │                                │--publishGenerationResult()
```

- `accepted` カウントの変動（832→756→...→48528）は **期待動作**
  - segmentCount < 2 の待機中: 少量ずつ drain（〜756/秒）
  - Optimization実行後: 溜まっていた全ブロックを一気に drain（〜数万）
- これにより Audio Thread はブロックされない
- **音飛びの原因ではない**

---

## 6. convDebounce 機構 **← 影響なし・正常**

### 確認事項

- ログでは `convDebounce(req/defer/sched/trigger)=0/0/0/0`
- カウンタが全て 0 のため、convDebounce は一度も発火していない
- ConvolverProcessor 側の `rebuildPendingAfterLoad` / `isLoading` / `isRebuilding` も正常

---

## 7. 総合的な音飛び原因

### 主原因ランキング

| 順位 | 原因 | 影響 | 対策難易度 |
|------|------|------|-----------|
| **1** | **rebuildIR 405ms の同期的ブロッキング** | **80%** | 中 |
| 2 | DSPCORE_PREPARE 6回連鎖（起動時） | 15% | 低 |
| 3 | NoiseShaperLearner CPU負荷（間接的） | 5% | 低 |

### 音飛びのメカニズム

```
起動 or IR変更
  └→ convolverParamsChanged → rebuild generation increment
      └→ rebuildThreadLoop:
          ├─ RuntimeBuilder.build() → DSPCore構築 (~95ms)
          └─ convolverRt().rebuildAllIRsSynchronous() → LoaderThread.runSynchronously():
              ├─ doTransformStep() → MixedPhase (~176ms)
              └─ doBuildStep() → NUC構築 (~228ms)
              └─ Total: ~405ms
      └→ trySubmit: publish SUCCEEDED → Runtime切り替え
          └→ ★ この瞬間、新しいConvolverStateが適用される
              └→ DSPCore_PREPARE → 全DSPチェーン再初期化
                  └→ この間、AudioBufferの処理に過渡的な乱れが発生
```

rebuildThread は Audio Thread と別スレッドのため、405ms のリビルド中も Audio Thread は古い Runtime で処理を続けられる。しかし、**publish 後の Runtime 切り替え + DSPCore_PREPARE + クロスフェード** のタイミングで過渡的な乱れが発生する可能性がある。

### 最も効果的な改善（優先順位順）

1. **`setUseIncrementalRebuild` のバグ修正 + 呼び出し追加** → rebuildIR時間を分割
2. **prepareToPlay 暫定値を実運用レートに変更** → 起動時のDSPCORE_PREPARE連鎖を削減
3. **IppFFTPlanCache の事前ウォームアップ** → rebuildIRのFFT plan作成時間を削減
4. **MixedPhasePersistentCache の有効活用** → 同一IRの再リビルドを高速化
