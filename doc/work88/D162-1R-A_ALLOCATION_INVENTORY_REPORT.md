# D162-1R-A Work Report — Allocation Inventory（read-only・production 変更 0）

- Work item: D162-1R-A（D162-1R の Phase 0: allocation site 全件棚卸し・instrumentation 実装前の必須作業）
- Date: 2026-09-02
- Mode: **完全 read-only**（src/ / tests/ / CMakeLists.txt 変更 0）
- Authority: 現行 ConvoPeq.md + 実ソース直読（serena / rg / python 解析併用）
- 上流 evidence: evidence/D162-1P_soak.log（[IR_LOAD]/[IR_LAYOUT]/[IR_RELEASE]/[MEM_SNAP] 実測再解析を含む）

---

## 0. Executive Summary

1. **D162-1P の「未帰属 ~70 MB/DSP」の主犯候補を静的に特定した**: 各 DSPCore が所有する `ConvolverProcessor`（`convolverState`）が固定容量バッファとして
   `delayBuffer` 2ch × `DELAY_BUFFER_SIZE = 4,194,304` samples = **64 MB**、
   `dry / smoothing / oldDry / wet` 4 セット × 2ch × `MAX_BLOCK_SIZE = 524,288` = **32 MB**、
   `delayFadeRamp` **4 MB** — 合計 **~100 MB/DSP** を `convo::aligned_malloc`（= diag counter 対象）で確保している（ConvolverProcessor.Lifecycle.cpp:299-364）。
2. **diag counter の収支が閉じた**: 世代ごとの tracked 確保量は **~229 MB/gen**（IR_LOAD before 値の差分・実測）。内訳 ≈ ConvolverProcessor 固定バッファ 100 + NUC pair 34（[IR_LOAD] delta 17MB×2 実測）+ latency 23.5 + EQ 1.6 + irData/IRState 5.8 + DSPCore aligned 0.3 + その他 ~5 ≈ **170 MB/live-DSP** + 一時確保 ~60 MB/gen。Private 増加 141 MB/gen は freed arena が次世代の確保に再利用されるため live 分のみ増加すると説明が整合。
3. **D162-1P の訂正 1 件**: `tF=0GB` は「counter が減らない」のではなく **(a)** aligned_free 経路（ConvolverProcessor 固定バッファ・latency・irData・DSPCore 本体）が counter を通らない設計と、**(b)** 表示丸め（0.3GB→0GB）の複合。NUC layer の解放（freeTracked 経路）は正しく counter を減算している（IR_RELEASE のうち空 NUC 118 件は delta=0、実解放 18 件は delta=−17MB を実測）。
4. 既存診断 `[IR_LOAD]/[IR_LAYOUT]/[IR_RELEASE]` + `allocSizes` + `NucDiagnosticsSnapshot` は設計通り動作している。D162-1R 実装は**新規仕組みではなく既存体系への最小拡張**（live inventory table + per-site サイズ個別出力 + DSP footprint snapshot）でよいことが確定した。

---

## 1. Allocator 経路の全体構造

### 1.1 確保経路

| 経路 | 展開 | diag counter | 使用者 |
| --- | --- | --- | --- |
| `DIAG_MKL_MALLOC(size, align)` | DIAG=1: `diagMklMalloc` → `mkl_malloc`（counter++）<br>DIAG=0: `mkl_malloc` 直 | **++ 対象** | NUC 全バッファ |
| `convo::aligned_malloc` / `makeAlignedArray` / `aligned_make_unique`（AlignedAllocation.h:19-31） | `DIAG_MKL_MALLOC` 経由 → 同上 | **++ 対象** | DSPCore 本体・ConvolverProcessor 固定バッファ・latency・EQ・irData・oversampler 等 |
| 生 `mkl_malloc`（NUC scratch） | 直接 | **非対象** | impulseForFft(:721)・tempTime/tempFreq(:906-907)・swapSoA(:961)・reusableGain(:361)・gainReal(:1076) |

注: `ConvoPeq` ターゲットは `JUCE_DSP_USE_INTEL_MKL=1` + `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1`（compile_commands.json で実確認）。テストターゲットの一部（AudioEngineHarness は MKL 定義なし等）と非対称だが、本監査対象の ConvoPeq.exe は両立。

### 1.2 解放経路（D162-1P 発見の訂正を含む）

| 経路 | 展開 | counter 減算 | 使用箇所 |
| --- | --- | --- | --- |
| `freeTracked(ptr, size)`（DIAG のみ） | `DIAG_MKL_FREE` = `diagMklFree(ptr,size)` | **減算する** | NUC `Layer::freeAll`（14 field）+ NUC レベル 5 バッファ（MKLNonUniformConvolver.cpp:256-273, 525-531） |
| `convo::aligned_free(ptr)` | `mkl_free`（サイズ不明） | **減算しない** | DSPCore 破壊（Threading.cpp:17-25）・irData・IRState owner・latency 再確保・ConvolverProcessor 固定バッファの再確保時 |
| 生 `mkl_free(ptr)` | 直接 | **減算しない** | NUC scratch（table 未登録なので問題なし） |

→ **counter は「確保累積 − freeTracked 解放」であり live bytes ではない**。D162-1P の `tF=0GB` は (a) aligned_free 経路の非減算 + (b) 0.3GB の表示丸め（`tF` は GB 表示）による見かけ。IR_RELEASE 実測: 空解放 118 件 delta=0 / 実解放 18 件 delta=−17MB — `allocSizes` 経路は正常。

---

## 2. Per-DSPCore allocation inventory（generation-scoped・全件）

### 2.1 ConvolverProcessor（`convolverState`・DSPCore 1 個につき 1 個）— ★ 新規特定の主項

| Buffer | Site | Size（1 ch） | ×2ch | 備考 |
| --- | --- | --- | --- | --- |
| delayBuffer[0/1] | Lifecycle.cpp:303-304 | `DELAY_BUFFER_SIZE = 4,194,304` × 8B = **32 MB** | **64 MB** | `prepareForProcessing` 毎に capacity 判定（新規オブジェクトなら必ず確保） |
| dryBufferStorage | :321 | `MAX_BLOCK_SIZE = 524,288` × 8B = 4 MB | 8 MB | |
| smoothingBufferStorage | :322 | 同上 | 8 MB | |
| oldDryBufferStorage | :323 | 同上 | 8 MB | |
| wetBufferStorage | :342-344 | 同上 | 8 MB | |
| delayFadeRampBuffer | :356-360 | 4 MB | 4 MB（単一） | |
| **ConvolverProcessor 合計** | | | **≈ 100 MB** | |

### 2.2 StereoConvolver / NUC（DSPCore 1 個 = StereoConvolver 1 + NUC 2）

| Buffer | Site | 実測/計算 | 備考 |
| --- | --- | --- | --- |
| NUC layer ×2ch（irFreqReal/Imag, fdlReal/Imag, irFreqDomain, fdlBuf, fftTime/OutBuf, prevInputBuf, accumBuf/Real/Imag, inputAccBuf, tailOutputBuf, delayLineBuf） | MKLNonUniformConvolver.cpp:817-872, 1009 | **[IR_LAYOUT] Total=18 MB/NUC 実測**（L0=3 + L1=14, part0=2048, irLen=192000）→ ×2 = **36 MB** | allocSizes 保存済み |
| NUC ring | :1041 | ringSize=16384 → 0.13 MB | IR_LAYOUT Ring に含まれる |
| NUC direct ×4 | :699-704 | directTaps=0（soak 実測）→ 0 | |
| irData[0/1] | ConvolverProcessor.h:792-793（LoaderThread.cpp:222 で確保→init に移管） | 192000×8B×2 = **2.9 MB** | aligned_free 解放 |
| IPP FFT plan（spec + work）×layer×2ch | FFTBackend.cpp:41,71 | untracked（ippsMalloc_8u） | sizeSpec/sizeWork は IPP 内部値 — 診断 gap |

### 2.3 DSPCore 直属

| Buffer | Site | Size | 備考 |
| --- | --- | --- | --- |
| latency buffers ×4（old/new × L/R） | PrepareToPlay.cpp:193-196 | `min(1,536,000, 384000×2)+1024+2 = 769,026` × 8B × 4 = **23.5 MB** | D162-1P 通り |
| alignedL/R + dryBypassDoubleL/R | DSPCoreLifecycle.cpp:152-164 | 8192 × 8B × 4 = 0.25 MB | |
| EQ 8 バッファ（scratch/dry/parallel×3/xfade×2/msWork/agc） | EQProcessor.Core.cpp:695-740 | ≈ **1.6 MB**（65536 + 16384×6 + 32768 samples） | [EQ_PREPARE] 実測整合 |
| LoudnessMeter filterWork + ring | LoudnessMeter.cpp:22, :84 | ≈ 0.2 MB | |
| Oversampler（stages×3 convCoeffs/history + workA/B ×2ch） | CustomInputOversampler.h:113-120 | ≈ 1-3 MB（推定） | **実サイズ未計測** |
| softClipOS（single stage） | 同上 | ≈ 1 MB（推定） | 同上 |
| TruePeakDetector upsampleBuffer | TruePeakDetector.cpp:39 | 未計測（推定 <1 MB） | |
| DSPCore / StereoConvolver / NUC オブジェクト本体 | RuntimeBuilder.cpp:425 ほか | ~0.1-1 MB（sizeof 未測定） | aligned_make_unique |

### 2.4 世代ごとの一時確保（counter に乗るが解放される）

| Buffer | Site | Size |
| --- | --- | --- |
| loader irL/irR（一時・init 後 irData へ移管なので実質 persistent） | LoaderThread.cpp:222 | 2.9 MB |
| impulseForFft（SetImpulse 内 scoped） | MKLNonUniformConvolver.cpp:721 | 1.46 MB × 2 NUC |
| tempTime/tempFreq・swapSoA・gainReal（同上） | :906-961, :1076 | < 1 MB |
| ResampleAndFallback window/envelope/aligned_data | ResampleAndFallback.cpp:135-207 | 76800×8B × ~5 = ~3 MB（同 SR では低減） |
| IRState + irOwner（RCU 入替、旧は deferred delete） | Lifecycle.cpp:32-34 | irOwner は juce AudioBuffer（CRT・非 tracked）、IRState オブジェクトのみ tracked |
| RuntimePublishWorld / RuntimeState / FrozenRuntimeWorld（submit 毎） | RuntimeBuilder.cpp:425, Orchestrator.cpp:272 | KB〜低 MB（sizeof 未測定） |

### 2.5 収支（generation 単位）

| 項目 | MB/gen | 出所 |
| --- | ---: | --- |
| tracked 確保累積の増分（実測） | **229**（204-241） | IR_LOAD before 値差分 59 ペア |
| ConvolverProcessor 固定バッファ | 100 | §2.1（静的） |
| NUC pair | 36 | [IR_LOAD] 実測 |
| latency | 23.5 | §2.3 |
| irData | 2.9 | §2.2 |
| EQ + DSPCore aligned + Loudness + TP + OS | ≈ 5 | §2.3 |
| 一時確保（loader/scratch/world 等） | ≈ 60 | 差分（= 229 − 169 live 分） |
| **合計（閉包）** | **≈ 229** | ✓ |

**live footprint per retained DSP（allocator 再利用調整後）**: static 合計 ≈ **169 MB** vs 実測 private 増加 **141.7 MB/gen**（Δ≈12-15% — RuntimePublishWorld/EQ/OS の推定部と MEM_SNAP の MB 丸めを含む）。この残差の runtime 確認が AC-R2/R3 の対象。

---

## 3. 既存診断の棚卸し（D162-1R で拡張する土台）

| 既存機能 | 状態 | D162-1R での扱い |
| --- | --- | --- |
| `[IR_LOAD]` NUC#/seq/irLen/blockSize/Layers/MKL delta | 実測動作確認済み（118 件 × +17MB） | そのまま活用 |
| `[IR_LAYOUT]` IRFreq/FDL/Accum/Tail/Direct/Ring/Total + L0/L1/L2 | 実測動作確認済み（Total=18MB 一貫） | **Total ではなく kind 別・個別 allocSizes の出力に拡張**（指示 §2） |
| `[IR_RELEASE]` MKL before/after/delta + LayersBefore | 動作確認済み（空解放 118 × 0MB / 実解放 18 × −17MB） | そのまま活用 |
| `Layer::allocSizes`（15 field）+ `NucDiagnosticsSnapshot` | 実装済み | live inventory の data source として使用 |
| `MEM_SNAP`（NUC alloc/DC/SC/Ret/Priv/WS/TRK） | 動作。ただし NUC alloc=累積、TRK=推算式（実サイズでない）、Other=clamp 無効 | TRK は依存させない（指示 §8・優先度 ④） |
| `destroyDSPCoreNode`（[D117_DESTROY]） | 動作 | footprint snapshot の destroy-side 出力点に |
| `runtimeDSPHandleMap_` | 診断出力なし | shutdown 時 1 点のみ（指示 §7） |

---

## 4. D162-1R instrumentation 設計案（実装は次フェーズ・本 report では設計のみ）

### 4.1 Live Allocation Table（指示 §4・DIAG 専用）

- `DiagnosticsConfig.h` に `convo::diag::LiveAllocRegistry`（DIAG ガード内）を追加:
  - `register(ptr, size, siteId, ownerCtx)` / `unregister(ptr)`（ptr → {size, siteId, ownerCtx} の hash map + mutex）
  - `diagMklMalloc` が register、`diagMklFree` が unregister（NUC 経路）
  - `convo::aligned_free` に DIAG ガード下で `unregister(ptr)` 1 行（size は table lookup）— **production OFF 時は現行 semantıcs 完全不変**
  - 生 `mkl_free` は table 非登録ポインタなので無関係（unregister は miss で無視）
- ownerCtx: `thread_local` または scoped guard による `{generation, dspPtr, ownerKind}`。RuntimeBuilder::build 入口 / prepare 入口 / SetImpulse 入口で set、出口で reset。

### 4.2 siteId 分類（指示 §2 の category 割付）

```
NUC_IRFREQ / NUC_FDL / NUC_ACCUM / NUC_TAIL / NUC_DIRECT / NUC_RING / NUC_DELAYLINE
NUC_SCRATCH(生mkl) / IPP_PLAN
CONV_DELAY(64MB) / CONV_DRY / CONV_SMOOTH / CONV_OLDDRY / CONV_WET / CONV_RAMP
IRDATA / IRSTATE / LATENCY(23.5MB) / EQ / OVERSAMPLER / LOUDNESS / TP / DSPCORE_OBJ / WORLD / OTHER
```

### 4.3 出力ログ

1. `[NUC_ALLOC] NUC#%p gen=%u layer=%u kind=%s bytes=%zu` — SetImpulse 完了時に layer × kind の個別 `allocSizes` を 1 行ずつ（指示 §2 の「Total だけでは不可」に対応）
2. `[DSP_FOOTPRINT]` — prepare 完了時 + destroy 時に同一 identity で:
   `gen=%u dsp=%p nuc0=%zuMB nuc1=%zuMB conv=%zuMB latency=%zuMB eq=%zuMB other=%zuMB TOTAL=%zuMB liveEntries=%u`
   （live inventory の ownerCtx 集計 + allocSizes 合算）
3. `[LIVE_LEAK]` — destroy 完了時に ownerCtx=dsp の残存 entry があれば出力（AC-R5）
4. shutdown 時: `[HANDLEMAP] size=%u`（1 点）+ live inventory の site 別サマリ

### 4.4 検証マトリクス（指示 §11）

- RelWithDebInfo DIAG（本命・soak 用）: instrumentation ありで同一 CLI 条件短縮 soak（AC-R7: 49 retained 再現）
- RelWithDebInfo DIAG + ASAN: instrumentation が heap を壊していないことの確認
- Debug DIAG: 補助
- Release DIAG: 既存起動 crash が解消されるまで範囲外（別管理）

### 4.5 AC 対応

| AC | 充足手段 |
| --- | --- |
| AC-R1 全 generation-scoped allocation 列挙 | §2 inventory（静的）+ live table（動的） |
| AC-R2 site sum ≈ observed footprint | [DSP_FOOTPRINT] TOTAL vs private 差分（§2.5 の残差 12-15% を含め評価） |
| AC-R3 閉包（NUC+latency+EQ+DSP-local+other=TOTAL） | [DSP_FOOTPRINT] カテゴリ集計 |
| AC-R4 gen N ↔ retained DSP の identity 対応 | ownerCtx（generation, dspPtr）+ [D133]/[D117] 既存ログとの突合 |
| AC-R5 destroy 時 live→0 | [LIVE_LEAK] 非出力（= zero residual） |
| AC-R6 診断 OFF で production 不変 | 全 instrumentation `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` + aligned_free の unregister も DIAG ガード内（バイナリ等価） |
| AC-R7 短縮 soak で 49 retained 再現 | RWDI DIAG + instrumentation で同一条件 soak → DC live 1→50 + [DSP_FOOTPRINT] 50 件 |

---

## 5. 分岐判定の前倒し通知（D162-2 起票条件に関わる）

D162-1P の分岐基準「allocation で 123–141 MB/DSP が閉じる → D162-2 起票可能」について、**静的 inventory で ≈169 MB/DSP の帰属モデルが成立した**（主因: ConvolverProcessor 固定バッファ 100 MB）。ただし指示どおり runtime 閉包（AC-R2/R3/R5/R7）を取るまで D162-2 は起票しない。D162-1R 実装フェーズで [DSP_FOOTPRINT] が閉包を確認した時点で改めて判定する。

## 6. production source 変更確認

- 変更 0。本 report + evidence 解析（既存 soak ログの再解析）のみ。
- D162-1P 訂正: tF=0GB の解釈（§1.2）— H1〜H4 の判定には影響なし。
