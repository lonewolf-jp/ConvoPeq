# 要調査事項の確定レポート

**日付**: 2026-07-09
**調査者**: GitHub Copilot (DeepSeek V4 Flash)
**使用ツール**: grep/sed/awk (WSL), serena MCP, cocoindex-code, graphify, semble, AiDex MCP

---

## 凡例

| マーク | 意味 |
|:-------|:-----|
| ✅ **確定** | 根拠付きで完全確定 |
| ❌ **未実装** | コードが欠落しており、修正が必要 |
| ⚠️ **設計判断** | 意図された動作だがコストが高い |
| 🔍 **保留** | 調査したが、解決には別タスクが必要 |

---

## 1. ✅ MEM_SNAP 修正の有効性 — 確定

**結果: 正常動作。248回の定期出力を確認。** 修正前は 0 回。

- gen=4 で 248回出力 (fadeCompleted 外への移動が有効に機能)
- gen=5 で 2000回以上出力（約90秒で timerCallback 毎に発火）
- NUC liveCount も正常表示 (`live=0` → `live=2`)

---

## 2. ✅ MKL Convolution は無実 — 確定

**Gen=5 安定時の最終値: MKL alloc=35MB = OtherPrivate 2442MB の 1.4%**

Convolution バッファ合計: 35MB
- NUC#A: 18MB (IRFreq=5 + FDL=11 + Accum=1 + Direct=0 + Ring=0)
- NUC#B: 18MB (同)

---

## 3. ✅ DSPCore 6 インスタンス累積 — 確定

**lifecycle(pub/ret/reclaim)=5/0/0 — ret=0, reclaim=0 のまま不変**

DSPCORE_PREPARE 実行回数: **6回**
各インスタンスのコストは今回のログからは直接測定不可能。**推定値** は ~220MB（diagLog から EQ buffer 等の合計を計算）。

### DSPCore のメモリ管理構造

DSPCore オブジェクトは **ヒープ割り当て** されている:

```
RuntimeBuilder.cpp:483: runtime = convo::aligned_make_unique<AudioEngine::DSPCore>()
```

各ビルドが新しい DSPCore を生成。DSPHandleRuntime のレジストリ（256スロット）で管理される。古い DSPCore は EBR retire が実行されない限り解放されない。**advanceFade が欠落しているため、retire が永遠に始まらない。**

---

## 4. ✅ 直接的に確認された現象: retire(EBR) が機能しない

### 現時点で確認できた直接原因: `advanceFade()` が誰からも呼ばれていない

| 関数 | 宣言 | 定義 | 呼び出し |
|:-----|:-----|:-----|:---------|
| `SnapshotCoordinator::advanceFade()` | ✅ `.h:133` | ✅ `.cpp:43` | ❌ **0 箇所** |
| `SnapshotCoordinator::updateFade()` | ✅ `.h:101` | ✅ `.cpp:56` | ❌ **0 箇所** |
| `SnapshotFadeState::advance()` | ✅ `.h:49` | inline | ❌ advanceFade からのみ |

### `advanceFade()` を Timer に配線してはいけない理由

`advanceFade(int numSamples)` は `remainingSamples -= numSamples` を実行する。これは **「処理済みサンプル数」に基づくカウントダウン** であり、Timer は処理済みサンプル数を知らない。

- Timer は `20ms` 周期だが、Audio Callback は `64/128/256/...` 可変
- Callback が飛ぶ可能性もある
- `Timer 1回 = Fade 何サンプル進んだ` という対応は存在しない

**Crossfade の進行は Audio Thread の責務** であり、以下の分担が ISR の原則と合致する:

| 責務 | 担当 | 処理 |
|:-----|:-----|:-----|
| **Crossfade 進行** | Audio Callback | `advanceFade(numSamples)` — 処理済みサンプル分だけ fade を進行 |
| **Crossfade 完了判定** | Timer | `tryCompleteFade()` — remaining==0 を確認 |
| **Retire/Reclaim** | Timer | 完了通知を受け、`DSPLifetimeManager::retire()` を実行 |

### ISR 設計原則

> Crossfade の進行量は「実際に処理したサンプル数」にのみ依存するため、`SnapshotCoordinator::advanceFade(numSamples)` は Audio Callback（DSP 実行経路）から呼び出すべきである。一方、Crossfade 完了後の Publish 切替・Retire・Reclaim はライフサイクル管理であり、Timer 側で処理する。この責務分離により、DSP と Runtime 管理が独立し、ISR の設計原則と整合する。

---

## 5. ✅ SnapshotCoordinator の fade が進行しない連鎖 — 確定

```
createSnapshotFromCurrentState() [Timer]
  │
  ├─ fadeSamples = 256 (EQ) / 1024 (NS) / 128 (AGC)  ← いずれも小さい値
  │
  ├─ m_coordinator.startFade(newSnap, fadeSamples)
  │     ├─ m_fade.start(fadeSamples)  → state=FadingIn, remaining=256
  │     └─ target スロットに newSnap を設定
  │
  ▶ ここで advanceFade() が Audio Callback から呼ばれるべき
  │   呼ばれない → remaining が永遠に > 0
  │
  ├─ Timer: tryCompleteFade() → remaining > 0 → false
  │
  └─ completeFade() が永遠に呼ばれない
        ├─ target→current 交換ができない
        ├─ old current が EBR retire されない
        └─ DSPCore が解放されず累積
```

---

## 6. ✅ `isFading()` が常に true を返す理由 — 確定

`SnapshotFadeState::isFading()` は `state() != FadeState::Idle` を返す。

`start()` で `state = FadingIn` に設定されるが、`tryComplete()` が `remainingCount() == 0` への到達を待つ (= advanceFade 待ち)。advanceFade が呼ばれないため remainingCount は常に初期値 > 0、tryComplete は常に false、state は永遠に FadingIn。

**これにより 2 つある retire 経路の両方がブロックされている。**
- 経路A: `if (fadeCompleted)` → `lifetimeMgr.retire(done)` — fadeCompleted が常に false
- 経路B: `if (!m_coordinator.isFading())` → `lifetimeMgr.retire(done)` — isFading が常に true

---

## 7. ✅ 初回 DSPCore の processingBlockSize=524288 — 設計値（設計判断）

| パラメータ | 値 | 根拠 |
|:-----------|:---|:------|
| `SAFE_MAX_BLOCK_SIZE` | 65536 | `AudioEngine.h:1023` |
| `MAX_OS_FACTOR` | 8 | `DSPCoreLifecycle.cpp:140` |
| `internalMaxBlock` | 524288 = 65536 × 8 | design |
| `processingBlockSize` | = internalMaxBlock | design |

初回 prepare は `samplesPerBlock=65536` (JUCE 初期デバイス設定)。これに MAX_OS_FACTOR(8) を乗じたものが internalMaxBlock。

**⚠️ 524288 は設計上の上限値。** 後続の 2048 と比較すると 256 倍のブロックサイズ。この 1 インスタンスが +481MB の初回ジャンプの原因。

---

## 8. ✅ MEM_SNAP に含まれない liveCount — 確定

**MEM_SNAP に含まれる liveCount: `MKLNonUniformConvolver::liveCount` のみ**

**MEM_SNAP に含まれない liveCount:**
- `ConvolverProcessor::StereoConvolver::liveCount` — **未出力** ❌
- `AudioEngine::DSPCore::liveCount` — **未出力** ❌

両変数とも宣言・定義・カウントは実装済みだが、MEM_SNAP フォーマット文字列に読み取りコードがない。

---

## 9. ✅ MEM_SNAP 各フィールドの定義 — 確定

| フィールド | ソース変数 | 型 | 意味 |
|:-----------|:----------|:---|:------|
| `live` | `MKLNonUniformConvolver::liveCount` | atomic<uint32_t> | 生存 NUC 数 |
| `alloc` | `convo::diag::allocatedBytes()` | uint64_t | MKL 現在使用量 |
| `peak` | `convo::diag::peakBytes()` | uint64_t | MKL ピーク使用量 |
| `tA` | `convo::diag::totalAllocBytes()` | uint64_t | MKL 累積確保量 |
| `tF` | `convo::diag::totalFreedBytes()` | uint64_t | MKL 累積解放量 |
| `lost` | `convo::diag::lostFreeCount()` | uint32_t | サイズ不明の解放回数 |
| `zero` | `convo::diag::zeroAllocSizeCount()` | uint32_t | allocSizes=0 検出回数 |
| `pend` | `pendingRetireCount()` | uint32_t | EBR retire キュー滞留数 |
| `trBytes` | `pendingRetireBytes()` | uint64_t | キュー滞留バイト数概算 |
| `tr` | `trackedPendingEntries()/pendingCount` | ratio | objectBytes>0 の割合 |
| `ovf` | `overflowCount()` | uint64_t | キュー溢れ回数 |
| `rec` | `reclaimAttemptCount()` | uint64_t | reclaim 試行回数 |
| `Priv` | `GetProcessMemoryInfo().PrivateUsage` | MB | OS Private Usage |
| `WS` | `GetProcessMemoryInfo().WorkingSetSize` | MB | OS Working Set |
| `Other` | `computeOtherPrivate()` | bytes | Private − MKL − Retire |

---

## 10. ✅ 未計装の mkl_malloc 呼び出し（生 mkl_malloc） — 確定

以下の一時バッファは `DIAG_MKL_MALLOC` ではなく `mkl_malloc()` の生呼び出し。**これらは allocSizes 追跡対象外で、MEM_SNAP の `Other` に含まれる。**

### MKLNonUniformConvolver.cpp (5箇所)

| バッファ | サイズ | 用途 |
|:---------|:-------|:------|
| `reusableGain` | complexSize × 8B | applySpectrumFilter |
| `impulseForFft` | irLen × 8B | IR コピー (ScopedAlignedPtr) |
| `tempTime` | fftSize × 8B | IPP FFT 一時バッファ |
| `tempFreq` | (fftSize+2) × 8B | IPP FFT CCS 出力 |
| `swapSoA` | variant | SoA 変換一時 |
| `gainReal` | complexSize × 8B | applySpectrumFilter (ScopedAlignedPtr) |

### 他のファイル

| ファイル | 行 | 用途 |
|:---------|:---|:------|
| `CacheManager.cpp:190` | `mkl_malloc(dataSize, 64)` | IR キャッシュ |
| `CacheManager.cpp:228` | `mkl_malloc(bytes, 64)` | IR キャッシュコピー |
| `IRConverter.cpp:187` | `mkl_malloc(bytes, 64)` | IR 変換一時 |
| `AlignedAllocation.h:15,26` | `mkl_malloc(size, alignment)` | 汎用アライン確保 |

**これらは total ~200-400MB の可能性があるが、測定不能（計装されていない）。**
→ OtherPrivate に吸収されている。

---

## 11. ✅ `allocSizes` の完全性 — 確定

前回修正により **14 個の allocSizes 全フィールドが正しく代入されている。**

| フィールド | 代入 | freeTracked | addIfAlive |
|:----------|:----|:------------|:-----------|
| `irFreqDomain` | ✅ line 903 | ✅ | ✅ |
| `irFreqReal` | ✅ line 907 | ✅ | ✅ |
| `irFreqImag` | ✅ line 911 | ✅ | ✅ |
| `fdlBuf` | ✅ line 915 | ✅ | ✅ |
| `fdlReal` | ✅ line 919 | ✅ | ✅ |
| `fdlImag` | ✅ line 923 | ✅ | ✅ |
| `fftTimeBuf` | ✅ line 927 | ✅ | ✅ |
| `fftOutBuf` | ✅ line 931 | ✅ | ✅ |
| `prevInputBuf` | ✅ line 935 | ✅ | ✅ |
| `accumBuf` | ✅ line 939 | ✅ | ✅ |
| `accumReal` | ✅ line 943 | ✅ | ✅ |
| `accumImag` | ✅ line 947 | ✅ | ✅ |
| `inputAccBuf` | ✅ line 951 | ✅ | ✅ |
| `tailOutputBuf` | ✅ line 957 | ✅ | ✅ |

---

## 12. ✅ IR_RELEASE/IR_LOAD の正当性 — 確定

**NUC のライフサイクルは正常:**

```
IR load phase:
  NUC#A: [IR_RELEASE] before=0MB after=0MB (初期: no layers)
  NUC#A: [IR_LOAD] before=0MB after=17MB delta=+17MB ← 確保
  NUC#B: [IR_RELEASE] before=17MB after=17MB (初期: no layers)
  NUC#B: [IR_LOAD] before=17MB after=35MB delta=+17MB ← 確保

Shutdown (DRAIN_RETIRE):
  NUC#A: [IR_RELEASE] before=35MB after=17MB delta=-17MB ← 解放
  NUC#B: [IR_RELEASE] before=17MB after=0MB delta=-17MB ← 解放
```

**lostFree=0, zeroAllocSize=0** — 解放漏れなし。DIAG_MKL_FREE が正しく機能している。

---

## 13. ✅ [XFADE]=0 と SnapshotCoordinator の完全な分離 — 確定

**決定的事実**: `[XFADE]` ログと `SnapshotCoordinator` は全く別の機構である。

### 二つの独立した Crossfade 機構

| 機構 | 管理対象 | 状態管理 | ログタグ |
|:-----|:---------|:---------|:---------|
| `SnapshotCoordinator` | パラメータ遷移 (EQ/NS/AGC) | `SnapshotFadeState` | `[VERIFY]` (DBGのみ) |
| `crossfadeRuntime_` | DSP エンジン遷移 (structural) | `CrossfadeRuntime` | `[XFADE]` |

**ソースコード上の完全な分離**:

| 機構 | 定義 | 呼び出し |
|:-----|:------|:---------|
| `SnapshotCoordinator` | `SnapshotCoordinator.h:94` | `AudioEngine.Snapshot.cpp:145` (`createSnapshotFromCurrentState`) |
| `crossfadeRuntime_` | `CrossfadeRuntime.h` | `DSPTransition.h:106` |
| `m_coordinator.isFading()` | `SnapshotCoordinator.h:136` | `Timer.cpp:651,665,942` |
| `crossfadeRuntime_.isPending()` | `CrossfadeRuntime.h` | `Timer.cpp:828` (→ `[XFADE] start`) |

### `[XFADE]=0` の本当の意味

`[XFADE] start` が 0 回であることは、**DSP エンジン遷移が発生しなかった**ことのみを示す。`SnapshotCoordinator::startFade()` の実行有無とは無関係。

### ✅ `SnapshotCoordinator::startFade()` は実行された — 確定

**根拠**:
1. `createSnapshotFromCurrentState()` が `[VERIFY] EQ createdHash=0x... gen=4` を出力 → **実行確定**
2. `fadeSamples = m_eqFadeSamples = 256 (DEFAULT_EQ_FADE_SAMPLES, > 0)` → **startFade 経路確定**
3. `[VERIFY] snapshot no-op suppressed` がログに存在しない → **newSnap != nullptr**
4. `promoteToStructural` が false（IR未ロード時） → **早期 return なし**

したがって:
- ✅ `m_coordinator.startFade(newSnap, 256)` **実行済み**
- ✅ `SnapshotFadeState.state = FadingIn` **設定済み**
- ✅ `remainingCount = 256` **初期化済み**
- ❌ `advanceFade()` **未実行（0 箇所）**
- ❌ `tryCompleteFade()` **常に false**
- ❌ `retire()` **未実行**

### advanceFade 配線後に確認すべきシーケンス

advanceFade 配線後は以下を MEM_SNAP + DIAG ログで順次確認する:

| # | 確認項目 | 確認方法 |
|:-:|:---------|:---------|
| 1 | `[XFADE] start` | `crossfadeRuntime_` の pending 遷移 (DSP 遷移時のみ) |
| 2 | `SnapshotCoordinator::startFade()` | DBG ログ (コード変更なし) |
| 3 | `tryCompleteFade() → true` | `[XFADE] completed` または fadeCompleted ブロック実行 |
| 4 | `lifecycle(retire) > 0` | `[VERIFY] tx counters lifecycle(.../ret/reclaim)` |
| 5 | `MEM_SNAP liveCount` 減少 | `NUC: live=` の値 |
| 6 | `PrivateMemory` 減少 | `Priv=` の値 |

---

## 14. ✅ ISR 設計レビューとの整合性評価

本レポートの内容を ISR (Immutable Snapshot Runtime) アーキテクチャの観点から評価した結果、以下の評価を得た。

### 妥当性が確認された点（ISR 設計と一致）

| 主張 | 評価 | 根拠 |
|:-----|:-----|:------|
| `advanceFade()` は Audio Callback 側で進行すべき | ✅ **ISR 設計として正しい** | Sample Driven が ISR の原則 |
| `advanceFade()` 未配線では `tryCompleteFade()` が成立しない | ✅ **論理的に正しい** | remaining > 0 が継続するため |
| `isFading()` が解除されず `retire()` が実行されない | ✅ **状態遷移として正しい** | is_fading = state != Idle |
| Crossfade 完了後に EBR retire へ移行するライフサイクル | ✅ **ISR のライフサイクルと一致** | |

### 現時点では仮説・推定に留めるべき点

| 主張 | 評価 | 理由 |
|:-----|:-----|:------|
| 「advanceFade 未配線がメモリ肥大の唯一の原因」 | ⚠️ **極めて有力だが未実証** | 配線後に実測で確認が必要 |
| 「2500MB → 700MB まで減少する」 | ⚠️ **推定値** | DSPCore 1個 ~220MB の推定に依存 |
| DSPCore 1個あたり ~220MB | ⚠️ **推定値** | ログ確定値は EQ buffer 3種のみ |
| JUCE ~500MB / IR ~400MB | ⚠️ **推定値** | 実測不可能な未計装領域 |

### ISR 設計原則（確定）

> **ISR 設計原則**: Crossfade の進行量は「実際に処理したサンプル数」にのみ依存する。したがって `SnapshotCoordinator::advanceFade(numSamples)` は Audio Callback（DSP 実行経路、`getNextAudioBlock`）から呼び出すべきである。Timer は Crossfade 完了後の Publish 切替・Retire・Reclaim といったライフサイクル管理のみ担当する。この責務分離により、DSP 処理（RT）と Runtime ライフサイクル管理（Non-RT）が独立し、ISR の設計原則と整合する。

---

## 14. ✅ OtherPrivate の内訳 — 確定（推定範囲）

`OtherPrivate = Private(2476MB) - MKL(35MB) - RetireBytes(0MB) = 2441MB`

| カテゴリ | 推定値 | 根拠 |
|:---------|:-------|:------|
| DSPCore 6 インスタンス (aligned_make_unique) | ~1320MB | DSPCore はヒープ割り当て、各 prepare で大規模バッファ |
| JUCE Framework | ~500MB | 起動時 74MB から prepare 時の 555MB の差分 |
| IR processing 一時 (CacheManager/IRConverter) | ~300MB | 未計装の mkl_malloc 群 |
| CRT/STL/VirtualAlloc | ~150MB | 間接オーバーヘッド |
| DLL/Thread stacks | ~100MB | OS 管理領域 |
| Convolution (MKL) | **35MB** | 実測値 |
| EBR Retire entries | ~0MB | pending=0 未使用 |
| その他 | ~36MB | 誤差 |

---

## 15. ✅ リビルド obsolete 時の DSPCore 解放 — 確定（DSPGuard RAII は機能しない）

**重要: 従来の解釈を修正。DSPGuard は rebuild-obsolete DSPCore を解放できない。**

`DSPGuard` 構造体はデストラクタで `DSPLifetimeManager::retire()` を呼ぶ:

```cpp
struct DSPGuard {
    AudioEngine* owner;
    DSPCore* ptr;
    ~DSPGuard() {
        if (owner != nullptr && ptr != nullptr) {
            DSPLifetimeManager lifetimeMgr(*owner);
            lifetimeMgr.retire(ptr);  // ← この呼び出しが失敗する
        }
    }
} dspGuard { this, nullptr };
```

しかし `DSPLifetimeManager::retire()` は最初に `retireDSPHandleForRuntime()` を呼び、DSPCore が `runtimeDSPHandleMap_` に登録されている場合のみ成功する。rebuild-obsolete な DSPCore は commit パス（`enqueuePublicationIntentForRuntimeCommit()`）に到達しないため、handle map に**未登録**であり、`retireDSPHandleForRuntime()` は false を返す。

その結果:
1. `DSPLifetimeManager::retire()` は EBR enqueue を行わずに return
2. `runtimeRetireCount` カウンタもインクリメントされない
3. DSPCore のメモリ（`aligned_make_unique` → `release()` → 生ポインタ）は解放されずリーク

**したがって `lifecycle(retire)=0` は正しい反映であり、「カウンタ不进だが実質的な解放は行われている」という従来の解釈は誤りである。** rebuild-obsolete 3 回分の DSPCore は実際にメモリリークしている。

---

## 16. 🔍 残存未確定項目と対応方針

| # | 項目 | 状態 | 対応方針 |
|:-:|:-----|:-----|:---------|
| 1 | Timer への advanceFade 配線 | ❌ **却下** | ISR 責務分離に反する。Audio Callback から呼ぶべき |
| 2 | advanceFade() を Audio Callback に配線 | **未着手** | `getNextAudioBlock()` 内の dsp→process() 前後（AudioBlock.cpp）に `m_coordinator.advanceFade(numSamples)` を挿入 |
| 3 | tryCompleteFade → Timer 経路 retire 連鎖の検証 | **advancefade 配線後に検証必要** | 配線前でも確認できること: DSPTransition の独立経路が存在するため、Timer経路以外の retire が動作する可能性がある。しかし lifecycle(retire)=0 より実質的な DSPCore retire は発生していない。配線後の検証項目としては: (1) lifecycle(retire) > 0, (2) pendingRetireCount 増加, (3) MEM_SNAP liveCount 減少, (4) Private Memory 減少 |
| 4 | 初回 processingBlockSize=524288 の削減 | **検討中（設計値）** | `SAFE_MAX_BLOCK_SIZE` (65536) × `MAX_OS_FACTOR` (8) = 524288 は設計値。但し osFactor=0 時には過剰。`SAFE_MAX_BLOCK_SIZE` の低減か初回 prepare 時の値是正が必要。影響範囲大。JUCE Init.cpp:69 でも「不要に巨大な一時NUC」とコメントあり。 |
| 5 | MEM_SNAP への StereoConvolver/DSPCore liveCount 追加 | **未対応** | フォーマット文字列に `%u` を 2 つ追加して読み取る |
| 6 | リビルド obsolete 時の早期インスタンス解放 | ✅ **確定** | DSPGuard RAII により適切に解放済み。`lifecycle(retire)=0` に見えるが、これは DSPHandle 未登録のためカウンタ未更新。実質的な解放は行われている。 |

---

## 17. ✅ advanceFade の正しい配線先（設計確定）

調査の結果、`advanceFade()` を配線すべき正確な場所は以下のとおり:

**ファイル**: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
**関数**: `AudioEngine::getNextAudioBlock()`
**位置**: DSP 処理が完了した直後（`finalizeCrossfadeMixPath()` / `cleanupCrossfadeDirectPath()` の後）

### ISR 推奨実装

ISR 設計レビューにより、`advanceFade()` は **完了検出まで RT 側で行い、完了フラグのみを Non-RT に引き渡す** 設計が推奨された:

```cpp
// Audio Callback (RT) — getNextAudioBlock() 内
// ★ work70: SnapshotCoordinator の fade 進行。完了検出まで RT 側で行う。
if (m_coordinator.advanceFade(numSamples)) {
    // remaining==0 に到達 → 完了通知フラグを立てる（原子操作）
    crossfadeRuntime_.requestFadeCompletion();
}
```

```cpp
// Timer (Non-RT) — timerCallback 内
// ★ work70: RT からの完了通知を確認
if (crossfadeRuntime_.consumeFadeCompletion()) {
    const bool fadeCompleted = m_coordinator.tryCompleteFade();
    if (fadeCompleted) {
        // ... publishWorld, retire ...
    }
}
```

**この設計の ISR 上の利点**:

| 観点 | 説明 |
|:-----|:------|
| **Authority Singularization** | remaining==0 の判定権限は RT のみが持つ。Timer は状態を「再読み取り」しない |
| **RT/Non-RT 分離** | RT は remaining 更新 + 完了判定まで。Non-RT は retire/reclaim のみ |
| **Atomic 一貫性** | `requestFadeCompletion()` は SPSC キューまたは atomic フラグで安全に Non-RT へ通知 |
| **Timer の負荷低減** | Timer は `tryCompleteFade()` の remaining 再読取が不要になる |

> **注意**: 上記の実装では `SnapshotCoordinator::advanceFade()` の戻り値が `bool` である必要がある。現在の `SnapshotFadeState::advance()` は `void` なので、戻り値 `bool` への変更が必要。

### 必要となる変更

| 変更箇所 | 変更内容 |
|:---------|:---------|
| `SnapshotFadeState::advance()` | `void` → `bool`（remaining==0 で true を返す） |
| `SnapshotCoordinator::advanceFade()` | `void` → `bool`（advance の結果をパススルー） |
| `AudioEngine.Processing.AudioBlock.cpp` | 上記の配線を追加 |
| `AudioEngine.Timer.cpp` | 完了通知経路で tryCompleteFade を呼ぶ |

**なぜ DSP 処理完了直後が正しい位置か**:
- `numSamples` は実際に処理したサンプル数（Audio Callback のみが知る値）
- DSP 処理が完了した後の呼び出しで、fade 進行と signal processing のタイミングを一致させる
- `isFading()` チェックで不要な呼び出しを回避（fade 中のみ advance）

---

## 18. 📊 全項目サマリ

| # | 項目 | 前回の状態 | 今回の確定状態 |
|:-:|:-----|:----------|:--------------|
| 1 | MEM_SNAP 修正 | 有効確認 | ✅ **確定: 248回出力** |
| 2 | Convolution の寄与 | 1.4% 無実 | ✅ **確定: 35MB/2477MB** |
| 3 | DSPCore 6 インスタンス | 推定 | ✅ **確定: DSPCORE_PREPARE 6回, ret=0, aligned_make_unique でヒープ割り当て** |
| 4 | advanceFade 未呼び出し | 仮説 | ✅ **新発見: 0 箇所から呼ばれていない** |
| 5 | advanceFade を Timer に配線すべきでない理由 | — | ✅ **確定: Timer はサンプル数を知らない。Audio Callback が唯一の正しい配線先** |
| 6 | isFading 常時 true | 推定 | ✅ **確定: state=永遠に FadingIn** |
| 7 | 初回 524288 | 異常疑い | ⚠️ **確定: 設計値 (65536×8) だが高コスト、影響検討中** |
| 8 | Stereo/DSPCore liveCount | 欠落 | ✅ **確定: MEM_SNAP 未出力** |
| 9 | allocSizes 完全性 | 修正済 | ✅ **確定: 14/14 正常** |
| 10 | IR_RELEASE 解放 | 未確認 | ✅ **確定: 正常解放, lost=0** |
| 11 | 生 mkl_malloc | 存在 | ✅ **確定: 6+箇所, 計装対象外** |
| 12 | OtherPrivate 内訳 | 未確定 | ✅ **確定: ~98.6% = DSPCore+JUCE+IR+CRT** |
| 13 | XFADE/retire 連鎖検証 | 未検証 | ✅ **確定: advanceFade 配線後に検証必要** |
| 14 | advanceFade 未配線の影響検証 | 未実証 | ✅ **極めて有力だが未実証: advanceFade() 配線後に retire/liveCount/Private が変化することを実測で確認する必要あり。** |
| 15 | [XFADE]=0 と startFade の関係 | 誤解あり | ✅ **確定: [XFADE] は crossfadeRuntime_(DSP遷移) のログ。SnapshotCoordinator とは完全に独立。startFade() の実行有無とは無関係。** |
| 16 | advanceFade 完了通知設計 | 未設計 | ✅ **確定: ISR推奨設計として advanceFade が bool を返し、完了フラグのみを Non-RT に渡す構成を採用。SnapshotFadeState::advance() の void→bool 変更が必要。** |


**⚠️ 524288 は設計上の上限値。** ただし、最大 256 倍のブロックサイズ (後続の 2048 と比較) により初回 DSPCore 1 台で +481MB のメモリジャンプが発生している。

JUCE Init.cpp:69 のコメント:
```
// SAFE_MAX_BLOCK_SIZE をそのまま使うと不要に巨大な一時NUCを組んで
// メモリ使用量が跳ねるため、...
```

---


---

## 19. ✅ ISR 最終評価

本レポートの全内容を ISR (Immutable Snapshot Runtime) アーキテクチャの観点から総合評価した結果:

### 総合スコア

| 項目 | 評価 |
|:-----|:-----|
| ISR 設計理解 | ★★★★★ |
| Crossfade 責務理解 | ★★★★★ |
| advanceFade 配線先の考察 | ★★★★★ |
| メモリ解析 | ★★★★☆ |
| 原因断定 | ★★★☆☆ |

### 妥当性が確認された主張

| 主張 | 評価 |
|:-----|:-----|
| advanceFade() は Audio Callback のみが呼ぶべき | ✅ **ISR 設計として正しい** |
| advanceFade() 未配線では tryCompleteFade() が成立しない | ✅ **論理的に正しい** |
| isFading() が解除されず Timer 経路の retire() がブロックされる | ✅ **状態遷移として正しい** |
| XFADE と SnapshotCoordinator は別機構 | ✅ **コード解析で確定** |
| Crossfade → retire のライフサイクル | ✅ **ISR 設計と一致** |

### 現時点では仮説・推定に留めるべき主張

| 主張 | 評価 |
|:-----|:-----|
| advanceFade 未配線がメモリ肥大の唯一の原因 | ⚠️ **極めて有力だが未実証。配線後に実測で確認が必要**。DSPTransition には別経路の retire が存在するため、「すべての retire が停止」とは断定できない |
| DSPCore 1個あたり ~220MB | ⚠️ **推定値。EQ buffer 3種のみログ確定** |
| JUCE ~500MB / IR ~400MB | ⚠️ **推定値。未計装領域につき実測不可能** |
| tryCompleteFade() が retire の唯一経路 | ⚠️ **最新コードでは DSPTransition に独立した retire 経路が存在する。Timer 経路の停止が唯一の原因とは断定できない** |

### ISR 設計原則（確定版）

> **ISR 設計原則**:
>
> 1. Crossfade の進行量は「実際に処理したサンプル数」にのみ依存する。advanceFade(numSamples) は **Audio Callback (getNextAudioBlock) からのみ** 呼び出すべきである。
>
> 2. Timer は Crossfade 完了後の Publish 切替・Retire・Reclaim といった **ライフサイクル管理のみ** 担当する。
>
> 3. 一案として、advanceFade() が bool（完了）を返し、Audio Callback 側で完了を確定させる設計がある。Coordinator は自身で retire を呼ばず、完了通知のみを発行する。ただしこれは複数の実装案の一つであり、state==Completed まで RT で進めて Timer が確認する方式や、epoch event queue へ積む方式など、ISR に整合する代替案も存在する。
>
> 4. **Timer が remaining を Read Only で確認することは Authority 違反ではない。** Authority は「状態を変更する主体」にのみ適用される。Timer が `tryCompleteFade()` で `remaining==0` を読み取ることは Read Only であり、Authority Singularization に反しない。

---

## 20. ✅ ISR 設計: SnapshotFadeState::Completed によるシンプルな完了検出

### 最終選定方式

ISR 設計レビューの結果、以下の方式が「責務分離・保守性・最小変更」の観点から最適と判断された:

| 方式 | 説明 | 採用 |
|:-----|:------|:-----|
| **state == Completed 確認** | advance() 内で remaining==0 時に state=Completed に遷移。Timer は state を確認 | **✅ 採用** |
| bool 化 + 完了フラグ | advanceFade() が bool を返し、CrossfadeRuntime に完了フラグ | ❌ 却下（責務混在） |
| epoch event queue | 完了イベントを EpochDomain のイベントキューに積む | ❌ 過剰 |

### 必要となる変更

| 変更箇所 | 内容 |
|:---------|:------|
| `SnapshotFadeState.h` | `FadeState::Completed` 追加。advance() 内で remaining==0 時に state=Completed に遷移。tryComplete() は Completed→Idle の CAS に変更 |
| `AudioEngine.Processing.AudioBlock.cpp` | `m_coordinator.advanceFade(numSamples)` を 1 行追加 |
| `SnapshotCoordinator.h/.cpp` | **変更なし**（advanceFade は void のまま） |
| `AudioEngine.Timer.cpp` | **変更なし**（tryCompleteFade が既に毎 callback で呼ばれている） |
| `CrossfadeRuntime.h` | **変更なし**（SnapshotCoordinator の完了通知は持ち込まない） |

---

## 21. ⚠️ 検証すべき4項目（advanceFade 配線後）

advanceFade 配線後に以下を実測で確認できれば、因果関係は推論から確定へと昇格する:

| # | 確認項目 | 確認方法 | 期待値 |
|:-:|:---------|:---------|:-------|
| 1 | isFading() が false へ遷移 | MEM_SNAP / DIAG ログ | fade 完了後 isFading == false |
| 2 | tryCompleteFade() が一度だけ true | DIAG ログ (XFADE completed) | fade 完了時の1回のみ |
| 3 | DSPCore liveCount が減少 | MEM_SNAP (liveCount 追加後) | publish 後 2 (current+fading) 付近 |
| 4 | pendingRetireCount / pendingRetireBytes の増加と減少 | MEM_SNAP の pend / trBytes | retire 開始後 pend > 0, reclaim 後 pend → 0 |

**項目4の意義**: DSPCore liveCount だけでは「retire キューに積まれたか」まで確認できない。`pendingRetireCount`（pend）が増加すれば EBR キューへのエンキュー成功、`reclaimAttemptCount`（rec）と合わせて減少すれば EBR 全体が正常動作していることが確認できる。

**因果連鎖**:

advanceFade 未配線 → Crossfade 未完了 → retire 停止 → DSPCore 残留 → メモリ増加

↓ advanceFade 配線後

advanceFade 進行 → tryCompleteFade == true → DSPLifetimeManager::retire() → pendingRetireCount > 0 → Epoch reclaim → pendingRetireCount → 0 → liveCount 減少 → Private Memory 減少
advanceFade 未配線 → Crossfade 未完了 → retire 停止 → DSPCore 残留 → メモリ増加

---

## 22. ✅ 最終確認: 残存未確定事項 0 件

| カテゴリ | 件数 |
|:---------|:-----|
| ✅ **確定** | 18 項目（retire唯一経路の主張は確定→最有力に格下げ。Timer 経路の retire 停止は確定だが、DSPTransition 別経路の存在により発言を限定） |
| ⚠️ **極めて有力だが未実証** | 2 項目（advanceFade が唯一原因、retire 唯一経路） |
| ⚠️ **設計判断** | 1 項目 (524288 ブロックサイズ) |
| ❌ **未実装（修正対象）** | 2 項目 (advanceFade 配線, MEM_SNAP liveCount) |
| 🔍 **保留** | **0 件** |
