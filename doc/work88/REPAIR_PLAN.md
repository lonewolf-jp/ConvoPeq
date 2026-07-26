# ConvoPeq 改修設計書 — BUG-011〜BUG-046 修正計画

> 3部構成。第一部「設計」のみ読めば実装可能。
> 本稿は4次レビュー（2026-07-26）までの全指摘を反映した最終版。

---

# 第一部: 設計

## 1. 改修方針

### 前提とする設計原則

| 原則 | 内容 |
|------|------|
| **Authority Singularization** | 各責務の Authority は一意。Timer は観測者 |
| **責務分離** | SnapshotFade と Crossfade は別機構 |
| **RT スレッドは観測者** | Audio Thread は観測するが判断しない |
| **Fail Closed** | エラー時はゼロ出力/安全側 |
| **CAS-based synchronization** | スロット排他には CAS を用い、追加フラグを増やさない |

### 実施グループ一覧

| グループ | 性質 | 件数 | 工数目安 |
|----------|------|------|----------|
| **A** | 即時実施可能 | 13件 | 2〜4時間 |
| **B** | 設計確定済み | 6件 | 1〜2日（B-2は要追加検証） |
| **C** | 計画的対応 | 7件 | 2〜4時間 |
| **D** | 余裕時 | 4件 | 1〜2時間 |

---

## 2. グループA: 即時実施可能（13件）

### A-1: BUG-038 — SpectrumAnalyzer FFT スケーリング誤差

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/SpectrumAnalyzerComponent.h:74` |
| **修正** | 定数値変更（1行） |

```cpp
static constexpr float FFT_MAGNITUDE_SCALE = 2.0f / NUM_FFT_POINTS;
```

---

### A-2: BUG-035 — applyComputedIR isLoading 固着 → RAII LoadingGuard

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/convolver/ConvolverProcessor.LoadPipeline.cpp` |
| **修正** | RAII LoadingGuard 導入 |
| **注意** | `isLoading` は複数箇所で管理されている。本修正は `applyComputedIR()` 内のスコープに限定する。 |

```cpp
// ★ applyComputedIR() 専用。isLoading には複数の書き込み元があり、
//   この Guard は applyComputedIR スコープ内の isLoading 管理のみを責務とする。
//   汎用 LoadingGuard と誤解されないよう用途限定名＋ final で継承禁止。
//   isLoading は UI 表示用の状態フラグであり（Runtime の State Machine ではない）、
//   release/acquire による可視性保証で十分（seq_cst 不要）。
class ApplyComputedIRLoadingGuard final {
    std::atomic<bool>& isLoading_;
public:
    explicit ApplyComputedIRLoadingGuard(std::atomic<bool>& flag) noexcept
        : isLoading_(flag)
    {
        // release: UI/スレッドが isLoading の true を観測可能にする。
        // isLoading は UI 表示用の状態フラグであり、メモリ順序保証は
        // release/acquire で十分（seq_cst 不要）。
        convo::publishAtomic(isLoading_, true, std::memory_order_release);
    }
    ~ApplyComputedIRLoadingGuard() noexcept
    {
        // release: UI にローディング完了を通知。
        convo::publishAtomic(isLoading_, false, std::memory_order_release);
    }
    ApplyComputedIRLoadingGuard(const ApplyComputedIRLoadingGuard&) = delete;
    ApplyComputedIRLoadingGuard& operator=(const ApplyComputedIRLoadingGuard&) = delete;
};
```

```cpp
void ConvolverProcessor::applyComputedIR(std::unique_ptr<ConvolverIRPayload> prepared)
{
    // ★ null/generation mismatch は Loading 開始前 → Guard 不要
    if (!prepared) { /* log */ return; }
    if (!convolverStateGeneration.isCurrentGeneration(prepared->generationId)) { /* log */ return; }

    // ★ ここから Loading 開始
    //   IMPORTANT: applyComputedIR() 内では isLoading を直接変更しないこと。
    //   isLoading の true/false はこの Guard のスコープのみが責務を持つ。
    //   Guard より前の return 経路（null、generation mismatch）では
    //   isLoading は変化しない。
    ApplyComputedIRLoadingGuard guard(isLoading);
    // ... 以降 return してもデストラクタが isLoading=false を保証 ...
}
```

**確認**: `isLoading` は他に `LoadPipeline.cpp:46,73,443,533,541,705,778` や
`LoaderThread.cpp:58,71` でも書かれているが、すべて別関数・別スコープであり、
本修正の LoadingGuard と競合しない。

**補足**: `finalizeNUCEngineOnMessageThread()` 内の `isLoading=false`（`LoadPipeline.cpp:533,541`）は
LoadingGuard のデストラクタと重複するが、冪等な保険として問題ない。

---

### A-3: BUG-036 — init() 失敗時に irL/irR がリーク

```cpp
double* irLRaw = irL.get();
double* irRRaw = irR.get();
if (newConv->init(irLRaw, irRRaw, length, sr, peakDelay, ...))
{
    irL.release();
    irR.release();
    // ... 正常時処理 ...
}
```

---

### A-4: BUG-034 — IPP FFT 戻り値未チェック（6箇所）

| 項目 | 内容 |
|------|------|
| **重要度** | CRITICAL |
| **ファイル** | `src/MKLNonUniformConvolver.cpp`（6箇所） |
| **修正** | `clearFFTOutputOnError()` ヘルパー導入＋全6箇所に適用 |
| **対象外** | `MklFftEvaluator.h` 内の4箇所は BUG-044（Rule of Five）の対象。戻り値チェックは別途 |

```cpp
// ★ FFT エラー時に出力バッファをゼロクリア（Fail Closed）。
//   エラーを検出したことよりゼロ出力で続行することを優先する。
//   ippStatus と stage は RuntimeHealthReporter 接続時に使用するため
//   将来の拡張に備えて引数で受け取っておく（現状は unused）。
static void clearFFTOutputOnError(double* buffer, size_t count,
                                  [[maybe_unused]] IppStatus status = ippStsNoErr,
                                  [[maybe_unused]] FFTStage stage = FFTStage::Unknown) noexcept
{
    if (buffer != nullptr)
        std::memset(buffer, 0, count * sizeof(double));
    // TODO(RuntimeHealth): Record FFT failure information (IppStatus + FFTStage).
    //   Zero-clear alone loses error context. Future RuntimeHealthReporter should
    //   collect these as diagnosable metrics (fail_count per stage).
}
```

**重要: CCS バッファサイズの注意**

IPP FFT の入出力形式により、ゼロクリアサイズが異なる（ソースコード確認済み: `MKLNonUniformConvolver.cpp:859`）

| # | 行 | 種類 | 用途 | ゼロクリア対象 | サイズ（double単位） |
|---|-----|------|------|--------------|---------------------|
| 1 | 1043 | Fwd | IR 周波数変換 | `tempFreq`（CCS出力） | `l.complexSize * 2` |
| 2 | 1060 | Inv | IR 反転 | `tempTime`（実出力） | `l.fftSize` |
| 3 | 1376 | Fwd | オーディオ処理 FDL | `currentFDLSlot`（CCS） | `l.complexSize * 2` |
| 4 | 1436 | Inv | オーディオ処理 IFFT | `l.fftOutBuf`（実出力） | `l.fftSize` |
| 5 | 1570 | Fwd | オーディオ処理 FDL2 | `currentFDLSlot`（CCS） | `l.complexSize * 2` |
| 6 | 1637 | Inv | オーディオ処理 IFFT2 | `l.fftOutBuf`（実出力） | `l.fftSize` |

**補足**: `l.complexSize = l.fftSize / 2 + 1`（CCS 出力は `[re0,im0,re1,im1,...]` 形式）。

**異常系テスト**: IPP の `pFFTSpec` に `nullptr` を渡す方法は推奨しない（IPP が nullptr を許容する保証がない。アクセス違反の可能性）。代わりに以下を推奨:
- FFT wrapper 関数を導入し、単体テストではモックに差し替え
- またはコードレビューで戻り値チェックの網羅性を確認

---

### A-5: BUG-011/012/013 — CMA-ES sigma クランプ欠如（3箇所）

```cpp
// A-5a: src/CmaEsOptimizer.h:79
sigma = std::clamp(inSigma, params.sigmaMin, params.sigmaMax);

// A-5b: src/CmaEsOptimizerDynamic.h:29
void setSigma(double s) noexcept { sigma = std::clamp(s, params.sigmaMin, params.sigmaMax); }

// A-5c: src/CmaEsOptimizerDynamic.cpp:204
sigma = std::clamp(inSigma, params.sigmaMin, params.sigmaMax);
```

---

### A-6: BUG-029 — DSPTransition Emergency Override exchangeFadingRuntimeDSP 欠落

```cpp
if (health == convo::ISRHealthState::Critical) {
    lifetime.activate(newDSP);
    if (oldDSP != nullptr) {
        // ★ TODO(BUG-030): Temporary Compatibility Authority.
        //   Remove after B-1 CAS-only claimFadingRuntimeDSP() implementation.
        //   Temporary Compatibility Authority — DSPTransition transiently manages
        //   the fading slot only because B-1 CAS-only has not yet been implemented.
        //   Does NOT constitute new permanent ownership authority.
        //   ★ ASSERT: After B-1 implementation, exchangeFadingRuntimeDSP()
        //     must be completely removed — no code path may fall back to it.
        //
        //   Execution sequence (temporary slot mgmt → state cleanup → lifetime):
        //   1. exchangeFadingRuntimeDSP()     — temporary slot management
        //   2. CrossfadeRuntime::complete()  — crossfade state cleanup
        //   3. lifetime.retire()              — DSP lifetime release
        auto* prevRaw = engine_.exchangeFadingRuntimeDSP(oldDSP);
        engine_.crossfadeRuntime_.complete();
        lifetime.retire(oldDSP);
        // ★ recoverDSPPtr 後に oldDSP と比較して二重 retire 防止
        if (auto* prev = recoverDSPPtr(prevRaw)) {
            if (prev != oldDSP) lifetime.retire(prev);
        }
    }
    return;
}
```

---

### A-7: BUG-028 — CrossfadeRuntime::complete() フラグリセット

```cpp
// complete: 通常のクロスフェード完了時のみ使用。
//   pending_ を false に戻し、stale フラグをクリアする。完了した fade は不要なため
//   queuedFadeTimeSec_/fadeStartTimestampUs_ は初期値へ戻す（= resetToInitialState）。
//   ★ queuedFadeTimeSec_ は「現在の fade 用 Queued 時間」であり、次回 fade の予定ではない。
//     そのため complete() で初期値に戻すのは正しい。
//   シャットダウン時は reset() を使用（complete より広範囲のフィールドをリセット）。
void complete() noexcept
{
    convo::publishAtomic(pending_, false, std::memory_order_release);
    convo::publishAtomic(useDryAsOld_, false, std::memory_order_release);
    convo::publishAtomic(firstIrDryPending_, false, std::memory_order_release);
    convo::publishAtomic(firstIrDryDone_, false, std::memory_order_release);
    static constexpr double kDefaultFadeTimeSec = 0.030;
    convo::publishAtomic(queuedFadeTimeSec_, kDefaultFadeTimeSec, std::memory_order_release);
    convo::publishAtomic(fadeStartTimestampUs_, 0, std::memory_order_release);
}
```

---

### A-8: BUG-015 — enqueueWithRetry 戻り値無視（3箇所）

```cpp
auto result = enqueueWithRetry(...);
if (result != RetireEnqueueResult::Success) { /* ★ Future: HealthMonitor */ }
```

---

### A-9: BUG-016 — CmaEsOptimizer sanitize NaN/Inf

```cpp
return (!std::isfinite(x) || std::abs(x) < 1e-15) ? 0.0 : x;
```

---

### A-10: BUG-042/044/046 — Rule of Five（3件）

各クラスに `= delete` / `= default`。詳細は初版参照。

---

### A-11: BUG-045 — IRConverter resample failure mislabels sample rate

```cpp
// ★ r8brain resample 失敗: IR データは sourceRate のまま → actualSampleRate も sourceRate。
converted = ir;
actualSampleRate = sourceRate;  // データ実態と一致させる
```

---

### A-12: BUG-039 — Oversampler バッファ過剰読み取り

```cpp
const size_t copySamples = std::min(static_cast<size_t>(targetSamples),
                                    static_cast<size_t>(upsampledBlock.getNumSamples()));
std::memcpy(dst, src, copySamples * sizeof(double));
```

---

### A-13: BUG-040 — NoiseShaperLearner 1 Hz フォールバック

```cpp
: ((block.sampleRateHz > 0) ? block.sampleRateHz : 48000);
```

---

## 3. グループB: 設計確定済み（6件）

### B-1: BUG-030 — Timer vs DSPTransition fading slot 競合（CAS-only）

| 項目 | 内容 |
|------|------|
| **重要度** | HIGH |
| **ファイル** | `src/audioengine/DSPTransition.h`, `AudioEngine.Timer.cpp`, `AudioEngine.h` |
| **設計** | **CAS-only** — `claimFadingRuntimeDSP` のみで完結。exchange() 不要 |

**設計根拠**: `CAS success → exchange()` の二段階は非原子的（CAS と exchange の間に Timer が介入する隙がある）。
CAS 成功時点でスロット所有権が確定するため、`compare_exchange(nullptr, oldDSP)` のみで十分。

```cpp
// === AudioEngine.h に追加（または exchangeFadingRuntimeDSP と併設） ===
// ★ claimFadingRuntimeDSP: fading slot の所有権を取得する。
//   expected は内部で nullptr に設定される（呼び出し側は意識不要）。
//   CAS 成功時、slot = desired となり、呼び出し元が唯一の fading slot 所有者となる。
//   これは DSP オブジェクト全体の所有権ではなく、fadingRuntimeDSPSlot に対する所有権。
inline bool claimFadingRuntimeDSP(DSPCore* desired) noexcept
{
    DSPCore* expected = nullptr;
    return convo::compareExchangeAtomic(fadingRuntimeDSPSlot, expected, desired,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire);
}

// === DSPTransition.h : onPublishCompleted() 通常パス ===
{
    // claimFadingRuntimeDSP(expected=nullptr は内部で生成)
    if (engine_.claimFadingRuntimeDSP(oldDSP))
    {
        // ★ CAS 成功 = この DSPTransition が fading slot の唯一の所有者。
        //   exchange() は不要（CAS が直接 slot = oldDSP を設定）。
        //   この後 crossfade 開始（Timer の介入不可）。
    }
    else
    {
        lifetime.retire(oldDSP);
    }
    // ... crossfade start ...
}

// === AudioEngine.Timer.cpp : Timer 側 ===
if (!m_coordinator.isFading())
{
    // ★ current の安全性（ライフタイム順序保証）:
    //   current は fadingRuntimeDSPSlot から acquire で読み込んだ値。
    //   CAS 成功時にのみ retire される（CAS 成功 = Timer が slot の唯一の観測者）。
    //
    //   ★ 設計目標（ISR 上の理想 retirement 条件）:
    //     1. DSPTransition::onPublishCompleted() → complete() 呼び出し
    //     2. complete() は pending_=false を publish
    //     3. ObservedRuntime / EpochDomain により oldDSP が全 Reader から参照不能と証明される
    //        （pending_=false 観測 + 該当 epoch が全 Reader の最小 epoch より過去になることを確認）
    //     4. Timer tick で上記条件確認 → CAS(nullptr) → retire(current)
    //
    //   ★ 現状実装:
    //     現在は isFading()==false のみ確認している。
    //     EpochDomain による safe-to-retire 証明は未実装。
    //
    //   ★ 重要: Epoch 条件の追加だけでは解決しない。
    //     retirementEpoch を retire path に運ぶ「Publication Metadata Propagation to Retire Path」
    //     が設計されていない。问题是「Authority の不足」ではなく「Retire Path まで伝わらない」。
    //     詳細らは別 Issue「Publication Metadata Propagation to Retire Path」を参照。
    DSPCore* current = convo::consumeAtomic(fadingRuntimeDSPSlot, std::memory_order_acquire);
    if (current != nullptr
        && convo::compareExchangeAtomic(fadingRuntimeDSPSlot, current, nullptr,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
    {
        // CAS 成功 → slot = nullptr。current を retire。
        // ★ 注意: 現時点では publicationEpoch を retire に渡す手段がない。
        //   別 Issue「Publication Metadata Propagation to Retire Path」参照。
        DSPLifetimeManager lifetimeMgr(*this);
        lifetimeMgr.retire(current);
    }
    // CAS 失敗 → DSPTransition が既に設定済み。何もしない。
}
```

---

### B-2: BUG-023 — SafeStateSwapper swap() vs tryReclaim() 競合

| 項目 | 内容 |
|------|------|
| **重要度** | HIGH |
| **ファイル** | `src/SafeStateSwapper.h` |
| **ステータス** | ⚠️ **未確定 — 実装前に追加検証必要** |

#### 検討案（未採用）

```cpp
void swap(ConvolverState* newState) noexcept
{
    // ... 2-step bump, exchangeAtomic activeState ...

    size_t t = convo::consumeAtomic(tail, std::memory_order_acquire);
    size_t next = (t + 1) % kMaxRetired;

    for (;;)
    {
        if (next == convo::consumeAtomic(head, std::memory_order_acquire))
        {
            std::lock_guard<std::mutex> lock(fallbackMutex);
            fallbackQueue.push({oldState, epoch2});
            return;
        }

        // ★ 書き込み順序: state → epoch → tail(CAS)
        convo::publishAtomic(retiredBuffer[t].state, oldState, std::memory_order_release);
        convo::publishAtomic(retiredBuffer[t].epoch, epoch2, std::memory_order_release);

        if (convo::compareExchangeAtomic(tail, t, next,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire))
            break;

        // CAS 失敗: 別スレッドが tail を変更。retry。
        t = convo::consumeAtomic(tail, std::memory_order_acquire);
        next = (t + 1) % kMaxRetired;
    }
}
```

#### 未確定理由

現在の SafeStateSwapper の ring buffer では `tail` が以下の3つの責務を兼任している:

1. **Queue position**: 次に書き込むべき slot の識领
2. **Reservation**: slot の確保状態
3. **Commit**: slot の 내용이確定し Reader に公開されていることを示す

このため、`swap()` での `state/epoch 書込み → tail CAS` という順序において、CAS 失敗時に「未確定の slot に有効な payload だけが残留する」という状態が発生する可能性がある。

**本質は CAS 順序問題ではなく Queue Protocol の未定義**。
`tail` の责務分担を再設計しないと、CAS 成功後に payload を書込み始める逆の競合も 발생하는。

**現段階での記載**: 現行 Queue Protocol の安全性を検証し、必要であれば reservation / payload visibility / commit の责務分離を含めて再設計する。

**three-phase protocol (reserved/write/commit) を採用案としては書かない**。有力な設計候補の一つであり結論ではない。

**実装前に確認すべき事項**:
- `getState()` が `activeState` のみを読み `retiredBuffer` を直接読まないこと（✅ 確認済み）
- `tryReclaim()` の Single Consumer 前提が現状の呼び出しパターンで成立すること
- 現行プロトコルにおいて `tail` が slot validity の唯一の公開情報源となっていることの証明
- `retiredBuffer` への直接アクセス経路が他にないこと（✅ `grep -rn "retiredBuffer" src/` で SafeStateSwapper.h 内のみ確認済み）

---

### B-3: BUG-031 — updateAudioThreadSnapshotFade スタブ

```cpp
inline bool updateAudioThreadSnapshotFade(int numSamples, float& snapshotAlpha,
    const convo::GlobalSnapshot*& snapshotFrom, const convo::GlobalSnapshot*& snapshotTo) noexcept
{
    // ★ advanceFade はこの関数内でのみ呼ぶ。BlockDouble.cpp 等からは呼ばない。
    m_coordinator.advanceFade(numSamples);
    return m_coordinator.updateFade(snapshotAlpha, snapshotFrom, snapshotTo);
}
```

**確認状況**:

- `updateAudioThreadSnapshotFade()` helper は定義されているが、**どこからも呼ばれていない**（grep 確認済み）
- `updateFade()` の唯一の呼び出し経路がこの helper のみ
- `snapshotAlpha` / `snapshotFrom` / `snapshotTo` は DSP 処理パスで参照されていない

**現時点で確認できる事実**: `updateFade()` が Audio Thread から未呼び出しであること。

**現時点で断定できないこと**: 「Snapshot Fade が未動作」。`CrossfadeRuntime` や `RuntimeProjection` など DSP 側が同等情報を別経路で取得している可能性は未解析。DSP 側全経路の解析後に評価する。

---

### B-4: BUG-032 — createSnapshotFromCurrentState torn-read

```cpp
// 修正後: GlobalSnapshot から一括取得（publish 時一貫性保証済み）
const auto* currentSnap = m_coordinator.getCurrentSnapshot();
if (currentSnap) { params = currentSnap->getParams(); }
```

**注意**: `getCurrentSnapshot()` のインターフェースは要確認（U-1）。

---

### B-5: BUG-024 — SnapshotFadeState advance() vs resetToIdle()

```cpp
void advance(int numSamples) noexcept
{
    if (state() != FadeState::FadingIn) return;
    const int remaining = remainingCount();
    if (remaining <= 0) return;

    const int newRemaining = remaining - numSamples;
    if (newRemaining <= 0) {
        convo::publishAtomic(remainingSamples_, 0, std::memory_order_release);
        return;
    }

    convo::publishAtomic(remainingSamples_, newRemaining, std::memory_order_release);
    if (state() != FadeState::FadingIn) return;  // ← resetToIdle 競合対策
    // ★ alpha update intentionally skipped after resetToIdle:
    //   state が Idle に戻った場合、alpha は resetToIdle() が 1.0 に設定済み。
    //   ここで alpha を上書きすると矛盾が生じるためスキップする。
    //
    // ★ fadeGeneration 導入必須（ISR 設計上の ABA 問題対策）。
    //   現在の state 再確認だけでは「remaining更新→reset→startFade」の
    //   シーケンスで新しい fade の remaining を古い alpha 計算に使う
    //   ABA 問題がある。
    //   fadeGeneration カウンタ（std::atomic<uint64_t>）を追加し、
    //   startFade() でインクリメント、advance() で取得した generation と
    //   現在値を比較して不一致なら alpha 計算をスキップすることで
    //   完全に解決する。
    //   実装手順:
    //   1. SnapshotFadeState に std::atomic<uint64_t> fadeGeneration_{0} 追加
    //   2. start() 内の更新順序（重要）:
    //      a) totalSamples_ = fadeSamples を先に publish (release)
    //      b) remainingSamples_ = fadeSamples を publish (release)
    //      c) alpha_ = 0.0 を publish (release)
    //      d) state_ = FadingIn を publish (release)
    //      e) 最後に convo::publishAtomic(fadeGeneration_, ++gen, release) ← ABA generation は最後
    //
    //      generation を最後にすることで、advance() が genAtStart を読んだ時点で
    //      total/remaining/alpha/state は全て commit 済みであることが保証される。
    //
    //   3. advance() で以下のように generation 比較:
    //
    //      // advance() 開始時の ABA generation を保存
    //      const uint64_t genAtStart = convo::consumeAtomic(fadeGeneration_,
    //                                          std::memory_order_acquire);
    //      // ... remaining 更新、state 再確認 ...
    //
    //      // alpha 計算直前に generation 再確認（startFade で increment されたら不一致）
    //      if (genAtStart != convo::consumeAtomic(fadeGeneration_,
    //                                     std::memory_order_acquire))
    //          return;  // 新しい fade が開始された → 古い remaining で alpha 計算しない
    //
    //   ★ generation は ABA detector（Seqlock ではない）:
    //     fadeGeneration は単純な ABA detector。Seqlock（odd→payload→even）ではなく、
    //     書き込み中を示す odd/even 状態は持たない。
    //     Writer が startFade() で increment し、Reader が advance() で前後比較する。
    //   Publish 対象は total/remaining/alpha/state。generation は最後に更新する。
    //   generation を最後に publish する理由:
    //   - generation を先に publish すると「新 generation, 旧 payload」を読む可能性がある
    //   - generation は単なる世代番号であり、payload の atomic 性は保証しない
    //
    //   ★ Reader 側も ABA generation 確認:
    //     advance() では generation 確認を以下の順序で行う:
    //       1. start 時: genAtStart = load(fadeGeneration_, acquire)  — ABA generation 取得
    //       2. payload 読取: remaining, state 等を acquire で読む
    //       3. alpha 計算前: load(fadeGeneration_, acquire) が genAtStart と一致するか確認
    //          → 不一致なら「startFade が別世代を開始した」ため alpha 計算をスキップ
    //     これにより Reader は「同一 generation の payload」のみを参照することが保証される。
    //
    //   ★ Reader の読取り順序は固定: generation → payload → generation の順に読むこと。
    //     payload より先に generation を取得し、alpha 計算直前に再確認する。
    //     この順序を変更すると、別 generation の payload を誤って参照する可能性がある。
    //
    //   ★ fadeGeneration は ABA 検出器であり、将来の状態遷移拡張に対する万能な整合性保証ではない。
    //     現状の FadeState は Idle/FadingIn の2値だが、今後増える場合は generation 単独では
    //     不十分な可能性がある。その場合、状態遷移ごとに generation をインクリメントするか、
    //     別途 FadeState 単位の Version 管理が必要になる。
    //
    const int total = totalCount();
    if (total > 0) {
        const double nextAlpha = 1.0 - static_cast<double>(newRemaining) / static_cast<double>(total);
        convo::publishAtomic(alpha_, nextAlpha, std::memory_order_release);
    }
}
```

---

### B-6: BUG-037 — loaderTrashBin UAF 防止（Generation）

```cpp
// ConvolverProcessor.h
std::atomic<uint64_t> loaderGeneration_{0};

// デストラクタ先頭（fetch_add で原子的に increment）
// ★ Generation は Owner validity のみ（ConvolverProcessor の有効性）。
//   スレッドの停止保証や join 完了は別責務（LoaderThread::stopThread 等）。
//   Generation が変わっても Loader が即座に停止するとは限らないため、
//   デストラクタでは stopThread(-1) によるブロッキング待機と併用する。
//   Generation は「この owner はもう使えない」という通知であり、
//   スレッド停止の完了は join が保証する。
convo::fetchAddAtomic(loaderGeneration_, 1, std::memory_order_acq_rel);

// LoaderThread::run()
void run() override {
    const uint64_t myGeneration = owner_.loaderGeneration_.load(std::memory_order_acquire);
    while (!threadShouldExit()) {
        if (owner_.loaderGeneration_.load(std::memory_order_acquire) != myGeneration) break;
        // ... 通常処理 ...
    }
}
```

---

## 4. グループC: 計画的対応（7件）

| ID | バグ | ファイル | 修正内容 | 工数 |
|----|------|----------|----------|------|
| C-1 | BUG-033 | `BlockDouble.cpp:400-427` | `dryScale` ラムダキャプチャ追加 | 30分 |
| C-2 | BUG-025 | `SnapshotCoordinator.cpp:57-72` | enqueueWithRetry 化 | 30分 |
| C-3 | BUG-018 | 3ファイル | `!=1.0` → `std::abs(x-1.0)>1e-12` | 15分 |
| C-4 | BUG-019 | `TruePeakDetector.cpp:102-111` | `int` → `size_t` | 15分 |
| C-5 | BUG-020 | `LoaderThread.cpp:198` | `if(targetLength<=0)return 0;` | 5分 |
| C-6 | BUG-021/022 | `Lifecycle.cpp` | RCU GlobalGuard 追加(2箇所) | 20分 |
| C-7 | BUG-026 | `ObservedRuntime.h:42-49` | `rootEnterSucceeded()`確認 | 10分 |

---

## 5. グループD: 余裕時（4件）

| ID | バグ | ファイル | 修正 | 工数 |
|----|------|----------|------|------|
| D-1 | BUG-041 | `NoiseShaperLearner.cpp:643` | VLA→ヒープ | 15分 |
| D-2 | BUG-043 | `IRConverter` | パラメータ名修正 | 10分 |
| D-3 | BUG-027 | `SnapshotCoordinator` | target==null時 state再確認 | 30分 |
| D-4 | BUG-046 | `PsychoacousticDither.h` | A-10に含む | 0分 |

---

## 6. テスト戦略

| ID | テスト方法 | 確認内容 |
|----|-----------|----------|
| A-1 | 目視 | 0 dBFS 正弦波が 0 dBFS 表示 |
| A-2 | UI操作 | 世代不一致後も Loading スピナーが消える |
| A-3 | メモリ | init() 失敗後のメモリ増加なし |
| A-4 | コードレビュー | clearFFTOutputOnError 適用確認（FFT wrapper モック化推奨） |
| A-5 | 単体 | sigma=0, negative, >sigmaMax でクランプ |
| A-6 | 異常系 | HealthState::Critical 時の遷移 |
| A-7 | 単体 | complete() 後フラグ全リセット確認 |
| B-1 | ストレス | Crossfade ownership stress test: IR Publish → Emergency Override → Crossfade → Timer → Shutdown → Restart をランダム順で 10000 回繰り返し、UAF・リークなし。TSAN + ランダム Sleep 注入 + CAS 成功直後の Timer 強制介入 + CAS 失敗高頻度発生 + Forced Epoch Delay（publish→50ms停止→Timer→Audio→Retire ランダム挿入）+ Epoch 遅延と Crossfade 終了のランダム組合せ + Publisher 再介入（CAS成功→complete→Timer→Publisher→Timer）を含む。
| B-2 | — | **要追加検証** |
| B-3 | 結合 | advanceFade 二重進行なし確認 |
| B-4 | 単体 | GlobalSnapshot 経由パラメータ一貫性 |
| B-5 | 単体 | advance+resetToIdle 競合確認 |
| B-6 | クラッシュ | プロセッサ破棄＋ローダー動作中安全 |

### ビルド計画

```
グループA: cmake --build build --config Debug && ctest -C Debug --output-on-failure
グループB: 同上 + Release ビルド（B-2 は検証後）
```

---

## 7. 推奨マージ戦略

| # | 項目 | ブランチ | 注意 |
|---|------|----------|------|
| 1 | A-1〜A-4 | `fix/phase1-critical` | 独立。A-4 は6箇所 |
| 2 | A-5〜A-9 | `fix/phase2-high` | 独立 |
| 3 | A-10〜A-13 | `fix/phase3-medium` | A-11 コメント同時修正 |
| 4 | B-1, B-3, B-4, B-5, B-6 | `fix/phase4-b` | B-1 CAS-only 確認 |
| 5 | B-2 | `fix/phase5-b2` | **要追加検証後** |
| 6 | C-1〜C-7 | `fix/phase6-c` | 計画的 |
| 7 | D-1〜D-4 | `fix/phase7-d` | 余裕時 |

**重要**: A-6 と B-1 は同一 `DSPTransition.h`。**A-6 を先にマージ**し、B-1 は A-6 をベースに。

---

# 第二部: 未確定・未決定事項

## U-1: `SnapshotCoordinator::getCurrentSnapshot()` インターフェース

B-4 で使用。`src/core/SnapshotCoordinator.h` の public メソッド一覧を確認。
存在しない場合は `m_slots.loadCurrent(std::memory_order_acquire)` を使用。

## U-2: B-3 updateAudioThreadSnapshotFade() の利用状況

B-3 の helper (`updateAudioThreadSnapshotFade`) は **未使用**（grep 確認済み）。
`updateFade()` の唯一の呼び出し経路がこの helper のみ。
`snapshotAlpha` / `snapshotFrom` / `snapshotTo` が DSP 処理で参照されていないことも確認済み。

**現時点で確認できること**: `updateFade()` が Audio Thread から未呼び出し。

**現時点で断定できないこと**: 「Snapshot Fade が未動作」—— `CrossfadeRuntime` や `RuntimeProjection` など DSP 側が同等情報を別経路で取得している可能性は未解析。DSP 側全経路の解析後に評価する。

## U-3: B-2 SafeStateSwapper Queue Protocol の安全性証明（未確定）

| 項目 | 内容 |
|------|------|
| **ステータス** | ❌ **未確定** — 実装前に追加検証が必要 |
| **本質** | CAS 順序問題ではなく Queue Protocol の未定義。`tail` が reservation / visibility / commit を兼任している |



**既に確認済み**:
- ✅ `retiredBuffer` へのアクセスは `SafeStateSwapper.h` 内のみ（grep 確認）
- ✅ `getState()` は `activeState` のみ読み、`retiredBuffer` を直接読まない
- ✅ `tryReclaim()` は Single Consumer 前提（コードコメントに明記）

**問題の本質**:
- `tail` が queue position / reservation / commit の3責務を兼任
- `write state/epoch → CAS tail` の順序では CAS 失敗時に未確定slotにpayload残留
- `CAS tail → write state/epoch` に変えると逆の競合（Reader 到達時にpayload未書込み）

**対応**: SafeStateSwapper の ring buffer protocol 全体を解析し、reservation / payload visibility / commit の責務分離が必要か検証する。three-phase protocol は有力な設計候補の一つだが、結論ではない。

## U-4: Publication Metadata Propagation to Retire Path

| 項目 | 内容 |
|------|------|
| **ステータス** | ❌ **未解決** — 設計前に追加検討が必要 |
| **本質** | Publication Metadata を Retire Path まで伝搬する設計が未確定 |

**問題**:

```
RuntimePublishWorld (Authority あり)
    ↓ publicationEpoch 生成
    ↓
    ... (伝搬経路が未設計)
    ↓
RetireQueue (epoch 到着なし)
```

**欲しい情報**:
- `fadingRuntimeDSPSlot` から DSPCore* は取得できる
- しかし DSPCore* から publicationEpoch を逆引きする手段がない
- `DSPCore` 自体には `retirementEpoch` を持たせない（Authority 分散になる）

**設計方向**（現段階では候補）:
- `PublishResult{ptr, publicationEpoch}` を Publish から Retire まで保持する
- `RuntimeHandle` ごと RetireRequest に渡す
- Lifetime Manager に epoch 取得責務を赋予する

**現時点で決めたくないこと**:
- `uint64_t publicationEpoch` 固定ではない。将来 `RuntimeGeneration` などの追加字段の可能性あり
- `DSPCore` への `retirementEpoch` 追加は禁止

**対応**: 別 Issue として Publication Metadata の Lifetime 設計を検討する。

## U-5: B-6 Generation インクリメントタイミング

デストラクタ先頭を推奨。`shutdown()` での Generation 更新も理論上は可能だが、
Loader の join 完了まで保証できる設計が必要。最も安全なのはデストラクタ先頭。

## U-6: A-4 FFT 異常系テスト方法

`fftSpec = nullptr` は **非推奨**（IPP が nullptr を許容する保証がない）。
代わりに FFT wrapper 関数を導入し、モック差し替え可能にすることを推奨。
またはコードレビューでの網羅性確認に留める。

---

# 第三部: Appendix

## A. レビュー履歴

| 版 | レビュー | 主な変更 |
|----|---------|----------|
| v1 | — | 初版。Phase 1〜4 分類 |
| v2 | 1次 | グループA/B/C/D 再分類。W-6/W-8/W-10 設計変更 |
| v3 | 1次 | 全調査確定。3部構成。B全6件設計確定 |
| v4 | 2次 | B-1: acquiringFadingSlot→CAS+exchange。B-2: publish順序。他 |
| v5 | 3次 | **B-1: CAS+exchange→CAS-only。A-4: CCS/FFT サイズ差異明記。B-2: 安全性証明追記** |
| **v6** | **4次** | **A-4: 「7箇所」→「6箇所」に修正（MKLNonUniformConvolver.cpp のみ）。異常系テスト方法を変更（nullptr 非推奨→モック推奨）。A-2: isLoading 全使用箇所を確認し競合なしと明記。B-2: 「未確定」に格下げ、残留期間問題を U-3 で詳細化。** |

## B. 4次レビューでの主要修正点

### 1. A-4 FFT件数: 「7箇所」→「6箇所」

原因: 元の BUG-034 レポートが「7箇所」と記載していたが、ソースコード実査の結果、
`MKLNonUniformConvolver.cpp` 内の IPP FFT 呼び出しは **6箇所** であることを確認。
（`MklFftEvaluator.h` 内の4箇所は BUG-044 の対象。）

### 2. A-4 異常系テスト: nullptr 非推奨→モック推奨

`fftSpec = nullptr` を IPP に渡すテストはアクセス違反のリスクがあるため非推奨。
代わりに FFT wrapper のモック化またはコードレビューを推奨。

### 3. A-2 isLoading 唯一性確認

`isLoading` の全書き込み箇所（10箇所）を grep 確認。すべて別関数・別スコープであり、
本修正の `applyComputedIR()` 内 LoadingGuard と競合しないことを確認。

### 4. B-2 「未確定」に格下げ

CAS 失敗時の `retiredBuffer[t]` 部分書き込み残留問題が解決していないため、
「理論上安全」ではなく「未確定」とした。U-3 に詳細を追記。

## C. 調査で使用したツール

| ツール | 用途 |
|--------|------|
| WSL grep | `ippsFFTFwd_RToCCS_64f` 全呼び出し箇所の検索（A-4 件数確定） |
| WSL grep | `isLoading` 全使用箇所の検索（A-2 競合確認） |
| WSL grep | `retiredBuffer` 全アクセス経路の検索（B-2 直接アクセス有無確認） |
| context-mode MCP | 並列コマンド実行による効率的な情報収集 |
| ソースコード実査 | `SafeStateSwapper.h` enterReader/getState (retiredBuffer非アクセス確認) |

---

*本設計書は ISR Runtime OS 設計原則に基づく。B-2/U-3 は実装前に追加検証を必須とする。*
