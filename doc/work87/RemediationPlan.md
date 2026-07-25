# ConvoPeq バグ改修計画書

> 作成日: 2026-07-25 | 最終改訂: 2026-07-25
> 元文書: `doc/work87/ConvoPeq_BugReport.md`
> 検証文書: `doc/work87/BugReport_ValidationReport.md`
> レビュー基準: `doc/Practical Stable ISR Bridge Runtime.md` + `doc/ISR Bridge Runtime 改修計画.md`
> 対象ブランチ: `main`

---

# 1. 設計 — 実装着手可能な事項

## 1.1 P0-1: EQProcessor shadow変数データ競合

| 項目 | 内容 |
|------|------|
| **原因** | Worker Thread→`rtBypassedShadow`, `rtAgcCurrentGainShadow`, `rtAgcEnvInputShadow`, `rtAgcEnvOutputShadow` への非atomic書込。Audio Threadが同時読 → C++ UB |
| **ISR整合** | ⚠️ `ShadowState*`構造体によるatomic pointer swap方式は **独立した第二Publication Domain** を形成しISR設計に反する。RuntimeWorld以外にPublication Domainを増やさないことが不変条件。代わりに**個別のstd::atomic変数**を使用する（個別atomicはPublication Domainではない） |

**修正設計: 個別 `std::atomic` 化（ShadowState構造体は採用しない）**

```cpp
// EQProcessor.h
// Worker↔RT間のデータ競合修正。個別atomicはPublication Domainを形成しない。
// memory_order_relaxed: WorkerとRTが同期対象とするのは「各atomic変数自身」のみ。
// 他メモリとの happens-before 構築不要のため relaxed で十分。
std::atomic<double> rtAgcCurrentGainShadow { 1.0 };
std::atomic<double> rtAgcEnvInputShadow { 0.0 };
std::atomic<double> rtAgcEnvOutputShadow { 0.0 };
std::atomic<bool> rtBypassedShadow { false };

// m_rtBypassShadow (RT only) は変更不要 — RT単一スレッドのみ書込
bool m_rtBypassShadow = false;
```

**メモリオーダー (`memory_order_relaxed` の選択根拠）:**
これらのatomic変数は各々が独立した意味を持ち、読み出し側は3変数の組（gain/envIn/envOut）を一貫したスナップショットとして扱っていない。各atomic値は他の共有状態と意味的に独立しており、読取り側も複数atomicを一つの論理状態として扱わない。そのため `memory_order_relaxed` で十分。C++メモリモデル上、`relaxed` が使えるのは「WorkerとRTが同期対象とするのは **各atomic変数自身** のみであり、他のメモリとの happens-before が不要」だからである。`EQProcessor.Processing.cpp` の実装確認:
- `rtBypassedShadow` → 独立した分岐判定にのみ使用（line 497, 1020）
- AGC shadow値 → 各変数は個別に読み取られ（lines 415-417）、個別に書き戻される（lines 434-436）。1つの論理状態として同時取得しない
したがって現状設計では `memory_order_relaxed` で十分。`seq_cst` は不要なバリアとなる。
**注意**: 将来これらを「同一スナップショット」として扱う設計に変更する場合は、`memory_order_acquire/release` または一括構造体atomic swapへの移行を検討すること。

**双方向更新の書込主体分析（最重要検討事項）:**

WorkerとRTの双方が同じatomic変数を書き込む（2W1R）。これはC++上UBではないが、ISR設計の観点から慎重に分析する必要がある。

**データフロー実態（ソースコード確認結果）:**

```
other (UI-side processor)                this (audio RT processor)
  ┌──────────────────┐                   ┌──────────────────┐
  │ agcCurrentGain   │──(atomic)──┐      │ rtAgcCurrentGain │←──Worker書込
  │ (atomic, UI RT)  │            │      │ (non-atomic→atomic化)│←──RT書込（処理結果）
  └──────────────────┘            └──→   │                  │
       ↑ UI RT書込                        └──────────────────┘
       ↑ resetToDefaults書込                    ↑ RT AGC algorithm
```

**各変数の書込主体:**

| 変数 | Worker書込値 | RT書込値 | 競合リスクと緩和理由 |
|------|------------|---------|-------------------|
| `rtBypassedShadow` | UI永続状態 (`syncedBypassed`) | RT過渡上書き（ramp中一時変更, lines 506-517） | **リスク**: WorkerがRTのramp途中を上書きする可能性。**緩和**: RTは `m_rtBypassShadow`（別変数）で独自ramp管理。`rtBypassedShadow`のRT書込は過渡的。次のWorker同期でUI状態に再設定される。 |
| `rtAgcCurrentGainShadow` | UI側AGC `other.agcCurrentGain` | RT AGC計算値 (`nextGain`) | **リスク**: WorkerがRTのAGC収束途中値をUI側値で上書き。**緩和**: AGCは時定数付き自己収束アルゴリズム。RTは毎ブロック（~1ms）処理、Worker syncは~10-100ms間隔。Worker上書き後、RTが次ブロックで即座に再収束。 |
| `rtAgcEnvInputShadow` | UI側envelope `other.agcEnvInput` | RT AGC計算値 (`envIn`) | 同上 |
| `rtAgcEnvOutputShadow` | UI側envelope `other.agcEnvOutput` | RT AGC計算値 (`envOut`) | 同上 |

**結論**: 本修正の目的は**Data Race除去のみ**である。Worker/RT両方が書き込む設計（複数の書込主体）は既存設計であり、今回は変更しない。

**注意**: 以下の変数は複数の書込主体を持つ。atomic化によりData Raceは解消されるが、書き込みの一意性までは保証されない。
- `rtBypassedShadow` → 書込主体: Worker（UI永続状態）＋RT（過渡的ramp上書き）
- `rtAgcCurrentGainShadow` → 書込主体: Worker（UI側AGC値）＋RT（RT計算値）
- `rtAgcEnvInputShadow` / `rtAgcEnvOutputShadow` → 同上

**参考（現状実装の観察）**: 現状実装ではRT処理周期（~1ms/block）に対しWorker同期周期が十分長い（~10-100ms）ため、Worker上書き後RTが次ブロックで再収束する。**ただしこれは実装依存の観測であり設計保証ではない。**

**確認試験**: AGC収束性・可聴品質試験を実施し、実測で保証する（詳細は検証計画参照）。試験項目:
- Worker同期100Hz/500Hz/1000Hz ストレス下でAGCゲインの不自然な跳躍ゼロ
- 既存実装（非atomic版）とのAGCゲイン波形の差分比較で異常なし
- クリック検出（波形エッジ検出）で可聴レベルの不連続なし
- 耳視確認でポンピング・アーティファクトなし
- 長時間（10分以上）の連続動作でAGC値の収束が持続すること

**設計上の判断**: 本修正はC++メモリモデル上の未定義動作を除去するものであり、書込主体の一意化を目的とした変更ではない。書込主体の一意化（例: Worker同期時にRT最新値を破棄しないプロトコル、またはWorker→RT片方向のみの設計）はアーキテクチャ変更が必要であり、将来のISRアーキテクチャ改善項目として扱う。

**なぜ `ShadowState*` を採用しないか:**
- `ShadowState` 構造体＋atomic pointer swap は **RuntimeWorldとは別の独立したPublication Domain** を形成する
- ISR不変条件「Publication DomainはRuntimeWorld以外に増やさない」に違反
- Coordinator/EQStateの既存Epoch経路と重複した退役管理が必要になる
- 個別 `std::atomic` は独立した値であり、Publication Domainではない

**Coordinator管理下の更新であること:**
- shadow値は `syncGlobalStateFrom()` 経由で設定される。これはCoordinator管理下のWorker Threadからの更新
- HealthMonitor連動: 退役滞留通知は既存の `m_epochDomain.tryReclaim()` 経路でカバー

**Shutdown時の安全保証:**
- Shadow値に `new`/`delete` がないため、Shutdown Drain は不要
- `std::atomic<double>` のデストラクタはトリビアル

**ファイル**: `src/eqprocessor/EQProcessor.h`（宣言変更）, `src/eqprocessor/EQProcessor.Core.cpp`（代入文はそのまま動作）, `src/eqprocessor/EQProcessor.Processing.cpp`（読み取りはそのまま動作）

**検証**（P0-1専用）:
| 項目 | 方法 |
|------|------|
| TSAN | TSAN対応ツールチェーン（clang-cl / icx 推奨。MSVC単体では非対応の可能性あり） |
| Release | MSVC Release + icx Release で競合なし確認 |
| 連続Preset切替 | 100回以上のPreset切替中にAudio callbackを実行しデータ競合なし |
| 長時間Audio Callback | 10分以上の連続Audio処理中にWorker同期を継続 |
| AGC収束性試験 | Worker `syncGlobalStateFrom()` を100/500/1000Hzで実行中、以下の入力信号でAGC値の収束性を確認: 無音→0dBFSステップ、0dBFS→-60dBステップ、1kHz正弦波、ピンクノイズ。**判定基準**: (1)ゲイン波形に不連続ジャンプなし (2)クリック検出なし (3)耳視確認で異常なし |
| Shutdown | 同期中にシャットダウンしてもデッドロック・クラッシュなし |

| 項目 | 内容 |
|------|------|
| **原因** | Writerのrelease順序（writePosition→totalSamples）とReaderのacquire順序が異なるため、更新系列の観測に不整合が生じる可能性 |
| **優先度** | P1（単一Writer。範囲外アクセスなし） |

**修正理由**: Writerは `writePosition`→`totalSamples` の順で更新している。Readerも同順で取得することで、Writerの更新系列をより一貫して観測できる設計とする。

**修正**: `copyLatest()` のacquire順序をWriterのrelease順序と一致させる。
```cpp
const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);
const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
```
**ファイル**: `src/AudioSegmentBuffer.h`

---

## 1.3 P1-1: CpuFeatureCheck.cpp

**修正**: `#define PF_AVX2_INSTRUCTIONS_AVAILABLE 40`（正値。旧値10はSSE2）
**ファイル**: `src/CpuFeatureCheck.cpp`

---

## 1.4 P1-2: CustomInputOversampler AVX2 OOB

**修正**: 境界チェックに `kLoadStride2Offset = 6` を明示
```cpp
static constexpr int kLoadStride2Offset = 6;
const int avxMinConvIdx = globalMinConvIdx - kLoadStride2Offset;
const bool convTapOk = (avxMinConvIdx >= 0) && (globalMaxConvIdx < capacity);
```
**ファイル**: `src/CustomInputOversampler.cpp`

---

## 1.5 P2-1: 整数オーバーフロー

**修正**: 4箇所の `numSamples * sizeof(double)` → `static_cast<size_t>(numSamples) * sizeof(double)`

| ファイル | 行 |
|---------|-----|
| `MKLNonUniformConvolver.cpp` | 1663 |
| `AudioEngine.Processing.DSPCoreIO.cpp` | 280 |
| `ConvolverProcessor.Runtime.cpp` | 1165 |
| その他同種 | — |

---

## 1.6 P2-5: std::atomic\<bool\> 初期化

**4ファイル**: `DeferredFreeThread.h:19`, `ISRRTExecution.cpp:14`, `ISRRTExecution.h:11`, `ISRRuntimePublicationCoordinator.h:211`
```cpp
std::atomic<bool> running{false};  // {false} で直接初期化
```

---

## 1.7 P3-1: JUCE_LEAK_DETECTOR 追加

**6クラス**: `ConvolverControlPanel`, `ConvolverSettingsComponent`, `EQControlPanel`, `MixedPhaseOptimizationComponent`, `NoiseShaperLearningComponent`, `SpectrumAnalyzerComponent`
```cpp
JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ClassName)
```

---

# 2. 未確定事項 — 設計検討中のため実装着手してはならない

## 2.1 P1-3: ScopedAlignedPtr factory化

**課題**: `ScopedAlignedPtr::reset(T*)` が任意ポインタを受入れ `mkl_free()` 解放。アライメントでは `_aligned_malloc` と区別不可。

**設計方針案**: `fromMklMalloc()` factory分離

**未確定理由**: 影響範囲19箇所（AudioEngine 6, ConvolverProcessor 8, LoaderThread 4, makeAlignedCopy 1）。`reset()`/`release()` との整合性、直接コンストラクタ呼出し移行方式未確定。

## 2.2 P2-2: FTZ/DAZ 設定

**確定**: `AudioBlock.cpp:101`/`BlockDouble.cpp:103` ✅。`DSPCoreFloat.cpp`/`DSPCoreIO.cpp` ❌追加必要。**未確定**: `DSPCoreDouble.cpp` のソース確認未完了。

## 2.3 P2-3: RetireEQStateDeferred Fallback統合

**確定方針**: 直接delete不可。`DeferredRetireFallbackQueue` + `DSPQuarantineManager` の3段階管理。PressureLevel 0-3まで段階的制御。

**未確定**: 結合コード・`HealthMonitor.injectBackpressureSignal()` シグネチャ・Coordinator連動プロトコル未設計。

## 2.4 P2-4: CacheManager volatile sink

**設計方針案**: `_mm_prefetch` + `volatile touch` + `std::atomic_signal_fence`
**未確定理由**: ICX LTO有効ビルドでの逆アセンブリ確認未完了。

## 2.5 P3-2: 軽微な改善（5件）

L-1〜L-6: 全般に未設計。P3のため優先度低。

---

# 3. Appendix

## 3.1 改修対象一覧

| ID | 内容 | 優先度 | Status |
|----|------|--------|--------|
| P0-1 | EQProcessor shadowデータ競合 | P0🔴 | **1.設計** |
| P1-0 | AudioSegmentBuffer メモリ順序 | P1🟠 | **1.設計** |
| P1-1 | CpuFeatureCheck #define 10→40 | P1🟠 | **1.設計** |
| P1-2 | Oversampler OOB (-6 offset) | P1🟠 | **1.設計** |
| P1-3 | ScopedAlignedPtr factory化 | P1🟠 | **2.未確定** |
| P2-1 | 整数オーバーフロー (size_t cast) | P2🟡 | **1.設計** |
| P2-2 | FTZ/DAZ 設定 | P2🟡 | **2.未確定** |
| P2-3 | Retire Fallback統合 | P2🟡 | **2.未確定** |
| P2-4 | CacheManager volatile sink | P2🟡 | **2.未確定** |
| P2-5 | std::atomic\<bool\> 初期化 | P2🟡 | **1.設計** |
| P3-1 | JUCE_LEAK_DETECTOR | P3🟢 | **1.設計** |
| P3-2 | 軽微な改善5件 | P3🟢 | **2.未確定** |

## 3.2 スケジュール

| Phase | 期間 | 対象 | 工数 |
|-------|------|------|------|
| 1 | 3-4日 | P0-1 | 4人日 |
| 2 | 2日 | P1-0, P1-1, P1-2, P2-1, P2-5, P3-1 | 2人日 |
| 3 | 3-4日 | 未確定→確定後 | 4人日 |
| 4 | 1日 | P3-2 | 1人日 |
| 検証 | 2-3日 | 全変更 | 2人日 |
| **合計** | **10-12日** | | **12人日** |

## 3.3 テスト計画

| テスト | 対象 | 方法 |
|--------|------|------|
| TSAN | P0-1, P1-0 | TSAN対応ツールチェーン（clang-cl / icx 推奨） |
| リニアライザビリティ | P1-0 | ランダムWriter/Reader |
| AVX2非対応CPU | P1-1 | `/arch:AVX2` なしビルド |
| Oversampler境界値 | P1-2 | 各種タップ数 |
| 逆アセンブリ | P2-4 | ICX LTO有効ビルド |
| MXCSR確認 | P2-2 | 全スレッドFTZ/DAZ |
| 回帰テスト | 全変更 | 既存スイート+CTest |
| 実機テスト | DAW | Cubase, Studio One, REAPER |

## 3.4 ISR設計不変条件チェックリスト

| # | 不変条件 | 確認方法 |
|---|---------|---------|
| 1 | RTでdeleteしない | `delete`/`free`/`mkl_free` 不在確認 |
| 2 | RTでlockしない | `mutex`/`lock_guard` 不在確認 |
| 3 | RTでmalloc/newしない | `malloc`/`new`/`mkl_malloc` 不在確認 |
| 4 | RetireはEpoch経由 | Shutdown時以外に直接deleteなし |
| 5 | Coordinator管理下のみ更新 | `retireDSP()` 直接呼出しなし。Coordinatorを経由せず状態を変更しない |
| 6 | Overflow即ドロップ防止 | Fallback→Quarantine段階的退避 |
| 7 | Shutdown完全Drain | `VerifyDrained` 全キュー確認 |
| 8 | Publication Domain一元化 | RuntimeWorld以外でatomic swapしない |

## 3.5 誤検知一覧（改修不要）

| カテゴリ | 件数 | 理由 |
|---------|------|------|
| メモリリーク/所有権 | 5 | `exchangeAtomic`+`unique_ptr` 適切管理 |
| AVXアライメント（大半） | 9 | `mkl_malloc`/`alignas(32)` 保証済み |
| noexcept（大半） | 14 | C関数・空実装で例外なし |
| 仮想デストラクタ | 2 | 全IFに `virtual ~ = default` |
| MessageManagerLock | 2 | `callAsync` フォールバック安全 |
| Vyukov MPMC fence | 2 | release/acquireでHB充足 |
| OutputFilter 零除算 | 2 | `Q>0` ガードで `1+alpha>=1` |
| Retired flag 死 | 1 | `exchangeAtomic` 戻値防止 |
| tryAddRef CASなし | 1 | `compareExchangeAtomic` はCAS |

**改修不要。**
