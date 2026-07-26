# バグ修正改修計画書 v2 — ConvoPeq work88（確定版）

**作成日**: 2026-07-26
**前版**: BUG-REMEDIATION-PLAN.md (v1)
**レビュー後確定**: 2026-07-26
**分類**: コード品質 / メモリ安全性 / データ競合 / パフォーマンス

---

## 0. 改修方針と採否判定

### レビューを踏まえた再評価

本計画書はレビューコメントを反映し、各BUGを「**採用 / 保留 / 却下**」に再分類する。

| 判定 | 意味 | 件数 |
|------|------|------|
| ✅ **採用** | 現行の修正方向で問題なし。実装着手可能 | |
| ⚠️ **保留** | 修正方向は妥当だが、追加の確認または設計検討が必要 | |
| ❌ **却下** | 現行の修正方向は誤り。設計から再検討が必要 | |

---

## 1. ✅ 採用 — 修正着手可能（5件）

### 1.1 BUG12: SafeStateSwapper enterStateReader/exitStateReader No-Op

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL — Use-after-free |
| **判定** | ✅ **採用** |
| **ファイル** | `ConvolverProcessor.h:268-269` |
| **修正難易度** | ★☆☆☆☆（4行の委譲追加） |
| **依存** | BUG13とセットで修正（BUG13の修正方向は別途再設計） |

**修正内容**（変更なし）:
```cpp
void enterStateReader(int readerIndex) const noexcept
{
    rcuSwapper.enterReader(readerIndex);
}
void exitStateReader(int readerIndex) const noexcept
{
    rcuSwapper.exitReader(readerIndex);
}
```

**レビュー評価**: 「実在性が高い。SafeStateSwapper への委譲を入れる方針自体は筋が通っている。」

---

### 1.2 BUG11: activeRuntimeDSPSlot / fadingRuntimeDSPSlot Non-Atomic Data Race

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL — C++未定義動作（データ競合） |
| **判定** | ✅ **採用**（センチネル互換性の確認事項を追記） |
| **ファイル** | `AudioEngine.h:1996-1999`, `AtomicAccess.h:31-33` |
| **確認事項** | `DSPTransition.h:93` のセンチネル判定 `reinterpret_cast<uintptr_t>(prevRaw) == (~static_cast<uintptr_t>(0))` は `std::atomic<DSPCore*>` から load したポインタに対しても同様に動作する |

**修正内容**:
```cpp
// AudioEngine.h
std::atomic<DSPCore*> activeRuntimeDSPSlot{nullptr};
std::atomic<DSPCore*> fadingRuntimeDSPSlot{nullptr};

inline DSPCore* exchangeFadingRuntimeDSP(DSPCore* value) noexcept
{
    return fadingRuntimeDSPSlot.exchange(value, std::memory_order_acq_rel);
}
inline DSPCore* getActiveRuntimeDSP() const noexcept
{
    return activeRuntimeDSPSlot.load(std::memory_order_acquire);
}
inline void setActiveRuntimeDSP(DSPCore* value) noexcept
{
    activeRuntimeDSPSlot.store(value, std::memory_order_release);
}
```

**センチネル互換性確認**:
```cpp
// DSPTransition.h:93 — 現行コード
if (auto* prev = (reinterpret_cast<uintptr_t>(prevRaw) == (~static_cast<uintptr_t>(0)))
    ? nullptr : reinterpret_cast<DSPCore*>(prevRaw))

// std::atomic<DSPCore*> 化後 → prevRaw が DSPCore* になるため下記でOK:
if (auto* prev = (reinterpret_cast<uintptr_t>(static_cast<void*>(prevRaw)) == (~static_cast<uintptr_t>(0)))
    ? nullptr : prevRaw)
// またはより簡潔に: センチネル値を nullptr に変更する設計判断も可
```

**レビュー評価**: 「妥当。`NonOwningPtr<DSPCore>` のままで、`exchangeFadingRuntimeDSP()` も直接読み書きしている。別箇所でセンチネル判定があるため、`std::atomic<DSPCore*>` 化だけで契約を壊さないか確認が必要。」

---

### 1.3 BUG4: xRunBuffer ACTIVATE イベント無条件プッシュ

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL — イベント喪失 |
| **判定** | ✅ **採用** |
| **ファイル** | `AudioBlock.cpp:605`, `BlockDouble.cpp:572` |
| **修正難易度** | ★☆☆☆☆（2行のif文追加） |

**修正内容**（変更なし）:
```cpp
if (!xRunBuffer.push(ev))
{
    convo::fetchAddAtomic(rtAuxMutable_.xRunDropCount,
        uint64_t{1}, std::memory_order_relaxed);
}
```

---

### 1.4 BUG-001: Float処理パスにPeak Limiterが未実装

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — 音質劣化（ハードクリップ歪み） |
| **判定** | ✅ **採用**（可視性確認済み） |
| **ファイル** | `AudioEngine.Processing.DSPCoreIO.cpp:506-514` |
| **確認結果** | `peakLimiter` は `AudioEngine::DSPCore` のメンバ（line 960）。DSPCoreIO.cpp/DSPCoreDouble.cpp は同じ DSPCore のメンバ関数を実装しているため、追加の参照なしで直接アクセス可能 |

**修正内容**:
```cpp
// DSPCoreIO.cpp processOutput() — NaN/Inf Scrub の後、Hard Clamp の前
constexpr double kPLThreshold = 0.8413951287507587;
constexpr double kPLKnee = 0.108748;
peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee);
```

**テスト期待値の修正**: 「FloatとDoubleの出力が一致」ではなく、メトリクス値の整合または許容誤差による判定を使用すること。

---

### 1.5 BUG-002/003: Float処理パスでLoudness Meter/TruePeak Detector未実行

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM |
| **判定** | ✅ **採用**（BUG-001と同時修正推奨） |
| **ファイル** | `AudioEngine.Processing.DSPCoreIO.cpp:375-378` 周辺 |

**修正内容**: DC Blocker適用後、NaN/Inf Scrub前に以下を追加:
```cpp
truePeakDetector.processBlock(dataL, dataR, numSamples);
loudnessMeter.processBlock(dataL, dataR, numSamples);
```

---

## 2. ⚠️ 保留 — 追加確認・設計検討が必要（5件）

### 2.1 BUG13: SafeStateSwapper Epoch Bump Before Pointer Swap

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — BUG12修正後にUAF顕在化の可能性 |
| **判定** | ❌ **v1案は却下** → 修正方向を再設計中 |
| **v1案の問題点** | 「SWAP → BUMP」への順序反転はRCUの退避期間を弱める可能性が高い |

**再検証結果**:
```
現行コード（bump→bump→swap, retire with epoch1）:
  epoch1 = bump前の値（旧データが有効だった時代）
  2-step bump により単調性確保 + 観測ズレ吸収
  retireEpoch = epoch1

問題のインターリーブ:
  t0: bump#1 → globalEpoch = N+1
  t1: Reader enterReader → epoch = N+1 を記録
  t2: bump#2 → globalEpoch = N+2
  t3: swap → activeState = newState, retire oldState with epoch=N
  t4: Reclaimer: isOlder(N, N+1) = true → reclaim!
  t5: Reader: getState() が旧ポインタを見ている → UAF 💥
```

**結論**: 問題は存在するが、v1の「SWAP→BUMP」案は誤り。正しい修正には以下の選択肢を検討中:
- **案A**: `retireEpoch` に `epoch1` ではなく `epoch2`（= epoch1+2, 2回目のbump後の値）を使用する
- **案B**: swap 後に retire epoch を取得する（swap→bump、ただしbumpはepoch1ではなくswap後の値で行う）
- **案C**: 現在の設計は意図通りであり、enterReader と getState の間のメモリ順序保証により問題が発生しないことを証明する

**⚠ 注記**: BUG12のenterStateReader修正を入れた場合にのみ顕在化する。現在はBUG12のno-opスタブによりマスクされている。

---

### 2.2 BUG-010: retireEQStateDeferred/retireBandNodeDeferred 戻り値無視

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — MKLメモリリーク（17箇所） |
| **判定** | ⚠️ **保留** — void化は設計変更。修正方向を再検討中 |
| **ファイル** | `EQProcessor.Core.cpp:26-108` 他、17箇所の呼び出し元 |
| **再検証結果** | retireEQStateDeferred は NonRT パスからのみ呼ばれる（EQProcessor.Processing.cppでは不使用）。したがって「直接deleteへのフォールバック」のRT安全性リスクは低い。しかし Coordinator/Router のISR epoch 追跡をバイパスすることになる |

**推奨修正案（v1からの修正）**:
```cpp
// 変更前: (void) で戻り値を破棄
(void)retireEQStateDeferred(oldState);

// 変更後: 失敗時にカウンタをインクリメント（リーク検出 + 診断）
if (!retireEQStateDeferred(oldState))
{
    convo::fetchAddAtomic(m_retireDropCount, uint64_t{1}, std::memory_order_relaxed);
}
```

**この修正の方針**:
- 戻り値の型は `bool` のまま維持（設計変更なし）
- 失敗時にドロップカウンタをインクリメント（診断可能に）
- Coordinator/Router のISR epoch 追跡をバイパスしない（直接 delete は行わない）
- リーク自体は発生しうるが、診断・監視が可能になり、かつ設計変更を伴わない

**レビュー評価**: 「void化は現状コードの修正というより設計変更。まずは返り値を捨ててよい経路か、失敗時のフォールバックが本当に安全か確認した方がよい。RTから呼ばれる可能性があるなら、直接deleteへ落とす案はPractical ISRの原則にも反する。」

---

### 2.3 BUG17: overflowDurationMs 単位不一致

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — 慢性OVF検出が1000倍早く発動 |
| **ファイル** | `AudioEngine.Retire.cpp:134-137` |
| **判定** | ✅ **採用**（MSVC chrono文献により確定） |

**再検証結果**:
- Microsoft公式ドキュメント: `steady_clock::duration` は MSVC 実装では `nanoseconds`（`duration<long long, nano>`）
- `now`（line 134）: `steady_clock::now().time_since_epoch().count()` → **ナノ秒**
- `overflowStart`（line 131）: 同じ `steady_clock::now()...count()` → **ナノ秒**（ISRRetire.cpp:71と同じソース）
- `(now - overflowStart) / 1000` → **マイクロ秒**
- 変数名 `overflowDurationMs` とコメント「>5秒」は誤り
- 実際の閾値: 5000 μs = **5 ms**（意図は5000 ms = 5秒）

**修正内容**:
```cpp
// 変更前:
const uint64_t overflowDurationMs = (now - overflowStart) / 1000;
chronicByDuration = (overflowDurationMs > 5000);  // 5ms で発動

// 変更後:
const uint64_t overflowDurationMs = (now - overflowStart) / 1'000'000;
chronicByDuration = (overflowDurationMs > 5000);  // 5000ms = 5秒
```

---

### 2.4 BUG16: Retire Overflow Timestamp 単位不一致

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — overflowAgeWarnコードパスがデッドコード |
| **判定** | ✅ **採用**（MSVC chrono文献により確定） |
| **ファイル** | `ISRRetire.cpp:55-57` (Producer), `ISRRuntimePublicationCoordinator.cpp:279-284` (Consumer) |

**再検証結果**:
- Producer（ISRRetire.cpp:56）: `steady_clock::now().time_since_epoch().count()` → **ナノ秒**を保存
- Consumer（Coordinator.cpp:279-282）: `duration_cast<microseconds>` で取得した **マイクロ秒** と比較
- フィールド名 `overflowTimestampUs`（`Us`=マイクロ秒）だが、格納値はナノ秒

**修正内容**:
```cpp
// ISRRetire.cpp:56 — Producer を microseconds に修正
RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()), 0};
```

---

### 2.5 BUG10: DeferredDeletionQueue uint32_t ラップアラウンド

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — カウンタラップでキュー誤動作 |
| **判定** | ✅ **採用**（3行の型変更） |
| **ファイル** | `DeferredDeletionQueue.h:80,120,172` |

**修正内容**（変更なし）:
```cpp
// 変更前:
intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

// 変更後:
int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
```

---

## 3. 🟡 後回し — 低リスク・性能対策（3件）

### 3.1 BUG-009: 9ファイルで `_mm256_zeroupper()` 欠落

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM — AVX→SSE遷移ペナルティ |
| **判定** | ✅ **採用（後回し可）** |
| **確認結果** | 9ファイル未対応。正当性の核心ではなく性能改善。低リスク |
| **修正難易度** | ★☆☆☆☆（9ファイル×1行挿入） |

**修正対象**: CustomInputOversampler, ConvolverProcessor.Runtime, MKLNonUniformConvolver, SpectrumAnalyzerComponent, EQProcessor.Processing, DSPCoreIO, EQResponse, ConvolverProcessor.LoaderThread, DSPCoreFloat(※スカラー名だが2026-07-26時点のコードベースではAVX2不使用確認済み)

### 3.2 BUG-004: DSPCoreIO.cpp の `_mm256_zeroupper()` 欠落

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM |
| **判定** | ✅ **採用（後回し可）** |
| **備考** | BUG-009の一部。独立管理 |

### 3.3 BUG9: EQProcessor シャドウ relaxed-only データ競合

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM — ARM64で顕在化リスク |
| **判定** | ✅ **採用（後回し可）** |
| **ファイル** | `EQProcessor.Core.cpp:592-596,655-659`, `EQProcessor.Processing.cpp` |

---

## 4. 🟢 最下位 — コーディング規約（1件、4サブカテゴリ）

### 4.1 BUG-005〜008: `static_cast<size_t>` 欠落（24箇所）

| 項目 | 内容 |
|------|------|
| **リスク** | LOW — MSVCでは実質安全、規約上の問題 |
| **判定** | ✅ **採用（最下位優先度）** |
| **確認結果** | 現行ソースの近傍では `static_cast<size_t>` がすでに広く使用されている。grepで機械的に不足箇所を確定してからまとめて処理する案件 |
| **対象** | 4ファイル、24箇所（完全棚卸し済み） |

---

## 5. 改修スケジュール（確定版）

### Phase 1 — 採用・即時修正

| 優先順位 | BUG-ID | 判定 | リスク | 工数 |
|---------|--------|------|-------|------|
| 🔴 1 | BUG12 | ✅ 採用 | CRITICAL - UAF | 45min |
| 🔴 2 | BUG11 | ✅ 採用 | CRITICAL - データ競合(UB) | 1.5h |
| 🔴 3 | BUG4 | ✅ 採用 | CRITICAL - イベント喪失 | 30min |
| 🔴 4 | BUG-001 | ✅ 採用 | HIGH - 音質劣化 | 2h |
| 🔴 5 | BUG-010 | ⚠️ 保留→カウンタ案 | HIGH - メモリリーク | 1h |
| 🔴 6 | BUG17 | ✅ 採用 | HIGH - OVF誤検出 | 15min |
| 🔴 7 | BUG10 | ✅ 採用 | HIGH - ロックフリー破綻 | 30min |
| 🟡 8 | BUG16 | ✅ 採用 | HIGH - デッドコード | 20min |

### Phase 2 — 追加設計検討後

| 優先順位 | BUG-ID | 判定 | 状態 |
|---------|--------|------|------|
| 🔴 9 | BUG13 | ❌ v1案却下→再設計中 | RCU退避期間の設計検証＋修正案の再設計 |
| 🔴 10 | BUG-002/003 | ✅ 採用（Phase1セット） | BUG-001と同時修正 |

### Phase 3 — 後回し

| 優先順位 | BUG-ID | 判定 | 工数 |
|---------|--------|------|------|
| 🟡 11 | BUG-009 | ✅ 採用（後回し） | 1h |
| 🟡 12 | BUG-004 | ✅ 採用（後回し） | 15min |
| 🟡 13 | BUG9 | ✅ 採用（後回し） | 1h |
| 🟢 14 | BUG-005〜008 | ✅ 採用（最下位） | 1h |

---

## 6. 総合見積り

| Phase | 工数 | 内訳 |
|-------|------|------|
| Phase 1（即時修正） | 6.25h | BUG12(45min)+BUG11(1.5h)+BUG4(30min)+BUG-001(2h)+BUG-010(1h)+BUG17(15min)+BUG10(30min)+BUG16(20min) |
| Phase 2（設計後） | 2h | BUG13(1h)+BUG-002/003(1h) |
| Phase 3（後回し） | 3h15min | BUG-009(1h)+BUG-004(15min)+BUG9(1h)+BUG-005〜008(1h) |
| **合計** | **〜11.5h** | |

---

## 7. 補足: BUG13 修正方向の設計メモ

### 問題の本質

`SafeStateSwapper::swap()` は epoch bump（2回）を **先に行い**、その後 `exchangeAtomic` でポインタを差し替える。これにより、bump と swap の間に enterReader したスレッドが「新しい epoch を記録しているが、旧ポインタを読む」ウィンドウが存在する。

### 修正案A: retireEpoch に epoch2 を使用する

```cpp
const uint64_t epoch1 = fetchAddAtomic(globalEpoch, 1, acq_rel);
const uint64_t epoch2 = fetchAddAtomic(globalEpoch, 1, acq_rel);  // = epoch1+1
ConvolverState* oldState = exchangeAtomic(activeState, newState, acq_rel);
if (oldState) retireEntry(oldState, epoch2);  // NOT epoch1
```

- epoch2 (= epoch1+1) は bump#2 後の値
- 2-step bump により epoch2 は epoch1 より必ず大きい
- Reader が epoch1+1 を記録（bump#1 後）→ retireEpoch(epoch1+1) と比較→ isOlder(epoch1+1, epoch1+1) = false → 安全
- Reader が epoch1+2 を記録（bump#2 後）→ retireEpoch(epoch1+1) と比較→ isOlder(epoch1+1, epoch1+2) = true → 解放、しかし Reader は epoch1+2 なので swap 前の可能性あり

うまくいかない。Reader が epoch1+2 を記録（bump#2 後、swap 前）でも旧ポインタを見る可能性がある。

### 修正案B: swap → bump（v1案、却下）

v1では提案したが、レビューで「RCU退避期間を弱める」と指摘された。2-step bump による safety margin が失われる。

### 修正案C: 現在の設計は正しい（要証明）

設計コメントにある通り、`epoch1` は「旧データが有効だった時代」を表す。Reader が epoch > epoch1 を記録した場合、理論的には「旧データを見ない」と仮定している。実際には swap 前に enterReader した Reader は旧データを見るが、その場合でも以下のメモリ順序保証により安全かもしれない:

```
swap: fetchAdd(acq_rel) epoch1 → enterReader: consume(acquire) epoch1+1
→ swap: exchangeAtomic(acq_rel) activeState → Reader: consume(acquire) activeState
```

acq_rel chain により、enterReader の acquire が swap の exchangeAtomic より後に実行される場合、Reader は新しい activeState を見る。enterReader が swap の exchangeAtomic より前に実行された場合、Reader の epoch は epoch1 以下である。

**この証明にはさらに詳細なメモリモデル分析が必要。Phase 2 で設計検討。**

---

## 8. 補足: 検証に使用したツール

| ツール | 使用目的 |
|--------|---------|
| **ripgrep/grep/sed/awk (WSL)** | 全パターン検索・コード解析 |
| **AiDex MCP** | コードインデックス・シンボル検索 |
| **serena MCP** | シンボル探索・型情報取得 |
| **semble-search** | コードコンテキスト検索 |
| **Web Search** | MSVC chrono ドキュメント確定 |
| **Python3 (WSL)** | 全memset/memcpyサイトの自動棚卸し |
