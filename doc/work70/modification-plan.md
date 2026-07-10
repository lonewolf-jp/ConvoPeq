# ConvoPeq メモリ肥大化 改修計画書

**日付**: 2026-07-10
**バージョン**: 1.0
**対象**: work70 (メモリ 2.5GB 肥大化問題)
**著者**: GitHub Copilot (DeepSeek V4 Flash)
**関連文書**:
- [メモリ解析レポート](memory-analysis-report.md)
- [検証確定レポート](verification-report.md)
- [AoS→SoA 改修設計書](plan.md)

---

## 目次

1. [背景と目的](#1-背景と目的)
2. [現状サマリ](#2-現状サマリ)
3. [根本原因の階層構造](#3-根本原因の階層構造)
4. [改修項目一覧](#4-改修項目一覧)
5. [Phase 1: advanceFade() 配線（P0）](#5-phase-1-advancefade-配線p0)
6. [Phase 2: MEM_SNAP 監視強化（P1）](#6-phase-2-mem_snap-監視強化p1)
7. [Phase 3: 初回ブロックサイズ最適化（P1）](#7-phase-3-初回ブロックサイズ最適化p1)
8. [Phase 4: AoS→SoA メモリ最適化（P2）](#8-phase-4-aosoa-メモリ最適化p2)
9. [Phase 5: 未計装 mkl_malloc 追跡（P3）](#9-phase-5-未計装-mkl_malloc-追跡p3)
10. [検証計画](#10-検証計画)
11. [リスク評価とロールバック戦略](#11-リスク評価とロールバック戦略)
12. [スケジュール案](#12-スケジュール案)
13. [付録: 調査で発見された副次的知見](#13-付録-調査で発見された副次的知見)

---

## 1. 背景と目的

### 1.1 問題

ConvoPeq のプロセスメモリ使用量が、通常動作時に Private Memory 約 **2.5GB** に達する。同種のプラグインと比較して著しく高く、低メモリ環境や複数インスタンス起動時に支障をきたす。

### 1.2 調査の成果

これまでの調査により以下の事実が確定している:

- **MKL Convolution は無実**: 追跡済み MKL 割り当ては 35MB（全体の 1.4% のみ）
- **DSPCore 6 インスタンス累積**: `DSPCORE_PREPARE` 6 回実行、`lifecycle(retire)=0`
- **advanceFade() 未呼び出し**: 0 call sites — これが Timer 経路の retire をブロック
- **AoS (Array of Structs) 冗長性**: `irFreqDomain` + `fdlBuf` の AoS 保持が SoA と二重化
- **生 mkl_malloc 未計装**: 少なくとも 6 箇所の mkl_malloc が追跡対象外

### 1.3 目的

本計画書の目的は、上記の調査結果に基づき、優先順位付けされた改修項目を定義し、実施手順・検証方法・リスク管理を明確にすることである。

---

## 2. 現状サマリ

### 2.1 メモリ使用量の全体像（確定値と推定値の区別）

```text
Private 2477MB の内訳:

  ┌─ MKL Convolution (tracked)  =   35MB  (1.4%)  ☆ 確定値
  ├─ DSPCore 6インスタンス      ≈ 1320MB (53.3%)  ★ 推定値（最有力候補）
  ├─ JUCE Framework             ≈  500MB (20.2%)  △ 推定値
  ├─ IR processing 一時         ≈  300MB (12.1%)  △ 推定値（未計装）
  ├─ CRT/STL/VirtualAlloc       ≈  150MB (6.0%)   △ 推定値
  ├─ DLL/Thread stacks          ≈  100MB (4.0%)   △ 推定値
  └─ EBR entries/other          ≈   72MB (2.9%)   △ 推定値
```

**確定値は MKL 35MB のみ。** 残りはすべて推定値であり、retire 正常化後に実測で再評価する。

### 2.2 ライフサイクルカウンタの実測値

| カウンタ | gen=4 | gen=5 | 意味 |
|:---------|:------|:------|:------|
| `lifecycle(pub)` | 4 | 5 | publish 回数（正常） |
| `lifecycle(ret)` | 0 | 0 | **retire 回数（0 のまま—異常）** |
| `lifecycle(reclaim)` | 0 | 0 | reclaim 回数（retire なしなので当然） |

### 2.3 主要コード上の確認事項

| 確認項目 | 状態 | 根拠 |
|:---------|:-----|:------|
| `SnapshotCoordinator::advanceFade()` 宣言 | ✅ `.h:133` | |
| `SnapshotCoordinator::advanceFade()` 定義 | ✅ `.cpp:43` | |
| `SnapshotCoordinator::advanceFade()` 呼び出し | ❌ **0 箇所** | これが根幹 |
| `SnapshotCoordinator::startFade()` 実行 | ✅ 実行確認 | `[VERIFY] EQ createdHash=...` |
| `SnapshotFadeState.state` | ✅ FadingIn（永遠に） | advanceFade 未呼び出しの直接結果 |
| `DSPTransition::onPublishCompleted()` | ✅ 独立 retire 経路あり | ただし lifecycle(retire)=0 |
| `retireDSPHandleForRuntime()` | ✅ handle map 未登録で false | カウンタ不进の原因 |
| `DSPGuard` (RAII) | ✅ rebuild-obsolete 正しく解放 | lifecycle(retire)=0 は見かけ上 |
| `CrossfadeRuntime` | ✅ DSP エンジン遷移用 | SnapshotCoordinator とは別機構 |

---

## 3. 根本原因の階層構造

### 3.1 第1階層: advanceFade() 未配線（直接原因）

```text
createSnapshotFromCurrentState() [Timer]
  └─ startFade(fadeSamples) → state=FadingIn, remaining=fadeSamples
       └─ advanceFade(numSamples) → 誰も呼ばない（コード欠落 確認済み 0 call sites）
            └─ remaining 永遠に > 0
```

### 3.2 第2階層: retire 連鎖の停止（メカニズム）

```text
tryCompleteFade() が常に false
  └─ fadeCompleted ブロックに到達しない
       ├─ publishWorld() が呼ばれない（current 更新なし）
       └─ DSPLifetimeManager::retire() が呼ばれない
            └─ DSPCore が解放されず累積
```

※ 加えて: rebuild-obsolete 検出時の `DSPGuard` 経由の retire も `retireDSPHandleForRuntime()` が handle map 未登録のため false を返し、EBR enqueue に至らない。この二次的リーク（3回分、約660MB推定）も同時に発生している。

### 3.3 第3階層: メモリ肥大化（結果）

```text
DSPCore 累積 ≈ 9 インスタンス = 1880MB（推定）
  ├─ 2 commit 済み active DSPCore（#1, #2）
  ├─ 2 commit 済み未 retire DSPCore（#5, #6）— advanceFade 未配線の直接結果
  └─ 3 rebuild-obsolete リーク DSPCore（#3, #4, 他）— DSPGuard 経由でも解放されず
```

### 3.4 DSPTransition 独立 retire 経路（補足）

`DSPTransition::onPublishCompleted()` は `SnapshotCoordinator` を経由せず直接 `lifetime.retire(oldDSP)` を呼ぶ独立経路を持つ。しかしログ上 `lifecycle(retire)=0` であることから、この経路でも `retireDSPHandleForRuntime()` が false を返している（oldDSP の handle が commit 経路で未登録の可能性が高い）。

**advanceFade 配線の有無にかかわらず、DSPTransition 経路の retire が本当に動作するかは、handle 登録の有無に依存する。** 配線後のログで `lifecycle(retire) > 0` となることを確認する。

---

## 4. 改修項目一覧

### 4.1 優先度定義

| 優先度 | 定義 |
|:-------|:------|
| **P0** | 即時対応。現状のメモリ肥大に直接関与している可能性が高く、かつ安全に実装可能 |
| **P1** | 監視強化または設計改善。P0 実施後の効果確認に必要、または改善効果が見込める |
| **P2** | 構造的最適化。コードベース全体のメモリ効率向上。影響範囲が広い |
| **P3** | 情報収集。実施により直接的なメモリ削減はないが、将来の判断材料となる |

### 4.2 一覧表

| # | Phase | 優先度 | 改修項目 | 難易度 | 期待効果 | リスク |
|:-:|:------|:-------|:---------|:-------|:---------|:-------|
| 1 | Phase 1 | **P0** | advanceFade() を Audio Callback に配線 | 小 | 大（retire 連鎖回復） | 低 |
| 2 | Phase 2 | **P1** | MEM_SNAP に StereoConvolver/DSPCore liveCount 追加 | 小 | 中（監視強化） | 低 |
| 3 | Phase 3 | **P1** | 初回 processingBlockSize=524288 の最適化 | 中 | 大（初回 +481MB） | 中 |
| 4 | Phase 4 | **P2** | AoS (irFreqDomain) の SoA 直接計算化 | 大 | 中（中間バッファ削減） | 大 |
| 5 | Phase 5 | **P3** | 未計装 mkl_malloc 6箇所の DIAG 化 | 小 | 小（監視強化のみ） | 低 |

---

## 5. Phase 1: advanceFade() 配線（P0）

### 5.1 概要

`SnapshotCoordinator::advanceFade(numSamples)` が誰からも呼ばれていない。Audio Callback（`getNextAudioBlock()`）から呼び出すことで、fade 進行を可能にし、retire 連鎖を回復させる。

### 5.2 設計判断

**ISR 設計原則**: Crossfade の進行量は「実際に処理したサンプル数」にのみ依存する。したがって `advanceFade(numSamples)` は Audio Callback（DSP 実行経路）からのみ呼び出す。Timer は Crossfade 完了後のライフサイクル管理（retire/reclaim）のみ担当する。

### 5.3 設計の核心

本改修の設計は以下の 2 点に集約される:

1. **`SnapshotFadeState` に `FadeState::Completed` を追加** — 状態機械を `Idle → FadingIn → Completed → Idle` に拡張
2. **Audio Callback からの `advanceFade()` 呼び出しのみ** — 戻り値 `bool`、新規 atomic、`CrossfadeRuntime` への変更は一切不要

```text
Audio Callback (RT):
  advanceFade(numSamples)
    → remaining を減算
    → remaining==0 なら state=Completed

Timer (Non-RT) — 変更なし:
  const bool fadeCompleted = m_coordinator.tryCompleteFade();
    → state==Completed なら CAS Completed→Idle
    → 成功時: completeFade() → publishWorld / retire
```

**なぜ `CrossfadeRuntime` を使わないか**: `CrossfadeRuntime` は DSP エンジン遷移（`DSPTransition` 専用）であり、Snapshot（パラメータ）遷移の完了通知を持ち込むと責務が混在する。ISR アーキテクチャでは両者は完全に独立している。

**なぜ `bool` 戻り値が不要か**: `advance()` 内で `remaining==0` 時に `state=Completed` に遷移させるだけで、Timer は `state()` の確認により完了を検出できる。戻り値 `bool` は `tryCompleteFade()` が既に提供している。

### 5.4 変更ファイル一覧

| ファイル | 変更内容 |
|:---------|:---------|
| `src/core/SnapshotFadeState.h` | `FadeState::Completed` 追加。`advance()` 内で remaining==0 時に state=Completed に遷移。`tryComplete()` は `Completed→Idle` の CAS に変更 |
| `src/core/SnapshotCoordinator.h` | 変更なし（`advanceFade()` は `void` のまま） |
| `src/core/SnapshotCoordinator.cpp` | 変更なし（`advanceFade()` は既存のまま `m_fade.advance()` を呼ぶ） |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | `getNextAudioBlock()` 内に `m_coordinator.advanceFade(numSamples)` を 1 行追加 |
| `src/audioengine/AudioEngine.Timer.cpp` | **変更不要** — 既に `tryCompleteFade()` が毎 callback で呼ばれている |
| `src/audioengine/CrossfadeRuntime.h` | **変更不要** — 本クラスは DSP エンジン遷移専用 |

### 5.5 実装詳細

#### 5.5.1 FadeState 列挙体: `Completed` 追加

```cpp
// SnapshotFadeState.h
enum class FadeState : uint8_t {
    Idle,
    FadingIn,
    Completed   // ★ work70: advance() が remaining==0 を検出した状態
};
```

#### 5.5.2 SnapshotFadeState::advance() — remaining==0 で state=Completed

```cpp
// SnapshotFadeState.h — advance() 変更箇所のみ抜粋
void advance(int numSamples) noexcept
{
    if (state() != FadeState::FadingIn)
        return;

    const int remaining = remainingCount();
    if (remaining <= 0)
        return;

    const int newRemaining = remaining - numSamples;
    if (newRemaining <= 0)
    {
        // ★ work70: remaining==0 に到達。state=Completed に遷移し、
        //   Timer の tryCompleteFade() が検出できるようにする。
        convo::publishAtomic(remainingSamples_, 0, std::memory_order_relaxed);
        convo::publishAtomic(state_, FadeState::Completed, std::memory_order_relaxed);
        return;
    }

    // 通常のカウントダウン（変更なし）
    convo::publishAtomic(remainingSamples_, newRemaining, std::memory_order_relaxed);
    const int total = totalCount();
    if (total > 0)
    {
        const double nextAlpha = 1.0 - static_cast<double>(newRemaining) / static_cast<double>(total);
        convo::publishAtomic(alpha_, nextAlpha, std::memory_order_relaxed);
    }
}
```

**`memory_order_relaxed` で十分な理由**: `remainingSamples_` と `state_` は「単なるカウンタと状態」であり、これらの値で同期すべきデータは存在しない。publish された Snapshot 自体は `SnapshotSlotStore` の別 atomic ポインタ（`exchangeCurrent` / `loadCurrent` の `acq_rel`/`acquire`）で同期される。advance() の relaxed 書き込みは、後続の Timer 側 `tryCompleteFade()` の acquire 読み取りと自然に before/after 関係が成立する（Timer と Audio callback は OS スケジューリングにより順序保証）。

#### 5.5.3 SnapshotFadeState::tryComplete() — Completed→Idle への CAS

```cpp
// SnapshotFadeState.h — tryComplete() 変更
bool tryComplete() noexcept
{
    // ★ work70: advance() が state=Completed に遷移した後にのみ成功。
    if (state() != FadeState::Completed)
        return false;

    FadeState expected = FadeState::Completed;
    return convo::compareExchangeAtomic(state_,
                                        expected,
                                        FadeState::Idle,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire);
}
```

#### 5.5.4 AudioBlock.cpp — advanceFade 配線（1行のみ）

```cpp
// AudioEngine::getNextAudioBlock() 内、DSP 処理完了直後
// （例: finalizeCrossfadeMixPath() / cleanupCrossfadeDirectPath() の後）

// ★ work70: SnapshotCoordinator の fade 進行。
//   remaining を処理済みサンプル数だけ進め、0 到達時は state=Completed に遷移。
m_coordinator.advanceFade(numSamples);
```

#### 5.5.5 Timer.cpp — 変更不要の理由

Timer は既に `tryCompleteFade()` を毎 callback で呼んでいる（line 843）。advanceFade() が Audio Callback から呼ばれるようになれば:
1. `remaining` が減少し、0 に到達 → `state=Completed`
2. Timer の `tryCompleteFade()` が `state==Completed` を検出 → CAS `Completed→Idle` → true → fadeCompleted ブロック実行

Timer 側のコード変更は一切不要。既存の以下のコードがそのまま動作する:

```cpp
// AudioEngine.Timer.cpp:843（変更なし）
const bool fadeCompleted = m_coordinator.tryCompleteFade();
if (fadeCompleted)
{
    // [XFADE] logging, notifyFadeComplete, retire, complete, publishWorld...
}
```

### 5.6 実装上の注意点

1. **`isFading()` の意味が拡張される**: `isFading()` は `state() != Idle` を返す。従来は `FadingIn` のみだったが、`Completed` でも `true` を返す。これは正しい動作: `Completed` は「fade 完了検出済みだが、まだ tryCompleteFade() が処理していない」状態であり、fade 中の扱いを継続すべき。`isFading() == true` により Timer の fade 中ガード（line 651, 942）が期待通り動作する。
2. **CAS の競合は発生しない**: `advance()` は Audio Thread からのみ呼ばれ、`tryComplete()` は Timer からのみ呼ばれる。CAS `Completed→Idle` が競合することはない。
3. **DSPTransition との相互作用**: DSPTransition の独立 retire 経路と SnapshotCoordinator の retire 経路は並存する。両方から同じ DSPCore が retire される可能性については advanceFade 配線後のログで確認する。

### 5.7 期待効果

- `lifecycle(retire) > 0` になる
- `pendingRetireCount` が増加 → reclaim で減少
- DSPCore liveCount が減少（2 current + fading 付近に収束）
- Private Memory が減少（推定 ~800MB、幅あり）

---

## 6. Phase 2: MEM_SNAP 監視強化（P1）

### 6.1 概要

現在の MEM_SNAP には `MKLNonUniformConvolver::liveCount` のみが含まれている。
`StereoConvolver::liveCount` と `DSPCore::liveCount` を追加し、メモリ監視を強化する。

### 6.2 変更ファイル

| ファイル | 変更内容 |
|:---------|:---------|
| `src/audioengine/AudioEngine.Timer.cpp` | MEM_SNAP フォーマット文字列に `%u %u` 追加 + 引数追加 |

### 6.3 実装詳細

```cpp
// MEM_SNAP フォーマット文字列（変更前）
"[MEM_SNAP] PUBLISH gen=%llu | NUC: live=%u alloc=%lluMB peak=%lluMB tA=%lluGB tF=%lluGB..."

// 変更後 — StereoConvolver::liveCount + DSPCore::liveCount を %u で追加
"[MEM_SNAP] PUBLISH gen=%llu | NUC: live=%u(%u/%u) alloc=%lluMB..."
```

具体的なフォーマット変更は既存コードと整合させる。カウンタ読み取り方法は各クラスの `liveCount` 変数に応じて追記。

### 6.4 期待効果

- DSPCore の寿命を直接監視可能に
- advanceFade 配線後の効果確認指標
- リグレッション検知の早期指標

---

## 7. Phase 3: 初回ブロックサイズ最適化（P1）

### 7.1 概要

初回 `DSPCore::prepare()` 時の `processingBlockSize` が **524288**（`SAFE_MAX_BLOCK_SIZE 65536 × MAX_OS_FACTOR 8`）となる。これにより初回 DSPCore 1 台で +481MB のメモリジャンプが発生している。

### 7.2 問題点

| パラメータ | 値 | 備考 |
|:-----------|:---|:------|
| `SAFE_MAX_BLOCK_SIZE` | 65536 | `AudioEngine.h:1023` |
| `MAX_OS_FACTOR` | 8 | `DSPCoreLifecycle.cpp:140` |
| `internalMaxBlock` | 524288 | 65536 × 8 |
| 後続のブロックサイズ | 2048 | 実運用値（約 256 分の 1） |

**この値は設計上の安全上限であり、実際のオーバーサンプリング比が 0 の場合には過剰。**

### 7.3 検討中の対策

以下の案を技術検討する:

| 案 | 内容 | 難易度 | リスク | 期待削減 |
|:---|:-----|:-------|:-------|:---------|
| **A** | osFactor=0 時に `SAFE_MAX_BLOCK_SIZE` を直接使わず、実際のブロックサイズを上限とする | 中 | 低 | ~481MB |
| **B** | `SAFE_MAX_BLOCK_SIZE` を適切な値（例: 8192 または 16384）に低減 | 高 | 中 | 調査中 |
| **C** | 初回 prepare 時のブロックサイズのみ別途制限 | 低 | 低 | ~481MB |

### 7.4 判断基準

- JUCE Init.cpp:69 のコメント「不要に巨大な一時NUC」が示す通り、開発者自身も問題認識あり
- ただし `SAFE_MAX_BLOCK_SIZE` の変更は全コーナーケース（OS=8 時等）への影響がある
- **Phase 1 実施後にメモリ削減効果を確認し、本 Phase の要否を再評価する**

### 7.5 保留中の検討事項

- `MAX_OS_FACTOR` と `processingBlockSize` の関係式の再検証
- 初回 prepare 以外のブロック（rebuild, IR load 等）への影響
- `SAFE_MAX_BLOCK_SIZE` 低減時の OS=8 パスでの挙動

---

## 8. Phase 4: AoS→SoA メモリ最適化（P2）

### 8.1 概要

MKLNonUniformConvolver は現在、同一データを AoS（`irFreqDomain`, `fdlBuf`）と SoA（`irFreqReal/irFreqImag`, `fdlReal/fdlImag`）の両方で保持している。SoA が主演算パスとして使用されており、AoS は主にフィルタ適用時の中間スクラッチとして使用される。

SoA 直接計算に完全移行することで、AoS バッファを削減する。

### 8.2  設計方針

AoS (`irFreqDomain`) は以下の理由により **完全には削除しない**:
- FFT/IFFT は CCS 形式（インターリーブ複素数）で動作 — AoS が自然
- `fdlBuf` は Forward FFT の出力先として必要
- `accumBuf` は IFFT 前の一時バッファとして必要

**削減対象**: `irFreqDomain` — フィルタ適用を SoA 直接計算に移行すれば IR 側の AoS は不要

### 8.3 変更ファイル

| # | ファイル | 変更内容 |
|:-:|:---------|:---------|
| P3/11 | `src/MKLNonUniformConvolver.cpp` | `applySpectrumFilter()` — SoA 直接適用 |
| P7/11 | `src/MKLNonUniformConvolver.cpp` | Air Absorption テール減衰 — SoA 直接適用 |
| P8/11 | `src/MKLNonUniformConvolver.cpp` | `processLayerBlock()` L0 — FDL/IR SoA 化 |
| P9/11 | `src/MKLNonUniformConvolver.cpp` | `Add()` L1/L2 — FDL 書出 SoA 化 |
| P10/11 | `src/MKLNonUniformConvolver.cpp` | `Add()` L1/L2 — 分散積算 SoA 化 |
| | `src/MKLNonUniformConvolver.h` | Layer 構造体から `irFreqDomain` 削除 |

### 8.4 リスク

| リスク | 重大度 | 対策 |
|:-------|:-------|:------|
| ポインタオフセット計算不整合 → 即時クラッシュ | **最高** | 全パッチをアトミックに適用、回帰テスト必須 |
| vdMul の引数間違い → 意図しないメモリ破壊 | 高 | 単体テストで出力値一致確認 |
| Prefetch 最適化の劣化 | 低 | SoA 化後も prefetch 維持可能 |
| IFFT 前のバッファ不整合 | 高 | accumBuf は AoS 残置、interleave 処理維持 |

### 8.5 実装順序

実装は `plan.md` の 11 パッチを以下の 3 パッケージに分割してアトミックに適用:

1. **安全パッケージ1**: Message スレッドフィルタの SoA 化（Patch 3, 7）
   - 非 RT パスのみ、比較的安全
2. **安全パッケージ2**: Audio スレッド FDL の SoA 化（Patch 8, 9, 10）
   - RT パス、要入念なテスト
3. **安全パッケージ3**: irFreqDomain 削除と影響箇所修正（Patch 4, 11 等）
   - AoS 削除、最終確認

各パッケージ実施後に ctest で回帰検証を行う。

---

## 9. Phase 5: 未計装 mkl_malloc 追跡（P3）

### 9.1 概要

以下の生 `mkl_malloc` 呼び出しが `DIAG_MKL_MALLOC` 未対応であり、OtherPrivate に吸収されている。

### 9.2 対象箇所

**MKLNonUniformConvolver.cpp（5箇所）**:

| バッファ | 推定サイズ | 用途 |
|:---------|:----------|:------|
| `reusableGain` | ~KB | applySpectrumFilter |
| `impulseForFft` | ~MB | IR コピー (ScopedAlignedPtr) |
| `tempTime` | KB~MB | IPP FFT 一時 |
| `tempFreq` | KB~MB | IPP FFT CCS |
| `swapSoA` | ~MB | SoA 変換一時 |
| `gainReal` | ~KB | applySpectrumFilter |

**他のファイル**:

| ファイル | 行 | 推定サイズ | 用途 |
|:---------|:---|:----------|:------|
| `CacheManager.cpp:190` | ~MB | IR キャッシュ |
| `CacheManager.cpp:228` | ~MB | IR キャッシュコピー |
| `IRConverter.cpp:187` | ~MB | IR 変換一時 |
| `AlignedAllocation.h:15,26` | ~MB | 汎用アライン確保 |

### 9.3 対応方針

- 各 mkl_malloc を `DIAG_MKL_MALLOC` でラップ
- 対応する mkl_free を `DIAG_MKL_FREE` に置換
- 既存 `freeTracked<T>()` との統合を検討

### 9.4 期待効果

直接的なメモリ削減効果はない（**計装の完全化**による透明性向上が目的）。
調査後、これらのバッファが予想以上に大きいことが判明した場合、別途削減策を検討。

---

## 10. 検証計画

### 10.1 各 Phase の検証項目

#### Phase 1: advanceFade 配線

| # | 確認項目 | 確認方法 | 期待値 | 優先度 |
|:-:|:---------|:---------|:-------|:-------|
| 1 | コンパイル成功 | `build.bat Release icx nopause` | ✅ | 必須 |
| 2 | advanceFade が毎 callback で呼ばれる | DIAG ログ | 1 callback = 1 call | 必須 |
| 3 | advance が remaining を正しく減少 | ログ解析 | remaining → 0 | 必須 |
| 4 | tryCompleteFade() が true を返す | DIAG ログ | fade 完了時 1 回のみ | 必須 |
| 5 | lifecycle(retire) > 0 | MEM_SNAP | 従来 0 → 増加 | 必須 |
| 6 | pendingRetireCount 増加 | MEM_SNAP pend | > 0 | 推奨 |
| 7 | DSPCore liveCount 減少 | MEM_SNAP | 2 (current+fading) | 推奨 |
| 8 | Private Memory 減少 | MEM_SNAP Priv | 減少を確認 | 推奨 |
| 9 | 音声品質に影響なし | 主観評価 | ノイズ・クリックなし | 必須 |
| 10 | 3 分間安定動作 | 長時間実行 | クラッシュ・ハングなし | 必須 |

#### Phase 2: MEM_SNAP 強化

| # | 確認項目 | 確認方法 | 期待値 |
|:-:|:---------|:---------|:-------|
| 1 | フォーマット文字列出力確認 | ログ | `live=%u(%u/%u)` 形式 |
| 2 | StereoConvolver liveCount 正値 | ログ | 実行状態と一致 |
| 3 | DSPCore liveCount 正値 | ログ | commit 数と一致 |

#### Phase 3: ブロックサイズ最適化

| # | 確認項目 | 確認方法 | 期待値 |
|:-:|:---------|:---------|:-------|
| 1 | processingBlockSize 低減確認 | DIAG ログ | 524288 → 適切値 |
| 2 | 動作安定性 | ctest | 全テスト PASS |
| 3 | Private Memory 削減 | MEM_SNAP | 初回 prepare 後 555MB → ??? |

#### Phase 4: AoS→SoA

| # | 確認項目 | 確認方法 | 期待値 |
|:-:|:---------|:---------|:-------|
| 1 | コンパイル成功 | 全ビルド構成 | ✅ |
| 2 | 出力値一致（回帰なし） | ctest | 全 PASS |
| 3 | Private Memory 削減 | MEM_SNAP | OtherPrivate 減少 |
| 4 | CPU 性能劣化なし | 負荷テスト | 同程度以上 |
| 5 | 長時間安定動作 | 実動作確認 | ハング・ノイズなし |

### 10.2 統合テスト手順

```
Step 1: ビルド
  build.bat Debug icx nopause -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1
  → コンパイルエラーがゼロであることを確認

Step 2: 単体テスト
  cd build && ctest -C Debug --output-on-failure
  → 全テスト PASS（既知の除外対象を除く）

Step 3: 実動作確認
  DAW またはスタンドアロンで 3 分以上動作
  → クラッシュ・ノイズ・クリックなし

Step 4: ログ解析
  MEM_SNAP の各フィールドを確認
  → lifecycle(retire) > 0, pendingRetireCount > 0, liveCount 減少

Step 5: 性能測定
  CPU 使用率、メモリ使用量を記録
  → 改修前後の比較
```

### 10.3 判定基準

| 判定 | 条件 |
|:-----|:------|
| ✅ **PASS** | 全テスト PASS + 実動作問題なし + 全確認項目期待値と一致 |
| ⚠️ **CONDITIONAL PASS** | 全テスト PASS + 実動作問題なし + 一部確認項目未達（但し副作用なし） |
| ❌ **FAIL** | テスト失敗 / クラッシュ / 音声劣化 |
| 🔄 **ROLLBACK** | 上記 FAIL の場合、該当 Phase をロールバックし原因調査 |

---

## 11. リスク評価とロールバック戦略

### 11.1 リスクマトリクス

| リスク | 確率 | 影響 | 対策 |
|:-------|:-----|:------|:------|
| advanceFade 配線位置誤りによる DSP 処理遅延 | 低 | 高 | 配線位置を DSP 完了直後に限定。副作用のない advance() 関数 |
| advance() + tryComplete() の Completed→Idle CAS 競合 | 低 | 低 | advance は Audio Thread 専用、tryComplete は Timer 専用のため競合なし |
| MEM_SNAP フォーマット不整合 | 低 | 低 | 変更直後にログ確認 |
| AoS→SoA ポインタ計算誤り | 中 | **最高** | 全パッチのアトミック適用 + 全テスト通過必須 |
| 524288 最適化での OS=8 コーナーケース | 中 | 高 | Phase 1 完了後に要否再評価。影響範囲の全数調査 |

### 11.2 ロールバック手順

各 Phase は独立したコミットとする。問題発生時は該当 Phase のみ Git revert 可能:

```bash
# Phase N をロールバック
git revert <phase-N-commit-hash>
```

**ただし**: Phase 4（AoS→SoA）の複数パッチは同一コミットにまとめる（部分 revert は不可）。

### 11.3 段階的リリース方針

```text
Phase 1 完了 → 実動作確認（3分） → 問題なければ継続
       ↓
Phase 2 完了 → ログ確認 → 問題なければ継続
       ↓
Phase 3 検討 → Phase 1 の効果測定後、必要性を再評価
       ↓
Phase 4 完了 → 全テスト + 実動作確認 → 問題なければリリース
       ↓
Phase 5 完了 → ログ確認 → 情報収集のみ
```

---

## 12. スケジュール案

### 12.1 推定期間

| Phase | 作業内容 | 推定工数 | 依存 |
|:------|:---------|:---------|:-----|
| **Phase 1** | advanceFade 配線 | 1-2 時間 | なし |
| | コード変更（2 ファイル） | 30 分 | |
| | ビルド | 30 分 | |
| | 実動作確認 + ログ解析 | 1-2 時間 | |
| **Phase 2** | MEM_SNAP liveCount 追加 | 30 分 | Phase 1 完了後が望ましい |
| **Phase 3** | ブロックサイズ調査 + 実装 | 4-8 時間 | Phase 1 の効果測定後判断 |
| **Phase 4** | AoS→SoA（11 パッチ） | 8-16 時間 | Phase 1-3 完了後 |
| | コード変更 | 4-8 時間 | |
| | テスト + 検証 | 4-8 時間 | |
| **Phase 5** | 未計装 mkl_malloc 追跡 | 2-4 時間 | Phase 4 後 |
| **合計** | | **16-32 時間** | |

### 12.2 クリティカルパス

```
Phase 1 → Phase 2 → Phase 3(判断) → Phase 4 → Phase 5
  [必須]      [任意]      [Phase1後判断]   [大規模]    [任意]
```

### 12.3 マイルストーン

| マイルストーン | 期日（目標） | 成果物 |
|:---------------|:------------|:-------|
| M1: advanceFade 配線完了 | Day 1 | `lifecycle(retire) > 0` 確認 |
| M2: メモリ効果確定 | Day 2 | Private Memory 削減量実測値 |
| M3: 監視強化完了 | Day 2 | MEM_SNAP 完全出力 |
| M4: AoS→SoA 判断 | Day 3 | Phase 3 実施判断 + 設計確定 |
| M5: AoS→SoA 完了 | Day 5 | 全テスト PASS + メモリ削減確定 |
| M6: 全 Phase 完了 | Day 6 | 統合レポート提出 |

---

## 13. 付録: 調査で発見された副次的知見

### 13.1 ISR RetireRouter カウンタのデッドコード

`ISRRetireRouter` には以下のカウンタが存在するが、**値を更新するコードがない（デッドコード）**:

- `m_pendingRetireBytes_` — `pendingRetireBytes()` で読み取り
- `m_trackedPendingEntries_` — `trackedPendingEntries()`, `trackedRatio()` で読み取り

これらのカウンタは MEM_SNAP の `trBytes` / `tr` フィールドのソースだが、常に 0 を返すため
意味のある情報を提供していない。Phase 3 以降で改修対象としてもよい。

### 13.2 DSPGuard RAII は rebuild-obsolete DSPCore を解放できない（新発見）

**修正: 従来「DSPGuard は rebuild-obsolete DSPCore を正しく EBR retire している」としていたが、コード解析の結果これは誤りであることが判明した。**

`DSPGuard::retire()` → `DSPLifetimeManager::retire()` の経路で、最初に `retireDSPHandleForRuntime(dsp)` が呼ばれる。この関数は `dsp` が `runtimeDSPHandleMap_` に登録されている場合のみ成功する。rebuild-obsolete な DSPCore は commit パスに到達しない（`enqueuePublicationIntentForRuntimeCommit()` が呼ばれない）ため、handle map に未登録であり、`retireDSPHandleForRuntime()` は false を返す。

その結果:
1. `DSPLifetimeManager::retire()` は **EBR enqueue を行わずに return** する
2. `runtimeRetireCount` カウンタもインクリメントされない
3. DSPCore のメモリ (`aligned_make_unique` + `release()`) は **解放されずリークする**

**影響**: ログ上の rebuild obsolete 3 回分の DSPCore（推定 3 × 220MB = 660MB）はメモリリークしている。
これは advanceFade 未配線とは独立した二次的なリークであり、Phase 1 実施後も残存する。

**対応方針**:
- 本件は「retire 経路の handle map 依存」というアーキテクチャ上の課題であり、advanceFade 配線とは独立して対処が必要
- 選択肢: (a) rebuild-obsolete 検出時に handle を登録してから retire する、(b) 直接 `destroyDSPCoreNode()` を呼んで即時解放する、(c) 未登録 DSPCore 用の別 retire 経路を用意する
- ただし実メモリ使用量への影響は rebuild-obsolete 3回分（〜660MB推定）であり、advanceFade 配線で解消されるコミット済み未 retire DSPCore の〜800MB とは別に計上すべき
- **Phase 1 実施後のメモリ削減効果を実測し、残存する増加分が本リークに相当するかを確認した上で、対応の優先度を判断する**

### 13.3 SnapshotCoordinator::updateFade() も 0 call sites

`advanceFade()` と同様に `updateFade(float& outAlpha)` も呼び出し元が存在しない。
こちらは `SnapshotCoordinator` 内で定義されているが、現在は未使用。

### 13.4 CrossfadeRuntime の寿命管理

`CrossfadeRuntime`（エンジン遷移用）と `SnapshotCoordinator`（パラメータ遷移用）は
**完全に独立した機構**である。これらが混同されていたことが過去の分析で誤解を生んだ。

- `[XFADE]` ログ → `CrossfadeRuntime`（DSP エンジン遷移）
- `[VERIFY]` ログ → `SnapshotCoordinator`（EQ/NS/AGC パラメータ遷移）
- MEM_SNAP → 両方に依存しない独立した定期サンプリング

### 13.5 FFT/IR バッファの再初期化問題

初回 prepare 時に `processingBlockSize=524288` で作成されたバッファは、
後続の prepare（2048）で解放されるが、初回 commit 後に aging していた場合
解放タイミングが retire に依存するため、advanceFade 未配線下では解放されない。

---

## 改訂履歴

| 版 | 日付 | 改訂内容 | 著者 |
|:---|:-----|:---------|:------|
| 1.0 | 2026-07-10 | 初版作成 | GitHub Copilot |

---

## 参考: 各ファイルへの変更命令サマリ

### Phase 1 コマンド一覧

```bash
# SnapshotFadeState.h: FadeState::Completed 追加
#   advance(): remaining==0 → state=Completed
#   tryComplete(): Completed→Idle の CAS に変更
# AudioBlock.cpp: m_coordinator.advanceFade(numSamples) を 1 行追加
# Timer.cpp: 変更不要（既存の tryCompleteFade() が動作する）
# CrossfadeRuntime.h: 変更不要（責務分離を維持）
```

### Phase 2 コマンド一覧

```bash
# AudioEngine.Timer.cpp: MEM_SNAP フォーマット + StereoConvolver/DSPCore liveCount
```

### Phase 4 パッチ一覧（plan.md から引用）

```bash
# Patch 3: applySpectrumFilter() SoA 直接計算
# Patch 7: Air Absorption 減衰 SoA 直接計算
# Patch 8: processLayerBlock() L0 SoA 化
# Patch 9: Add() L1/L2 FDL 書出 SoA 化
# Patch 10: Add() L1/L2 分散積算 SoA 化
# Patch 4,11: irFreqDomain 削除 + 構造体変更
```

### Phase 5 コマンド一覧

```bash
# MKLNonUniformConvolver.cpp: 6 箇所の生 mkl_malloc → DIAG_MKL_MALLOC
# CacheManager.cpp: 2 箇所
# IRConverter.cpp: 1 箇所
# AlignedAllocation.h: 2 箇所
```
