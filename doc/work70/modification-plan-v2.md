# ConvoPeq メモリ肥大化 改修計画書 v3.0

**日付**: 2026-07-10
**バージョン**: 3.0
**対象**: work70 (メモリ 2.5GB 肥大化問題)
**前版**: v2.0 (2026-07-10)

## 凡例

本文中では以下のラベルで**事実・仮説・設計案**を明確に区別する:

| ラベル | 意味 | 例 |
|:-------|:-----|:----|
| ✅ **FACT** | コード解析またはログ解析で確定した事実 | `advanceFade()` に呼び出し元が存在しない |
| 🔍 **HYPOTHESIS** | ログやコードから強く示唆されるが未確定 | `lifecycle(retire)=0` の原因が handle map 未登録である |
| 💡 **PROPOSAL** | 改善のための設計案（複数案あり得る） | `FadeState::Completed` の追加 |
| ⚠️ **CAVEAT** | 注意・制約・未解決の懸念 | memory_order 変更リスク |

---

## 目次

1. [背景と目的](#1-背景と目的)
2. [現状: 確定事実の整理](#2-現状-確定事実の整理)
3. [現状: 未確定の仮説](#3-現状-未確定の仮説)
4. [優先順位と全体ロードマップ](#4-優先順位と全体ロードマップ)
5. [P0-1: lifecycle(retire)=0 の原因特定](#5-p0-1-lifecycleretire0-の原因特定)
6. [P0-3: RuntimeDSPHandleMap Dump](#6-p0-3-runtimedsphandlemap-dump)
7. [P0-2: advanceFade 配線の要否検証](#7-p0-2-advancefade-配線の要否検証)
8. [P1: advanceFade 配線（最小実装）](#8-p1-advancefade-配線最小実装)
9. [P1: MEM_SNAP 監視強化](#9-p1-mem_snap-監視強化)
10. [P2-1: 初回ブロックサイズ調査](#10-p2-1-初回ブロックサイズ調査)
11. [P2-2: 初回ブロックサイズ実装](#11-p2-2-初回ブロックサイズ実装)
12. [設計方針: SnapshotCoordinator と CrossfadeRuntime の責務分離](#12-設計方針-snapshotcoordinator-と-crossfaderuntime-の責務分離)
13. [P3: AoS→SoA メモリ最適化](#13-p3-aosoa-メモリ最適化)
14. [P4: 未計装 mkl_malloc 追跡](#14-p4-未計装-mkl_malloc-追跡)
15. [付録: 調査で発見された副次的知見](#15-付録-調査で発見された副次的知見)

---

## 1. 背景と目的

### 1.1 問題

ConvoPeq のプロセスメモリ使用量が、通常動作時に Private Memory 約 **2.5GB** に達する。

### 1.2 調査の成果

| 項目 | 状態 | エビデンス |
|:-----|:-----|:-----------|
| MKL Convolution は 35MB (1.4%) のみ | ✅ **FACT** | MEM_SNAP `alloc=35MB`, `Other=2442MB` |
| `advanceFade()` が 0 call sites | ✅ **FACT** | grep 確認: `.cpp` からの呼び出しなし |
| `SnapshotCoordinator::startFade()` は実行された | ✅ **FACT** | `[VERIFY] EQ createdHash=` ログ |
| `lifecycle(retire)=0` | ✅ **FACT** | MEM_SNAP ログ |
| `DSPCORE_PREPARE` 6回実行 | ✅ **FACT** | DIAG ログ |
| DSPTransition に独立した retire 経路が存在する | ✅ **FACT** | `DSPTransition.h:49` `onPublishCompleted()` |
| `SnapshotCoordinator` と `CrossfadeRuntime` は別機構 | ✅ **FACT** | コード確認: 前者はパラメータ遷移、後者はエンジン遷移 |
| `AudioEngine::retireDSP()` に呼び出し元がない | ✅ **FACT** | grep 確認: 0 callers — デッドコード |
| `DSPLifetimeManager::retire()` が事実上の retire 経路 | ✅ **FACT** | DSPTransition/Timer/RebuildDispatch 全てがこちらを使用 |
| rebuild-obsolete → DSPGuard → DSPLifetimeManager::retire → retireDSPHandleForRuntime | ✅ **FACT** | `retireDSPHandleForRuntime()` が handle map 未登録で false を返す |

### 1.3 本計画書のスタンス

本計画書は v1.0 からの改訂版である。以下の方針で構成する:

- **事実（FACT）と仮説（HYPOTHESIS）と設計案（PROPOSAL）を明確に分離**
- 各 Phase の前に「なぜこの変更が必要か」の根拠を FACT ベースで示す
- 設計案には複数案がある場合は代替案も併記する
- 優先順位は「原因特定 → 症状修正 → 最適化」の順とする

---

## 2. 現状: 確定事実の整理

### 2.1 メモリ使用量

```
Private 2477MB の内訳（確定値は MKL 35MB のみ）:

  ┌─ MKL Convolution (tracked)  =   35MB  (1.4%)  ☆ 確定値
  ├─ 残り (DSPCore/JUCE/IR/他)  ≈ 2442MB (98.6%) △ 内訳はすべて推定値
```

### 2.2 advanceFade 未呼び出し

✅ **FACT**: `SnapshotCoordinator::advanceFade()` が 0 call sites。

```
SnapshotCoordinator.cpp:43  void advanceFade(int numSamples) noexcept
                     → この関数を呼ぶコードが存在しない

SnapshotFadeState の状態遷移:
  start(fadeSamples)
    → state=FadingIn, remaining=fadeSamples (>0)
      → advance() が呼ばれない
        → remaining が永遠に初期値のまま
          → tryComplete(): remainingCount() > 0 → false
```

### 2.3 lifecycle(retire)=0

✅ **FACT**: MEM_SNAP で `lifecycle(retire)=0` が一貫して観測されている。

✅ **FACT**: `runtimeRetireCount` がインクリメントされる経路は以下:

| コードパス | インクリメント条件 | 実使用 |
|:----------|:------------------|:-------|
| `AudioEngine::retireDSP()` (AudioEngine.h:3786) | `retireDSPHandleForRuntime()` 成功後 | **0 callers — デッドコード** |
| `DSPLifetimeManager::retire()` (DSPLifetimeManager.h:55) | `retireDSPHandleForRuntime()` 成功 + EBR enqueue 成功後 | DSPTransition/Timer/RebuildDispatch で使用 |

つまり **`runtimeRetireCount` が 0 の理由は、`DSPLifetimeManager::retire()` が `retireDSPHandleForRuntime()` で false を返されているか、そもそも呼ばれていない** のいずれかである。

### 2.4 publish 開始点の2種類

✅ **FACT**: publish の最終的な公開処理は共通の `RuntimePublicationCoordinator` を通るが、開始点として2種類の経路がある:

| 開始点 | 使用箇所 | DSPTransition 経由 | Handle 登録 | retire 発生 |
|:-------|:---------|:------------------|:------------|:------------|
| Coordinator 直接呼び出し | Init, PrepareToPlay, ReleaseResources, Timer(fadeCompleted), Transition | ❌ No | ❌ No | ❌ No |
| Orchestrator 経由 | RebuildDispatch → enqueuePublicationIntentForRuntimeCommit | ✅ Yes | ✅ Yes(newDSP) | ⚠️ oldDSP の handle 次第 |

両者とも最終的には `RuntimePublicationCoordinator::publishWorld()` で公開処理が行われる。違いは **DSPTransition を経由して oldDSP の retire まで行うかどうか** のみ。

🔍 **HYPOTHESIS**: Timer 経由の fadeCompleted publish は Coordinator 直接呼び出しのため、DSPTransition を経由せず oldDSP の retire が行われない。ただし fadeCompleted ブロック内には `lifetimeMgr.retire(done)` の直接呼び出しが存在する（Timer.cpp:878）ため、retire 自体は行われている可能性がある。

### 2.5 rebuild-obsolete と EBR retire 経路

✅ **FACT**: rebuild-obsolete 検出時の DSPGuard デストラクタは `DSPLifetimeManager::retire(ptr)` を呼ぶ。

✅ **FACT**: rebuild-obsolete な DSPCore は `enqueuePublicationIntentForRuntimeCommit()` に到達しないため `registerDSPHandleForRuntime()` が呼ばれず、handle map に未登録。したがって `DSPLifetimeManager::retire()` → `retireDSPHandleForRuntime()` は false を返し、**EBR retire 経路からは解放されない**。

```cpp
// RebuildDispatch.cpp: DSPGuard のデストラクタ
~DSPGuard() {
    DSPLifetimeManager lifetimeMgr(*owner);
    lifetimeMgr.retire(ptr);
    // → retireDSPHandleForRuntime(ptr) → handle map に未登録 → false → return
    // → EBR enqueue されない
}
```

🔍 **HYPOTHESIS**: EBR 経路から解放されないため、他経路（デストラクタや明示的解放）がなければメモリリークとなる。rebuild-obsolete 3回分の DSPCore（推定 ~660MB）のリーク可能性があるが、**最終的なリークであるかは他経路からの解放有無を含めた総合判断が必要**。

⚠️ **CAVEAT**: `DSPCore` は `aligned_make_unique` で確保され `release()` で生ポインタ化されている。現時点で確認できた解放経路は EBR enqueue による `destroyDSPCoreNode()` のみである。handle map 未登録により EBR enqueue が行われない場合、この解放経路が機能しない。ただしコード全体を完全に網羅したわけではないため、「唯一の解放経路」とは断定しない。

### 2.6 DSPTransition の retire 経路 (独立経路の実態)

✅ **FACT**: `DSPTransition::onPublishCompleted()` は以下で `lifetime.retire()` を呼ぶ:

| 条件 | 行 | 内容 |
|:-----|:---|:------|
| Emergency Override | 63 | oldDSP を即時 retire |
| Crossfade 有 | 98 | fading slot 置き換え後 prev を retire |
| Crossfade 不要 | 111 | oldDSP を即時 retire |

✅ **FACT**: `DSPTransition::onTransitionComplete()` (Timerから呼ばれる) も line 131 で retire を呼ぶ。

🔍 **HYPOTHESIS**: DSPTransition 経路の retire が `retireDSPHandleForRuntime()` で失敗している可能性がある（oldDSP が Coordinator 直接 publish 由来のため handle map 未登録）。

---

## 3. 現状: 未確定の仮説

### 3.1 lifecycle(retire)=0 の原因 — ログ解析により仮説 B が最有力に

2026-07-10 の追加ログ解析により、以下の事実が確定:

✅ **FACT**: Orchestrator publish は 3 回実行され、全て SUCCEEDED（gen=1, gen=4, gen=8）
✅ **FACT**: DSPTransition は上記 3 回の Orchestrator publish で呼ばれている
✅ **FACT**: lifecycle(pub)=5 に対し lifecycle(ret)=0 が不変 — **retireDSPHandleForRuntime() は 3 回全てで false を返している**
✅ **FACT**: Coordinator 直接 publish（gen=3, prepareToPlay）は DSPCore を `runtimeDSPHandleMap_` に **登録しない**
✅ **FACT**: この未登録 DSPCore が「current」になった後、後続の Orchestrator publish の DSPTransition が oldDSP を retire しようとしても handle map に見つからず失敗する

**したがって仮説 B（handle map 未登録）が原因として確定した。** 残りの仮説 A/C/D/E は排除された。

```text
Publish sequence:
  gen=2: Orchestrator gen=1 → newDSP_A registered → published → current=A
  gen=3: Coordinator direct  → newDSP_B NOT registered → published → current=B ★問題
  gen=4: Orchestrator gen=4 → newDSP_C registered → DSPTransition(oldDSP=B)
    → retireDSPHandleForRuntime(B): handle map に未登録 → false
    → lifecycle(retire)=0 のまま
  gen=5: Orchestrator gen=8 → newDSP_D registered → DSPTransition(oldDSP=C?)
    → retireDSPHandleForRuntime(C?): handle map に未登録 → false
    → lifecycle(retire)=0 のまま
```

⚠️ **CAVEAT**: Coordinator 直接 publish は Init/PrepareToPlay/ReleaseResources/Timer(fadeCompleted)/Transition の各所で使用される。gen=3 以外の Coordinator 直接 publish も同様の handle 未登録問題を引き起こす可能性がある。Timer(fadeCompleted) の publish が gen=? で発生したかは今回のログでは未確認。

### 3.2 advanceFade が retire 連鎖の唯一のボトルネックとは断定できない

✅ **FACT**: DSPTransition は SnapshotCoordinator とは独立した retire 経路を持つ。

したがって:

```
advanceFade 未呼び出し
  → SnapshotCoordinator の fade は進行しない (確定)
    → Timer 経路の retire() はブロックされる (確定)
      → しかし DSPTransition 経路の retire() は独立して動作可能 (確定)
```

🔍 **HYPOTHESIS**（v2.0）: DSPTransition 経路が実際に動作していれば、advanceFade 未配線でも一部の retire は発生するはず。lifecycle(retire)=0 は DSPTransition 経路も機能していないことを示唆する。

✅ **FACT**（v3.0 確定）: DSPTransition 経路は **動作している**。Orchestrator publish 3 回全てで DSPTransition::onPublishCompleted() が呼ばれ、lifetime.retire(oldDSP) が実行されている。しかし `retireDSPHandleForRuntime(oldDSP)` が handle map 未登録により false を返している。したがって lifecycle(retire)=0 の原因は **DSPTransition の停止ではなく handle map 未登録** である。

⚠️ **CAVEAT**: `RuntimePublicationOrchestrator` → `DSPTransition` → `retire` の経路は advanceFade とは**完全に独立**している。advanceFade の有無にかかわらず、この経路が正しく動作すれば oldDSP の retire は発生する。「advanceFade を直せば全部直る」という前提は誤り。advanceFade は **Timer 経路の publish に伴う retire** 専用であり、Orchestrator 経路の retire には影響しない。

### 3.3 DSPCore 1個あたりのメモリコスト

🔍 **HYPOTHESIS**: DIAG ログから EQ buffer 3種 (scratch=32MB, msWorkBuffer=16MB, dryBypass=8MB) は確定しているが、合計 ~220MB/インスタンスは **未確定の概算**。

### 3.4 Completed 状態追加の要否

✅ **FACT**: 現在の `tryComplete()` は `remainingCount() > 0` を直接チェックし、CAS `FadingIn→Idle` を行う。advance() が release で remaining=0 を書き込めば、既存の設計で完了検出できる。

💡 **PROPOSAL A** (v1.0): `FadeState::Completed` を追加 — **不要。** 既存の remaining==0 チェックで十分。

💡 **PROPOSAL B** (v1.0): `memory_order_relaxed` に変更 — **危険。** 現在の release/acquire は remaining→state の HB を形成しており、relaxed 化すると Timer 側が remaining==0 を正しく観測できない可能性がある。

---

## 4. 優先順位と全体ロードマップ

### 4.1 優先度定義

| 優先度 | 定義 |
|:-------|:------|
| **P0** | **原因特定**: コード解析と検証により仮説を FACT に昇格させる。コード変更を伴わない。 |
| **P1** | **症状修正**: 確定した原因に対する最小限の修正。 |
| **P2** | **改善・最適化**: 影響範囲を評価した上で実施。 |
| **P3** | **構造的最適化**: 大規模リファクタリング。回帰リスクが高い。 |
| **P4** | **情報収集**: 直接の削減効果はないが将来の判断材料。 |

### 4.2 優先順位（v3.0 確定版）

| 優先度 | Phase | 内容 | 変更ファイル数 | リスク | v1.0 比 |
|:-------|:------|:------|:-------------|:-------|:--------|
| **P0** | 完了 | lifecycle(retire)=0 の原因特定 — **原因確定: handle map 未登録**（Coordinator direct publish に起因） | 0 (解析済) | なし | **確定済** |
| **P1** | 実装 | Coordinator direct publish に handle 登録を追加（DSPTransition retire 回復） | 1-2 | 中 | **新設（v2.0のP1を置換）** |
| **P1** | 実装 | advanceFade 配線（AudioBlock.cpp 1行） | **1** | 低 | 縮小 |
| **P1** | 実装 | MEM_SNAP 監視強化 (Stereo/DSPCore liveCount) | 1 | 低 | 同 |
| **P2-1** | 調査 | 初回 BlockSize (524288) 調査と影響分析 | 0 (分析) | なし | 降格+分割 |
| **P2-2** | 実装 | BlockSize 最適化（P2-1結果次第） | TBD | 中 | 降格 |
| **P3** | 実装 | AoS→SoA (11パッチ) | 2+ | 高 | 降格 |
| **P4** | 実装 | 未計装 mkl_malloc DIAG 化 | 4 | 低 | 同 |

### 4.3 v1.0/v2.0 からの主な変更点

| 変更点 | v1.0 | v2.0 | v3.0 | 理由 |
|:-------|:-----|:------|:------|
| Phase 1 実装規模 | 6ファイル変更, bool 化, CrossfadeRuntime 改造 | 実質1ファイル, 1行追加のみ | P1分割: (a) Coordinator direct handle登録 (b) advanceFade配線 | lifecycle(retire)=0 の原因確定に伴う再編 |
| P0 調査 | なし | P0-1/P0-2/P0-3 追加 | **完了: 原因確定** | ログ解析の結果、仮説 B が確定。残りの仮説は排除 |
| 優先順位 | advanceFade P0, BlockSize P1 | advanceFade P1, BlockSize P2 | **Coordinator direct handle 登録が最優先 P1** | これが ret=0 の直接原因。advanceFade は副次的 |
| Completed | 設計案として採用 | 不要と判断 | 不要（確定） | 既存の remaining==0 チェックで十分 |
| memory_order | release 提案 | 現状維持（relaxed 化は却下） | 現状維持（確定） | 現状の release/acquire が正しい同期を提供 |

---

## 5. P0-1: lifecycle(retire)=0 の原因特定

### 5.1 目的

`runtimeRetireCount` が 0 の直接原因を確定させる。**コード変更なし** — ログ分析と DIAG 出力の追加のみ。

### 5.2 調査手順（5段階の原因切り分け — register 側も含む）

retire が機能しない原因を特定するため、**register 側の世代管理**も含めて段階的に調査する:

#### 第1段階: `registerDSPHandleForRuntime()` は呼ばれているか

retire の前提として handle map にエントリが存在する必要がある。register が行われていない場合、retire 以前の問題。

| # | 調査項目 | 方法 |
|:-:|:---------|:------|
| 1-1 | `registerDSPHandleForRuntime()` が呼ばれた回数とそのDSPCore* | 関数内に DIAG ログ追加（DSPCore*, generation, slot を出力） |
| 1-2 | 各 publish 時点の `runtimeDSPHandleMap_` サイズ | map size をログ出力 |
| 1-3 | register された DSPCore* と generation の対応 | `dspHandleRuntime_.create()` の戻り値を記録 |

#### 第2段階: `DSPLifetimeManager::retire()` は呼ばれているか

retire の入口となる関数。これが呼ばれていなければ、handle map や enqueue 以前の問題。

| # | 調査項目 | 方法 |
|:-:|:---------|:------|
| 2-1 | `DSPLifetimeManager::retire()` が呼ばれた回数 | 関数先頭に DIAG ログを追加。DSPCore* と generation を出力 |
| 2-2 | DSPTransition::onPublishCompleted() が呼ばれているか | 関数内に DIAG ログ追加 or `[DIAG] trySubmit: publish SUCCEEDED/FAILED` ログ確認 |
| 2-3 | Timer の fadeCompleted ブロックが実行されているか | `[XFADE] completed` ログの有無を確認 |

#### 第3段階: `retireDSPHandleForRuntime()` は成功しているか

```cpp
bool retireDSPHandleForRuntime(DSPCore* dsp) {
    std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);
    const auto it = runtimeDSPHandleMap_.find(dsp);
    if (it == runtimeDSPHandleMap_.end()) return false;  // ← ここで false
    const auto handle = it->second;
    // handle.slot, handle.generation
    dspHandleRuntime_.retire(handle);
    dspHandleRuntime_.reclaim(handle);
    return true;
}
```

| # | 調査項目 | 方法 |
|:-:|:---------|:------|
| 3-1 | retireDSPHandleForRuntime() の戻り値 | false ケースで DIAG：「DSPCore* not found in map」 |
| 3-2 | register 時と retire 時の generation 一致 | retire 側で handle.generation を出力（register 側の generation と比較） |
| 3-3 | generation mismatch の有無 | register 時 gen=17 → retire 時 gen=18 のような世代ズレの検出 |
| 3-4 | oldDSP が handle map に存在するか | false ケースで map 内エントリ一覧をログ出力（最大数制限） |

#### 第4段階: `router_->enqueueRetire()` は成功しているか

| # | 調査項目 | 方法 |
|:-:|:---------|:------|
| 4-1 | `router_->enqueueRetire()` の戻り値 | 戻り値をログ出力（Success/QueuePressure/QueueFull/Shutdown） |
| 4-2 | `currentEpoch()` の値 | publishEpoch との相関確認 |
| 4-3 | どの publish 経路がどの gen で使われたか | Coordinator direct / Orchestrator の使用状況をログに出力 |

#### 第5段階: `runtimeRetireCount++` は実行されているか

| # | 調査項目 | 方法 |
|:-:|:---------|:------|
| 5-1 | `runtimeRetireCount` のインクリメント有無 | `fetchAddAtomic(runtimeRetireCount, ...)` の直前に DIAG ログを追加（DSPLifetimeManager.h:55, AudioEngine.h:3786） |
| 5-2 | インクリメント条件が満たされているか | `retireDSPHandleForRuntime()` 成功直後にしか進まないことを確認 |

#### 検証の全体フロー

```text
registerDSPHandleForRuntime() 呼ばれた? (DSPCore*, generation, slot)
  ↓ Yes (map に登録)
DSPLifetimeManager::retire() 呼ばれた?
  ↓ Yes
retireDSPHandleForRuntime() 成功? (lookup: DSPCore* match)
  ↓ → No: (a) not found (b) generation mismatch
router_->enqueueRetire() 成功?
  ↓ Yes
runtimeRetireCount++ 実行?
  ↓ Yes
→ lifecycle(retire) > 0 に変化
  ↓ No (いずれか)
→ 該当段階の原因に応じた対応
```

### 5.3 仮説検証マトリクス

| 仮説 | 検証条件 | 確定した場合の対応 |
|:-----|:---------|:-------------------|
| A: DSPLifetimeManager::retire() が呼ばれていない | 関数先頭でログ出力 | DSPTransition 経路の呼び出し条件を調査 |
| B: handle map 未登録 | retireDSPHandleForRuntime の false 確認 | 全 publish 経路で handle 登録を必須化 |
| C: enqueueRetire 失敗 | 戻り値確認 | EBR キューの状態を調査 |
| D: epoch 問題 | currentEpoch と publishEpoch の比較 | epoch 管理の設計確認 |
| E: Orchestrator 未使用 | Coordinator direct vs Orchestrator の使用比率 | Timer path の DSPTransition 適用を検討 |

### 5.4 期待成果

- lifecycle(retire)=0 の原因が A〜E のいずれかに確定する
- advanceFade 配線が本当に有効か否かが判断できる
- 不要なコード変更を回避できる

---

## 6. P0-3: RuntimeDSPHandleMap Dump（全レジスタ内容の可視化）

### 6.1 目的

P0-1 の調査に加えて、`runtimeDSPHandleMap_` の内容を publish 毎に可視化する。これにより register 側の世代・ポインタ・slot と retire 側の generation mismatch を直接確認できる。

### 6.2 調査内容

各 publish 完了後に以下の情報を DIAG ログに出力:

| # | 出力項目 | 内容 |
|:-:|:---------|:------|
| 1 | マップ全体のエントリ数 | `runtimeDSPHandleMap_ size` |
| 2 | 各エントリの `DSPCore*` | ポインタアドレス（0x1234ABCD） |
| 3 | 各エントリの `DSPHandle.slot` | レジストリスロット番号 |
| 4 | 各エントリの `DSPHandle.generation` | 世代番号（64bit） |
| 5 | 各エントリの DSPState | Active/Constructing/Retired/Reclaimed |
| 6 | publish 側の generation | `runtimeWorld->generation` |

### 6.3 期待される効果

以下の仮説を直接検証できる:

- **generation mismatch**: register 時 gen=17 → retire 時 gen=18 のような世代ズレ
- **handle map 枯渇**: レジストリスロットが 256 を使い切り新規登録できない
- **未登録 DSPCore**: retireDSP が参照する DSPCore* が map に存在しない
- **複数 oldDSP の残留**: 1回の publish で複数の oldDSP が retire されるべき状況

### 6.4 期待成果

- lifecycle(retire)=0 の原因が register 側にあるのか retire 側にあるのかを特定
- generation mismatch による retire 失敗の有無を確定
- handle map のライフサイクル全体の可視性向上

---

## 7. P0-2: advanceFade 配線の要否検証

### 7.1 目的

advanceFade の配線が retire 連鎖を回復することを検証する。**コード変更なし**で可能な限り判断する。

### 7.2 検証すべき因果連鎖

```
advanceFade() が毎 callback で呼ばれる (P1 実装後)
  → SnapshotFadeState::advance() が remaining を減少
    → remaining == 0 に到達
      → tryComplete(): remainingCount()==0 → CAS FadingIn→Idle → true
        → fadeCompleted ブロック実行
          → completeFade() → publishWorld + DSPLifetimeManager::retire()
            → pendingRetireCount > 0 → EBR reclaim → liveCount 減少
```

🔍 **HYPOTHESIS**: 上記の連鎖は論理的には正しいが、以下の要因で実際の効果が異なる可能性がある:

| 要因 | 影響 |
|:-----|:------|
| DSPTransition の独立経路が advanceFade なしでも retire を実行している | advanceFade 配線後の追加効果が小さい |
| 「fade 完了後の publish」が Coordinator 直接経路（DSPTransition なし）で行われる | retire が発生しない（Timer.cpp:900 の経路） |
| `completeFade()` 内の retire 対象が常に nullptr | retire が発生しない |

### 7.3 判断基準

P1 実装後に以下を**段階的**に確認する。成功条件は「前段階が全て PASS した上で次へ進む」こと。

#### 第1段階: advanceFade 正常動作の確認

| # | 確認段階 | 確認項目 | 合格基準 |
|:-:|:---------|:---------|:-------|
| 1-1 | **advanceFade の正常動作** | DIAG ログ | advanceFade が毎 callback で呼ばれ、remaining が正しく減少する |
| 1-2 | **remaining==0 到達** | DIAG ログ | remainingCount()==0 で tryComplete() が true を返す |
| 1-3 | **completeFade() 実行** | DIAG ログ or ブロック到達確認 | fadeCompleted ブロック内の completeFade() が実行される |

#### 第2段階: retire 連鎖の確認

| # | 確認段階 | 確認項目 | 合格基準 |
|:-:|:---------|:---------|:-------|
| 2-1 | **retire() 呼び出し** | DIAG ログ | fadeCompleted ブロック内で `DSPLifetimeManager::retire()` または `lifetimeMgr.retire()` が実行される |
| 2-2 | **pendingRetireCount 増加** | MEM_SNAP `pend` | retire キューにエントリが追加され、一時的に >0 になる |
| 2-3 | **EBR reclaim 実行** | MEM_SNAP `rec` | reclaim 試行が行われ、pending が減少する |
| 2-4 | **liveCount 減少** | MEM_SNAP `live` | NUC/DSPCore/Stereo liveCount が減少する |
| 2-5 | **runtimeRetireCount 増加** | MEM_SNAP `lifecycle(ret)` | ⚠️ P0-1 の原因が advanceFade 由来の場合のみ増加 |

⚠️ **CAVEAT**: `runtimeRetireCount` の増加は advanceFade 配線成功の必要条件ではない。`retireDSPHandleForRuntime()` が handle map 未登録で false を返す場合、retire() は呼ばれてもカウンタは増加しない。

したがって advanceFade 配線の成功判定は「**SnapshotCoordinator のフェードライフサイクルが完結すること**（advanceFade → remaining==0 → completeFade() まで到達）」とし、runtimeRetireCount の改善は「P0-1 で特定した原因が advanceFade 由来である場合の副次効果」として位置づける。

#### Phase 1 Success（全体成功条件）

| # | 条件 | 詳細 |
|:-:|:-----|:------|
| ✅ | ① advanceFade 正常動作 | 毎 callback で呼ばれ remaining 減少 |
| ✅ | ② completeFade() 実行 | fadeCompleted ブロック到達、completeFade() 成功 |
| ✅ | ③ retire() 呼び出し | DSPLifetimeManager::retire() が実行される |
| ⚠️ | ④ reclaim 実行 | EBR reclaim により pending 減少（publish 経路次第） |
| ⚠️ | ⑤ liveCount 減少 | DSPCore/NUC liveCount 減少（publish 経路次第） |

---

## 8. P1: advanceFade 配線（最小実装）

### 8.1 前提条件

- P0-1 で lifecycle(retire)=0 の原因が特定されていること
- P0-2 で advanceFade 配線が有効と判断されたこと

### 8.2 設計判断

✅ **FACT**: 現在の `SnapshotFadeState` の同期設計（release/acquire）は正しい。変更不要。

✅ **FACT**: `tryComplete()` は既に `remainingCount() > 0` を直接チェックしている。`FadeState::Completed` は不要。

✅ **FACT**: Timer は既に毎 callback で `tryCompleteFade()` を呼んでいる（line 843）。変更不要。

✅ **FACT**: `CrossfadeRuntime` は DSP エンジン遷移専用。完了通知を持ち込まない。

### 8.3 変更内容

**コード変更ファイル**: 1 ファイルのみ（AudioBlock.cpp）

#### 8.3.1 AudioBlock.cpp — 唯一の変更箇所

追加するコードは以下の1行のみ:

```cpp
// AudioEngine::getNextAudioBlock() 内、DSP 処理完了直後
// ★ work70: SnapshotCoordinator の fade 進行。
//   remaining を処理済みサンプル数だけ進める。
m_coordinator.advanceFade(numSamples);
```

**他の全ファイルは変更不要。**

### 8.4 変更ファイル一覧

| ファイル | 変更内容 | 行数 |
|:---------|:---------|:-----|
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | `getNextAudioBlock()` 内に `m_coordinator.advanceFade(numSamples)` 1行追加 | +1 |

**上記1ファイルのみがコード変更の実体。** SnapshotFadeState.h / SnapshotCoordinator.h / .cpp / Timer.cpp / CrossfadeRuntime.h は変更不要。

### 8.5 v1.0 からの変更点

| 項目 | v1.0 | v2.0 | 理由 |
|:-----|:-----|:------|:------|
| `FadeState::Completed` 追加 | 必須 | 不要 | 既存の remaining==0 チェックで完了検出可能 |
| `advance()` bool 化 | 必須 | 不要 | 戻り値を使う設計は CrossfadeRuntime への責務混入を招く |
| `memory_order_relaxed` 化 | 推奨 | 却下 | 現状の release/acquire が正しい同期を提供 |
| 完了通知 atomic 追加 | 必要 | 不要 | state の release store で十分 |
| 変更ファイル数 | 6 | **実質1** | 最小変更を徹底 |

### 8.6 期待効果

期待効果は P0 の調査結果に依存する:

| P0-1 の結果 | P1 実施後に期待される状態 |
|:-----------|:--------------------------|
| 原因が advanceFade 未配線 | **SnapshotCoordinator のフェードライフサイクルが完結する**: advanceFade → remaining==0 → tryComplete→completeFade まで正常動作 → 該当経路の retire 回復 → lifecycle(retire)>0 の可能性 |
| 原因が advanceFade 以外 | **フェードライフサイクルは完結する**（advanceFade → completeFade まで正常動作）が lifecycle(retire) は不変。別途対応が必要 |

したがって P1 実施後の成功条件は「**SnapshotCoordinator のフェードライフサイクルが完結すること**（advanceFade → remaining==0 → completeFade() まで到達）」であり、`runtimeRetireCount` の改善は「**P0-1 で原因が advanceFade 由来と特定された場合の副次効果**」として位置づける。

---

## 8. P1: MEM_SNAP 監視強化

### 8.1 前提条件

P0-1 で lifecycle(retire)=0 の原因が特定され、advanceFade の配線が有効と判断されたこと。

### 8.2 advanceFade(numSamples) のサンプル数単位

✅ **FACT**: `getNextAudioBlock()` の `numSamples` は JUCE ホストから渡されるコールバックブロックサイズ（例: 512 samples）である。`SnapshotFadeState` のカウンタ単位も同一のコールバックサンプル数を前提としている。

✅ **FACT**: `fadeSamples`（`DEFAULT_EQ_FADE_SAMPLES=256` 等）は createSnapshotFromCurrentState() から固定値で設定される。advanceFade(numSamples) はこの remaining をコールバックブロックサイズずつ減算する。

✅ **FACT**: オーバーサンプリング（OS=2/4/8）は DSPCore 内部の処理レートの話であり、SnapshotFadeState の remaining 減算とは無関係。fade 完了時間は「コールバックサンプル数ベースの実時間」で決まる。

**したがって、advanceFade(numSamples) に OS 補正は不要。** コールバックサンプル数のまま減算して問題ない。

### 8.3 変更内容

`AudioEngine.Timer.cpp` の MEM_SNAP フォーマットに `StereoConvolver::liveCount` と `DSPCore::liveCount` を追加。

- DSPCore の寿命を直接監視可能に
- advanceFade 配線後の効果確認指標
- リグレッション検知の早期指標

### 8.4 期待効果

- DSPCore の寿命を直接監視可能に
- advanceFade 配線後の効果確認指標
- リグレッション検知の早期指標

---

## 9. P2-1: 初回ブロックサイズ調査

### 9.1 問題

初回 `DSPCore::prepare()` 時の `processingBlockSize=524288`（`SAFE_MAX_BLOCK_SIZE 65536 × MAX_OS_FACTOR 8`）。

✅ **FACT**: この値は設計上の安全上限であり、osFactor=0 時には過剰。

⚠️ **CAVEAT**: 影響範囲が広い。Phase 1 実施後にメモリ削減効果を確認し、本 Phase の要否を再評価する。

### 9.2 検討中の対策案

| 案 | 難易度 | リスク | 期待削減（推定） |
|:---|:-------|:-------|:---------|
| A: osFactor=0 時に実ブロックサイズを上限とする | 中 | 低 | 🔍 約481MB（推定値） |
| B: `SAFE_MAX_BLOCK_SIZE` の低減 | 高 | 中 | 要調査 |
| C: 初回 prepare 時のブロックサイズ制限 | 低 | 低 | 🔍 約481MB（推定値） |

🔍 **HYPOTHESIS**: 約481MB は EQ scratch(32MB) + msWorkBuffer(16MB) + dryBypass(8MB) 等の積み上げ推定。実測値ではなく、Phase 1 実施後のメモリ削減効果と合わせて再評価すべき概算。

---

---

## 10. 設計方針: SnapshotCoordinator と CrossfadeRuntime の責務分離

### 10.1 本設計方針の目的

ConvoPeq の ISR アーキテクチャには、**責務が完全に独立した2つの「Crossfade的な機構」** が存在する。これらを混同すると設計判断を誤るため、本計画書における改修方針の根拠として明確に区別する。

### 10.2 2つの独立した機構

✅ **FACT**: `SnapshotCoordinator` と `CrossfadeRuntime` は責務・管理対象・状態管理・ログタグの全てで独立している。

| 機構 | 管理対象 | 状態管理 | ログタグ | 呼び出し元 |
|:-----|:---------|:---------|:---------|:----------|
| **`SnapshotCoordinator`** | パラメータ遷移 (EQ/NS/AGC) | `SnapshotFadeState` (`Idle`/`FadingIn`) | `[VERIFY]` | `createSnapshotFromCurrentState()` (Timer) |
| **`CrossfadeRuntime`** | DSP エンジン遷移 (structural) | `pending_` boolean + `LinearRamp` | `[XFADE]` | `DSPTransition::onPublishCompleted()` (Orchestrator) |

✅ **FACT**: ソースコード上も完全に分離されている:

| 確認項目 | SnapshotCoordinator | CrossfadeRuntime |
|:---------|:-------------------|:-----------------|
| 定義ファイル | `SnapshotCoordinator.h:94` | `CrossfadeRuntime.h` |
| advance 機構 | `advanceFade(numSamples)` | `LinearRamp` (sample-by-sample) |
| 完了検出 | `tryCompleteFade()` (remaining==0) | `isPending()` (fade 完了後 false) |
| 完了後処理 | `completeFade()` → publish + retire | `complete()` → status クリア |
| DSPTransition との関係 | **なし**（独立） | DSPTransition から開始される |

### 10.3 本計画書における意義

この分離に基づき、本計画書では以下の設計判断を行う:

| 判断 | 根拠 |
|:-----|:------|
| `CrossfadeRuntime` に SnapshotCoordinator の完了通知を持ち込まない | 責務違反。v1.0 で検討したが v2.0 で却下 |
| advanceFade の配線先は Audio Callback のみ | Timer はサンプル数を知らない。ISR 原則に基づく |
| Timer の tryCompleteFade() は変更不要 | Timer は既に毎回呼んでおり、advance() が remaining を減らすだけで動作する |
| `[XFADE]=0` と advanceFade 未配線は無関係 | `[XFADE]` は DSP エンジン遷移のログ。Snapshot(=パラメータ)遷移とは別機構 |

### 10.4 参考: コード上の根拠

```cpp
// DSPTransition.h:49 — DSP エンジン遷移のみを扱う
void onPublishCompleted(DSPCore* newDSP, DSPCore* oldDSP, ...) {
    // CrossfadeAuthority が判断した要否に基づき:
    // - 要: crossfadeRuntime_.start(decision.fadeTimeSec, sampleRate);
    // - 不要: crossfadeRuntime_.complete(); lifetime.retire(oldDSP);
}

// SnapshotCoordinator.cpp:43 — パラメータ遷移のみを扱う
void advanceFade(int numSamples) noexcept {
    m_fade.advance(numSamples);  // SnapshotFadeState のカウンタ減算
}
```

`DSPTransition` は `SnapshotCoordinator` を一切参照しない。`SnapshotCoordinator` は `CrossfadeRuntime` を一切参照しない。互いに完全に独立した責務である。

---

## 12. P3: AoS→SoA メモリ最適化

**変更なし** — 詳細は `plan.md` の 11 パッチ設計を参照。

### 12.1 設計方針の補足

💡 **PROPOSAL**: `irFreqDomain` の削減方式として以下の3案を検討:

| 方式 | リスク | 削減効果 |
|:-----|:-------|:---------|
| 完全削除（irFreqDomain メンバ削除） | 高: 全参照箇所の修正必須 | 最大: 常時確保なし |
| スクラッチ化（IR load 時のみ確保、フィルタ適用後解放） | 中: 寿命管理の追加 | 中: 通常時は解放 |
| **Small Buffer Optimization**（Layer build 中のみ AoS 保持、build 完了後即破棄） | 中: build パス限定の寿命管理 | 中: IRロード時以外はゼロ |

---

## 12. P4: 未計装 mkl_malloc 追跡

### 11.1 対象箇所

MKLNonUniformConvolver.cpp の 5 箇所 + CacheManager/IRConverter/AlignedAllocation の生 `mkl_malloc` を DIAG 化。

### 11.2 期待効果

直接的な削減効果はない。**計装の完全化**による透明性向上が目的。

---

## 13. 付録: 調査で発見された副次的知見

### 13.1 AudioEngine::retireDSP() はデッドコード

✅ **FACT**: `AudioEngine::retireDSP()` は `DSPLifetimeManager::retire()` とは別の関数であり、呼び出し元が存在しない。DSPCore の retire はすべて `DSPLifetimeManager::retire()` 経由で行われている。

### 13.2 DSPGuard は rebuild-obsolete DSPCore を解放できない

✅ **FACT**: `DSPGuard::~DSPGuard()` → `DSPLifetimeManager::retire()` → `retireDSPHandleForRuntime()` → handle map 未登録 → false → return。EBR enqueue されず唯一の解放経路が機能しない。

🔍 **HYPOTHESIS**: 他経路からの解放がなければメモリリークとなる。

◀ **v1.0 では「実質的に解放されている」としていたが、コード解析の結果これは誤りと判明。**

### 13.3 ISRRetireRouter のカウンタはデッドコード

✅ **FACT**: `m_pendingRetireBytes_` と `m_trackedPendingEntries_` に値を書き込むコードが存在しない。MEM_SNAP の `trBytes` / `tr` フィールドは常に 0 を返す。

### 13.4 Coordinator 直接 publish と Orchestrator publish の差異

✅ **FACT**: Timer.cpp の fadeCompleted publish は `makeRuntimePublicationCoordinator().publishWorld()` (Coordinator 直接) を使用しており、`RuntimePublicationOrchestrator` (DSPTransition あり) を経由しない。

🔍 **HYPOTHESIS**: この経路差異が lifecycle(retire)=0 の一因である可能性がある。fadeCompleted ブロック内で DSPTransition 相当の処理を行うため、Coordinator 直接経路でも retire は実行されている（lifetimeMgr.retire(done) at line 878）が、`runtimeRetireCount` が increment されない可能性がある。

### 13.5 改訂履歴

| 版 | 日付 | 改訂内容 |
|:---|:-----|:---------|
| 1.0 | 2026-07-10 | 初版 |
| 2.0 | 2026-07-10 | 事実/仮説/設計案を分離。Phase 1 最小化。優先順位再編。Completed 状態と CrossfadeRuntime 改造を不要と判断。retireDSP デッドコード発見を反映。|
