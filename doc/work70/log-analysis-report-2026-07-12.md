# ConvoPeq ログ解析レポート

**日付**: 2026-07-12
**対象ログ**: `ConvoPeq.log` (起動 09:48:48 〜 09:51:23+ まで観測)
**解析目的**: メモリ消費の経時変化の詳細把握 + [未確定] 7 記載事項の検証 + バグ・不適切動作の特定

---

## 1. セッション全体のメモリタイムライン

### 1.1 起動〜初期化 (gen=1)

| 時刻 | Event | Private | WS | PF (累積) | 備考 |
|:---|:------|:-------|:---|:---------|:-----|
| 09:48:48 | `[MEM] publish gen=1` | **75MB** | 70MB | — | EQ_CTOR ×1 |
| 09:48:48 | `BUILD_PHASE gen=1` (rebuildIR=0ms) | **299MB** | 283MB | 78,564 | 初回 DSPCore prepare, SR=48kHz→48kHz |
| 09:48:48 | `[MEM] gen=2` | 299MB | 283MB | 78,565 | gen=2 publish SUCCEEDED |
| 09:48:48 | `[MEM] gen=3` | 299MB | 284MB | — | prepareToPlay, SR→192kHz |

### 1.2 初期 prepareToPlay + AUTH_CONTRACT FAIL (gen=3, ~6秒間)

BUILD_PHASE gen=3 後、AUTH_CONTRACT FAIL により publish 停止。DSPCore は silent retire/recycle を繰り返す:

| 時刻 | Seq | Private | WS | PF Delta | 累積PF | 備考 |
|:---|:---:|:-------|:---|:--------|:------|:-----|
| 09:48:50.280 | 1 | **391MB** | 373MB | +0 | 154,968 | gen=3 初回 MEM_SNAP (NUC live=0) |
| 09:48:51.294 | 4 | 405MB | 386MB | +6,624 | 161,592 | WS成長 |
| 09:48:52.305 | 6 | 406MB | 396MB | +3,215 | 164,807 | |
| 09:48:53.310 | 8 | 415MB | 430MB | +10,304 | 175,111 | WS急増 |
| 09:48:54.323 | 10 | 415MB | 431MB | +814 | 175,925 | |
| 09:48:55.331 | 12 | 416MB | 432MB | +1,048 | 176,973 | |

**MEM_SNAP gen=3 遷移:**

| rec# | NUC live | alloc | Priv | TRK(total) | 備考 |
|:---:|:--------:|:-----|:----|:----------|:-----|
| 2 | 0 | 452MB | 391MB | 5.0MB | DC=1 |
| 22 | 0 | 452MB | 405MB | 5.0MB | 緩やかな成長 |
| 60 | 0 | 452MB | 415MB | 5.0MB | |
| 112 | 2 | 560MB | 524MB | 5.0MB | DC=2 (IR準備) |

> **この間、NUC live=0 (published world なし)**。gen=3 の publish は AUTH_CONTRACT FAIL で全滅。DC (DSPCore) が 1→2 に増加。

### 1.3 MixedPhase + IR ロード (約 09:48:56)

**最大メモリスパイク:**

| イベント | Private | WS | alloc | 備考 |
|:---------|:-------|:---|:-----|:-----|
| MixedPhase 開始直前 | 416MB | 432MB | 452MB | Seq=12 |
| MEM_SNAP rec=112 | **524MB** | 535MB | **560MB** | DC=2, SC=0 |
| MEM_SNAP rec=114 | 523MB | 533MB | **667MB** | tA=1GB |
| IR_RELEASE (ch0旧) | 534MB | — | 896MB | delta=0MB |
| IR_LOAD (ch0新) | 534MB | — | **914MB** | +17MB |
| MEM_SNAP rec=118 | 424MB | 437MB | 700MB | 一時的な再編 |
| MEM_SNAP rec=120 | **558MB** | 565MB | 830MB | DC=2, IR確立中 |
| MEM_SNAP rec=122 | **566MB** | 563MB | 838MB | |
| MEM_SNAP rec=124 | **589MB** | 584MB | 894MB | MixedPhase完了前 |
| MEM_SNAP rec=126 | **597MB** | 604MB | **894MB** | MixedPhase中 |
| MEM_SNAP rec=128 | 566MB | 576MB | 932MB | gen=6 publish SUCCEEDED |
| **Seq=17** | **589MB** | **580MB** | — | **PageFault surge: +166,330** (EWMA=3,667) |

**MixedPhase 詳細:**
```
[MixedPhase] State -> Optimizing
  → channel optimization enabled (reuse ch0 for identical channels)
  → Linear IR peak delay: 488 samples
  → Target group delay range: -494.449 to -488 samples
  → GD range (post-fix): 5.01266 to 11.4494 samples
  → starting GreedyAdaGrad with 12 freq points, maxIter=4
  → design result = 0 (成功)
[MixedPhase] State -> Completed (191.0ms)
```

**IR メモリ内訳:**
```
ch0: IRFreq=5MB FDL=11MB Accum=1MB Ring=0MB Total=18MB (L0=3MB L1=14MB)
ch1: IRFreq=5MB FDL=11MB Accum=1MB Ring=0MB Total=18MB (L0=3MB L1=14MB)
          ──
          36MB total (2ch × 18MB) IR data + 931-896=35MB MKL delta
```

### 1.4 gen=6 publish 成功〜steady state (09:48:56.580〜)

| 時刻 | Seq | Private | WS | PF Delta | 累積PF |
|:---|:---:|:-------|:---|:--------|:------|
| 09:48:56.580 | — | 566MB | 576MB | — | 373,222 | gen=6 publish, XFADE開始 |
| 09:48:57.389 | 21 | **566MB** | 576MB | +32,123 | 375,426 |
| 09:48:58.402 | 24 | 568MB | 577MB | +3,295 | 378,721 |
| 09:48:59.408 | 28 | **569MB** | 577MB | +2,732 | 381,453 |
| 09:49:00.419 | 31 | 576MB | 579MB | +3,505 | 384,958 |
| 09:49:01.429 | 33 | 575MB | 579MB | +2,544 | 387,502 |
| 09:49:02.438 | 35 | 575MB | 579MB | +2,838 | 390,340 |
| 09:49:05.461 | 41 | 575MB | 580MB | +2,550 | 398,437 |
| 09:49:10.509 | 52 | 574MB | 580MB | +2,700 | 412,026 |

### 1.5 gen=7 IR 解放後〜長期 steady state (~09:49:37〜09:51:23)

| 時刻 | Seq | Private | WS | PF Delta | 累積PF | 備考 |
|:---|:---:|:-------|:---|:--------|:------|:-----|
| 09:50:00.753 | ~175 | 456MB | 469MB | +~800 | ~686,000 | IR解放完了 |
| 09:50:31.023 | ~255 | 456MB | 469MB | +~500 | ~709,000 | XRUN#5 発生 |
| 09:50:54.427 | 280 | 456MB | 469MB | +847 | 711,026 | |
| 09:51:10.585 | 316 | 456MB | 469MB | +375 | 718,362 | |
| 09:51:23.731 | 345 | **455MB** | 468MB | +327 | 724,950 | **最終 steady state** |

**IR 解放時のメモリ開放:**
```
[IR_RELEASE] NUC#FBF97FC0: MKL 931MB→914MB (-17MB) OS Private 359→341MB (-18MB)
[IR_RELEASE] NUC#EC5B6600: MKL 914MB→896MB (-17MB) OS Private 341→323MB (-18MB)
                                   Total: -34MB MKL, -36MB OS Private
```

**TRK (TrackedMemory) 内訳 (steady-state gen=7):**
```
OS=0.0MB  EQ=0.2MB  AL=0.2MB  LT=0.3MB  total=1.2MB
```
※ この値は aligned_malloc 非計装分を含まず、実 Private (456MB) のごく一部のみ追跡。

---

## 2. [未確定] 7 の検証結果

### 2.1 NoiseShaper accepted=0 の真因 → ✅ **否定的に解決 (Not-a-Problem)**

**ログ確認:**
```
[NoiseShaperLearner] Waiting diagnostics:
  accepted=3012  dropSession=0  dropSampleRate=0  dropBank=0
  bufferedSamples=771072  sessionId=0  sampleRateHz=192000
  bankIndex=107  generation=39  queueDepthBlocks=0

[NoiseShaperLearner] Waiting diagnostics:
  accepted=3004  dropSession=0  dropSampleRate=0  dropBank=0
  bufferedSamples=1540096  sessionId=0  sampleRateHz=192000
  bankIndex=107  generation=39  queueDepthBlocks=0
```

| 項目 | 値 | 判定 |
|:-----|:---|:-----|
| `accepted` | **3012 / 3004** | ❌ `accepted=0` ではない。正常に学習ブロックを受理中。 |
| `dropSession` | 0 | sessionId=0 によるドロップは 0。 |
| `dropSampleRate` | **0** | block.sampleRateHz(192000) = session.sampleRateHz(192000)。**不一致なし。** |
| `dropBank` | 0 | bankIndex 一致。 |
| `queueDepthBlocks` | 0 | 滞留なし。1回の tick で全 Queue 消化。 |
| `bufferedSamples` | 771K→1.5M | バッファは成長しているが Queue 深度は 0 → **WM トレンドは問題なし**。 |

**結論**: `accepted=0` の仮説は **実測で完全に否定された**。NoiseShaper は正常動作。設計書の懸念 (block.sampleRateHz と session.sampleRateHz の不一致) は発生していない。

### 2.2 EBR lifecycle(retire)=0 → ✅ **原因確認（大きい要因特定済み、細部は検証継続）**

**発見事実:**
- `AUTH_CONTRACT FAIL` により gen=3 以降の publish が全滅(gen=4〜6)
- NUC live=0 の間は retire 機会が発生しない → `lifecycle(retire)=0` はこれを主因として説明可能
- gen=6 publish 成功後、`Ret: pend=1` を確認 → `lifetimeMgr.retire(oldDSP)` による enqueue は正常
- 再度 gen=7 publish 成功後、`Ret: pend=0` に解消

**確認できたこと／未確認の境界:**
| ステップ | 確認可否 | エビデンス |
|:---------|:---------|:----------|
| publish 成功 | ✅ 確認 | gen=6/7 publish SUCCEEDED |
| DSPTransition | ✅ 推測可 | World=1/1/1 → 1/0/0 |
| `lifetimeMgr.retire()` enqueue | ✅ 確認 | Ret: pend=1 |
| EBR epoch advance | ⚠️ 未確認 | `Ret: pend=1→0` のタイミングのみ |
| Reader leave drain | ⚠️ 未確認 | このログのみでは確認不可 |
| EBR reclaim 実行 | ✅ 推測可 | reclaim counter 0→13 増加 |
| 物理メモリ解放 | ⚠️ 未確認 | 解放は確認できるが reclaim callback の完全性は別確認が必要 |

**結論**: `lifecycle(retire)=0` の主因は AUTH_CONTRACT FAIL であり、その点は確定。gen=6 以降の retire chain は動作しているが、「完全に正常な EBR drain が行われた」と断言するには別ログ（reclaim callback 詳細）が必要。

### 2.3 BlockSize 削減効果 / 680MB Other 内訳 → ⚠️ **限定的確認**

**BlockSize 関連:**
- 初回 prepare: `spb=4096 internalMaxBlock=32768` (SR=48kHz)
- prepareToPlay 後: `spb=1024 internalMaxBlock=8192` (SR=192kHz)
- P2-1 の `kInitialPrepareMaxBlock=4096` は今回のログでは未確認 (新実装のため)

**680MB Other（推定 → 🔍 現状の計装では検証不可）:**
- 今回の最大 Private = **589MB** (MixedPhase中) → 設計書の 680MB より低い
- steady-state **456MB** で安定
- MEM_SNAP 上の `Other=0MB` は DIAG 未計装の aligned_malloc 領域を反映
- **追跡対象外メモリ**: Private(456MB) − TRK total(1.2MB) = **~455MB** が DIAG 未計装
- この ~455MB の内訳をこれ以上分解することは、現状のログ計装では不可能

> **重要**: 従来の設計書でも「OtherPrivate ≒ DSPCore + JUCE + CRT + IR」は推定と明記されている。
> 本ログから具体的なMB配分（例: DSPCore 200MB, MKL 100MB, CRT 120MB など）を導出する根拠は一切ない。

### 2.4 その他の確認

**XRUN イベント:**
| # | 時刻 | Callback | Interval | Drift | 状況 |
|:-:|:-----|:--------|:--------|:-----|:------|
| 1 | 09:48:59.813 | 0.55ms | 8.44ms | +3,489us | gen=6 xfade 中, World=1/1/1 |
| 2 | 09:49:00.216 | 0.47ms | 8.05ms | +3,410us | gen=6 xfade 中 |
| 3 | 09:49:37.752 | 0.89ms | 8.26ms | +3,453us | gen=7 steady, World=1/0/0 |
| 4 | 09:49:47.437 | 1.36ms | 8.04ms | +3,786us | |
| 5 | 09:50:32.034 | 0.94ms | 8.11ms | +3,552us | |
| 6 | 09:50:42.319 | 1.57ms | 8.01ms | +3,444us | |
| 7 | 09:51:22.424 | 1.44ms | 8.72ms | +4,297us | |

全 XRUN とも callback タイミングの ~3-4ms の drift による。`Pressure=0` で backpressure 制御は正常。Callback 実時間は 0.47〜1.57ms と予算(5.33ms) 内。

**DIAG_STAT (tick=100〜1500):**
| tick | pushed/popped | cbW/cbD | 状態 |
|:----|:------------|:--------|:-----|
| 100 | 0/16 | 17/0 | steady, 全 cb 正常処理 |
| 200 | 4/16 | 12/6 | 一部 cbD (drop) 発生 |
| 400 | 0/16 | 16/2 | cbD は 2まで減少 |
| 500-1500 | 0/16 | 16/1-2 | **steady, 安定** |

---

## 3. 総合サマリ

### メモリ経過 3行要約
1. **起動**: 75MB → 初回 DSPCore 299MB → prepareToPlay 後 391MB
2. **AUTH_CONTRACT FAIL 期間**: 391MB→416MB へ緩やか増加 (NUC live=0, DC=1→2)
3. **MixedPhase + IR Load でピーク 589MB** → IR 解放後 **steady-state 456MB に収束**

### [未確定] 7 解決状況一覧

| # | 項目 | 結果 | 確度 | エビデンス |
|:-:|:-----|:-----|:----|:---------|
| 7.1 | NoiseShaper accepted=0 真因 | ✅ **NOT-A-PROBLEM** | **確定** | `accepted=3012/3004`, `dropSampleRate=0` |
| 7.2 | EBR lifecycle(retire)=0 原因 | ✅ **主因特定済み**（AUTH_CONTRACT FAIL）、細部は別検証必要 | **主因:確定**、drain完全性:未確認 | gen=6 Ret: pend=1→0 確認、ただし EBR epoch advance 詳細は別ログが必要 |
| 7.3 | BlockSize 削減 / 680MB Other | ⚠️ **限定的確認**（最大Private 589MB、Other のMB配分は推定不可） | **589MB < 680MB は事実**、内訳は推定 | max Private=589MB, Other=0MB は DIAG 限界、Private−TRK=~455MB は追跡対象外 |
| 7.4 | collectTrackedMemoryStatistics | ✅ **MEM_SNAP 統合確認** | **確定** | TRK: 全カテゴリ表示 (OS/EQ/AL/LT) |

### 今後の注視点
- steady-state 456MB Private (旧版推定 2,477MB 対比) → 差異の原因は改修有無だけでなく、IRサイズ/FFT長/チャンネル数/MixedPhase設定/Debug-Release/MKL設定など多数の要因があり得るため、特定不可
- MEM_SNAP の `Other=0MB` は DIAG 非計装が原因 → P4 進行で改善見込み
- XRUN drift 3-4ms は許容範囲だが、特に crossfade 中の XRUN (#1, #2) はオーディオ影響の可能性あり
- **MMCSS `mmcssApplied_` 設計**: compare-exchange 成功後に apply を呼ぶ方式のため、初回失敗時は prepareToPlay まで再試行されない。改善検討の余地あり

---

## 4. バグ・不適切動作の分析

### 4.1 [CRITICAL] AUTH_CONTRACT FAIL — Runtime 状態不整合（原因は複数経路の可能性）

**現象**: `[AUTH_CONTRACT] FAIL fadingNode=0 hasFadingByUuid=1` により gen=3〜5 の publish が全滅。約6秒間、有効な RuntimePublishWorld が存在しない状態が継続した。

**ログ抜粋**:
```text
[CONV_STATUS] gen=3 irLoaded=0 irLen=0 convBypass=0 sr=192000.0 osFactor=2
[BUILD_PHASE] gen=3 build=62.0ms rebuildIR=0.0ms e2e=64.1ms
  → memBuild=Private=499MB,WS=477MB,PF=154874
[AUTH_CONTRACT] FAIL fadingNode=0 hasFadingByUuid=1
[DIAG] runPublicationPrecheckNonRt: reject reason=runtime_graph_authority_contract gen=4 seq=4
[PUBLISH] commitRuntimePublication FAILED gen=4 ownership=2
```

**確定している事実**:
- `world.graph.fadingNode == nullptr` かつ `world.topology.fadingRuntimeUuid != 0` のため AUTH_CONTRACT チェックで FAIL
- **publish 直前に検査された World が契約違反だった**ことは確定
- しかし矛盾が **Spec生成 → Coordinator → Builder → publish のどの段階で作られたか**はログのみでは特定不能（Builder は Spec を World へコピーしているだけであり、Builder 自身のバグとは限らない）

**一貫性条件の確認**: AUTH_CONTRACT チェック（`AudioEngine.Commit.cpp:75-98`）は以下3項目の一致を検証:
```cpp
// (1) graph.fadingNode != nullptr  ?= topology.fadingRuntimeUuid != 0
if (hasGraphFadingNode != hasFadingByUuid) → FAIL

// (2) execution.transitionActive ?= hasFadingByUuid
if (world.execution.transitionActive != hasFadingByUuid) → FAIL
```

**可能性のある経路** — Builder または Spec 生成段階の不整合:
`RuntimeBuilder.cpp:210` では `fadingRuntimeUuid` を `next`（=fadingDSP）から無条件に設定する一方、`makeEngineRuntimeState`（`AudioEngine.h:2758-2759`）では `graph.fadingNode` を `transitionActive` に依存して条件付きで設定する。この不統一が矛盾の原因である可能性がある。ただし Spec 生成段階（`RuntimePublicationOrchestrator.cpp`）で `spec.topology.fadingDSP` 自体が誤って設定された可能性も排除できない。

ログだけでは「Builder の写像処理」「Coordinator による Spec 生成」のどちらで矛盾が生じたかは特定不能。

**【ただし原因箇所は Builder だけとは限らない】** — 矛盾が挿入され得る経路:
1. **Builder 経路**: 上記の条件不統一
2. **Coordinator / Orchestrator 経路**: `trySubmit` での `oldDSP` 解決方法の誤り
3. **Spec 生成段階**: `RuntimePublicationOrchestrator.cpp` での `spec.topology.fadingDSP` 設定の誤り

※ ISR Runtime では Builder は同一 Snapshot から Spec を構築するため、Builder 実行中に他スレッドが crossfade 状態を変更する可能性は極めて低い。したがって Coordinator/Spec生成/Builder の各段階を等しく候補として扱うべきである。

**確定に必要な DIAG（4点で同一 UUID 付き Spec/World ダンプ、Coordinator 出口を含めることで Builder 以前と Builder 以後を完全に切り分け）**:
```
// Coordinator 出口: 生成された Spec の状態（Coordinator → Builder 境界）
[DIAG_AUTH] CoordExit uuid=XXX transitionActive=? currentUuid=? nextUuid=? spec.fadingRuntimeUuid=?

// Builder 入口: Builder が受け取った Spec の状態
[DIAG_AUTH] BuilderEnter uuid=XXX transitionActive=? currentUuid=? nextUuid=? ...

// Builder 出口: 生成された World の状態
[DIAG_AUTH] BuilderExit  uuid=XXX graph.currentNode=? graph.fadingNode=? graph.pendingNode=? fadingRuntimeUuid=?

// Commit 直前: publish される World の最終状態
[DIAG_AUTH] PreCommit  uuid=XXX graph.fadingNode=? fadingRuntimeUuid=? transitionActive=?
```
CoordinatorExit がないと「Spec が既に壊れていた」のか「Builder で壊した」のかの区別がつかない。

---

### 4.2 [MEDIUM] MMCSS 登録失敗 (Error 1552) — 1回の障害が継続する可能性

**現象**: `[MMCSS] FAILED: GetLastError=1552 taskIndex=0`

**エラーコード**: 1552 = `ERROR_NO_MORE_ITEMS` — Windows Multimedia Class Scheduler (`AvSetMmThreadCharacteristics`) へのタスク登録に失敗。

**ログ**: ログ全体でこの1件のみ。成功/再試行のログは確認できない。

**コード調査で判明した設計（`AudioEngine.Processing.AudioBlock.cpp:36-50` 実コード）**:
```cpp
{
    bool expected = false;
    if (convo::compareExchangeAtomic(mmcssApplied_, expected, true, std::memory_order_acq_rel)) {
        applyMmcssPriority();  // ← 失敗しても mmcssApplied_ は既に true
    }
}
```

**重要な実装詳細**: `compareExchangeAtomic` は **`mmcssApplied_` を `true` に設定してから** `applyMmcssPriority()` を呼ぶ。
- つまり `applyMmcssPriority()` が失敗（MMCSS FAILED）しても `mmcssApplied_` は `true` のまま
- **同一 prepareToPlay 期間中に再試行は発生しない**
- `prepareToPlay()` で `mmcssApplied_` が `false` にリセットされるため、次回 prepareToPlay（デバイス再初期化時）で再試行される

**影響**:
- MMCSS が効いていない場合、オーディオスレッドは通常のスレッドとしてスケジュールされる（リアルタイム保護なし）
- 全7件の XRUN すべてに正の drift (+3,410〜+4,297μs) があることと整合するが、**因果関係の証明にはならない**（drift の原因は MMCSS 以外にも多数ある = Windows Audio、USB、ASIO、DPC、ISR、CPU C-state 等）

---

### 4.3 [MEDIUM] publish 停止中の Rebuild 浪費 — prepare 処理の累積実行時間 約354ms（経過時間ベース）

**現象**: AUTH_CONTRACT FAIL で publish が停止しているにもかかわらず、rebuild 要求が発行され続け、DSPCore の prepare が何度も実行されて即座に obsolete 扱いされる。

**内訳（prepare 処理の累積時間。CPU 占有時間そのものではなく、実時間ベースの経過）**:
| 時刻 | wasted | フェーズ | 原因 |
|:----|:------|:--------|:------|
| 09:48:50.1 | 72.0ms | prepare | gen=1 prepare → gen=3 に追い越される |
| 09:48:55.8 | 94.9ms | prepare | gen=4 prepare → gen=3 が current (→ AUTH_CONTRACT FAIL) |
| 09:48:56.1 | 187.5ms | warmup | gen=6 warmup 中に gen=7 要求到着 |
| **合計** | **354.4ms** | — | — |

**各 rebuild obsolete の DSPCore prepare 時間（実時間ベースの累積経過時間。CPU 占有時間そのものではない）**:
```
1. 56.44ms (gen=1, SR=48k, spb=4096)
2. 58.77ms (gen=3, SR=192k, spb=1024, OS=768k)
3. 57.51ms (gen=3, SR=192k, spb=1024, OS=384k = 意味の異なる2回目)
4. 83.04ms (gen=4, IR apply後)
5. 73.01ms (gen=6, IR転送)
6. 83.66ms (gen=7, IR Loaded)
```

**問題点**: 3回目の prepare (57.51ms) は gen=3 の2回目の prepare だが、OS倍率が 768k から 384k に変化している。

**【2026-07-12 調査で特定】**: 原因はノイズシェイパータイプ変更による処理構成の変化。
```
L97:  [AudioEngine] setNoiseShaperType: newType=2 wasAdaptive=1
      ↓ noise shaper type 0→2 の変更が入る
L99:  [DSPCORE_PREPARE] enter → type=0 で prepare（古い設定）
      processingRate=768000
L158: ~EQProcessor（最初の準備を破棄）
L179: [DSPCORE_PREPARE] enter → type=2 で prepare（新しい設定）
      processingRate=384000
```
OS倍率の差はノイズシェイパータイプの違い（type=0 → 2）が原因であり、**Spec 再生成は正常な動作**。AUTH_CONTRACT とは無関係。

**発見日**: 2026-07-12

---

### 4.4 [INFO] EQ Cache Miss カウンタ — コード確認により正常動作と判明

**観測**: VERIFY カウンタ `eqCacheMiss(create/lookup)=0/0` が全22回の VERIFY で一貫して 0/0。

**コード調査結果**:
- `AudioEngine.Cache.cpp:53-115` に `EQCacheManager::getOrCreate()` の完全な実装あり
- `AudioEngine.Snapshot.cpp:56` で `getOrCreate()` を呼び出し → キャッシュが使用されないわけではない
- `VERIFY eqCacheMiss(create/lookup)` は **snapshotCreateMissCount と runtimeLookupMissCount の累積値**

**正しい解釈（当初の誤解を修正）**:
```
eqCacheMiss(create/lookup)=0/0
                          ^      ^
                          |      └── runtime での lookup miss = 0 → 全ルックアップが hit
                          └───────── snapshot 作成時の cache miss = 0 → 全 snapshot 作成が cache hit
```

✅ **EQ Cache は正常動作中。`miss=0` は cache hit rate 100% を示しており、問題ではない。**

**発見日**: 2026-07-12（本解析にて当初の誤解を訂正）

---

### 4.5 [INFO] COEFF_AUTH Lag ±1 — 正常範囲内の診断情報

**観測**: COEFF_AUTH 診断出力の lag が +1 から -1 に変化。

**コード調査結果**:
- `AudioEngine.Timer.cpp:574-594` — Timer tick 毎の純粋な診断出力
- Adaptive NoiseShaper の係数バンク (`bank.generation`) と World Generation (`runtimeWorld->generation`) を比較
- 両者は**非同期に進行**する別々の番号体系
- publish 途中で ±1 程度の lag は **正常範囲**

**ADAPTIVE_SWITCH カウント**: `dspUuid=1 count=0` → `dspUuid=6 count=1` → `dspUuid=6 count=2`
- count=0: 起動直後 (gen=1) で adaptive switch 未発生
- count=1→2: gen=6 で 2回の adaptive switch 発生
- adaptive switch は学習器が係数バンクを切り替えた回数であり、異常ではない

**結論**: 🔍 COEFF_AUTH lag は情報であり、±1 は遷移中に起こり得る。異常と断定する根拠はない。

---

### 4.6 [INFO] XRUN の非対称性 — 常に Positive Drift（観測事実）

**全 XRUN の Drift 分布**:
| XRUN# | Drift | 方向 |
|:-----|:------|:----|
| 1 | +3,489μs | 遅延 |
| 2 | +3,410μs | 遅延 |
| 3 | +3,453μs | 遅延 |
| 4 | +3,786μs | 遅延 |
| 5 | +3,552μs | 遅延 |
| 6 | +3,444μs | 遅延 |
| 7 | +4,297μs | 遅延 |

**事実**: 7件すべてが callback の **遅延** (到着が期待より遅い)。早期到着による XRUN はゼロ。

**考えられる原因（複数の仮説、ログのみでは特定不能）**:
1. 🔍 MMCSS 未適用によるスレッド競合（可能性あり、ただし証明不可）
2. 🔍 Windows Audio Engine / WASAPI / ASIO のジッター
3. 🔍 USB オーディオインターフェースのアイソクロナス転送遅延
4. 🔍 DPC/ISR によるプリエンプション
5. 🔍 CPU Package C-state 遷移によるレイテンシ
6. 🔍 メモリ帯域競合（MixedPhase 時の PageFault surge 時に顕著）

**CALLBACK_STAGE の driftUs 改善傾向**: gen=3 (publish停止中) は -14K〜-17K、gen=6 (正常publish後) は drift ±2K 以内に改善 → gen=3 の大きな drift は publish 停止による不整合状態の副次的影響の可能性。

> ⚠️ **注意**: MMCSS 欠如 → XRUN という因果関係は証明できない。複数の要因が考えられる。

---

### 4.7 [INFO] lifecycle reclaim カウンタ — 経過観察中

**VERIFY カウンタ** で reclaim 値が 0→13 まで単調増加している。

| 時点 | pub/ret/reclaim | 意味 |
|:----|:---------------|:-----|
| gen=3 FAIL 中 | 4/0/0 | publish 停止中、retire/reclaim 発生せず |
| gen=6 publish 成功直後 | 6/0/0 | 退役未発生 |
| gen=6 xfade 完了後 | 6/1/0 | 1件 retire、reclaim 未実行 |
| gen=7 publish 成功後 | 7/1/1〜13 | EBR reclaim が timer tick 毎に polling |

**判定**: 🔍 EBR reclaim カウンタが 0→13 に増加しているが、これは **polling または reclaim 処理の進行回数**を示すものであり、13件の物理破棄を意味するとは限らない（コード設計によっては polling 開始毎にカウントアップされる可能性がある）。ただし `Ret: pend=1→0` は RetireQueue 上のエントリが消費されたことを示す。EBR epoch advance → Reader leave → 物理破棄(destroy) の完全性はこのログのみでは確認不可能。

---

### 4.8 バグ重要度サマリ

| # | 重要度 | 項目 | 確度 | ステータス |
|:-:|:------|:-----|:----|:---------|
| 4.1 | **CRITICAL** | AUTH_CONTRACT FAIL — Runtime 状態不整合（現象確定、原因は Builder/Spec生成/Coordinator のいずれか） | **現象: 確定**、原因: Builder または Spec 生成段階（要追加DIAGで確定） | 設計書 P6 として Backlog 登録済み |
| 4.2 | **MEDIUM** | MMCSS 登録失敗 (Error 1552) — コード引用: compareExchange→apply の1ショット設計 | **現象: 確定**、XRUN因果: 仮説 | compareExchange成功後にapply失敗、mmcssApplied_はtrue固定 → prepareToPlayまで再試行なし |
| 4.3 | MEDIUM | Rebuild 浪費 354ms — prepare処理の累積実行時間（経過時間ベース、CPU占有時間ではない） | **確定**（obsolete 時間の合計） | AUTH_CONTRACT FAIL の副次影響 |
| 4.4 | **INFO** | EQ Cache Miss カウンタ (miss=0/0) — **cache hit rate 100%を示す** | **誤解を訂正**（正常動作） | 問題なし |
| 4.5 | **INFO** | COEFF_AUTH Lag ±1 | **正常範囲**（診断情報） | 問題なし |
| 4.6 | **INFO** | XRUN 非対称性 (常に正 drift) — 事実のみ確認 | **事実: 確定**、原因: 複数仮説 | 原因特定には別ログが必要 |
| 4.7 | INFO | lifecycle reclaim カウントアップ — `Ret: pend=1→0` は RetireQueue 離脱まで。物理破棄は未確認。 | 🔍 polling正常推定、callback完全性は未確認 | `Ret: pend=0` 確認済みだが、ReclaimQueue → drain → destroy の完全性は別ログが必要 |

### 改善推奨項目

1. **MMCSS `mmcssApplied_` のリカバリ不能設計**: `compareExchangeAtomic` 成功後に `applyMmcssPriority()` を呼ぶ設計（`AudioEngine.Processing.AudioBlock.cpp:40-44`）のため、MMCSS 登録が失敗しても再試行されない。prepareToPlay でリセットされるまで MMCSS 保護が得られない。

   **注意**: 改善案として「`applyMmcssPriority()` 成功時のみ `mmcssApplied_ = true`」が考えられるが、Audio callback 内で毎回 `AvSetMmThreadCharacteristics()` を試行すると RT 性能に悪影響を与える。代わりに以下を推奨:
   - `enum class MmcssState { NeverTried, Applied, Failed }` による状態管理
   - 失敗時の再試行は NonRT タイマ（Timer callback）で行い、Audio callback では試行しない

2. **AUTH_CONTRACT 三層不整合**: `graph.fadingNode`, `topology.fadingRuntimeUuid`, `execution.transitionActive` の3者が一致しない状態が発生している。Coordinator/Spec生成/Builder の各段階を等しく候補として扱い、確定には Coordinator 出口/Builder 入口/Builder 出口/Commit 直前の4箇所で同一 UUID 付き Spec/World ダンプを出力する DIAG を追加する必要がある:
   ```
   [DIAG_AUTH] CoordExit   uuid=XXX transitionActive=? currentUuid=? nextUuid=? spec.fadingRuntimeUuid=?
   [DIAG_AUTH] BuilderEnter uuid=XXX transitionActive=? currentUuid=? nextUuid=? ...
   [DIAG_AUTH] BuilderExit  uuid=XXX graph.fadingNode=? fadingRuntimeUuid=? ...
   [DIAG_AUTH] PreCommit   uuid=XXX graph.fadingNode=? fadingRuntimeUuid=? transitionActive=?
   ```
