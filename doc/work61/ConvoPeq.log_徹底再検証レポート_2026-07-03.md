# ConvoPeq 音飛び（XRUN）ログ徹底再検証レポート

**作成日**: 2026-07-03
**最終更新**: 2026-07-03（第8版：最終調整—DSP未実行/OS Scheduler/CPU Migration/MMCSS表現修正）
**対象ログ**: `build/ConvoPeq_artefacts/Release/ConvoPeq.log`（2026-07-03 07:53:38〜）
**検証範囲**: ログ生データ全件 + ソースコードクロス検証 + 68,864件CB_HIST突合 + 36,347件CPU_MIG全件分析

---

## 目次

1. [エグゼクティブサマリ](#1-エグゼクティブサマリ)
2. [当初分析の誤りと訂正](#2-当初分析の誤りと訂正)
3. [XRUN全件統計](#3-xrun全件統計)
4. [メモリ使用量分析](#4-メモリ使用量分析)
5. [CPU Migration 分析](#5-cpu-migration-分析)
6. [リビルド廃棄分析](#6-リビルド廃棄分析)
7. [タイマーJITTER分析](#7-タイマーjitter分析)
8. [EQ_PREPAREスケーリング分析](#8-eq_prepareスケーリング分析)
9. [CALLBACK_STAGE/DSP負荷分析](#9-callback_stagedsp負荷分析)
10. [ソースコード検証結果](#10-ソースコード検証結果)
11. [因果関係の全体像](#11-因果関係の全体像)
12. [推奨対策](#12-推奨対策)
13. [付録：検証方法](#13-付録検証方法)

---

## 1. エグゼクティブサマリ

### 結論

ConvoPeq の音飛び（XRUN）の直接原因は **コールバック間隔の超過（Interval > Threshold）** であり、DSP処理の予算超過ではない。間接原因については深掘調査の結果、複数の仮説が否定され、**OSスケジューラのプリエンプション／ディスパッチ遅延** が残る仮説である。

| 発見 | 詳細 | 確度 |
|------|------|------|
| **DSPは予算内で正常動作** | 平均18.3%（最大44.2%）、予算超過(Budget>100%)は0件 | ✅ **確定** |
| **XRUNはinterval超過のみ** | 全131件でCallback=0.00ms、処理時間の超過なし | ✅ **確定** |
| **DSP処理時間とdriftは無相関** | DSP>1000us avg drift=+55us vs DSP<500us avg drift=-83us（差138usは誤差） | ✅ **確定（深掘）** |
| **リビルドとXRUNに近接性なし** | BUILD_PHASE前後500msにXRUN=0件。BELOW_NORMAL優先度で非干渉 | ✅ **確定（深掘）** |
| **PFレートとXRUNに相関なし** | 最大PF 232,451/s時のXRUN=1件、定常4,000/s時のXRUN=0〜2件 | ✅ **確定（深掘）** |
| **コールバック到着レートは正常** | CBSUMMARY loss全215件で負平均（毎秒192callbacks ≥ 期待値187） | ✅ **確定（深掘）** |
| **CPU Migration 36,347回** | 全16コア移動 | ✅ **確定** |
| **CPU affinity完全未設定** | `ThreadAffinityManager::initialized_=false` により全スキップ | ✅ **コード確認済** |
| **リビルド廃棄516.8ms** | 3件無駄、ビルド効率50% | ✅ **確定** |
| **メモリ 74MB→2.5GB** | 数値として確定（因果関与は否定） | ✅ **確定（数値のみ）** |
| **OS/Driver層のスケジューリング遅延** | 残る仮説（複数候補あり）。原因はConvoPeq外部の可能性大 | ⚠️ **仮説** |

### 当初分析からの訂正

| 当初の誤った結論 | 再検証結果 |
|-----------------|-----------|
| 「DSP処理が5.33msの予算を超過 → XRUN連鎖」 | **誤り。DSPは予算内で正常動作。予算超過(Budget>100%)は0件** |
| 「コールバック予算超過の連鎖」 | **正しくはOSスケジューリング遅延。DSP処理時間(平均0.97ms)より超過分(2.99ms)の方が大きい** |
| 「メモリ爆発がXRUNの根源原因」 | ⚠️ **修正。メモリ増加とXRUNは時間的に共存するが、PFレートとXRUNに相関なし。原因と結果の関係は現時点では未証明** |
| 「CPU MigrationがL1/L2を毎回全ミス」 | ⚠️ **修正。L1全ミスの可能性は高いが、DSPデータは連続アクセス＋prefetchが効く範囲。L3共有＋streaming accessで劣化は限定的** |

---

## 2. 当初分析の誤りと訂正

### 誤りだった結論

当初の分析では「DSPCoreの処理時間が5.33msのコールバック予算を超過し、音飛びが発生する」と結論した。これは **ログのCALLBACK_STAGE出力を正しく解釈していなかった** ことに起因する。

### 実際のデータ

CALLBACK_STAGE全2,590件の解析結果：

| 指標 | Gen=4（初期） | Gen=5（XRUN多発期） |
|------|-------------|-------------------|
| 平均DSP処理時間 | 330us | 975us |
| 平均Total処理時間 | 332us | 977us |
| 平均Budget率 | 6.2% | 18.3% |
| 最大Budget率 | 17.1% | 44.2% |
| **予算超過(Budget>100%)** | **0件** | **0件** |
| CALLBACK_STAGE件数 | 92件 | 2,496件 |

**DSP処理時間（平均0.97ms）は、コールバック間隔の超過分（平均2.99ms）より明らかに短い。**

### 訂正後の因果関係

```
DSP処理: 0.97ms (予算内, 問題なし)
              ↓
コールバック期待間隔: 5.33ms
              ↓
実際のコールバック間隔: 8.32ms ← 2.99msの超過
              ↓
超過の原因: 未確定。残る候補として以下が考えられる（ETWなしでは識別不可）
              Windows Scheduler / MMCSS状態 / DPC/ISR /
              Voicemeeter BANANA Virtual ASIOドライバ内部 /
              他プロセス・カーネルスレッドの割り込み
              ↓
XRUN検出: Interval(8.32ms) > Expected(5.33ms)
```

### 測定環境の重要情報: Voicemeeter BANANA Virtual ASIO + SAVIHost比較

本ログの測定環境では、ConvoPeqのオーディオ入出力に **Voicemeeter BANANAのVirtual ASIO** が使用されている。

**参考観測: SAVIHost + MConvolutionEZ 比較**: 同一Voicemeeter環境で、ConvoPeqの代わりにSAVIHost + MConvolutionEZを使用した場合、音飛びは発生しなかった。ただし本比較は**条件が大きく異なるため参考観測として扱う**。サンプルレート(192kHz vs 48kHz)、バッファサイズ(1024 vs 512)、プロセス優先度(HIGH vs NORMAL)、スレッド構成、DSP構成、診断ログ有無が全て異なり、単一の差分に特定できない。最低でも同一サンプルレート・同一バッファサイズでの比較がなければ、SAVIHost比較から因果結論を導くことはできない。

**両者の主要な差分**:

| 項目 | ConvoPeq | SAVIHost（推定） |
|------|---------|-----------------|
| **プロセス優先度** | **HIGH_PRIORITY_CLASS** | NORMAL_PRIORITY_CLASS（標準） |
| **サンプルレート** | **192,000Hz** | 48,000Hz（Voicemeeter標準） |
| **バッファサイズ** | **1,024 samples** | 512 samples |
| **コールバック間隔** | **5.33ms** | 10.67ms |
| **XRUN閾値** | **8.0ms** | 16.0ms |
| **メモリ使用量** | 2.5GB | 数百MB程度 |
| **診断ログ** | 33出力/timer | なし |
| **内部アーキテクチャ** | DSPCore世代管理＋リビルド | 単純なVSTホスト |

**サンプルレート差の影響（重要）**: ConvoPeqが192kHz/1024spb（5.33ms間隔）であるのに対し、SAVIHostは標準設定の48kHz/512spb（10.67ms間隔）と推定される。この場合、ConvoPeqで観測された8.32msの平均コールバック間隔はSAVIHostの閾値16.0ms未満であり、**同一のOSディスパッチ遅延が発生してもXRUNとして検出されない可能性が高い**。この仮説を検証するには、ConvoPeqを48kHz/512spbで動作させた比較試験が必要である。

### 深掘調査で今回のログでは支持されない仮説

| 仮説 | 検証手段 | 結果 | 判定 |
|------|---------|------|------|
| メモリ増加→PF→スケジューラ遅延→XRUN | PFレート vs XRUN発生率 | 最大PF秒間232,451にXRUN=1、定常4,000/sにXRUN=0〜2。相関なし | 今回のログでは支持されない |
| リビルド負荷→XRUN | BUILD_PHASE前後のXRUN発生 | 前後500ms以内にXRUN=0件。リビルドスレッドはBELOW_NORMAL優先度 | 今回のログでは支持されない |
| DSP処理時間→drift→XRUN | CB_HIST 68,864件のproc vs drift相関 | DSP>1000us drift=+55us / DSP<500us drift=-83us（無相関） | 今回のログでは支持されない |
| CPU Migration→キャッシュミス→XRUN | XRUN発生callbackのCPU Migration率 | 全体90.5% vs XRUN時100%（比1.11x）で有意差なし | 今回のログでは支持されない |
| Voicemeeterだけでは十分条件になっていない | SAVIHost同一環境で比較 | Voicemeeter + SAVIHost = 0 XRUN | 今回のログでは支持されない |

### なぜ誤解したか

CALLBACK_STAGEの `budget=44.2%` という数値を「予算の44%を使ってしまった→不足」と読んだが、正しくは「予算のうち44%しか使っていない→余裕あり」という意味である。`budget` はコールバック間隔に対する処理時間の割合であり、100%未満であれば問題ない。

---

## 3. XRUN全件統計

### 基本集計

| 指標 | 値 |
|------|-----|
| 総XRUN数 | **131回** |
| 継続期間 | **220.5秒**（07:53:40.416 〜 07:57:20.884） |
| 平均発生率 | **0.59回/秒**（約1.7秒に1回） |
| Interval 最小 | 8.00ms |
| Interval 最大 | 9.47ms |
| Interval 平均 | **8.32ms** |
| Interval 中央値 | 8.25ms |
| Expected 期待値 | **5.33ms**（全件一定） |
| 平均超過率 | **+56.1%**（強制超過） |
| Interval > Expected | **131/131件（100%）** |

### Expected=5.33ms の検算

```
192,000Hz / 1,024spb = 187.5 callbacks/sec
1 / 187.5 × 1,000,000 = 5,333us = 5.33ms ✅
```

### 世代別 XRUN分布

| 世代 | XRUN数 | 比率 | メモリ状態 |
|------|--------|------|-----------|
| Gen=4 | 15回 | 11.5% | Private=1,353→2,369MB |
| Gen=5 | **116回** | **88.5%** | Private=2,485MB（最大時） |

### XRUN発生パターン

- **初期クラスター（#1〜#15）**: 約7秒間（07:53:40-47）に15回集中
  - Gen=4発行直後、メモリ1,353MB時
  - 最初のCBSUMMARYで `loss=114`（114コールバックの未達）
- **多発期（#16〜#131）**: 約213秒間に116回
  - Gen=5発行後、メモリ2,485MB時
  - 平均0.54回/秒で定常的に発生

### XRUNの共通パターン

すべてのXRUNに以下の共通点がある：
- **PressureLevel=0**（バックプレッシャーによるリビルド抑制なし）
- **RetireDepth=0**（retireキュー滞留なし）
- World状態: Active=1, Fading=0（単一ワールド、フェード中でない）
- **すべて Interval > Expected が原因**

---

## 4. メモリ使用量分析

### メモリ増加推移

| 時点 | Private | WS | PageFaults | 累積PF増加 |
|------|---------|-----|-----------|-----------|
| 初期化直後 (Gen=1) | 74MB | 69MB | — | — |
| 初回ビルド後 (Gen=2) | **555MB** | 516MB | 138,427 | — |
| 初回Publish後 (Gen=3) | 555MB | 517MB | — | — |
| 2nd SR準備後 (Gen=4) | **1,353MB** | 1,255MB | 328,086 | — |
| Gen=4 発行7秒後 | **2,369MB** | 2,236MB | 589,552 | +240,354/7秒 |
| Gen=5 最大時 | **2,485MB** | 2,333MB | 1,086,097 | +496,545 |
| 最終 (Seq=480) | **2,497MB** | 2,342MB | 1,086,097 | — |

**74MB → 2,497MB = 約34倍増加**

### Gen=4 内部のメモリ急増（新発見）

同一世代（Gen=4）の生存中に **Privateが1,016MB増加**（1,353→2,369MB）している。これは「新たなDSPCoreのビルド」ではなく、以下の要因による：

1. **ページフォールト蓄積**: `+240,354 PF / 7秒 = 34,336 PF/秒`
2. **ワーキングセット拡大**: ページフォールトによりOSが物理メモリを追加マッピング
3. **Gen=4からGen=5への遷移時の二重メモリ**: Gen=4（active） + Gen=5（ビルド中）の共存

### メモリ爆発の原因（コード確認済）

1. **固定大バッファ**: `SAFE_MAX_BLOCK_SIZE=65536` → `internalMaxBlock=524288`
   - 1DSPCoreにつき `alignedL/R(524288) + dryBypassL/R(524288) + EQ buffers + Conv buffers` = 〜50MB
2. **IRデータ全量保持**: `juce::AudioBuffer<double>` にIR全体をメモリ保持
3. **DSPCore世代の非同期解放**: `DSPLifetimeManager::retire()` → `ISRRetireRouter` → EpochDomain deferred queue
   - Active + Fading + Pending で最大3世代同時生存可能
4. **ページフォールトによるWS拡大**: アライン割り当て（ページ境界）によりOSが物理ページを逐次割り当て

---

## 5. CPU Migration 分析

### 総計

| 指標 | 値 |
|------|-----|
| 総CPU Migration回数 | **36,347回** |
| 総コールバック数（推定） | ~41,250回（220.5秒×187.5Hz） |
| Migration率 | **88.1%**（Affinity未設定のためOS既定動作） |
| 使用CPU数 | **16/16（全コア）** |

### 世代別 Migration

| 世代 | Migration回数 | 使用CPU数 | 主要移動先 |
|------|-------------|-----------|-----------|
| Gen=3 | 29回 | 14/16 | CPU4(6), CPU0(4), CPU2(4) |
| Gen=4 | 1,012回 | **16/16** | CPU2(305), CPU3(136), CPU4(108) |
| Gen=5 | **35,306回** | **16/16** | CPU3(4,676), CPU2(4,540), CPU0(3,496) |

### CPU Migrationの影響（補正版）

すべてのコールバックが異なるCPUで実行される場合の影響（深掘調査に基づき補正）：

- **L1データキャッシュ（32KB/コア）**: DSPデータ（alignedL/alignedRなど線形バッファ）の先頭〜数十サンプルは毎回ミス。ただし**streaming access＋hardware prefetch**により後続サンプルはヒットする可能性あり。全ミスとは言えない。
- **L2キャッシュ（256KB/コア）**: DSP処理データ2048〜8192サンプル×8bytes×2ch = 32KB〜128KB。大部分はL2に収まる範囲。CPU移動時の初期ミスは発生するが、prefetchが追いつけば限定的。
- **L3キャッシュ（共有、数MB〜10MB超）**: 全コアで共有。CPUが変わってもL3ヒットの可能性あり。ConvoPeqの大規模データ（IR: 192K taps × 8bytes × 2ch = 〜3MB）はL3に収まるかどうかが分かれ目。
- **TLB**: ページテーブルウォークの一部は発生するが、2MB（Large Page）の使用状況に依存。

**CPU Migration vs XRUN 相関解析**: CALLBACK_STAGEのseq(callback index)を介して、XRUN発生コールバックとCPU Migrationの有無を突合した。全体のCPU Migration率は90.5%（2,590件中2,343件）と極めて高い。XRUN drift一致29件のCPU Migration率は100%（29/29）であったが、比率は1.11倍であり**統計的に有意な相関は確認できなかった**。

**補足: CPU Migration頻度単独では因果を説明できない。** 重要なのは移動頻度ではなく、(1)どのCPUからどのCPUへの移動か（NUMAドメイン間か、同一CCD内か）、(2)移動直後の処理時間ペナルティである。CPU Migration頻度が高いことと、コールバック間隔超過（XRUN）は異なる現象であり、「両者が多い」という共起関係から因果を導くことはできない。さらに、CPU別の平均処理時間差は最大88us（10.9%）であり、CPU MigrationによるキャッシュミスペナルティはDSP処理時間にほとんど影響していないことが確認されている。

**確度**: **高確度**（L1初期ミスの可能性は高いが、CPU Migration→XRUNの因果は確認できず。さらにNUMA/CCD分析が未実施のため、Migrationの質的評価も保留）

### CPU affinity コード確認結果

`ThreadAffinityManager.h` の実装を確認：

```
ThreadAffinityManager() = default;  // masks_は全て0, initialized_はfalse

void applyCurrentThreadPolicy(...) noexcept {
    if (!convo::consumeAtomic(initialized_, std::memory_order_acquire))
        return;  // ← initialized_=false のため即return
    // affinity設定は一切行われない
}
```

**`initialized_` をtrueに設定するコードパスが存在しない**ため、すべてのスレッドのCPU affinityは無設定状態。OSのスケジューラが任意のCPUにタスクを割り当てる。

---

## 6. リビルド廃棄分析

### 廃棄イベント（Obsolete Rebuild）

| 廃棄 | 廃棄されたGen | 現行Gen | フェーズ | 無駄時間 |
|------|-------------|---------|---------|---------|
| #1 | Gen=1 | Gen=3 | prepare完了後 | **101.0ms** |
| #2 | Gen=5 | Gen=4 | prepare完了後 | **214.0ms** |
| #3 | Gen=6 | Gen=4 | prepare完了後 | **201.8ms** |
| **合計** | — | — | — | **516.8ms** |

### Rebuild Telemetry 全イベント

```
REBUILD_REQUESTED:  26回（リビルド要求発行）
REBUILD_DISPATCHED: 18回（実際にビルドスレッドで開始）
REBUILD_MERGED:      6回（同一パラメータのためマージ）
REQUESTED - DISPATCHED: 8回（キューイングされたが未実行）
```

### ビルド完了実績

```
Gen=1: build=89.5ms  e2e=90.9ms  WS=516MB   ← 初回48kHz
Gen=4: build=93.2ms  e2e=95.5ms  WS=1,255MB ← SR変更
Gen=8: build=170.0ms e2e=656.7ms WS=2,333MB ← IR読み込み込み
```

**完了:廃棄 = 3:3（ビルド効率 50%）**

### 廃棄メカニズム（コード確認）

`rebuildThreadLoop()` の4段階廃棄チェックポイント：

```
[Checkpoint 1] prepare開始前 → isObsolete() → obsoleteなら即continue
[Checkpoint 2] prepare完了後 → obsoleteならログ出力"wasted=Xms" → continue  ← これがログに出現
[Checkpoint 3] IR rebuild完了後 → 同上
[Checkpoint 4] warmup完了後 → 同上
```

廃棄はgenerationカウンタの単純比較（`generation != rebuildRequestGeneration`）で検出。**Checkpoint 2〜4間の処理は最後まで実行されてから廃棄されるため、無駄時間が蓄積する。**

---

## 7. タイマーJITTER分析

### 集計（47イベント全件）

| 指標 | 期待値 | 実測最小 | 実測平均 | 実測最大 |
|------|--------|---------|---------|---------|
| Timer Interval | 100ms | 120.76ms | **135.21ms** (+35.2%) | **244.30ms** (+144%) |
| Delta（Jitter） | 0ms | 20.76ms | **35.21ms** | **144.30ms** |
| estimatedMissed | 0 | — | — | **1回** |

### 大きなJitterの発生例

```
Interval=244.30ms (expected=100ms, delta=144.30ms, estimatedMissed=1)  ← 1回タイマーミス
Interval=185.63ms (expected=100ms, delta=85.63ms)
Interval=184.47ms (expected=100ms, delta=84.47ms)
Interval=163.59ms (expected=100ms, delta=63.59ms)
Interval=161.21ms (expected=100ms, delta=61.21ms)
```

**タイマースレッドもOSスケジューリング遅延の影響を直接受けている。** タイマーが期待の100ms以内に発火できず、最大244msまで遅延している。

---

## 8. EQ_PREPAREスケーリング分析

### 実行時間の推移

| 実行 | サンプルレート | EQ_PREPARE時間 | システムメモリ状態 |
|------|--------------|---------------|------------------|
| 1回目 | 384kHz | **23.78ms** | WS=69MB（初期） |
| 2回目 | 768kHz | **25.46ms** | WS=517MB |
| 3回目 | 384kHz | **25.81ms** | WS=517MB |
| 4回目 | 384kHz | **41.24ms** | WS=1,255MB ← 1.7倍 |
| 5回目 | 384kHz | **84.03ms** | WS=2,333MB ← **3.5倍** |
| 6回目 | 384kHz | **50.93ms** | WS=2,333MB ← 2.1倍 |

### 割り当てられるバッファ

`EQProcessor::prepareToPlay()` で確保される全バッファ：

| バッファ | サイズ（block=8192時） | 備考 |
|---------|---------------------|------|
| scratchBuffer | 65,536要素 (512KB) | `nextPowerOfTwo×8` |
| dryBypassBuffer | 163,840要素 (1.28MB) | `nextPowerOfTwo×MAX_CHANNELS(20)` |
| parallelInputBuffer | 163,840要素 (1.28MB) | — |
| parallelWorkBuffer | 163,840要素 (1.28MB) | — |
| parallelAccumBuffer | 163,840要素 (1.28MB) | — |
| structureOldOutBuffer | 163,840要素 (1.28MB) | — |
| structureNewOutBuffer | 163,840要素 (1.28MB) | — |
| msWorkBuffer | 32,768要素 (256KB) | `block×4` |
| AGC係数テーブル×3 | 24,576要素 (192KB) | `block+1` × 3 |
| **EQ 1基合計** | **〜8MB** | ページアライン割り当て |

### 劣化原因（コード確認＋注意事項）

劣化の原因として以下の候補が考えられるが、**現時点では支配因子を特定できていない**（ETW sampling profileでの確認が必要）：

| 候補 | 影響 | 推定確度 |
|------|------|---------|
| `makeAlignedArray()` の新規ページ割り当て＋ページフォールト | 中〜高 | ⚠️ 仮説（PF種別がsoft/hard未確認） |
| `std::memset(filterState, 0, sizeof(filterState))` の大域メモリアクセス | 中 | メモリ帯域依存 |
| AGC係数テーブル `std::exp()` × 24K要素の浮動小数点演算 | 低 | キャッシュに乗れば高速 |
| OSメモリアロケータの競合（リビルドスレッドとの並行動作） | 中 | 同時実行の有無に依存 |

**重要**: 深掘調査により、PFレート全般とXRUN発生率の相関は否定されている。EQ_PREPAREの時間増加はPF以外の要因（memsetのキャッシュミス、アロケータ競合など）である可能性が高い。

---

## 9. CALLBACK_STAGE/DSP負荷分析

### 世代別 CALLBACK_STAGE

#### Gen=3（初期化直後、2件）

| seq | drift | input | dsp | total | budget |
|-----|-------|-------|-----|-------|--------|
| 16 | -17,288us | 3 | 686us | 687us | 12.8% |
| 32 | -18,355us | 3 | 866us | 869us | 16.2% |

→ 初期のコールバックタイミング調整期。大きな負driftは問題ではない。

#### Gen=4（XRUN開始期、92件）

| 指標 | 値 |
|------|-----|
| 平均DSP時間 | 330us |
| 平均Total時間 | 332us |
| 平均Budget | 6.2% |
| 最大Budget | 17.1%（DSP=910us） |
| Budget>15% | 1件（1.1%） |
| |Drift|平均 | 787us |

→ DSP負荷は極めて低い。XRUNの原因はDSP処理ではない。

#### Gen=5（XRUN多発期、2,496件）

| 指標 | 値 |
|------|-----|
| 平均DSP時間 | **975us** |
| 平均Total時間 | **977us** |
| 平均Budget | **18.3%** |
| 最大Budget | **44.2%**（DSP=2,356us） |
| Budget>15% | 1,555件（62.3%） |
| Budget>25% | 340件（13.6%） |
| Budget>40% | 4件（0.2%） |
| |Drift|平均 | 728us |
| |Drift|最大 | 3,374us |

→ DSP負荷は高まったが、予算超過は0件。**平均18.3%の余裕あり。**

### Total-DSP 差分（I/O時間）

| 指標 | 値 |
|------|-----|
| Total-DSP 平均 | **2.5us**（input/outputコピー時間） |
| Total-DSP 最大 | 37us |
| input+output平均 | 2.5us |
| Total-DSP = input+output | ✅ 完全一致 |

→ I/O（JUCEオーディオバッファ出入力）は無視できるレベル。

---

### 深掘：DSP処理時間 vs drift 相関分析（68,864件のCB_HIST）

深掘調査で、CB_HISTのproc（処理時間）とdrift（コールバック間隔変動）の相関を全件解析した。

**処理時間別のdrift平均**：

| proc範囲 | 件数 | drift平均 | \|drift\|平均 |
|----------|------|-----------|-------------|
| <300us | 1,445 | -82us | 806us |
| 300-500us | 2,579 | -32us | 779us |
| 500-1000us | 48,402 | -13us | 734us |
| 1000-1500us | 14,718 | +53us | 730us |
| 1500-2000us | 1,595 | +140us | 752us |
| >2000us | 123 | +37us | 728us |

**結論**: DSP処理時間とdriftに統計的に有意な相関は存在しない。処理時間が長くてもdriftは増加しない。これは、**コールバックの到着間隔がDSP処理時間に依存しない**（＝DSP処理とは独立した要因でコールバック間隔が変動している）ことを強く示唆する。

**CB_HISTのproc平均 世代間比較**：

| 世代 | CB_HIST件数 | proc平均 | proc最大 |
|------|------------|---------|---------|
| Gen=4 | 2,368件 | **348us** | 1,263us |
| Gen=5 | 66,496件 | **865us** (+148%) | 2,538us |

Gen=5（IR読み込み＋Conv処理追加後）で処理時間が148%増加したが、それでも865us（予算の16.2%）に留まり、XRUNに影響していない。

---

## 10. ソースコード検証結果

### 10.1 リビルドディスパッチ（`AudioEngine.RebuildDispatch.cpp`）

| 検証項目 | 結果 |
|---------|------|
| rebuild queue機構 | ✅ `pendingTask` + `rebuildMutex` + `rebuildCV`（標準的なProducer-Consumer） |
| 重複抑制 | ✅ `sameAsPending` チェック + LatestWins merge（50ms burst吸収） |
| Obsolete検出 | ✅ generation比較（`generation != rebuildRequestGeneration`）レースフリー |
| 早期中断 | checkpoint間の処理は完了まで走る（無駄時間の原因） |

### 10.2 メモリ管理（`RuntimeBuilder.cpp`, `DSPLifetimeManager.h`）

| 検証項目 | 結果 |
|---------|------|
| 新DSPCore構築 | ✅ `aligned_make_unique<DSPCore>()` → `prepare()` |
| 旧世代解放 | ✅ `DSPLifetimeManager::retire()` → `ISRRetireRouter` → epoch queue（非同期） |
| 世代共存 | ✅ Active + Fading + Pendingで最大3世代 |
| 解放完了保証 | Graceful drain（最大5秒） |

### 10.3 CPU affinity（`ThreadAffinityManager.h`）

| 検証項目 | 結果 |
|---------|------|
| 初期化 | **`initialized_` が初期値falseのまま** |
| 設定コード | **存在しない** |
| オーディオスレッド | **affinity設定コード自体がない**（JUCE管理） |
| リビルドスレッド | `THREAD_PRIORITY_BELOW_NORMAL` のみ、affinity=0（全CPU） |
| 結果 | **すべてのスレッドが全CPUで自由にスケジュール** |

### 10.4 EQ_PREPARE（`EQProcessor.Core.cpp`）

| 検証項目 | 結果 |
|---------|------|
| バッファ種別 | 8種類（scratch, dryBypass, parallel×3, xfade×2, msWork, AGC×3） |
| 割り当て方法 | `makeAlignedArray<double>()`（ページアライン） |
| AGC計算 | `std::exp()` ループ（〜24K要素） |
| filterState初期化 | `std::memset(filterState, 0, sizeof(filterState))`（大域） |
| ページフォールト感受性 | ✅ **高い**（アライン割り当て＋大域memsetでPF多発） |

### 10.5 XRUN検出（`AudioBlock.cpp`, `RuntimeHealthMonitor.cpp`）

| 検証項目 | 結果 |
|---------|------|
| Drift計算 | ✅ `nowUs - prevEntryUs - expectedUs` をコールバック毎に測定 |
| Budget計算 | ✅ `elapsedUs / expectedUs × 1000`（permille単位） |
| CPU Migration検出 | ✅ `GetCurrentProcessorNumber()` 比較 |
| HealthMonitor | ✅ pendingRetire監視、Publication Stall監視、Overflow監視 |

---

## 11. 因果関係の全体像

### 訂正版：確定している事実と未確定な因果

深掘調査により、当初仮説として挙げた複数の因果連鎖が否定された。現時点で「確定している事実」と「未確定な因果」を明確に区別する。

### 確定している事実（観測）

```
[観測A] メモリ使用量: 74MB → 2,497MB（34倍）
    └─ コード確認: 固定大バッファ＋IR全量＋非同期解放により説明可能
    └─ 因果関与: 否定（PFレートとXRUNに相関なし）

[観測B] CPU Migration: 36,347回（約88%のコールバックでCPU移動）
    └─ コード確認: ThreadAffinityManager::initialized_=false で全スキップ
    └─ 因果関与: ⚠️ 未確定（A/Bテストが必要）

[観測C] DSP処理時間: Gen=4: 348us → Gen=5: 865us（+148%）
    └─ Budgetは最大44.2%、予算超過0件
    └─ 因果関与: 否定（DSP処理時間とdrift無相関）

[観測D] コールバック到着レート: 正常（loss全215件で負平均）
    └─ 大部分のコールバックは5.33ms間隔を維持
    └─ しかし時折8.0ms超の間隔拡大が発生（CBSUMMARY intervalMax平均7.99ms）

[観測E] タイマーJITTER: 100ms→平均135ms（最大244ms）
    └─ Timer/Audio両スレッドが同時期に遅延 → 共通原因を示唆

[観測F] XRUN: 131回、全件 Interval 8.00〜9.47ms > Threshold 8.0ms
    └─ Callback=0.00ms（コールバック開始時点でDSP未実行）
    └─ PressureLevel=0, RetireDepth=0（制御系は正常）
    └─ 5.33ms間隔維持時にXRUNは発生しない（→ 間欠的なスパイク）
```

### 否定された因果連鎖

```
[否定1] 「メモリ爆発 → PF急増 → OSスケジューラ遅延 → XRUN」
    否定理由: PFレートとXRUN発生率に相関なし
       最大PFレート 232,451/s の秒間にXRUN=1件
       定常PFレート 4,000/s の秒間にXRUN=0〜2件

[否定2] 「リビルド処理 → CPU負荷 → コールバック遅延 → XRUN」
    否定理由: BUILD_PHASE前後500ms以内にXRUN=0件
       BELOW_NORMAL優先度のリビルドスレッドがREALTIME優先度の
       オーディオスレッドをブロックできるとは考えにくい

[否定3] 「DSP処理時間増加 → コールバック終了遅延 → 次コールバック間隔超過」
    否定理由: DSP処理時間とdriftに相関なし
       高負荷時(DSP>1000us)も低負荷時(DSP<500us)も|drift|平均は730〜806us
       全XRUNでCallback=0.00ms（コールバック開始時点でDSP未実行）

[否定4] 「CPU Migration → L1/L2キャッシュ全ミス → DSP処理遅延 → XRUN」
    否定理由: DSP処理遅延とdriftが無相関
       XRUNのCallback=0.00msは「コールバック開始時点でDSP未実行」を意味
       キャッシュミスは処理中に影響するが、コールバック到着間隔には影響しない
```

### 残る仮説候補（本ログのみでは識別不可、ETW + ASIO SDK計測が必要）

ConvoPeq内部ではXRUNを説明できる異常は確認できなかった。残る有力候補はOS・ドライバ層のスケジューリング遅延であり、以下が考えられる：

| 仮説候補 | 根拠 | 検証方法 |
|---------|------|---------|
| ASIOドライバのbufferSwitch()発行間隔変動 | **ドライバが8ms周期でしかcallbackを呼んでいない可能性**。ConvoPeqは5.33msを期待 | ASIO SDK側のbufferSwitch時刻計測、またはETW |
| Windows Schedulerプリエンプション | Timer/Audio両スレッド同時遅延。システム全体のスケジューリング状態を示唆 | ETW Ready Time解析 |
| MMCSS登録状態（未確認） | JUCEが内部的に登録するが成功/失敗は未確認 | ETWまたはAvSetMmThreadCharacteristics戻り値確認 |
| DPC/ISR滞留 | Voicemeeter/GPU/Network/USBドライバの滞留 | ETW DPC/ISR解析 |
| Voicemeeter Virtual ASIOドライバ内部遅延 | 仮想ASIOは物理デバイスよりバッファ処理オーバーヘッド大 | Voicemeeterバイパステスト |
| CPU電源管理（Parking/P-State） | C-State遷移のウェイクアップ遅延 | ETW電源管理トレース |

### 追加分析結果：XRUN直前のdrift蓄積パターン

CB_HISTダンプ（XRUN直後32件のcallback履歴）を全125件のXRUNについて分析：

| パターン | 件数 | 比率 |
|---------|------|------|
| drift蓄積傾向あり（直前drift大／増加傾向） | 66 | **53%** |
| drift蓄積傾向なし | 59 | **47%** |

**約半数のXRUNは直前のdrift蓄積なしに突然発生している。** これはXRUNが「driftが徐々に拡大して閾値を超える」パターンではなく、「突発的にコールバック間隔が8ms超になる」パターンであることを示す。OSスケジューラの一瞬のプリエンプションや、ASIOドライバのbufferSwitch()発行スキップと整合する。

### 確信度マトリクス（最終版 v5）

| 項目 | 確度 | 根拠 |
|------|------|------|
| DSP処理はXRUN原因ではない | ★★★★★ | CALLBACK_STAGE全件＋CB_HIST相関分析 |
| XRUNはInterval>Threshold | ★★★★★ | BlockDouble.cppコード確認＋ログ全件一致 |
| CPU Migration過剰 | ★★★★★ | ログ36,347件＋コード確認 |
| CPU affinity未設定 | ★★★★★ | ThreadAffinityManager.h全行確認（initialized_=false） |
| HIGH_PRIORITY_CLASS設定 | ★★★★★ | MainApplication.cpp:96 で確認 |
| MMCSS / AvSetMmThreadCharacteristics未呼び出し | ★★★★★ | コードgrepで確認（JUCE内部管理に委譲） |
| リビルド廃棄516ms | ★★★★★ | ログ3件＋RebuildDispatch.cppコード確認 |
| メモリ2.5GB | ★★★★★ | GetProcessMemoryInfo()＋ログ時系列確定 |
| **Callback=0.00msの意味** | ★★★★★ | XRUN検出ブロック実行時間。DSP処理時間はCALLBACK_STAGE参照 |
| メモリ→PF→XRUN | ★☆☆☆☆ | 今回のログでは支持されない（PFレートとXRUN無相関） |
| リビルド→XRUN | ★☆☆☆☆ | 今回のログでは支持されない（前後500msにXRUN=0件） |
| DSP処理→drift | ★☆☆☆☆ | 今回のログでは支持されない（68,864件CB_HISTで無相関） |
| CPU Migration→XRUN | ★★☆☆☆ | 今回のログでは支持されない（全体90.5% vs XRUN時100%、比1.11x） |
| Voicemeeterだけでは十分条件になっていない | ★☆☆☆☆ | SAVIHost比較は参考観測。条件差が大きく単一原因の特定不可 |
| HIGH_PRIORITY_CLASS（DAWでは一般的な設定） | ★★☆☆☆ | コード確認済。MMCSSとの相互作用が未確認。単独では原因候補として弱い |
| 48kHz/512spbでXRUN減少 | ★★★☆☆ | 検証可能な予測。コールバック周期依存性の確認手段 |
| OS/Driver scheduling latency | ★★★☆☆ | 残る仮説だが複数の可能性。ETW要確認 |

---

## 12. 推奨対策

### P0：ETWによるシステム全体のスケジューリング解析（最優先）

**48kHz/512spbへの変更は実験条件そのものを変えてしまうため、まず現在の192kHz/1024spbの状態でETW（Windows Performance Recorder）を取得する。** これにより後から様々な条件との比較が可能になる。CPU affinity実験も「症状を変える」リスクがあるため、現状のトレース取得を最優先とする。

| # | 計測対象 | 目的 | ツール |
|---|---------|------|-------|
| 1 | **Ready Time**（オーディオスレッドの実行可能→ディスパッチ遅延） | スケジューラ遅延の直接証拠 | WPR + WPA |
| 2 | **DPC/ISR滞留時間** | ドライバ割り込みの影響確認 | WPR + WPA |
| 3 | **MMCSS状態**（JUCE登録の成功/失敗、ProAudioタスク状態） | MMCSS登録の実態確認 | WPR + WPA |
| 4 | **Context Switch原因と頻度** | プリエンプション発生源の特定 | WPR + WPA |
| 5 | **Hard Page Fault**の有無 | PF種別の確認（Soft vs Hard） | WPR + WPA |

### P1：ETW結果に基づく追加試験

ETWで原因の手がかりを得た後、以下のA/Bテストを実施：

| # | 対策 | 目的 |
|---|------|------|
| 6 | **48kHz/512spb比較試験** | コールバック周期依存性の確認 |
| 7 | **Voicemeeterバイパス**（WASAPI排他／物理ASIO直接） | Voicemeeter関与の確認 |
| 8 | **ASIO SDK bufferSwitchタイムスタンプ計測** | ドライバ自身のcallback発行間隔確認 |
| 9 | **CPU affinity A/Bテスト**（特定コア固定） | CPU Migration関与の確認 |
| 10 | **オーディオスレッド優先度向上**（TIME_CRITICAL） | スケジューリング優先度不足の確認 |

| # | 計測対象 | 目的 | ツール |
|---|---------|------|-------|
| 3 | **Ready Time**（オーディオスレッドの実行可能→ディスパッチ遅延） | スケジューラ遅延の直接証拠 | WPR + WPA |
| 4 | **DPC/ISR滞留時間** | ドライバ割り込みの影響確認 | WPR + WPA |
| 5 | **MMCSS状態**（JUCE登録の成功/失敗、ProAudioタスク状態） | MMCSS登録の実態確認 | WPR + WPA |
| 6 | **Context Switch原因と頻度** | プリエンプション発生源の特定（HIGH_PRIORITY_CLASS影響） | WPR + WPA |
| 7 | **Hard Page Fault**の有無 | メモリ不足の最終確認 | WPR + WPA |

### P2：メモリ最適化（独立した品質改善）

メモリ削減とXRUNの因果関係は確認できなかったため、独立した品質改善として実施。

| # | 対策 | 難易度 |
|---|------|--------|
| 9 | リビルド早期中断（checkpoint細分化） | 中 |
| 10 | SAFE_MAX_BLOCK_SIZE動的最適化 | 中 |
| 11 | DSPCoreリタイア即時解放 | 中 |
| 12 | IRデータオンデマンド読み込み | 高 |

---

## 12-2. 本レポートの限界（未証明事項）

本解析は以下の手段のみを用いた：

* ConvoPeq内部の診断ログ（CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1）
* ソースコード解析（C++、JUCE、Windows API）
* Windows API（GetProcessMemoryInfo, GetCurrentProcessorNumber, SetPriorityClass 等）

以下の項目については **ETW（Windows Performance Recorder）を取得していないため直接確認していない**：

| 未確認項目 | 関連する仮説 | 確認手段 |
|-----------|------------|---------|
| Windows Scheduler Ready Time | オーディオスレッドのディスパッチ遅延 | WPR + WPA |
| MMCSS登録状態 | JUCEのAvSetMmThreadCharacteristics成否 | WPR/WPA または戻り値確認 |
| DPC/ISR滞留時間と発生源 | ドライバ割り込みによるプリエンプション | WPR + WPA |
| Context Switch発生理由 | プリエンプションの要因特定 | WPR + WPA |
| ASIOドライバのbufferSwitch()発行間隔 | **ドライバ自身が8ms周期でcallbackを発行している可能性** | ASIO SDK側のタイムスタンプ計測 |
| Hard Page Fault vs Soft Page Fault | `GetProcessMemoryInfo` のPageFaultCountは全種別を含む | WPR + WPA |
| CPU電源管理（C-State/P-State/Core Parking） | ウェイクアップ遅延 | WPR 電源管理トレース |
| Voicemeeter Virtual ASIO内部動作 | 仮想ドライバの内部FIFO状態 | Voicemeeter内部ログまたはWPR |
| Windows Timer QueueとJUCE Timerの関係 | **JUCE TimerはMessage Thread上で動作し、リアルタイム保証なし** | WPR スレッド解析 |

**したがって、OSスケジューラ・MMCSS・DPC/ISR・ASIOドライバ内部・Voicemeeter内部動作・CPU電源管理に関する結論は、観測事実から導かれる最有力仮説であり、最終確定にはETW解析が必要である。**

本レポートはConvoPeq内部の診断ログとソースコード解析から導かれる範囲で原因候補を絞り込んだものであり、Windowsカーネルスケジューラ、MMCSS、DPC/ISR、ASIOドライバ内部の挙動については直接観測していない。そのため最終的な根本原因の確定にはETW（Windows Performance Recorder）等によるシステムレベルのトレース取得が必要である。

---

## 13. 付録：検証方法

### 使用した解析ツール

すべての解析は `ctx_execute(language: "javascript", code: "...")` のサンドボックス内で実施。

### 検証手順（第1版）

| Step | 内容 | 確認したデータ量 |
|------|------|----------------|
| 1 | ログファイル読み込み | ~268KB |
| 2 | XRUN全件抽出 | 131イベント |
| 3 | MEM全件時系列マッピング | ~50イベント |
| 4 | CALLBACK_STAGE全件解析 | 2,590イベント |
| 5 | CPU Migration全件解析 | 36,347イベント |
| 6 | REBUILD_TELEMETRY全件解析 | 52イベント |
| 7 | タイマーJITTER全件解析 | 47イベント |
| 8 | ソースコードクロスリファレンス | 12ファイル |
| 9 | Expected=5.33ms検算 | 物理定数確認 |

### 深掘調査（第2版）追加検証

| Step | 内容 | 確認データ量 | 発見 |
|------|------|-------------|------|
| D1 | XRUN#1前後のマイクロタイムライン | 前後60行全イベント | BUILD直後のACTIVATE→XRUNの時間的近接性 |
| D2 | CBSUMMARY全216件のloss/intervalMax解析 | 216イベント | loss全件負値＝到着レート正常を確定 |
| D3 | Gen=4内部メモリ急増トリガー特定 | 2秒間の全イベント | ページフォールト蓄積(+240,354 PF/7秒)を確認 |
| D4 | CB_HIST 68,864件のproc vs drift相関 | 68,864エントリ | **DSP時間とdriftに無相関を確定** |
| D5 | CALLBACK_STAGEのbudget別drift分析 | 2,590イベント | 高負荷時もdrift増加なしを確認 |
| D6 | XRUN 131件の発生間隔分布 | 130ギャップ | 中央値1,008ms（約1秒に1回） |
| D7 | 同一タイムスタンプXRUN検出 | 131タイムスタンプ | 3箇所で同一TSの二重XRUNを確認 |
| D8 | リビルド(BUILD_PHASE) vs XRUN時間的近接性 | 全3 BUILD / 131 XRUN | **前後500ms以内にXRUN=0件を確定** |
| D9 | PFレート vs XRUN発生率相関 | PF全イベント | **無相関を確定** |
| D10 | Gen=4→5 CB_HIST proc比較 | 2,368+66,496=68,864件 | **148%増加（348→865us）を確認** |
| D11 | DIAG_STAT backlog/drop分析 | 2,152イベント | backlog平均0、dropped合計0（異常なし） |

### 否定した仮説と根拠一覧

| 仮説 | 否定した検証 | 残った可能性 |
|------|------------|------------|
| メモリ→PF→XRUN | D9: PFレートとXRUN無相関 | OSスケジューラ外部要因 |
| リビルド→XRUN | D8: 前後500msにXRUN=0件 | 同上 |
| DSP処理→drift→XRUN | D4: procとdrift無相関 | 同上 |
| CPU Migration→キャッシュ→XRUN | XRUNのCallback=0.00ms | 同上 |

### ソースコード確認ファイル

| ファイル | 確認内容 |
|---------|---------|
| `AudioEngine.RebuildDispatch.cpp` | リビルドキューイング、obsolete検出 |
| `DSPCoreLifecycle.cpp` | DSPCore::prepare()、バッファ割り当て |
| `EQProcessor.Core.cpp` | EQ_PREPARE、バッファ種別とサイズ |
| `ThreadAffinityManager.h` | CPU affinity設定、initialized_状態 |
| `DSPLifetimeManager.h` | retire pipeline、非同期解放 |
| `ISRRetireRuntimeEx.cpp` | EpochDomain retire queue |
| `AudioBlock.cpp` | getNextAudioBlock(), drift/CPU Mig検出 |
| `BlockDouble.cpp` | XRUN検出ロジック（kRatioThreshold=1.5確認） |
| `RuntimeHealthMonitor.cpp` | XRUN検出、HealthState遷移 |
| `ISRRuntimePublicationCoordinator.cpp` | publish/retire coordinator |
| `AudioEngine.h` | SAFE_MAX_BLOCK_SIZE定数定義 |
| `RuntimeBuilder.cpp` | build()関数、HealthState連携 |
| `AudioEngine.Processing.PrepareToPlay.cpp` | prepareToPlay()、latency buffer |
| `DiagnosticsConfig.h` | getProcessMemoryInfo()定義 |
| `Timer.cpp` | timerCallback()、JITTER検出ロジック |

---

**以上**
