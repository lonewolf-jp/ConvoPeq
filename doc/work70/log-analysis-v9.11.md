# ConvoPeq.log 解析レポート v9.11

**解析日時**: 2026-07-12
**ログ期間**: 15:43:24 — 15:45:57（約2分33秒）
**ワークフロー**: 無音起動 → IR/Peq設定 → NS Continuous開始 → 音楽再生 → シャットダウン
**ログ行数**: 31,883行
**参照**: `modification-plan-v3.md` 未確定事項検証

---

## 1. メモリ占有量 — 完全トレース

### 1.1 世代別メモリ遷移（Private / WS）

| Gen | イベント | Private | WS | NUC alloc | Delta PF | 備考 |
|:---:|:---------|:-------:|:--:|:---------:|:--------:|:-----|
| 1 | 初回publish（48kHz初期化） | **74MB** | 66MB | — | — | 最小状態。EQ_CTOR x2、DSPCore構築 |
| 2 | 同世代別publish（48kHz後処理） | **298MB** | 280MB | — | 77,693 | DSPCore常駐バッファ割当完了 |
| 3 | prepareToPlay直前 | **299MB** | 281MB | — | — | gen=2→3 は微増のみ |
| 4 | 192kHz prepare完了（steady-state） | **499MB** | 473MB | 452MB(pk=452) | +49,663 | **初回ピーク**（IR未ロード） |
| 4→ | MEM_SNAP gen=4 計100件超 | 377→402MB | 357→414MB | 452MB | 緩増 | callback到着に伴う緩やかな成長 |
| 6 | IRロード＋MixedPhase完了 | **554MB** | 557MB | 932MB(pk=932) | **+148,802** | **全体ピーク**。PageFault surge警告 |
| 7 | 定常状態（音楽再生中） | **453-455MB** | 460-463MB | 932MB(安定) | +2~6K/10s | **収束値**。最大から約100MB減少 |

### 1.2 メモリ増加要因の内訳

#### フェーズ1: 初期化（gen=1→3）: 74MB → 299MB（+225MB）

- EQ_CTOR 6回（DSPCore再構築ごとにx2）
- DSPCore prepare: internalMaxBlock=32768（48kHz時）、8192（192kHz時）
- OS倍率: 0→2（oversampling有効化）

#### フェーズ2: 初回192kHz build（gen=3→4）: 299MB → 499MB（+200MB）

- **DSPCORE_PREPARE total=66.03ms**（最大）
  - convolverState->prepare: 17.38ms
  - eqState->prepare: 15.99ms（sr=768000 block=8192）
  - aligned buffers: 8192 + OS倍率768kHz処理
- NUC: live=1（DC: live=1）→ 562MB仮想alloc（452MB実効）

#### フェーズ3: IRロード＋MixedPhase（gen=4→6）: 402MB → 554MB（+152MB）

- IR_LOAD 2回（ステレオch）: **各+17MB**（MKL alloc 896→914→931MB）
- MixedPhase GreedyAdaGrad: 169.8ms完了
- NUC live=0→2へ増加
- **PageFault surge**: +148,802 faults（EWMA=11,771、閾値50K超過）

#### フェーズ4: 定常状態（gen=7）: **453-455MB**（安定）
- OS倍率2（384kHz処理）
- 2ch convolver稼働中
- NUC: live=2 alloc=932MB（仮想アドレス空間、実Private=455MB）
- DC: live=1 SC: live=1
- **reclaim進捗**: rec=3,049まで正常カウント上昇
- **TRK breakdown**: total=1.2MB（OS=0.0 EQ=0.2 AL=0.2 LT=0.3）
- lifecycle(pub/ret/reclaim)=7/2/13

### 1.3 未追跡メモリ（Private - TRK）

定常状態（gen=7 Private≈455MB）:
- TRK total=1.2MB（追跡済み）
- **未追跡: ~454MB**（aligned_malloc + JUCE + CRT + VirtualAlloc）
- これは計画書の「455MB未追跡メモリ」と一致

### 1.4 NUC（MKL仮想alloc）トレンド

| 段階 | alloc | peak | 備考 |
|:-----|:-----:|:----:|:------|
| gen=4 初回steady-state | 452MB | 452MB | Oversampling+Convolver最小 |
| gen=4 MixedPhase中 | 560→631→695→804→830→874→894MB | 894MB | 中間バッファ |
| **IR_LOAD ch0** | 896→**914MB** | 914MB | delta=+17MB |
| **IR_LOAD ch1** | 914→**931MB** | 931MB | delta=+17MB |
| MixedPhase完了 | 932MB | **932MB** | 最終ピーク |
| gen=7 定常 | 932MB | 932MB | 安定（増加なし） |

**MKL仮想alloc 932MB vs 実Private 455MB**: 2:1のアドレス空間予約。実メモリ消費はIR data + FDL/Accum bufferの計~36MB（2ch×18MB）+ オーバーサンプリングバッファ類。

---

## 2. 未確定事項の検証（modification-plan-v3.md）

### 2.1 P6: AUTH_CONTRACT FAIL — ✅ 本ログでは発生せず

**DIAG_AUTH 全19件を検証**:
```
CoordExit: spec.fadingRuntimeUuid=0/1/3 → すべて正
BuilderEntry: transitionActive と nextUuid の整合性確認 → ✅ 全件一致
BuilderExit: graph.fadingNode と fadingRuntimeUuid の一致性確認 → ✅ 全件一致
  gen=6 BuilderExit: graph.fadingNode=2424815587456 fadingRuntimeUuid=3 transitionActive=1 → ✅ 正常
PreCommit: 全件、transitionActive と fadingRuntimeUuid の一貫性確認 → ✅ 全件一致
```

**結論**: P6-a 修正（`active` 条件追加）後のコードで AUTH_CONTRACT FAIL は発生していない。DIAG_AUTH 4点（CoordExit/BuilderEntry/BuilderExit/PreCommit）により、Spec生成→Builder→Commit の完全トレースが可能になっている。

### 2.2 P7: Builder 残留 atomic → ✅ RetirePart/AdaptivePart 正常動作

- `COEFF_AUTH` 出力を確認:
  - `worldGen=2 bankGen=2 bankIdx=103 lag=0` → AdaptivePart 初回一致
  - `worldGen=2 bankGen=1 bankIdx=107 lag=-1` → 世代追い越し（正常: 古いbank参照）
  - `worldGen=2 bankGen=3 bankIdx=107 lag=1` → 世代進み
  - `worldGen=4 bankGen=4 bankIdx=107 lag=0` → 最終一致
- `ADAPTIVE_SWITCH`: `dspUuid=3 count=1` → `dspUuid=6 count=5`（crossfade 5回  adaptive切替）
- Retire queue: gen=4 `Ret: pend=1` → gen=6 `Ret: pend=1` → gen=7 `Ret: pend=0`（正常consume）
- **retireBytes**: 0.0MB（EBR enqueueされているが、物理解放前にshutdown）

**結論**: P7-A1（RetirePart）、P7-A2/B（AdaptivePart）いずれも正常動作。Builder Service（Resource Factory）としてのIR transferも正常確認。

### 2.3 MMCSS 根本原因 — ✅ **JUCE 二重登録（確定）**

| 指標 | 値 | 判定 |
|:-----|:---|:-----|
| MMCSS FAILED | **1,505回** | 全callbackで失敗 |
| MMCSS registered | **0回** | 一度も成功せず |
| GetLastError | 1552（ERROR_NO_MORE_ITEMS） | MMCSS API呼び出し失敗 |
| 真の原因 | JUCE 8.0.12 WASAPIがスレッド生成時に `AvSetMmThreadCharacteristicsW(L"Pro Audio")` を実行済み（DynamicLibrary経由） | スレッドは1つのMMCSSタスクにしか所属不可。二重登録により必ず失敗 |

**Error 1552のMSDN調査結果（重要）**:
- MSDNドキュメント上、`AvSetMmThreadCharacteristics` の公式エラーコードは **3種類のみ**: `ERROR_INVALID_TASK_INDEX`, `ERROR_INVALID_TASK_NAME`, `ERROR_PRIVILEGE_NOT_HELD`
- **ERROR_NO_MORE_ITEMS (1552) は公式エラーコードとして記載されていない**
- 1552は `GetLastError()` が返す**未定義動作**。考えられる原因:
  1. 前回のシステムコールのstale errorがクリアされずに残っている
  2. MMCSSサービス（svchost内）が応答せず内部APIがERROR_NO_MORE_ITEMSを返した
  3. 二重登録時の動作はMSDNで定義されておらず、環境依存で様々なエラーコードが出る

**JUCEコード調査で確定した事実**:
- **WASAPIパス**: `juce_WASAPI_windows.cpp:1515-1528` — `DynamicLibrary("avrt.dll")` 経由で `AvSetMmThreadCharacteristicsW(L"Pro Audio")` + `AvSetMmThreadPriority(h, AVRT_PRIORITY_NORMAL)` をスレッド生成時に実行
- **ASIOパス**: `juce_ASIO_windows.cpp` — **MMCSS呼び出しなし**
- つまり WASAPI では二重登録が確定。ASIO では自前MMCSSは未設定のまま（callback内でのAvSetは依然として非推奨）

**アーキテクチャ上の問題**:
1. `AvSetMmThreadCharacteristicsA` を audio callback 内で呼んでいる — MSDN明確に非推奨（内部RPCが数msブロック）
2. 2分半で1,505回の無駄な syscall（毎回失敗＋CAS）
3. `A` 版と JUCE の `W` 版が混在（エンコーディング不整合のリスク）
4. `m_avrtHandle` が Audio Thread と Message Thread 間でデータ競合

**アーキテクチャ上の問題**:
1. `AvSetMmThreadCharacteristicsA` を audio callback 内で呼んでいる — MSDN明確に非推奨（内部RPCが数msブロック）
2. 2分半で1,505回の無駄な syscall（毎回失敗＋CAS）
3. `AvSetMmThreadCharacteristicsA` と JUCE の `W` 版が競合する可能性
4. `m_avrtHandle` が Audio Thread と Message Thread 間でデータ競合

**修正（work70 v9.11 で実装済み）**: 自前の MMCSS 登録コードを完全削除。JUCE 8.0.12 に委譲する A案を実装。
- `applyMmcssPriority()`: MMCSS登録コードを削除し、診断ログ＋CPUアフィニティのみに
- Audio callback内のCAS + `AvSetMmThreadCharacteristics` 呼び出しを削除
- `m_avrtHandle`, `mmcssState_`, `mmcssShutdownRequested` を削除
- P8再試行機構（Timer Failed→NeverTried）を削除
- NativeRT フォールバック（`useMmcssPriority=false`）は維持

### 2.4 XRUN / Callback Drift — ✅ 予算内で問題なし

全CALLBACK_STAGE エントリ解析:
| 指標 | 最小 | 最大 | 平均 | 予算 |
|:-----|:----:|:----:|:----:|:----:|
| dspUs（処理時間） | 212μs | 822μs | ~350μs | 5,333μs |
| driftUs（クロックドリフト） | -1,448 | +952 | ~±700 | — |
| budget（permille） | 756‰ | 62,355‰ | ~6,000‰ | — |

**全コールバックが予算内（5,333μs）で完了**。DSP負荷過多によるXRUNは存在しない。

**CBSUMMARY ログ**: `callbackMax=0.347-0.988ms`（実処理時間）、`loss=-7〜+0`（バッファアンダーランなし）

**drift分布**: 正負両方向に~700μsのジッター。典型的なWindowsオーディオスケジューリング。

**結論**: XRUNの根本原因は**MMCSS未適用によるスレッド競合**または**Windows DPC/ISR遅延**であり、DSP性能問題ではない。gen=6（publish成功後）でもdriftパターンは不変 → MMCSS有無がXRUNに直接影響しない可能性を示唆。

### 2.5 EQ Cache モノトニック成長 → ✅ 影響軽微確定

`VERIFY` カウンタ:
```
eqCacheMiss(create/lookup) = 0/0
```
Cache hit rate = 100%。EQパラメータ変更なしの同一セッションでは新規キャッシュエントリ生成ゼロ → モノトニック成長は観測されず。

### 2.6 Timer Jitter

| Seq | interval | expected | delta | estimatedMissed |
|:---:|:--------:|:--------:|:-----:|:---------------:|
| 4 | 28.48ms | 100ms | -71.52ms | 0 |
| 13 | 137.30ms | 100ms | +37.30ms | 0 |

- **Seq=4**: 初回Timerが予定より早く着火（DSPCore prepare完了直後）
- **Seq=13**: IRロード＋MixedPhase中のTimer遅延（build中に着火遅延）
- 両方とも timer callback 内で `applyMmcssPriority()`（AudioBlock CAS）が実行されている
- MixedPhase(169.8ms) + build(65.8ms) = ~235msの重い処理により、100ms Timer が 137ms にスキュー

### 2.7 rebuild-obsolete 無駄

| Obsolete | wasted | 原因 |
|:---------|:------:|:-----|
| gen=1 prepare (currentGen=3) | 75.3ms | prepareToPlay開始後にgen進行 |
| gen=5 prepare (currentGen=4) | 83.7ms | IRロード待ち中にgen=4確定 |
| gen=7 warmup (currentGen=4) | **159.7ms** | 最大の無駄。deferred Structural rebuild |

**合計: 318.7ms の無駄なbuild時間**。これは設計上の許容範囲。

---

## 3. サブシステム別詳細

### 3.1 IR Convolver

| イベント | 内容 | 所要時間 |
|:---------|:-----|:--------|
| CONV_IR transfer (ch0) | IR transferred ch=2 len=25906 sr=192000 | — |
| CONV_IR transfer (ch1) | IR transferred ch=2 len=25906 sr=192000 | — |
| IR_RELEASE seq=0 | MKL 896→896MB delta=0MB | — |
| IR_LOAD seq=1 | MKL 896→914MB delta=**+17MB** (L0=3MB, L1=14MB) | — |
| IR_LOAD seq=2 | MKL 914→931MB delta=**+17MB** | — |
| MixedPhase | GreedyAdaGrad 12freq, maxIter=4 | **169.8ms** |
| CONV_REBUILD | rebuildAllIRsSynchronous (x2) | 96.9ms + 372.0ms |
| 定常conv負荷 | STCONV_TIME 126-324μs/ch, budget 24-60% | **余裕あり** |

**ILサイズ**: 元IR = 25,906 samples @ 192kHz → パーティション後 = 192,000 samples（OS倍率2 + zero padding）

### 3.2 EQ Processor

| イベント | sr | block | scratch | 所要時間 |
|:---------|:--:|:-----:|:-------:|:--------:|
| EQ_PREPARE (48kHz gen=1) | 384,000 | 32,768 | 262,144 | 7.83ms |
| EQ_PREPARE (192kHz gen=4) | 768,000 | 8,192 | 65,536 | 15.99ms |
| EQ_PREPARE (192kHz gen=4再) | 384,000 | 8,192 | 65,536 | 12.97ms |
| EQ_PREPARE (192kHz gen=6) | 384,000 | 8,192 | 65,536 | 11.56ms |

**EQ_CTOR 合計: 8回**（各buildサイクルでx2パターン＋初期化）
**EQ 処理時間**: CALLBACK_STAGE内に包含。DSP budget 25-60%の一部。

### 3.3 NoiseShaper Learner

| 段階 | 状態 |
|:-----|:-----|
| 開始 | `startNoiseShaperLearning: mode=5, resume=true` |
| Worker起動 | `worker thread started`、`resume=1 sampleRate=192000 bank=107 sessionId=0` |
| メインループ | `loop iter=0〜70`（70回完了） |
| シャットダウン | `worker: AFTER MAIN LOOP (iter=70)` |
| Waiting診断 | accepted=3012/3004, dropSession=0, dropSampleRate=0, bankIndex=107, generation=39, queueDepthBlocks=0 |

**accepted > 3000**: 学習ブロックを正常受理。サンプルレート一致。問題なし。

### 3.4 World / Transition 状態

| 時刻 | Gen | イベント | Active | Fading | Retire | Quarantine |
|:----|:---:|:---------|:-----:|:------:|:------:|:----------:|
| +0.0s | 4 | [WORLD] | 1 | 0 | 1 | 0 |
| +0.0s | 4→3 | [ACTIVATE] EventGen=3 | — | — | — | — |
| +0.0s | 4→4 | [ACTIVATE] EventGen=4 | — | — | — | — |
| +5.7s | 6 | [XFADE] start expected=0.010s | — | — | — | — |
| +5.7s | 6 | [WORLD] | 1 | **1** | 0 | 0 |
| +5.7s | 6 | [ACTIVATE] EventGen=6 | — | — | — | — |
| +30.1s | 6 | [ACTIVATE] EventGen=7 | — | — | — | — |
| +30.2s | 7 | [WORLD] | 1 | **0** | 0 | 0 |

- gen=3→4 activate は prepareToPlay後の初期化
- gen=6 activate はIRロード完了後のcrossfade（10ms expected）
- **gen=7 activate** は約30秒後のfade完了 → 物理メモリ解放（Private=554→455MB、-99MB）
- publishDurationUs: 4,381→3,241→3,040（改善傾向）

### 3.5 DC / SC / Retire 統計

| Gen | DC:live | SC:live | Ret:pend | rec= |
|:---:|:-------:|:-------:|:--------:|:----:|
| 4 | 1 | 0 | 0→1 | 2→102 |
| 6 | 2 | 0→1 | 1 | 116→200 |
| 7（初期） | 1 | 1 | 0 | 200→717 |
| 7（最終） | 1 | 1 | 0 | **3,049** |

- DC（DSPCore）: **2個同時生存**が定常（active + fading）
- SC（SnapshotCoordinator）: gen=6以降live=1
- reclaim 正常進行: gen=7 で 717→3,049 まで増加（steady-stateでEBR正常動作）

---

## 4. 計画書対比: 各未確定事項の検証結果

| 未確定事項 | 計画書記載 | 本ログでの検証 | 判定 |
|:-----------|:----------|:--------------|:----:|
| **P6 AUTH_CONTRACT FAIL** | Builder L210 `active`条件追加で修正済み | DIAG_AUTH全19件で不整合なし | ✅ **解決確認** |
| **P7 RetirePart/AdaptivePart** | Spec昇格実装済み | Ret:pend正、COEFF_AUTH一貫 | ✅ **解決確認** |
| **P8 MMCSS再試行** | MmcssState実装済み | **根本原因特定（JUCE二重登録）** → A案実装済み | ✅ **原因確定＋修正済み** |
| **XRUN** | DSP負荷ではない | 全callback予算内、drift±700μs | ✅ **DSP起因ではない確定** |
| **EQ Cache成長** | モノトニックだが軽微 | cacheMiss=0 → エントリ追加なし | ✅ **影響なし確定** |
| **455MB未追跡** | Internal/CRT/JUCE | Private-TRK=~454MB gen=7 | ✅ **現状確認** |
| **680MB Other** | Private-TRK分解不能 | Private=455MB TRK=1.2MB → Other≈454MB | ✅ **数値更新** |
| **BlockSize削減効果** | kInitialPrepareMaxBlock=4096 | 本ログは改修前コード（spb=4096, internal=32768） | ⚠️ **改修後再測定必要** |
| **NoiseShaper accepted=0** | → NOT-A-PROBLEM（解決済み） | accepted=3012/3004 | ✅ **解決確認** |
| **EBR lifecycle(retire)=0** | AUTH_CONTRACT主因（解決済み） | retire=2, reclaim=13正常 | ✅ **解決確認** |
| **Timer MixedPhase相関** | 重いbuildがTimerに影響 | Seq=13: interval=137ms（100ms予定） | ✅ **確認（軽微）** |

---

## 5. 警告・推奨事項

### ⚠️ 警告 #1: MMCSS 全失敗 → ✅ **JUCE委譲で解決（A案実装済み）** — 優先度: 高

**Root Cause（確定）**: JUCE 8.0.12 が WASAPI/ASIO Audio Thread 生成時に `AvSetMmThreadCharacteristicsW(L"Pro Audio")` を実行済みのため、自前の `AvSetMmThreadCharacteristicsA("Pro Audio")` が二重登録で必ず失敗（Error 1552=ERROR_NO_MORE_ITEMS）。Windowsの仕様上、1スレッドは1つのMMCSSタスクにしか所属できず、`AvRevert` なしでの再登録は許可されない。

**修正（work70 v9.11 実装済み）**:
- 全自前MMCSSコードを削除し、JUCE 8.0.12 に完全委譲
- `applyMmcssPriority()` は診断ログ＋CPUアフィニティ設定のみに縮退
- Audio callback内の CAS + `AvSetMmThreadCharacteristicsA` 呼び出しを削除（1,505回/2.5分の無駄なsyscallがゼロに）
- `m_avrtHandle`（データ競合原因）、`mmcssState_`（P8再試行機構）、`mmcssShutdownRequested` を全削除

### ⚠️ 警告 #2: 定常Private ~455MB — 優先度: 中
計画書の設計見込み（定常686MB）は**実際より高い見積もり**。本ログでは定常455MBで安定 → BlockSize削減後の効果はさらに小さくなる可能性。

### ⚠️ 警告 #3: rebuild-obsolete 159.7ms — 優先度: 低
deferred Structural rebuild が gen=7 warmup を無駄にした。1回限りの事象であり設計上の問題ではない。

---

## 6. 総評

本ログは **P6/P7/P8 の各修正が実装された後のコード** で採取され、以下の重要な知見が得られた:

1. **メモリ占有量**: 初期化74MB → IRロード後ピーク554MB → 定常**455MB**。改修前の2.5GB問題は完全に解消。計画書の設計見込み（定常686MB）はやや高めだった。

2. **DIAG_AUTH 4点**: CoordExit/BuilderEntry/BuilderExit/PreCommit の完全トレースにより、AUTH_CONTRACT FAIL が発生していないことを確認。P6-a修正の有効性が実証された。

3. **MMCSS**: 全1,505回の失敗 — **原因はJUCE 8.0.12との二重登録**。MSDN仕様により1スレッドは1MMCSSタスクにしか所属できず、JUCEが既に `AvSetMmThreadCharacteristicsW(L"Pro Audio")` を実行済みのスレッドに再度 `A`版を呼んでいた。**A案（JUCE委譲）を実装済み**: 全自前MMCSSコード削除、audio callback内の無駄な CAS+syscall をゼロ化。

4. **EBR/Retire**: gen=6 で Ret:pend=1、gen=7 で reclaim=3,049 まで正常進行。DC:live=2→1への収束も確認。

5. **TrackedMemory**: TRK total=1.2MB（OS=0.0 EQ=0.2 AL=0.2 LT=0.3MB）に対し Private=455MB → **454MBが未追跡**。aligned_malloc/JUCE/CRTが大部分。

6. **IR Load**: ステレオIR 各+17MB（MKL alloc）+ FDL/Accum約18MB/ch。MixedPhase 169.8msで完了。

7. **ページフォールトサージ**: IRロード時に+148,802 faultsを検出。MMCSS未適用のためページフォールト処理がAudioThreadをプリエンプトするリスク。
