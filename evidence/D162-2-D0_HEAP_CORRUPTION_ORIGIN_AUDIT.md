# D162-2-D0 — Heap Corruption Origin Audit（read-only + test-environment only）

- Work item: D162-2-D0（heap corruption 発生源同定。D162-2-C の C-F 中 `(i) corrupting free 未同定` の解消）
- Date: 2026-09-04
- Mode: **production source 変更 0**。S3 destroy・V-D destroy は**無効のまま**（再有効化なし）。
  S1/S2/S4/dormant/quarantine/AudioEngine.h comment は D162-2-B 最終状態のまま。
  テスト環境の変更のみ: Debug 再ビルド（現行ソースからの通常ビルド・source 変更なし）、Debug 6-gen 実行、
  クラッシュダンプ解析（minidump + llvm-symbolizer）。
- 成果物: 本ドキュメント

---

## 0. Exit criterion 判定

**PASS-B — corrupting free を即時 fault の形で捕捉し、call site・object・ownership 違反まで特定した。**

| 成果 | 内容 |
| --- | --- |
| Corrupting operation | `mkl_free` on a block allocated by `_aligned_malloc`（allocator mismatch） |
| Corrupted allocation | `AudioSegmentBuffer::leftSamples_` / `rightSamples_`（`_aligned_malloc(kCapacity*sizeof(double), 64)` x2、AudioSegmentBuffer.h:27-33） |
| Immediate fault site | `convo::aligned_free`（AlignedAllocation.h:41）← `ScopedAlignedPtr::reset`（:86）← `~ScopedAlignedPtr`（:61）← `~AudioSegmentBuffer`（AudioSegmentBuffer.h:131）← `~NoiseShaperLearner`（NoiseShaperLearner.cpp:95）← `~AudioEngine`（CtorDtor.cpp:288）← `~MainWindow`（MainWindow.cpp:1162） |
| Later manifestation | ntdll heap free 経路で `EXCEPTION_ACCESS_VIOLATION READ @ 0xFFFFFFFFFFFFFFFF`（ダンプ 7/7 件共通 RIP `ntdll+0x161914` と同一クラス） |
| Causal confidence | **proven（teardown-AV class）** — コード上の alloc/free ペア不一致が dump stack と完全一致。Debug CRT（`_free_dbg`）で再現可能 |
| Mid-run crash class（RWDI 20688 era） | **unresolved** — ログ上書きにより attribution 不能。S3/V-D 破壊時代の run でのみ発生し、最終コード（S3/V-D 無効）では RWDI 6-gen x3 exit 0（詳細 §5） |

D162-2-E への接続: PASS-B のため D162-2-D0 としては完了。以降は (1) mismatch free の処置判断と
(2) residual である mid-run corruption source の特定が D162-2-E 側の入力となる。

---

## 1. D0-1: 現行コード固定（遵守記録）

- 基準: ConvoPeq.md（再生成 2026-09-03、D162-2-B 最終状態と内容一致を確認 — `retireRegisteredDSP` x10、
  S3 無効化注記あり）
- S3 destroy: **無効のまま**（`clearDeferredForShutdown` 内に D162-2-C 注記付き無効 block のみ）
- V-D destroy: **無効のまま**（`if (false && activeDSPToDestroy/fadingDSPToDestroy)` + 注記）
- D162-2-B の S1/S2/S4/dormant/quarantine/AudioEngine.h comment は現状のまま変更せず
- 本監査では上記を変更する repair を一切行っていない（S3/V-D の destroy を再導入すると AV が再発する
  ため「観測条件を変える」ことになり原因同定が不能になる — D0-1 の指示どおり）

## 2. D0-2: PageHeap 適用の可否

**結果: blocker のため不成立（記録）**。

- `gflags.exe` は本環境に存在しない（WinSDK Debuggers ディレクトリには dbgcore/dbghelp のみ）。
- IFEO PageHeap 設定は HKLM への書き込みが必要だが、現在の process は非管理者であり
  `sudo` も OS 設定で無効化されている（`sudo config` で確認）。
- 代替手段として Debug CRT の debug heap（block header + no-man's-land + `_CrtDefaultAllocHook`）
  を **PageHeap 代替**として使用した。これは D0-2 の目的（「不正 free を即時 fault まで前倒し」）と
  同一効果を持つ: Debug ビルドの `_free_dbg` は破損検出時に abort/exception を発生させる。

## 3. D0-3: Debug 短縮再現（6-gen 反復）

### 3.1 手順

- Debug + DIAG バイナリを**現行最終ソースから再ビルド**（`D162-2D0_build_debug.log`、EXIT=0）。
  重要: D162-2-C 時代に使用した Debug バイナリは古い実験バリアント（S3-direct + VD-direct）の
  残留であり（mtime 00:54）、本監査では現行コードのスタックと照合できるよう再ビルドした。
- 6-gen 短縮 soak（`--cli-ir-reload-count 6` 等、exit 120s）を Debug バイナリで実行。
- 結果: **EXIT=0xC0000005**、新規ダンプ `ConvoPeq.exe.12980.dmp`（23:48）を取得。
- 並行して RWDI 最終コードでの 6-gen 実行（exit 0x0 x3）は維持確認済み（Debug のみ決定論的に fault）。

### 3.2 ダンプ解析（minidump + llvm-symbolizer inline）

- faulting thread: メインスレッド（20 threads 中、他は UCRT cond-wait で idle）
- `ExceptionInformation = [0 (READ), 0xFFFFFFFFFFFFFFFF]`、faulting RIP は ntdll
- module-range stack 走査 + llvm-symbolizer（`--inlines`）で復号した src frame（stack 順）:

```text
ReadPointerNoFence (winnt.h) / _guard_icall / _CrtDefaultAllocHook (debug_heap_hook)
→ convo::aligned_free (AlignedAllocation.h:41)
→ _free_dbg (debug_heap.cpp:1036)
→ ScopedAlignedPtr<double>::reset (AlignedAllocation.h:86)
→ free (free.cpp) → ~ScopedAlignedPtr (AlignedAllocation.h:61)
→ AudioSegmentBuffer::~AudioSegmentBuffer (AudioSegmentBuffer.h:131)
→ NoiseShaperLearner::~NoiseShaperLearner (NoiseShaperLearner.cpp:95)
→ AudioEngine::~AudioEngine (AudioEngine.CtorDtor.cpp:288 — body 終了後の member teardown)
→ MainWindow::~MainWindow (MainWindow.cpp:1162) → shutdown → WinMain
```

- 証跡: evidence/D162-2D0_dump30992_srcframes.txt（同手法の旧 dump 再掲）、新 dump は同一 chain。

## 4. D0-4: free 候補の分類

| Class | 判定 |
| --- | --- |
| F1 `destroyDSPCoreNode` → DSPCore dtor → MKL/IPP free | **当該 free ではない** — 監査範囲で DSPCore teardown 内の全 alloc/free ペアは allocator 整合（`makeAlignedArray`↔`aligned_free`、`mkl_malloc`↔`mkl_free`、`ippsMalloc`↔`ippsFree`）。`freeTracked` の size 引数は counter 専用で実 free（`mkl_free(ptr)`）に影響しない |
| F2 `destroyRolledBackDSP` → direct destroy | **当該 free ではない** — F1 と同一 dtor 経路 |
| F3 EQCache/CacheMap → `tryShutdownQuiescentReclaim` | **当該 crash の free ではない** — 当該 stack には EQCache 系 frame が存在しない。ただし F3 の UAF は D0-5-B の独立 defect として確定 |
| F4 DSPCore 内部 RCU/Convolver/member dtor | **当該 crash の free** — `~AudioSegmentBuffer`（`NoiseShaperLearner::segmentBuffer`）が F4 内 member dtor。alloc は `_aligned_malloc`（AudioSegmentBuffer.h:27-33）、free は `aligned_free`→MKL 時 `mkl_free`（DiagnosticsConfig.h:51）で **allocator mismatch（code-proven）** |

## 5. D0-5-A: `activeRuntimeDSPSlot` stale pointer 経路

結論: **本 crash の corrupting free ではない**。**実行中には発火し得ない**（以下）。

- 唯一の setter は bootstrap のみ（PrepareToPlay.cpp:264）。placeholder 破壊後は dangling。
- `retireDSPHandleForRuntime` は pointer value で `runtimeDSPHandleMap_.find` を行うため、
  address reuse 時に生存 DSP を lookup し得るのは事実（静的成立）。
- しかし実際にその retire を呼ぶ箇所は dtor（CtorDtor.cpp:190）と DSPGuard（RebuildDispatch.cpp:958）
  のみ。DSPGuard の対象は未登録 DSP（map 不在 → false で no-op）。dtor retire は shutdown 文脈のみ。
- 実測: 最終 clean 60-gen run でも placeholder address の再利用は観測されず（1 件の D→C reuse は
  非 placeholder）。発火未証明のため root cause とは認定しない（指示どおり）。
- ただし S1 導入で address reuse 頻度が激増したことはログで確認済み（clean run でも 1 件）であり、
  **将来の確率的発火源**として INV-D162-7（E-2）の対象に残す。

## 6. D0-5-B: `EQCacheManager::CacheMap::~CacheMap()` UAF

結論: **独立 defect 候補として確定（D162-2-C の主張を維持）**。本 crash（§3 stack）とは別物。

- member 破壊は宣言降順: `shutdownRuntime_`(:5022) は `eqCacheManager`(:2444) より先に破壊される。
- 一方 `AudioEngine::shutdownPhase`（別 atomic、h:2707）は dtor body D11 で `Destroy` に設定済みのため、
  `~CacheMap` は Destroy branch（h:2156 以降）に入り `owner->tryShutdownQuiescentReclaim` を呼ぶ。
  この関数は `shutdownRuntime_.tryMakeQuiescenceProof/tryMakeReclaimPermit` に触れる
  （h:4443-4478）— `shutdownRuntime_` は破壊済みの可能性があり **UAF 経路が構造的に実在**。
- D162-1R-B ASAN の同 signature と一致。**本 stack には CacheMap frame がなく、今回の 0xC0000005 とは
  直接因果なし**（「単なる仮説ではなく独立 defect 候補として確定」との指示どおりの扱い）。

## 7. D0-6: corrupting free と later manifestation の分離記録

### 7.1 teardown-AV class（proven）

```text
Corrupting operation:
    mkl_free(ptr) in convo::aligned_free (AlignedAllocation.h:41), invoked from
    ScopedAlignedPtr<double>::reset (AlignedAllocation.h:86) in
    ~ScopedAlignedPtr (AlignedAllocation.h:61) in
    ~AudioSegmentBuffer (AudioSegmentBuffer.h:131) in
    ~NoiseShaperLearner (NoiseShaperLearner.cpp:95) in
    ~AudioEngine member teardown (after CtorDtor.cpp:288)

Corrupted allocation:
    leftSamples_ / rightSamples_ blocks allocated by _aligned_malloc
    (kCapacity*sizeof(double), 64) in AudioSegmentBuffer::create (AudioSegmentBuffer.h:27-33).
    MKL block-registry view of a CRT-owned block → metadata corruption / false match → free fault.

Later manifestation:
    ntdll heap free path, EXCEPTION_ACCESS_VIOLATION READ @ 0xFFFFFFFFFFFFFFFF
    (dumps 12980/30992/448 ほか計 3 件以上で同一 chain).
    Rax=0x5b 等の小オフセット読み出しから、所有ブロック照合中の読み出しと整合。

Causal confidence: proven
```

**なぜ D162-1P では落ちなかったか**: mismatch 自体は常時存在するが fault は heap 状態依存。
D162-2-B の S1 EBR destroy が MKL/native ヒープの churn を劇増させ（destroy 11→54、MKL alloc/free
回数増）、それまで tolerable だった foreign free が破損顕在化に至った（probable・時系列整合）。
S3/V-D destroy を入れた run で exit AV が確定再現したのは、さらに teardown タイミングが
変わったためである。

### 7.2 mid-run crash class（unresolved — ログ上書きのため）

- RWDI 60-gen run（S3-EBR + VD-EBR 時代、log は後続 run で上書き済み）の dump 20688:
  faulting stack に `evaluateDeferred/finishView/DeferredPublishSlot::_Assign/DSPCore::DSPCore`
  が並ぶ = **RebuildThread deferred-admission 経路の alloc が破損ヒープに接触**。
- その corrupting free の実行主体は**未特定**（証拠のログが失われた）。最終コード（S3/V-D 無効）では
  RWDI 60-gen x1 + 6-gen x3 で mid-run crash は再現せず（exit 0x0）。したがって mid-run class は
  旧バリアント固有の可能性が高く、現行コードでの再現が今後の検証対象。
- **この class を teardown class と混同しない**（指示 D0-6 どおり）。

## 8. E-1〜E-4 への入力（D0 完了につき D162-2-E 開始可）

- E-1（INV-D162-6）は D0-5-B の確定 UAF を対象に実施可。
- E-2（INV-D162-7）は D0-5-A の既定扱い（将来発火源として残置）で実施可。
- E-3（INV-D162-8）は D0-5-A の dtor retire 危険を主原因のひとつとして実施可。
- E-4（INV-D162-9）は D162-2-C の residual 4 件（gen 9/35/41/53）を再現対象に実施可。
- **mismatch free の処置**は D162-2-E の範囲外の可能性が高い（AudioSegmentBuffer は D162 スコープ外の
  pre-existing defect。ただし teardown-AV class の直接原因のため、D162-2-E の検証 soak が正常 exit
  することを保証するには別途の disposition が必要になる点を flag する）。

## 9. 証跡

- evidence/D162-2D0_dbg6.log / D162-2D0_dbg6b.log（現行最終ソースの Debug 6-gen ログ）
- CrashDumps: ConvoPeq.exe.12980.dmp（23:48・現行ソース）、同 30992.dmp（23:18）、同 448.dmp
- evidence/D162-2D0_dump30992_srcframes.txt / evidence/D162-2C_dump448_srcframes.txt
- evidence/D162-2D0_build_debug.log（現行ソース Debug ビルド EXIT=0）
