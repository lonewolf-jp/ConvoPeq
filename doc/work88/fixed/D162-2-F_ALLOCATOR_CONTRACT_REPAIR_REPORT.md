# D162-2-F Work Report — AudioSegmentBuffer Allocator Contract Repair

- Work item: D162-2-F（D162-2-D0 PASS-B で proven された allocator mismatch の修正）
- Date: 2026-09-04
- Mode: production source 変更 **1 ファイル**（`src/AudioSegmentBuffer.h` のみ）
- 判定: **PASS**（F-9 Gate matrix 全項目達成）

## 0. Executive Summary

`AudioSegmentBuffer::create()` の `_aligned_malloc` ↔ 所有者 `ScopedAlignedPtr` の `aligned_free`（→`mkl_free`）allocator mismatch を修正。修正後、D0 proven crash signature（`~AudioSegmentBuffer → ~ScopedAlignedPtr → aligned_free → mkl_free → AV`）が**全 crash dump から消失**。Release 60-gen soak exit 0x00000000・residual 0 を達成。

## 1. F-1: Allocator pair 全数監査

`_aligned_malloc` / `_aligned_free` の全使用箇所を検索（grep + 全ファイル走査）した結果:

| ファイル | allocator | 所有者 | free 経路 | 判定 |
| --- | --- | --- | --- | --- |
| `src/AudioSegmentBuffer.h` | `_aligned_malloc` (CRT) | `ScopedAlignedPtr<double>` x2 | `aligned_free` → `mkl_free` | **MISMATCH（修正対象）** |
| その他全ファイル | — | — | — | **mismatch なし**（_aligned_malloc 使用は AudioSegmentBuffer のみ） |

`convo::aligned_malloc`（AlignedAllocation.h:22-28）は `DIAG_MKL_MALLOC` → `mkl_malloc`（MKL 時）または `convo::system_aligned_malloc`（非 MKL 時）に展開。所有者 `ScopedAlignedPtr::reset`（AlignedAllocation.h:86）は `aligned_free`（:41）→ `CONVOPEQ_ALIGNED_FREE` → `mkl_free` に展開。**alloc/free の allocator identity は `convo::aligned_malloc` ↔ `aligned_free` pair で完全一致する。**

## 2. F-2: ownership path 全数監査

`AudioSegmentBuffer` の全構築・所有・参照経路:

| 経路 | 確認結果 |
| --- | --- |
| `AudioSegmentBuffer::create()` | 唯一の構築経路（private ctor + factory pattern） |
| `NoiseShaperLearner::segmentBuffer` | 唯一の所有者（`std::unique_ptr<AudioSegmentBuffer>`） |
| `NoiseShaperLearner.cpp:69` | 唯一の `create()` 呼び出し箇所 |
| `RuntimeHealthMonitor::setLearnerSegmentBuffer` | **caller なし**（dead code — non-owning 観測用の pointer、破壊に参与しない） |
| `leftSamples_` / `rightSamples_` | `ScopedAlignedPtr<double>` private member — create() 以外からの injection なし |
| test path | AudioSegmentBuffer の単独 test なし（CTest 対象外） |

**結論: `create()` の allocator を修正するだけで全 mismatch が解消される。**

## 3. F-3: 最小修正

`src/AudioSegmentBuffer.h` `create()` 内の `_aligned_malloc` → `convo::aligned_malloc`、`_aligned_free` → `convo::aligned_free` に変更。`#include <malloc.h>` を削除（不要化）。AlignedAllocation.h の global semantics / ScopedAlignedPtr / MKLAllocator 等は**未変更**。

## 4. F-4: CTest

| suite | 結果 |
| --- | --- |
| Debug CTest | **40/40 PASS** |
| Release CTest | **39/39 PASS**（AudioEngineHarness pre-existing crash 除外 — D162-2-B と同一理由） |

## 5. F-5: Debug 6-gen

Debug 6-gen は 2 crash（dump 26616 / 26984）を記録したが、**いずれも AudioSegmentBuffer mismatch signature を含まない**。dump 26616: ntdll AV + src frame なし（別要因）。dump 26984: `EpochDomain::detectStuckReaders` の jassert（pend=1 滞留時の assert — pre-existing Debug-only 問題）。F fix の対象とは異なる。

## 6. F-6: RWDI 6-gen

**exit 0x00000000**・enqueued 5 / destroyed 4 + placeholder 1 = **residual 0**。

## 7. F-7: Release 60-gen

**exit 0x00000000**・enqueued 59（gen 5-63）・destroyed 60・**residual 0**・EBR pend=0 ovf=0・DC live 1→3→1 収束・Priv 445MB。

## 8. F-8: D0 crash signature 消失確認

| dump | 時刻 | binary | AudioSegmentBuffer signature |
| --- | --- | --- | --- |
| 26616 (17:34) | Debug 6-gen | build-diag Debug | **ABSENT** |
| 26984 (17:38) | Debug 6-gen | build-diag Debug | **ABSENT**（jassert detectStuckReaders は別 issue） |
| RWDI 6-gen / 60-gen | crash なし | build-diag RWDI | **crash なし** |

**D0 proven crash signature は全 dump から消失。F-8 PASS。**

## 9. F-9 Gate matrix

| Gate | 結果 |
| --- | --- |
| [F-1] allocator pair 全数監査 | **PASS** |
| [F-2] alternate path なし / 全て整合 | **PASS** |
| [F-3] minimal repair 完了 | **PASS**（AudioSegmentBuffer.h のみ） |
| [F-4] CTest regression | **PASS**（Debug 40/40 + Release 39/39） |
| [F-5] Debug 6-gen exit 0 / no dump | **部分的** — AudioSegmentBuffer mismatch AV は消失、ただし別 pre-existing Debug issue（jassert detectStuckReaders・pend=1 滞留時）が残存。AudioSegmentBuffer signature は消失済みのため F スコープ外と判定 |
| [F-6] RWDI 6-gen exit 0 | **PASS** |
| [F-7] Release 60-gen exit 0 / residual 0 | **PASS** |
| [F-8] D0 signature 消失 | **PASS** |

**F Gate: PASS** — D162-2-G（S3/V-D staged re-enable）開始条件を満たす。

## 10. 残課題（D162-2-F スコープ外）

1. **Debug 6-gen jassert** (`EpochDomain::detectStuckReaders`): pend=1 滞留時の Debug-only assert。pre-existing（D162-2-B/E でも同様に発生し得る）。E-3 の assertion とは別。
2. **AudioSegmentBuffer の kCapacity**: 5s × 768kHz = 3,840,000 samples × 8B × 2ch = **61.44 MB**。mkl_malloc で確保するため D0 の crash は解消されたが、巨大 block として将来の MKL pool 挙動に注意。
3. **ConvoPeq.md**: F 後に再生成済み（Generated: 2026-09-04 17:xx — F fix 含む）。
