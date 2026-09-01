# D154-F2 — Existing DSPHandle atomic-backend comment audit（別トラック・read-only・comment-only patch 仕様確定）

**Date:** 2026-08-31 (+09:00)
**Type:** read-only audit + comment-only patch 仕様確定。**Production source: 0 変更 / Test source: 0 変更 / CMake: 0 / build: 0。**
**原則:** `コメント訂正のみ ≠ std::atomic<DSPHandle> の実装変更`。T3c トラック（D152-R2/D154-R2/ST-1/RT affinity）とは混在しない。

---

## 0. 判定（先出し）

- 既存 DSPHandle 系コメントの「`std::atomic<DSPHandle>` は CMPXCHG16B で lock-free」「`is_lock_free()==false` は保身的判定」は**事実誤り**（D154-F1 実証 + 本セッションの STL 14.51.36231/14.52.36615 両バージョン実物検証）。
- ただし **安全性影響なし**: `std::atomic<DSPHandle>` の消費者は全て NonRT（§3 実測）であり、lock-pool backend は D150 §12 と同型の「非RT affinity ゆえ許容」条件を満たす。
- 訂正は **comment-only patch（7 サイト）** として実装可能。実装変更・CMake 変更・静的 assert の追加/削除は不要。**await user go**。

## 1. 事実の基準（本セッションで確定済み）

| 事実 | 根拠 |
|---|---|
| `std::atomic<16B by-value>`（DSPHandle も RecoveryLifecycleWord も同型）は MSVC STL のジェネリック locking 版 `_Atomic_storage`（`_Guard _Lock{_Spinlock}` per-object spinlock）で実装される | MSVC 14.51.36231 include/atomic:530 / 14.52.36615:527-660 — by-value 16B 専用特殊化は両バージョンに存在しない（検索 0 件） |
| `is_lock_free()==false` は**保身ではなく正確な報告** | `atomic::is_lock_free()` は `_Is_always_lock_free<sizeof(_Ty)>`（compile-time 定数）を返し、実装も実際に lock-pool |
| 「lock-free using 16-byte intrinsics」特殊化（`_Atomic_storage<_Ty&, 16>`）は **atomic_ref 用参照形のみ** | 14.51.36231:1120 / 14.52.36615:1102 + `atomic_ref : _Choose_atomic_base_t<_Ty, _Ty&>`（:2276） |
| D154-F1 disasm: T3c exe 内に cmpxchg16b 0 件 | evidence/D154_T3C_BUILD_GATE_VERIFICATION.md（D154-F1） |

## 2. 誤コメント在庫（comment-only patch 対象・7 サイト）

| # | 位置 | 現在の誤記述 | 訂正方針 |
|---|---|---|---|
| S-5a | `ISRDSPHandle.cpp:16-20` | 「STL の保身的判定。実際は InterlockedCompareExchange128 (CMPXCHG16B) で lock-free に動作するため、MSVC ではアサートを回避する」 | 「MSVC では 16B by-value atomic は STL の lock-pool（spinlock）で実装され、`is_lock_free()==false` は**正確な報告**。CAS 意味論（原子的相互排他 + フルバリア）は保たれる。全アクセスは NonRT のため lock-pool 許容（D150 §12 同型根拠）」 |
| S-6a | `ISRDSPHandle.h:22` | 「atomic<DSPHandle> が CMPXCHG16B を使用…」 | alignas(16) の目的を「将来の lock-free 切替（atomic_ref 参照形 / wrapper 案）への備え + 整列コピー効率」に訂正 |
| S-6b | `ISRDSPHandle.h:204-205` | 「std::atomic<DSPHandle> must be lock-free, which on x64 requires 16-byte alignment (CMPXCHG16B…)」 | 「16B atomic は原子的 CAS semantics が要件。MSVC では lock-pool 実装（`is_lock_free()==false` が正確）・全消費者 NonRT。alignas(16) は切替備え」 |
| S-6c | `ISRDSPHandle.h:212` static_assert メッセージ | 「so atomic<DSPHandle> uses CMPXCHG16B on x64」 | メッセージ文言のみ訂正（assert 自体は維持） |
| S-6d | `ISRDSPHandle.h:214-218` | （MSVC 分岐コメントは「runtime 検証」で概ね正しい） | 「verified at runtime」+「MSVC では false が期待値（正確な報告・異常ではない）」を追記 |
| S-1〜S-4 | `ISRRuntimePublicationCoordinator.h:79-83 / 98 / 413-417 / 423` | T3c 側同一誤り（D152-R2 §6 登録済み） | 同一 patch で訂正（lock-pool 事実・false は正確・記録のみ） |
| S-7 | `ISRRuntimePublicationCoordinator.h:93 / 96-97` static_assert メッセージ文言 | メッセージ中の「CMPXCHG16B」 | 文言清掃（assert 自体は維持 — alignas(16) は切替備えとして有効） |

## 3. `std::atomic<DSPHandle>` の消費者スレッド実測（安全性確認）

アクセス点: `ISRDSPHandle.cpp:90/100/101/116/117`（publishAtomic = store）、`221/223/256/261`（consumeAtomic = load）。外部消費者:

| 消費者 | 関数 | スレッド |
|---|---|---|
| `AudioEngine.Processing.ReleaseResources.cpp:444-445` | `releaseResources` | MessageThread |
| `AudioEngine.Timer.cpp:970/1123` | `timerCallback` | MessageThread |
| `AudioEngine.Timer.cpp:1704` | `onHealthEvent` | MessageThread |
| `AudioEngine.Timer.cpp:1944` | `retirePublishedDSP` | MessageThread（Timer 経路） |
| `RuntimePublicationOrchestrator.cpp:82` | Orchestrator | RebuildThread / CoordinatorLoop |

→ **RT callback（processBlock/getNextAudioBlock 系）からのアクセス 0 件**。lock-pool（spinlock）backend は NonRT affinity 下で安全（T3c と同型の根拠）。`isSlotInCrossfade` の本番呼び出しは 0 件（宣言のみ）。既存コメントの「全操作は NonRT」の主張自体は正しい — 誤りは lock-free 実装経路の記述のみ。

## 4. LatencySnapshot 前例の再評価（訂正不要）

`src/convolver/ConvolverProcessor.StateAndUI.cpp:960-966` `debugCheckAtomicLockFree()`: `std::atomic<LatencySnapshot>{}`（16B — int32×3+bool+padding）の `is_lock_free()` が false なら DBG で「not lock-free … continuing with implementation-provided atomic semantics」と記録して継続 — **この前例は当初から正しい**（lock-free を主張せず、実装提供の意味論で継続と明記）。以後この前例を「CMPXCHG16B lock-free の根拠」として引用してはならない（D152-R1 §1 で既に撤回済みの引用形）。

## 5. patch 実施時の運用（await go）

1. 上記 7 サイトの comment-only 編集（コード・静的 assert の論理は一切変更しない）
2. `python output_sourcecode_markdown.py` で ConvoPeq.md 再生成（派生スナップショット更新）
3. comment-only のため build/CTest は意味論上不要だが、実施する場合は D154 の手順を 1 回のみ踏襲（無目的な 40/40 再実行はしない）
4. 適用後の証跡: git diff が comment 行のみであることの確認（`git diff --word-diff` 等）

**本監査は production source を変更しない。実施はユーザーの明示 go を待つ。**
