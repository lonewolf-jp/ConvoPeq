# ConvoPeq 改修後 適合性検証 超詳細チェックリスト for AI

## Deterministic Concurrent Immutable DSP Runtime System Compliance Checklist

本書は、ConvoPeq を

* Immutable RuntimeWorld
* Deterministic Concurrent DSP Runtime
* Lock-free / Wait-free RT Pipeline
* RCU/Epoch Based Lifetime Management
* One-way Dataflow Architecture

へ完全移行した後、そのソースコードがアーキテクチャ不変条件（IR-1〜7）、実装基準書 v2.0、AI 実装規約書 v1.0 に完全適合していることを、AI が静的解析・AST解析・call graph解析・grep監査によって検証するための超詳細監査基準である。

---

# 0. 基本原則

## 0.1 本チェックリストの目的

本チェックリストは以下を保証する。

* Audio Thread の hard real-time safety
* RuntimeWorld immutable 性
* publication consistency
* deterministic concurrency
* lock-free lifetime safety
* shutdown correctness
* ownership integrity
* SIMD determinism
* AI-generated code safety

---

# 0.2 AI 監査原則

AI は以下を遵守すること。

| 原則                    | 内容                               |
| --------------------- | -------------------------------- |
| Conservative 判定       | 不明なコードは「違反候補」とみなす                |
| grep 優先               | まず grep、その後 AST                  |
| Callgraph 必須          | RT path は transitively 解析        |
| Runtime 側優先           | UI より Runtime の安全性を優先            |
| hidden allocation 疑い  | STL/JUCE は allocation 疑い扱い       |
| mutable suspicion     | mutable/static/thread_local は要監査 |
| publication suspicion | atomic/store は要監査                |

---

# 0.3 RT Path 定義

以下を RT Path と定義する。

* `getNextAudioBlock`
* `processBlock`
* `process`
* audio callback
* SIMD processing callback
* realtime worker
* Audio Thread から transitively 呼ばれる全関数

---

# 0.4 監査レベル

| レベル | 内容             |
| --- | -------------- |
| G   | grep           |
| A   | AST            |
| C   | call graph     |
| O   | ownership flow |
| M   | memory model   |

---

# 1. 一方向データフロー

## 1.1 逆流禁止

| ID    | 項目                            | 監査  | 期待状態               |
| ----- | ----------------------------- | --- | ------------------ |
| 1.1.1 | Audio→Builder 呼出禁止            | G/C | 検出されない             |
| 1.1.2 | Audio→Blueprint mutate 禁止     | G/C | 検出されない             |
| 1.1.3 | Runtime→Blueprint feedback 禁止 | G/A | 検出されない             |
| 1.1.4 | Runtime→UI direct callback 禁止 | G/C | 検出されない             |
| 1.1.5 | UI→Runtime direct mutate 禁止   | G/C | command queue 経由のみ |

注記 (1.1.5 運用明確化):

* `AudioEngine` が保持する **UI staging オブジェクト** (`uiConvolverProcessor` / `uiEqEditor`) への Message Thread setter は許可。
* ただし、Audio Thread が参照する RuntimeWorld への反映は **必ず** `requestRebuild(...)` / snapshot publish / command queue のいずれかの非RT経路を通すこと。
* Audio Thread から UI staging setter を直接呼ぶことは引き続き禁止。

---

## 1.2 正方向のみ許可

許可される流れ:

```text
UI/Input
↓
Blueprint
↓
Command
↓
Builder
↓
RuntimeWorld Build
↓
Warmup
↓
Publish
↓
Audio Thread Read-only Access
↓
Retire
↓
Epoch Reclaim
```

これ以外は禁止。

---

# 2. Audio Thread Hard RT Safety

# 2.1 動的メモリ確保禁止

| ID     | 項目                           | 監査  | 禁止対象                           |
| ------ | ---------------------------- | --- | ------------------------------ |
| 2.1.1  | new 禁止                       | G/C | new/new[]                      |
| 2.1.2  | malloc 禁止                    | G/C | malloc/calloc/realloc          |
| 2.1.3  | aligned allocation 禁止        | G/C | aligned_malloc/_aligned_malloc |
| 2.1.4  | vector resize 禁止             | G/C | resize/reserve                 |
| 2.1.5  | push_back 禁止                 | G/C | push_back/emplace_back         |
| 2.1.6  | map insertion 禁止             | G/C | insert/emplace/operator[]      |
| 2.1.7  | std::function 禁止             | G/A | hidden allocation              |
| 2.1.8  | std::any 禁止                  | G/A | hidden allocation              |
| 2.1.9  | std::variant dynamic path 禁止 | A   | hidden allocation              |
| 2.1.10 | juce::String 禁止              | G/A | hidden allocation              |
| 2.1.11 | AudioBuffer::setSize 禁止      | G/C | allocation                     |
| 2.1.12 | FFT plan creation 禁止         | G/C | DftiCreateDescriptor           |
| 2.1.13 | FFT commit 禁止                | G/C | DftiCommitDescriptor           |

---

# 2.2 ロック禁止

| ID     | 項目                    |
| ------ | --------------------- |
| 2.2.1  | std::mutex 禁止         |
| 2.2.2  | lock_guard 禁止         |
| 2.2.3  | unique_lock 禁止        |
| 2.2.4  | shared_mutex 禁止       |
| 2.2.5  | condition_variable 禁止 |
| 2.2.6  | future/promise 禁止     |
| 2.2.7  | async 禁止              |
| 2.2.8  | CriticalSection 禁止    |
| 2.2.9  | ScopedLock 禁止         |
| 2.2.10 | WaitableEvent 禁止      |

---

# 2.3 例外禁止

| ID    | 項目           |
| ----- | ------------ |
| 2.3.1 | throw 禁止     |
| 2.3.2 | try/catch 禁止 |
| 2.3.3 | SEH 禁止       |
| 2.3.4 | noexcept 必須  |

---

# 2.4 libm 禁止

| ID    | 項目              |
| ----- | --------------- |
| 2.4.1 | pow 禁止          |
| 2.4.2 | exp 禁止          |
| 2.4.3 | log/log10 禁止    |
| 2.4.4 | sin/cos/tan 禁止  |
| 2.4.5 | atan/atan2 禁止   |
| 2.4.6 | std::complex 禁止 |
| 2.4.7 | sqrt 使用監査       |

---

# 2.5 ログ/I/O 禁止

| ID    | 項目                       |
| ----- | ------------------------ |
| 2.5.1 | Logger::writeToLog 禁止    |
| 2.5.2 | DBG 禁止                   |
| 2.5.3 | printf 禁止                |
| 2.5.4 | std::cout 禁止             |
| 2.5.5 | File I/O 禁止              |
| 2.5.6 | MessageManager access 禁止 |

---

# 2.6 hidden allocation 検査

| ID    | 項目                          |
| ----- | --------------------------- |
| 2.6.1 | lambda capture allocation   |
| 2.6.2 | SBO overflow                |
| 2.6.3 | virtual dispatch allocation |
| 2.6.4 | JUCE hidden allocation      |
| 2.6.5 | temporary string allocation |

---

# 3. Atomic / Memory Model

# 3.1 RT Atomic 制約

| ID    | 項目                  |
| ----- | ------------------- |
| 3.1.1 | store 禁止            |
| 3.1.2 | exchange 禁止         |
| 3.1.3 | compare_exchange 禁止 |
| 3.1.4 | fetch_add 制限        |
| 3.1.5 | fence 禁止            |

---

# 3.2 Publication Memory Order

| ID    | 項目                    | 期待               |
| ----- | --------------------- | ---------------- |
| 3.2.1 | publish store         | release          |
| 3.2.2 | reader load           | acquire          |
| 3.2.3 | relaxed misuse        | 無し               |
| 3.2.4 | publication helper 集約 | publishAtomic のみ |
| 3.2.5 | double publish race   | 無し               |

---

# 3.3 ABA / Epoch

| ID    | 項目                    |
| ----- | --------------------- |
| 3.3.1 | Runtime generation 必須 |
| 3.3.2 | retire generation 保存  |
| 3.3.3 | epoch compare 正当性     |
| 3.3.4 | ABA 防止                |
| 3.3.5 | reclaim safety        |

---

# 4. Immutable RuntimeWorld

# 4.1 Immutable 保証

| ID    | 項目                         |
| ----- | -------------------------- |
| 4.1.1 | RuntimeWorld const access  |
| 4.1.2 | publish後 mutate 禁止         |
| 4.1.3 | syncParametersFrom 禁止      |
| 4.1.4 | lazy init 禁止               |
| 4.1.5 | singleton mutable state 禁止 |

---

# 4.2 RuntimeWorld 完全性

RuntimeWorld は以下を全て包含すること。

* PEQ
* Convolver
* Oversampler
* Delay
* TransitionState
* Filter coefficients
* FFT plans
* SIMD state
* smoothing state
* latency state
* monitoring snapshot
* automation state

RuntimeWorld 外 mutable state 禁止。

---

# 4.3 hidden mutable state

| ID    | 項目                      |
| ----- | ----------------------- |
| 4.3.1 | static mutable 禁止       |
| 4.3.2 | thread_local mutable 禁止 |
| 4.3.3 | singleton cache 禁止      |
| 4.3.4 | DSP hidden cache 禁止     |

---

# 5. Publication

# 5.1 Publication Sequence

必須順序:

```text
build
↓
warmup
↓
publishAtomic
↓
advanceEpoch
↓
retire
```

逆順禁止。

---

# 5.2 warmup

| ID    | 項目              |
| ----- | --------------- |
| 5.2.1 | SIMD warmup     |
| 5.2.2 | FFT warmup      |
| 5.2.3 | cache warmup    |
| 5.2.4 | denormal warmup |

---

# 6. RCU / Lifetime

# 6.1 Reader Safety

| ID    | 項目                  |
| ----- | ------------------- |
| 6.1.1 | RCUReader copy禁止    |
| 6.1.2 | thread_local reader |
| 6.1.3 | guard 必須            |
| 6.1.4 | nested guard 禁止     |

---

# 6.2 Retire Queue

| ID    | 項目                    |
| ----- | --------------------- |
| 6.2.1 | bounded capacity      |
| 6.2.2 | overflow policy       |
| 6.2.3 | deterministic reclaim |
| 6.2.4 | reclaim race 無し       |

---

# 6.3 Reclaim

| ID    | 項目                                 |
| ----- | ---------------------------------- |
| 6.3.1 | reclaimAllIgnoringEpoch shutdown限定 |
| 6.3.2 | join後 reclaim                      |
| 6.3.3 | UAF 検査                             |
| 6.3.4 | dangling observer 検査               |

---

# 7. Ownership

# 7.1 delete 禁止

| ID    | 項目                  |
| ----- | ------------------- |
| 7.1.1 | raw delete 禁止       |
| 7.1.2 | delete[] 禁止         |
| 7.1.3 | manual free 禁止      |
| 7.1.4 | placement delete 禁止 |

注記 (7.1 運用明確化):

* `delete` / `free` の直接実行は原則禁止。
* 例外として、**非RT遅延解放キュー** (epoch/deferred reclaim) に登録する deleter 内での解放は許可。
* この例外は「Audio Thread で解放しない」「所有権が retire 済みである」ことを満たす場合に限る。

---

# 7.2 所有権整合性

| ID    | 項目                       |
| ----- | ------------------------ |
| 7.2.1 | unique_ptr ownership     |
| 7.2.2 | release後 retire          |
| 7.2.3 | raw pointer container 禁止 |
| 7.2.4 | ownership escape 禁止      |

---

# 8. Blueprint

# 8.1 Immutable Blueprint

| ID    | 項目                     |
| ----- | ---------------------- |
| 8.1.1 | setter 禁止              |
| 8.1.2 | mutable 禁止             |
| 8.1.3 | runtime reverse ref 禁止 |
| 8.1.4 | value semantics        |

---

# 9. Command Queue

# 9.1 Queue Architecture

| ID    | 項目                     |
| ----- | ---------------------- |
| 9.1.1 | bounded queue          |
| 9.1.2 | lock-free              |
| 9.1.3 | wait-free              |
| 9.1.4 | deterministic overflow |
| 9.1.5 | queue segregation      |

---

# 9.2 Queue False Sharing

| ID    | 項目                      |
| ----- | ----------------------- |
| 9.2.1 | producer index isolated |
| 9.2.2 | consumer index isolated |
| 9.2.3 | alignas(64)             |

---

# 10. Builder

# 10.1 Builder Single Ownership

| ID     | 項目                 |
| ------ | ------------------ |
| 10.1.1 | single publisher   |
| 10.1.2 | partial publish 禁止 |
| 10.1.3 | build cancellation |
| 10.1.4 | rollback safety    |

---

# 11. Transition / Crossfade

# 11.1 Isolation

| ID     | 項目                       |
| ------ | ------------------------ |
| 11.1.1 | old/new shared memory 無し |
| 11.1.2 | transition encapsulation |
| 11.1.3 | ownership correctness    |
| 11.1.4 | deterministic fade       |

---

# 12. Shutdown

# 12.1 Lifecycle

| ID     | 項目                           |
| ------ | ---------------------------- |
| 12.1.1 | single lifecycle state       |
| 12.1.2 | deterministic shutdown order |
| 12.1.3 | callback stop確認              |
| 12.1.4 | join before reclaim          |

---

# 13. SIMD / DSP

# 13.1 SIMD Safety

| ID     | 項目                |
| ------ | ----------------- |
| 13.1.1 | ScopedNoDenormals |
| 13.1.2 | FTZ enabled       |
| 13.1.3 | DAZ enabled       |
| 13.1.4 | MXCSR fixed       |

---

# 13.2 Determinism

| ID     | 項目                |
| ------ | ----------------- |
| 13.2.1 | AVX/SSE/scalar一致  |
| 13.2.2 | tolerance定義       |
| 13.2.3 | FMA policy固定      |
| 13.2.4 | NaN/Inf detection |

---

# 14. Cache / False Sharing

| ID   | 項目                     |
| ---- | ---------------------- |
| 14.1 | hot atomic alignas(64) |
| 14.2 | cacheline isolation    |
| 14.3 | UI/RT false sharing 無し |
| 14.4 | queue index separation |

---

# 15. AI生成コード監査

# 15.1 危険コメント

禁止:

* temporary
* quick fix
* workaround
* FIXME
* TODO
* just for now

---

# 15.2 mutable 化検査

| ID     | 項目                |
| ------ | ----------------- |
| 15.2.1 | mutable 追加禁止      |
| 15.2.2 | const 除去禁止        |
| 15.2.3 | static mutable 禁止 |

---

# 15.3 Debug隠蔽禁止

| ID     | 項目                      |
| ------ | ----------------------- |
| 15.3.1 | DEBUG only RT safety 禁止 |
| 15.3.2 | assert-only safety 禁止   |
| 15.3.3 | release build unsafe 禁止 |

---

# 16. AST / Callgraph 必須監査

grepだけでは不十分。

必須:

| 種類                  | grep | AST      |
| ------------------- | ---- | -------- |
| allocation          | ○    | optional |
| const violation     | ×    | 必須       |
| ownership escape    | ×    | 必須       |
| mutable alias       | ×    | 必須       |
| RT call propagation | ×    | 必須       |
| hidden allocation   | △    | 必須       |
| publication race    | ×    | 必須       |

---

# 17. 最終適合条件

以下を全て満たした場合のみ適合。

* grep violations = 0
* AST violations = 0
* ownership violations = 0
* publication violations = 0
* RT violations = 0
* shutdown race = 0
* deterministic mismatch = 0
* hidden allocation = 0
* mutable runtime state = 0

---

# 18. 最終アーキテクチャ不変条件

ConvoPeq は最終的に以下を満たすこと。

## IR-1

Audio Thread は immutable RuntimeWorld の read-only consumer である。

## IR-2

RuntimeWorld は publication 後 immutable である。

## IR-3

状態変更は Blueprint→Builder→Publish のみ。

## IR-4

Lifetime は RCU/Epoch により deterministic reclaim される。

## IR-5

RT path は hard realtime safe。

## IR-6

Publication は single ownership / acquire-release consistency を持つ。

## IR-7

Shutdown は deterministic lifecycle state machine により管理される。

---

本チェックリストを完全通過したコードのみが、ConvoPeq の Deterministic Concurrent Immutable DSP Runtime System 実装としてマージ可能である。
