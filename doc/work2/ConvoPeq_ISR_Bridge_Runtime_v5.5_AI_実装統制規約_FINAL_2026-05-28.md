# ConvoPeq ISR Bridge Runtime v5.5

# AI 実装統制規約（Final）

## 0. 目的

本規約は、ConvoPeq ISR Bridge Runtime v5.5 を基準設計として、
AI に詳細設計・実装・リファクタリング・修正を行わせる際の
「破綻防止規約」を定義する。

本規約の最優先目的は：

- DAW 実運用で破綻しないこと
- shutdown determinism を維持すること
- rebuild storm / saturation / stale runtime reuse を防止すること
- RT safety を維持すること
- 段階移行を維持すること

である。

理論的 purity や architecture aesthetics は目的ではない。

---

# 1. 最上位原則

## Rule-0

# 実運用耐性最優先

以下を最優先すること：

- plugin unload safety
- long-running DAW stability
- deterministic shutdown
- rebuild storm survivability
- stale runtime non-observability

以下は禁止：

- architecture purity 優先
- 理論上の完全性優先
- 大規模rewrite優先
- “きれいな設計” のための破壊的変更

---

## Rule-1

# 全面rewrite禁止

以下禁止：

- RuntimeGraph 全再設計
- AudioEngine 全置換
- RuntimePublicationCoordinator 全再構築
- DSP pipeline 全置換
- lock-free 全面化
- immutable purity 強制

既存コードを段階 hardening すること。

---

## Rule-2

# RT Safety 最優先

Audio Thread / RT path において以下禁止：

- mutex
- condition_variable
- blocking wait
- heap allocation
- unordered_map
- shared_ptr refcount mutation
- filesystem access
- logging with allocation
- dynamic polymorphic allocation

RT path は bounded / deterministic mandatory。

---

## Rule-3

# Mutable execution state 許容

以下は禁止しない：

- execution-local mutable state
- fade progression
- ramp state
- history buffer
- scratch buffer
- DSP transition state

ただし：

- runtime visibility object
- rebuild dependency source
- publication state

とは分離すること。

---

# 2. Runtime / Snapshot 規約

## Rule-4

# rebuild worker の runtime pointer 参照禁止

worker thread から以下禁止：

- activeRuntime 参照
- fadingRuntime 参照
- task.currentDSP 参照
- oldRuntime pointer traversal
- shareConvolutionEngineFrom(oldRuntime)

worker が参照可能なのは：

- RuntimeBuildSnapshot
- immutable finalized snapshot
- deterministic rebuild input

のみ。

---

## Rule-5

# Snapshot Seal Contract mandatory

snapshot lifecycle は：

1. capture
2. normalize
3. finalize
4. immutable handoff

のみ。

finalize 後変更禁止。

---

## Rule-6

# Finalize Determinism mandatory

same semantic input
must produce
same finalized snapshot。

禁止：

- wall clock dependency
- thread timing dependency
- allocation order dependency
- pointer identity dependency
- unordered traversal dependency

---

## Rule-7

# rebuildFingerprint version mandatory

fingerprint は：

- fingerprintVersion
- normalized DSP params
- IR identity
- oversampling config
- topology class
- runtime policy version

を必須含有。

---

# 3. Admission / Publication 規約

## Rule-8

# Admission state machine 単一路線 mandatory

admission 用独立状態機械追加禁止。

以下へ統合すること：

- lifecycleState
- shutdownPhase
- shutdownRuntime_

のみ。

---

## Rule-9

# Publication Gate mandatory

publication 前に必ず：

acceptsRuntimePublication()

を通過すること。

---

## Rule-10

# shutdown 中 publication 禁止

以下 phase 以降 publication 禁止：

- StopAcceptingWork
- Releasing
- Destroying
- Destroyed

---

## Rule-11

# publication resurrection 禁止

drained 後禁止：

- append retry restart
- late publication
- deferred replay
- commit resurrection

---

# 4. Backpressure / Saturation 規約

## Rule-12

# Saturation stabilization mandatory

saturation state 中は：

stabilization direction only。

許可：

- HWM increase
- LWM increase
- reject aggressiveness increase
- rebuild coalescing increase
- obsolete rebuild discard increase

禁止：

- threshold relaxation
- rebuild expansion
- queue growth encouragement

---

## Rule-13

# Stepwise recovery mandatory

recovery は conservative stepwise mandatory。

禁止：

- immediate full recovery
- oscillating recovery
- aggressive threshold drop

---

## Rule-14

# Scaling Clamp mandatory

mandatory：

0.75 <= scale <= 1.50

対象：

- sampleRateScale
- irComplexityScale
- oversamplingScale
- memoryPressureScale

---

## Rule-15

# Runtime-local metrics only

memoryPressureScale は以下のみ：

- retire queue depth
- fallback queue depth
- rebuild backlog
- reclaim latency
- publication backlog
- quarantine residency

禁止：

- OS global memory usage
- DAW process memory usage
- external allocator heuristics

---

# 5. Rebuild Collapse 規約

## Rule-16

# latest-generation-wins mandatory

obsolete rebuild collapse は：

latest generation wins mandatory。

---

## Rule-17

# Replaceable rebuild strict definition mandatory

collapse 可能なのは：

- replaceable
- non-committed
- same fingerprint
- same rebuild class
- externally invisible

のみ。

---

## Rule-18

# MustExecute rebuild collapse 禁止

以下 collapse 禁止：

- prepareToPlay rebuild
- topology migration rebuild
- shutdown rebuild
- runtime recovery rebuild
- safety rebuild

---

# 6. Crossfade / Runtime State 規約

## Rule-19

# Shared mutable authority 排除

crossfade progression は：

execution-local state mandatory。

engine-global shared mutable fade authority 禁止。

---

## Rule-20

# RuntimeGraph は visibility object

RuntimeGraph は ownership object ではない。

禁止：

- mutable progression ownership
- execution authority ownership
- rebuild dependency storage

---

# 7. Retire / Shutdown 規約

## Rule-21

# Retire saturation hardening mandatory

mandatory：

- HWM/LWM
- hysteresis
- fallback handling
- telemetry
- backlog accounting

---

## Rule-22

# Drained definition strict mandatory

shutdown complete 条件：

- retireQueue empty
- fallbackQueue empty
- quarantine empty
- publicationCoordinator drained
- rebuildWorker stopped

すべて mandatory。

---

## Rule-23

# Drained resurrection prohibited

shutdown drained 後禁止：

- runtime reactivation
- publication restart
- rebuild restart
- retire replay

---

# 8. 実装方式規約

## Rule-24

# 小規模段階移行 mandatory

1 commit = 1 responsibility。

巨大変更禁止。

---

## Rule-25

# Behavior-preserving refactor mandatory

refactor は behavior preserving mandatory。

DSP 音響挙動変更禁止。

---

## Rule-26

# 既存責務維持 mandatory

既存 thread responsibility を壊さないこと。

特に：

- Audio Thread
- Message Thread
- rebuild worker
- timer thread

の責務混線禁止。

---

## Rule-27

# Telemetry first

hardening 前に：

- queue depth
- reclaim latency
- saturation frequency
- rebuild collapse count
- publication reject count

を可視化すること。

---

# 9. AI 実装禁止事項

## Rule-28

# AI 推測実装禁止

ソース確認なし推測実装禁止。

必ず既存コード確認。

---

## Rule-29

# 暗黙契約変更禁止

既存 sequencing / ordering / ownership を
無断変更禁止。

---

## Rule-30

# “安全化のつもり” rewrite 禁止

以下禁止：

- shared_ptr 化
- mutex 導入
- 全 atomics 化
- 全 lock-free 化
- 全 immutable 化

---

## Rule-31

# RT path instrumentation 汚染禁止

Audio Thread へ：

- verbose logging
- heap logging
- debug allocation
- std::string formatting

禁止。

---

## Rule-32

# 実装前後の invariants mandatory

変更前後で：

- shutdown determinism
- rebuild causality
- RT boundedness
- runtime visibility ordering

を確認すること。

---

# 10. 完了判定

以下すべて満たした場合のみ ISR Bridge Runtime hardening 完了：

1. worker が runtime object を直接参照しない
2. shutdown 中 publication 不可能
3. saturation policy 実装済み
4. rebuild collapse deterministic
5. stale runtime reuse 不可能
6. crossfade authority shared mutable state 不在
7. RT path mutex/allocation 不在
8. deterministic shutdown 成立
9. fallback queue を含め drain deterministic
10. finalize deterministic

---

# 11. AI 実装手順 mandatory

AI は必ず以下順序で作業すること：

1. 現状コード解析
2. 影響範囲列挙
3. invariants 列挙
4. failure mode 列挙
5. 最小変更案提示
6. telemetry 追加
7. 小規模実装
8. saturation/shutdown/rebuild 検証
9. rollback possibility 確保
10. commit 単位検証

---

# 12. 最終原則

ConvoPeq ISR Bridge Runtime の目的は：

“理論上もっとも美しい runtime”
ではない。

目的は：

「DAW 実運用で長時間破綻しない runtime」

である。

すべての詳細設計・実装・レビューは、
この原則を最優先基準として判断すること。
