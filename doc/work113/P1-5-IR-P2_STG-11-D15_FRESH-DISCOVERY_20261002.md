# STG-11-D15 Fresh Discovery — Runtime Ownership / Authority / Queue Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D15_FRESH-DISCOVERY_20261002.md`
- Work item: **STG-11-D15** — D13/D14 と重複しない領域の concrete defect 探索
- Date: 2026-10-02
- Authority: リポジトリルート `ConvoPeq.md`（監査 AI が参照する `ConvoPeq(20261002-115916).md` と同一ファイル。
  ファイル名の違いを理由に別世代として扱わない。Owner 確認済みの同一最新統合ソース）
  - baseline commit: `00013d64aff00812abed6d3d85df779f0272dc6f`（D14 docs）
  - `Generated: 2026-10-02 20:56:11` / 5,928,120 B
  - SHA-256: `41E27B57205FB50E6F9CAA76EFE6E9F199A3E6571481FF87ED94ADDDFC82E52F`
  - `--check`: `NEWER_SRC_COUNT = 0` / **FRESH**（調査開始時に確認）
  - 過去資料・記憶・旧行番号は根拠として使用していない。結論はすべて現行ソースから導出。
- 本 work item は **read-only Fresh Discovery**。**production / test / build ファイルの変更は一切行っていない。**
- 本書は **未 commit**（Owner の repair GO 判断待ち）

---

## 0. 判定

```
STG-11-D15 — GO（concrete defect 1 件: D15-1）
ただし実装には進まない。Owner 承認までは監査報告のみ。
```

| 項目 | 結果 |
| --- | --- |
| production change | **0 件**（read-only） |
| 新規 concrete defect | **1 件**（D15-1） |
| D7〜D14 修正済み箇所の再発 | **該当なし** |
| ISR / Authority への影響 | Ownership transfer 層（NonRT のみ）。RT path 変更不要 |
| 実装 | **未着手**（Owner 承認待ち） |

---

## 1. D15-1: OwnerChannel の SPSC 設計と 3 スレッド producer の現実の不一致

### 1.1 concrete trigger

次の 2 系統の publish が時間的に重なって `ownerChannel().enqueue` に入る。

- Message thread 側: fade 完了時の idle publish（`AudioEngine.Timer.cpp:1167`）、
  device 開始/停止時の publish（`PrepareToPlay.cpp:174/304`、`ReleaseResources.cpp:223`）。
  JUCE の `timerCallback`（`AudioEngine.h:1277` override）は message thread で実行される。
- RebuildThread 側: `trySubmitImpl`（`RuntimePublicationOrchestrator.cpp:41`）→
  `executor_.publish`（`:283`）→ `commitRuntimePublication` → enqueue。
- CoordinatorLoop worker thread 側: 専用 `juce::Thread`
  （`AudioEngine.Threading.cpp:277-281`、「MessageThread Timer は observe-only」化により
  processIntent / deferred resubmit が移管されている）→ fire-and-forget → enqueue
  （`RuntimePublicationOrchestrator.cpp:277-280` のコメントが明示）。

具体的重なり例: 再生中にユーザーが EQ を操作して rebuild publish が走っている最中に
crossfade が完了して timer idle publish が走る。両者は独立スレッドであり、
`enqueueRuntimePublicationFireAndForget`（`AudioEngine.h:4848-4963`）に同時に入りうる。
同関数内の `worldAuthority_.ownerChannel().enqueue`（`:4916`）が唯一の enqueue  site であり、
この呼び出しを serialize する lock は存在しない（`AudioEngine.h` /
`RuntimePublicationOrchestrator.h` / `ISRRuntimePublicationCoordinator.h` に
当該経路を守る mutex / CriticalSection / SpinLock は無い。
`rebuildMutex` は `recoveryPending` フラグ専用（`AudioEngine.h:4774`）である）。

### 1.2 actual state / data flow

```text
[contract] src/audioengine/OwnerChannel.h:2
           "Single Producer (Non-RT publish thread) -> Single Consumer (ISR/audio thread)"
           :79  "free slot (SPSC: sole producer on this path)"
           :100 "match: single-transfer drain (SPSC: sole consumer)"
           :12  "enqueue(key, owner&&) -> false if key already queued or full (caller keeps owner)"

[reality]  上記 3 スレッドが同一 ownerChannel_（capacity 256、:41）へ enqueueする。
           enqueue 本体（:67-87）は lock なし。Slot::key は non-atomic であり、
           2 スレッドが同一 free slot に入ると key への concurrent write（data race）と
           owner の release-store 競合が起きる。

[site]     AudioEngine.h:4916（唯一の enqueue 呼び出し）
           AudioEngine.h:4947（queue-full 時の producer 側 take-back。consumer 側 take は
           RuntimePublishExecutor.h:31。同一 key での両者の競合は、intent push 失敗時は
           consumer が当該 key を知り得ないため実質起きない）
```

### 1.3 failure mechanism（2 通り）

**(a) enqueue × enqueue の同一 slot 競合（primary）**

T1 と T2 が同一 free slot S に入る。SPSC 前提の手順は
`key 書込 → owner の release-store → unique_ptr::release` であるが、
相手スレッドの介入により次のいずれかになる。

- T1 の `owner.release()` 後に T2 が同 slot に上書き → T1 の owner が誰にも
  take されずリークする（`RuntimeState` の aligned allocation が回収されない）。
  registry unregister / retire のいずれも当該 owner を知らないため、
  shutdown 時の `drainAllNonRt` にも拾われない可能性がある
  （drain は owner==nullptr でない slot を回収するが、上書きで失われた pointer 自体は
  どこにも残らない）。
- key と owner の組合せが食い違う（例: key は T2 の k2、owner は T1 の o1）。
  `take(k2)` が o1 を k2 の world として返し、**誤った world が publish される**。
  本来の o2 / k1 側は stranded（リーク）する。

**(b) enqueue × take のすれ違い（secondary）**

consumer の `take` が slot の owner 読込と key 照合の間に、
producer が同 slot の key を書き換える。照合が偶然一致すれば誤った owner を drain し、
(a) と同じ誤 publish / リークに至る。non-atomic key の torn read もありうる。

### 1.4 concrete harm

1. **leak**: `RuntimeState`（aligned allocation）が回収不能になる。1 件あたり小さいが、
   蓄積する。DSPCore 自体は handle 管理のため連鎖リークはしないが、world の漏れは確定である。
2. **wrong-world publish**: 誤った world が `take` されて commit されると、
   ユーザーが意図しない設定（旧 EQ / 旧 IR / 旧 dither 等）が live になる。
   user-visible な誤動作である。
3. **stranded entries**: 照合に失敗した owner が slot に残り、後続の同 key publish を
   `already enqueued → reject`（`:75-76`）で弾き続ける可能性がある
   （同一 key の再 publish が恒久 reject される stuck の一形態）。

### 1.5 existing guard が無いこと

- `OwnerChannel::enqueue` に lock / CAS claim は無い。
- 呼び出し側（`enqueueRuntimePublicationFireAndForget`）に serialize は無い。
- 既存 test（`src/tests/OwnerChannelTests.cpp`、197 行）は single-thread のみであり、
  ファイル内に `thread` の文字は 1 件も無い。concurrency は未検証である。
- intent queue 側（`LockFreeRingBuffer` + residency counter）は MPSC 対応だが、
  OwnerChannel は別構造であり、その対応は継承されない。

### 1.6 violated invariant

`OwnerChannel.h` 自身が文書化する SPSC 契約（:2 / :79 / :100）。
3 producer スレッドの現実と矛盾している。

### 1.7 発生確率の正直な評価

同時性（µs window の重なり）と同一 slot への probe 衝突（256 slot、in-flight 少数時は
低確率）の積であり、通常運用での発生は稀である。しかし構造的欠陥であり、
stress test（衝突 key での複数スレッド hammer）では再現可能である。
稀少性は深刻度を下げるが、defect の存在自体は消さない。

### 1.8 minimal repair の成立性（定義のみ・未実装）

NonRT のみで閉じる案が 3 つある。いずれも RT path・authority 境界の変更を伴わない。

| 案 | 内容 | 効果 | 欠点 |
| --- | --- | --- | --- |
| α | facade（`enqueueRuntimePublicationFireAndForget`）の enqueue/take-back を mutex で serialize | 最小差分。全 producer が 1 関数に集約済みのため 1 箇所で足りる。NonRT のみなので lock 可 | わずかな直列化（publish は低頻度のため実害なし） |
| β | `OwnerChannel::enqueue` の slot 確保を CAS 化（MPSC-safe 化） | 構造的解決。key write の atomic 化も併せて必要 | 差分がやや大きい。SPSC 最適化の放棄 |
| γ | 全 producer を単一スレッドに funnel | 根本的だが設計変更が大きい | 本 repair の範囲外 |

**推奨は案 α**（Owner の minimal repair 方針に合致）。最終選択は Owner 判断。

### 1.9 reproducible test scenario（未実装）

`src/tests/OwnerChannelTests.cpp`（JUCE 非依存・既存の `MockOwner::alive` 会計あり）に
次の multi-thread hammer を追加する。

- 2 スレッドが衝突 key 群で `enqueue` を大量反復し、終了後に全 owner を `take` して
  `alive == 0` と id 一致を assert する。現行実装では leak / mismatch で FAIL する。
- consumer 役の第 3 スレッドを加えた enqueue × take 版も同様に FAIL する。
- TSan が使える環境では `Slot::key` の data race が即時検出される
  （本環境は MSVC のため TSan 不可。stress による実証が代替手段）。

---

## 2. D15 で NO-GO とした面（証拠付き）

| 面 | 確認内容 | 判定理由 |
| --- | --- | --- |
| retire 所有権 | Bridge retire 3 経路（`AudioEngine.h:3832-3861`）は Published/Rejected で分離、deferred delete | 二重 retire なし |
| shutdown 順序 | destructor の 3 段階停止、`clearDeferredForShutdown` の EBR disposition | 順序違反なし |
| receipt timeout | timeout ≠ failure、所有権 Transferred 維持（`:4985-4989`、work88 X2 §6.2） | 設計どおり |
| intent queue / registry | MPSC 対応（residency counter + fallback）。registry 上書きは payload fallback で吸収 | 新規証拠なし |
| deferred obligation | identity `(generation, obligationId)`、terminal でのみ invalidate | 二重解決の穴なし |
| convolver state tree | `setConvolverStateTree` は guard 済み setter 経由。filter mode は AudioEngine 管轄 | 不整合なし |
| learning mode enum | session / XML の復元対象外。clamp 経由の consumer | 到達経路なし |

D7〜D14 の再発なし。

---

## 3. 未確定事項

1. **repair 案の選択**（§1.8 α/β/γ）。Owner 判断。
2. **stress test の反復回数と CI 時間**。hammer test は意図的に race を踏むため、
   実行時間が読めない。回数の上限と flake 扱いの要否は Owner 判断。
3. **発生確率の定量化**。本 R1 は静的追跡のみであり、実測の発生頻度は未測定。
   repair 後に hammer test が PASS することをもって閉じたとみなす。

---

## 5. Follow-up — D15-1 repair 完了記録（2026-10-02）

- repair commit: 本 commit に同梱（`fix(runtime): serialize non-RT OwnerChannel producers`）。
  本書（発見記録）は削除せず残す。
- repair 内容: `AudioEngine::enqueueRuntimePublicationFireAndForget`
  （`AudioEngine.h`）に `ownerChannelProducerMutex_`（NonRT 専用）を追加し、
  `ownerChannel().enqueue` と producer-side take-back を直列化。
  `ASSERT_NON_RT_THREAD()` を facade 入口に追加。
  `OwnerChannel.h` の SPSC コメントを「1 論理 producer」として明確化（構造変更なし）。
- 対応関係:
  - initial defect → 本書 §1
  - minimal repair（案 α）→ 上記
  - tests → `src/tests/OwnerChannelTests.cpp` tests 9/10/11
    （Test A: 8-racer same-key hammer / Test B: producer rollback /
    Test C: 2 producers + lock-free consumer）
  - negative control → serialization を外すと Test A が 3/3 FAIL（RC=3）。
    復元は SHA-256 照合で確認。
  - Debug → build BUILD_EXIT=0 / CTest 全件 PASS（下記報告参照）
  - Release → build BUILD_EXIT=0 / CTest 全件 PASS（下記報告参照）
  - static checks → RT consumer lock-free 維持、RT への mutex/確保/wait 追加なし、
    `convo::` wrapper 規約維持、authority 不変（目視 + rg 確認）
  - ConvoPeq regeneration → 当該 tree で再生成、`--check` FRESH

## 6. 本 D15 の作業記録（discovery は read-only、repair は別途検証済み）

```text
変更ファイル: 0 件（production / test / build / CMakeLists.txt / ConvoPeq.md すべて無変更）
追加ファイル: 本書 1 件（未 commit。Owner の repair GO 判断待ち）

実施した操作:
  - 読み取り（git show / rg / Read）: 多数
  - ビルド / テスト: なし（ソース未変更のため不要）
  - reset / clean / rebase / amend / squash / force push: いずれも不使用

main checkout (C:\VSC_Project\ConvoPeq): 未改変（HEAD = bda43034）
作業 worktree: HEAD = 00013d64（= origin/main）、clean、未追跡 1 件（本書のみ）
```
