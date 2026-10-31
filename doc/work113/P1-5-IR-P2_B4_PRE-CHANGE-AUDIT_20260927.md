# B-4 — Pre-Change Audit / Implementation Contract

```text
record type   audit / contract record.  NOT a change record.
outcome       NO IMPLEMENTATION.  The task premise is contradicted by the
              authoritative record, and the item is already CLOSED.
production source change   0
build / test / measurement 0
git mutation               0
authority   ConvoPeq.md  5,550,764 B
            C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
            Generated 2026-09-27 00:36:31   HEAD e565019d
```

---

## 1. Finding — B-4 is CLOSED, and the DspNumericPolicy retention is by design

The two `cachedThreadHash()` call sites in `DspNumericPolicy.h` are **not
unmigrated remnants**. They are an **intentional exclusion**, decided on
2026-09-09 and recorded in five places.

```text
src\core\RCUReader.h  L58568-58571   (in the source itself)
  // ★ work92 B-4 (big 2-6): token 生成を単調 ID 採番に統一。
  //   衝突時の RCU reader 二重登録 / epoch 停止（reclaim 停止）の潜在を排除。
  //   DspNumericPolicy.h の役割タグ（isAudioThread 等）は衝突許容のため
  //   cachedThreadHash() を維持する（RECONCILIATION §5-3 設計確定）。

doc\work92\RECONCILIATION_20260909.md  L61-63, L136
  L61  PLAN にない消費点を 1 件発見: src/DspNumericPolicy.h:44（currentThreadTag）
       ・:120（isAudioThread）が convo::cachedThreadHash() を直接呼ぶ
  L62  設計確定（v3.1 提案）: acquireUniqueThreadId() を ThreadHash.h に追加し、
       RCUReader/EpochDomain の ownerThreadId 系を新 ID に統一。
       DspNumericPolicy の tag は「[衝突許容]…」→ 統一対象外
  L136 DspNumericPolicy.h:44/:120 は cachedThreadHash 維持の設計判断を明記
       （RCU/Epoch token 系のみ acquireUniqueThreadId へ統一）

doc\work92\IMPLEMENTATION_REPORT_20260910.md  L19, L70
  L19  | B | B-4 acquireUniqueThreadId | ✅ CLOSED | CTest 40/40
       （RCU/Epoch のみ・DspNumericPolicy 維持）|
  L70  DspNumericPolicy.h:44/:120 は cachedThreadHash 維持
       （衝突許容の役割タグ・RECONCILIATION §5-3 設計）

doc\work92\PLAN.md  L3（v3.1 補訂日 2026-09-09）
  …B-4 設計補完（DspNumericPolicy は cachedThreadHash 維持）
```

**`IMPLEMENTATION_REPORT` は CTest 40/40 をcited しています。** つまり B-4 には
validation 記録が存在します。AC-B4-2（既存 RCU/Epoch テスト全通過）は
CTest 40/40 として記録されています。

---

## 2. Correction to my own inventory

My inventory reported B-4 as `INCOMPLETE`, on the evidence that
`DspNumericPolicy.h` still called `cachedThreadHash()`. **That classification was
wrong.** I read `PLAN.md` §B-4 — the **pre-v3.1** design — and did not check
whether a superseding decision existed, even though the superseding decision was
written into the source file I was auditing, six lines above one of the two call
sites' own thread-token helper.

The error class is the one this project has recorded repeatedly: status inferred
from a plan document without testing whether the plan still governs.

---

## 3. B4-1 — the two consumers, item by item

Both consumers read the same function, `currentThreadTag()`, which is
`DspNumericPolicy.h:64441-64444`.

```cpp
inline uint64_t currentThreadTag() noexcept
{
    return convo::cachedThreadHash();          // L64443
}
```

### 3.1 Consumer 1 — `ScopedThreadRole::acquireAudioThreadSlot()` (L64465-64497)

```text
1  identity/tag の意味      「このスレッドは Audio Thread である」という役割タグ。
                           4 要素スロット registry の占有键。
2  比較対象                 L64473-64474  existingTag == tag  （等値）
                           L64486-64490  compareExchangeAtomic(slot.tag, expected=0, tag)
3  永続化されるか           いいえ。serialization 対象外。ただし process 寿命の
                           グローバル state（inline namespace の std::array, L64433）。
4  thread lifetime を越えて保持されるか
                           はい。slot はグローバルに保持され、thread 消滅後も
                           releaseAudioThreadSlot() が tag を 0 に戻すまで残る。
5  container key か          はい。4 要素配列の等値検索 + CAS による占有。
6  特殊値意味論               **ある。0 が「空き」sentinel。**
                             L64485  uint64_t expected = 0;
                             L64486  compareExchangeAtomic(slot.tag, 0, tag, …)
                             L64509  publishAtomic(slot.tag, 0, release)   ← 解放
7  呼び出し thread context    ScopedThreadRole は
                             L13283 / L14068  src\audioengine\AudioEngine.Processing.*
                             の 2 箇所で生成される = audio processing callback 内。
8  Audio Thread から到達可能か **はい。** これが RT パス。
9  first-use 時の動作         cachedThreadHash() の thread_local dynamic init が
                             1 回。std::hash<std::thread::id>{}(this_thread::get_id())
                             を評価。
10 steady-state 時の動作      thread_local const uint64_t の読み取り 1 回。
```

### 3.2 Consumer 2 — `isAudioThread()` (L64517-64526)

```text
1  identity/tag の意味      「現在のスレッドは Audio Thread か」という判定器。
2  比較対象                 L64522  consumeAtomic(slot.tag, acquire) == tag
3  永続化                   いいえ（process 寿命の runtime state のみ）
4  lifetime 越保持          はい（slot registry を参照するため）
5  container key            はい（4 要素配列の等値検索）
6  特殊値意味論             間接的に yes。tag==0 の slot は「空き」なので
                           0 は Artist  Edgeless thread と一致しうる。
7  呼び出し thread context  Raise 側 = RT 自身、または NonRT（ASSERT 用）
8  Audio Thread から到達可能か **はい。かつこれが RT 判定器そのもの。**
9  first-use                cachedThreadHash() の thread_local init 1 回
10 steady-state            thread_local 読み取り 1 回 ＋ 最大 4 要素の等値比較
```

### 3.3 `isAudioThread()` は ISR 防御の判定器である

```text
L29469  src\audioengine\ISRRetireRouter.cpp
        #include "../DspNumericPolicy.h" // ★ P-4: isAudioThread() —
                                        //    RT 防御（synchronous destruction 禁止）
L29978  const bool isRt = convo::numeric_policy::isAudioThread();  // ★ P-4: RT 防御
L64581  #define ASSERT_AUDIO_THREAD()   jassert(convo::numeric_policy::isAudioThread())
L64582  #define ASSERT_NON_RT_THREAD()  jassert(!convo::numeric_policy::isAudioThread())
```

**この 2 箇所は受動的な consumer ではありません。** プロジェクトが RT 文脈を
検出する機構そのものです。tag の出所を変えることは RT 判定器を変えることに
なります。

### 3.4 衝突時の機械的帰結（traceable from source、判断は含めない）

```text
collision: thread X と Y が同一 tag を持つ場合

  isAudioThread()
    X（RT）      → true   正解
    Y（NonRT）   → true   ★誤判定。Y は Audio Thread とみなされる

  ASSERT_NON_RT_THREAD() を Y が実行
    → jassert(!true) = jassert(false)  → debug で assertion 発火
    → 逆方向（RT が NonRT と判定されれば assertion は沈黙）ではない

  slot registry
    X と Y が同一 slot を共有 → depth が合算され、release 時の
    previousDepth<=1 判定（L64506）が skewed になり得る

  判定：帰結は有界。確率は 64-bit ハッシュの衝突という極小。
  「衝突許容」という設計判断は、reclaim liveness 系の ID
  （RCUReader token / EpochDomain ownerThreadId）とは要求水準が異なるという
  認識に基づいている。
```

**この「衝突許容」という判断自体が妥当かは Owner の決定事項です。** 本 audit は
機構の帰結を追跡しただけで、判断の当否は述べていません。既に 2026-09-09 に
確定し、実装され、CTest 40/40 で検証済みの設計です。

---

## 4. B4-2 — `cachedThreadHash()` vs `acquireUniqueThreadId()`

Both in `src\core\ThreadHash.h`.

| 項目 | `cachedThreadHash()` L60382-60388 | `acquireUniqueThreadId()` L60405-60415 |
| --- | --- | --- |
| initialization | `static thread_local const uint64_t` 直接初期化 | `static thread_local const uint64_t` を lambda で初期化 |
| thread_local state | あり | あり |
| atomic counter | **なし** | **あり** `static std::atomic<uint64_t> s_counter { 0 }` |
| memory ordering | **原子操作なし** | `fetch_add(1, std::memory_order_relaxed)` |
| first-call | `std::this_thread::get_id()` と `std::hash<std::thread::id>` を評価 | `fetch_add(relaxed)` 1 回 |
| steady-state | thread_local 読み取り 1 回 | thread_local 読み取り 1 回 |
| failure / exhaustion | なし | **counter は uint64_t。2^64 で折り返す。折り返し処理は無い**（実用上到達しない） |
| ID uniqueness | **保証されない**（L60393 が明記） | **保証される**（単調採番） |
| monotonicity | **なし** | **あり**（1 起点） |

### 4.1 「collision-free」と「monotonic」は別条件

```text
collision-free   異なるスレッドが同一 ID を持たない
                 cachedThreadHash    ✗ 保証されない
                 acquireUniqueThreadId ✓ 採番により構造的に成立

monotonic        採番された値が単調増加する
                 cachedThreadHash    ✗ 順序概念が無い（hash）
                 acquireUniqueThreadId ✓ ただし memory_order_relaxed のため
                                     「採番の順序と可視性の契約なし」と
                                     L60409 が明記。返り値の単調性は
                                     保証されるが、その値の公開順序は契約外。
```

### 4.2 0 の扱い — 両者で意味論が一致しない

```text
acquireUniqueThreadId   1 起点、0 は無効値予約（L60410 コメント、
                        EpochDomain.h:571 の ownerThreadId 初期値 0 と整合）

DspNumericPolicy        0 は「slot 空き」sentinel（L64485 / L64509）

→ cachedThreadHash は 0 を返す可能性が排除されていない。
  0 と返ると「空き slot」と衝突し得る。
  acquireUniqueThreadId は構造的に 0 を返さない。
```

**この差は「改善」であり「意味論の保存」ではありません。** 単純置換によって
DspNumericPolicy の 0-sentinel との整合は实际上は**強化**されます。

---

## 5. B4-3 — ID 系統の境界

**答えは既に確定しています：別 ID 系統を維持する。**

```text
単調 ID 系統（acquireUniqueThreadId）
    RCUReader::currentThreadToken()      RCUReader.h L58572-58575
    EpochDomain ownerThreadId 公開        EpochDomain.h L57188-57191
        「★ work92 B-4: ownerThreadId を単調 ID 採番に統一（BUG-063/BIG 2-6）」
    IEpochProvider::ownerThreadId        IEpochProvider.h L58007（uint64_t, 初期値 0）

hash 系統（cachedThreadHash）— 意図的に別系統として維持
    DspNumericPolicy::currentThreadTag() DspNumericPolicy.h L64443
    DspNumericPolicy::isAudioThread()    DspNumericPolicy.h L64519

第三の系統 — B-4 の対象外
    ObservedRuntime の ownerThreadId      ObservedRuntime.h L58358 / L58371 /
                                          L58389 / L58398
                                          std::thread::id 型であり uint64_t ではない
```

```text
ThreadHash.h L60390-60391 の関数コメントは
  「RCU reader token / EpochDomain ownerThreadId 専用の thread 識別子」
と書いており、これは現状と一致しています。
DspNumericPolicy を統合Apabilaこのコメントの書き換えREQUIRED ですが、
統合すべきでないという §1 の確定判断が先立ちます。

ownerThreadId は既に統合済み（L57188-57191）であり、
本 contract は ownerThreadId を変更対象へ拡張しません。
```

---

## 6. B4-4 — ISR / RT 契約

`acquireUniqueThreadId()` が RT で実行され得る場合的规定。

```text
                       cachedThreadHash        acquireUniqueThreadId
RT first-use           thread_local init。     static std::atomic の
                       std::hash<std::thread::id>  relaxed fetch_add 1 回。
                       の評価。標準は          「定数初期化可能」なので
                       allocation-free を     guard 変数・lock は生成されない。
                       規定しない。            x86-64 = lock xadd
                                              ARM64 = LDAXR/STLXR
                                              → bounded / wait-free
RT steady-state        thread_local read       thread_local read
lock                    なし                   なし（lock primitive ではない）
allocation              なし（保証されない）   なし
blocking                なし                   なし（OS 呼び出し・lock なし）
OS call                 なし                   なし
```

```text
「lock-free アルゴリズム」を名乗るべきではない。
fetch_add(relaxed) は cache line の排他を取得しうるため
形式的には lock-free ではない。ただし wait-free で
別スレッドの進行を待つblocking ではなく、OS にも触れない。

そしてこれは RCUReader / EpochDomain が既に RT で承受している契約と同一である。
「既存 cachedThreadHash() と同一タイミング・同一の thread_local 機構に依存し、
  新規 blocking/allocation なし」は L60402-60403 に明記されている。

Practical Stable ISR Bridge Runtime との照合
  ISR L710  RT で delete が発生しない        → 影響なし
  ISR L711  RT で lock が発生しない         → fetch_add は lock primitive ではない
  ISR L781  RT スレッドは動的確保禁止       → 発生しない
  ISR L798  RT スレッドはログ出力禁止       → 発生しない
  ISR L1258 RT スレッドは実行のみ、判断しない → 影響なし
```

**結論： `acquireUniqueThreadId()` は RT 契約に適合します。**
ただし §1 のとおり DspNumericPolicy を用它へ移行する必要はありません。

---

## 7. B4-5 — first-call semantics

```text
audio callback の最初の invocation
  現状   cachedThreadHash の thread_local dynamic init が 1 回
         std::hash<std::thread::id> の評価
  移行した場合
         acquireUniqueThreadId の thread_local dynamic init が 1 回
         relaxed fetch_add が 1 回
  → 差分は「hash 評価」→「relaxed atomic RMW」。
     前者の allocation-free 性が標準規定されていないのに対し、
     後者は定数初期化可能な atomic であり guard を出さない。
     RT 観点では悪化ではなく改善 방향。

worker thread の最初の invocation
  同上。thread ごとに 1 回。counter を 1 消費する。
  影響なし（採番であり状態ではない）。

steady-state で実行されるもの
  thread_local const uint64_t の読み取り 1 回。
  これは cachedThreadHash と完全に同一の機械である。

「steady-state では thread_local token の read のみ」の契約が
DspNumericPolicy の 2 consumer にも適用できるか
  → 適用可能。機械が同一（L60385 と L60407 はどちらも
    static thread_local const uint64_t）であり、
    RCUReader / EpochDomain が既に同じ契約の下で通っている。
```

---

## 8. B4-6 — replacement scope

`cachedThreadHash()` の全 source occurrence（authority 実測・6 件）:

```text
COMMENT  L58571  src\core\RCUReader.h          設計判断の記述（移行しない旨）
CODE    L60382  src\core\ThreadHash.h         定義
COMMENT  L60393  src\core\ThreadHash.h         衝突性の記述
COMMENT  L60402  src\core\ThreadHash.h         初回初期化タイミングの記述
CODE    L64443  src\DspNumericPolicy.h        ★ 移行対象外（§1 の確定設計）
CODE    L64519  src\DspNumericPolicy.h        ★ 移行対象外（§1 の確定設計）
```

**一括置換はしていません。置換自体を行わないのが本 audit の結論です。**

```text
Owner が指定した scope   DspNumericPolicy.h L64443 / L64519
本 audit の判定          移行不要。意図的な除外。
production source change  0 lines / 0 files
```

---

## 9. B4-7 — acceptance criteria, as they actually stand

```text
AC-B4-1  100+ threads で ID 重複ゼロ
  対象範囲 = RCUReader token / EpochDomain ownerThreadId
  状態     = IMPLEMENTED。validation 記録は
             IMPLEMENTATION_REPORT_20260910 L19「CTest 40/40」
  本 audit での再実行は未実施（test execution 禁止）

AC-B4-2  単調採番
  状態     = IMPLEMENTED（L60408-60413、1 起点 + 0 予約）

AC-B4-3  RT steady-state で lock / allocation / blocking / OS sync が発生しない
  状態     = IMPLEMENTED。steady-state は thread_local read のみ。
             first-use の relaxed fetch_add は wait-free・非 blocking・
             非 allocation・非 OS。RCUReader/EpochDomain と同一契約。
             本 audit は ISR 文書と照合した（§6）。

AC-B4-4  tag semantics の維持
  状態     = 該当なし。移行しないため semantic change は発生しない。
             0-sentinel（L64485 / L64509）との整合も現状のまま維持される。

AC-B4-5  既存 consumer（RCUReader / EpochDomain）の semantics を変更しない
  状態     = 満足。両 consumer は既に移行済みで、本 contract は
             それらに触れない。ownerThreadId にも触れない。
```

### 9.1 テスト

```text
existing test    今回の authority 内に、thread token / DspNumericPolicy の
                 slot / ScopedThreadRole / currentThreadTag / acquireUniqueThreadId を
                 対象とする test は検出されなかった。
                 符号の test 領域（>90000 行）での出現はいずれも 0。
                 IMPLEMENTATION_REPORT が cited する CTest 40/40 は
                 実行環境側の記録であり、test の source は
                 今回の authority からは特定できなかった。

new test         必要ない（本 audit の結論が「移行不要」のため）

test location    該当なし

test setup       該当なし

acceptance       AC-B4-1..5 は上記のとおり accounting 済み。
                 追加 test の設計は不要。
```

---

## 10. Stop condition invoked

`Owner §14` の該当項目:

```text
「cachedThreadHash() と ID semantics が単純置換できない」
    → 該当。ただし理由は「置換不能」ではなく「置換してはならない」。
      tag は container key であり 0 が空き sentinel であり、
      isAudioThread() は RT 判定器である。加えて §1 の確定設計が
      意図的な除外を明示している。

「既存 consumer との ID collision / ordering dependency がある」
    → 該当。RCU/Epoch と DspNumericPolicy は意図的に異なる ID 系統で
      ある。統合すると 0-sentinel と採番の意味論が同一系統内で
      混在する。

「追加 source modification が必要になる」
    → 該当。統合如果要なら ThreadHash.h L60390-60391 の関数コメント
      （「RCU reader token / EpochDomain ownerThreadId 専用」）の
      書き換えが scope 外の変更として必要になる。
```

**実装しません。停止します。**

---

## 11. State

```text
B-4                     CLOSED（authoritative record により確定）
  移行済み               RCUReader token / EpochDomain ownerThreadId
  意図的除外             DspNumericPolicy の 2 consumer
  validation 記録        IMPLEMENTATION_REPORT_20260910 L19 — CTest 40/40
  設計確定               RECONCILIATION_20260909 L62 / L136
  source 内の記録        RCUReader.h L58568-58571

production source change 0
build / test / measure   0
ConvoPeq.md regeneration 0
git mutation             0
files created            1（この record）
files modified           0
```
