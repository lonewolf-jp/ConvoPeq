# N1（Reclaimer scan visibility）候補方式の比較監査（READ-ONLY）

作業日時: 2026-11-12 21:04〜
基準 commit: `173e6995`（HEAD = origin/main）
対象: 前回監査で残存証明義務 N1 とした「Reclaimer が区画中の Reader を観測できない」問題
性質: **読み取り専用**。production source / test / CMake / git を変更していない。build / test / debugger / replay も実行していない。

---

## 0. 判定

```
N1-CANDIDATE-A = PROVED
N1-CANDIDATE-B = NOT PROVED
N1-RECOMMENDATION = A
```

* **A（seq_cst フェンス）**: Reader 側 1 箇所・Reclaimer 側 1 箇所に seq_cst フェンスを置くだけで、N1 の二者択一（§2.2 の A ∨ B）を C++ メモリモデル上で証明できる（§3）。
* **B（publication state の統合）**: 「active と epoch を 1 語にする」こと自体は同期辺を増やさない。未使用スロットを走査対象から外すための「制約なし」表現が残る限り、同じ穴が残る（§4）。
* 推奨は **A**。変更は 2 行で、RT 側の追加コストは 1 enter あたりフェンス 1 回、他の証明義務（N2 / N6 / N7）への影響も無い（§5）。

「PROVED」は **A を採用した場合に N1 が証明できる**という意味であり、A を実装したとか、Gate 1 全体が承認されたという意味ではない。
A の証明は既存義務 W1（= N2。publication と epoch 前進の単一スレッド順序）を前提にする（§3.6）。W1 が崩れる構成では A の証明も崩れる。

---

## 1. ソース最新性

| 項目 | 値 |
|---|---|
| HEAD | `173e69957b046290f7077797782f031ed8633fd0` |
| origin/main | 同上（ahead/behind 0 0） |
| ConvoPeq.md ヘッダ stamp | `2026-11-12 00:48:16` |
| ConvoPeq.md サイズ | 7,011,745 B |
| ConvoPeq.md SHA-256 | `1801bbbacf4d65484418a1cf978acf0c09fc728028b3c8c8105ddce95f25e14f` |
| 生成器再計算との一致 | **全文一致**（正規化後、7,011,745 B 同士） |
| 監査対象 12 ファイル | すべて HEAD と同一（`git status --porcelain` が clean） |
| 設計差分 v2.9 | 53,979 B / `c3ae5099…48427`（不変） |

`--check`（mtime 基準）は STALE（84 件）を返すが、commit 時の checkout による mtime 更新で説明でき、内容一致は生成器の再計算で確認した。**最新性は証明できている**（`evidence/source_identity.txt`）。

---

## 2. N1 の不変条件

### 2.1 記号

| 記号 | 内容 |
|---|---|
| `P` | `currentStateBits`（publication。`std::atomic<std::uintptr_t>`） |
| `G` | `globalEpoch`（`std::atomic<uint64_t>`） |
| `x` | `slot.depth`（`std::atomic<uint32_t>`） |
| `y` | `slot.epoch`（`std::atomic<uint64_t>`） |
| `R` | Reader が `enterReader` で読んだ epoch 値（= その区画の保護 epoch） |
| `R1` | Reader の `x` への +1（active 公開。RMW） |
| `S1` | object O を置き換えた publication の store（unlink。`exchange` の一部） |
| `T(O)` | O を enqueue したときの `entry.epoch`（retire epoch） |
| `M0` | 走査の最初の `currentEpoch()`（`min` の初期値） |
| `r1` | Reader の publication load |
| `r2` | 走査の `x` load |
| `F_R` / `F_S` | Reader 側 / Reclaimer 側の seq_cst フェンス |
| `hb` / `sb` / `sw` | happens-before / sequenced-before / synchronizes-with |

### 2.2 不変条件 N1

> Reader がある object O を取得して使用する場合、Reclaimer が O を reclaim 可能と判断して `deleter` を実行する時点で、次のいずれかが成立する。
>
> **A.** Reclaimer の走査が、その Reader の区画開始を観測している（その結果、Reader の O へのアクセスが free に happens-before する）。
> **B.** Reader は unlink 後の publication を観測しており、O を使用していない。

### 2.3 A と B の中身（HB まで含めた定義）

* **A の実体は「観測」ではなく「HB の確立」である。** 走査の `r2` が Reader の store（`R1` または退出時の store）を読めば、[atomics.order]/2 により `sw` が成立し、Reader の O へのアクセス（当該 store より `sb` 前）が free に `hb` する。これが無いと、Reader の O への読み取りと Reclaimer の `delete` は data race であり、UB になる。したがって N1 は「`min` が小さいこと」ではなく「HB が成立すること」の要求である。
* **B は「Reader が読んだ値が unlink 以降の store のもの」**である。`P` の store は全順序（modification order）を持つので、「`S1` より前」か「`S1` 以降」の二択であり、後者なら Reader は O を参照しない。
* A の副次条件として、走査が `R1` を読んだ場合はその場で `min <= R` となり O は解放されない（解放は後の走査で、退出 store を読んだときに起きる）。その後の走査は退出 store を読むので HB が成立する。この 2 段構えで A は「解放時の HB」を保証する。

### 2.4 現行コードが N1 を満たさない理由（前回監査の要約）

`r2`（`EpochDomain.h:379` の acquire load）と `R1`（`:266` の acq_rel RMW）の間に `sw` を強制する辺が無い。HB の無い load は [intro.races]/10 により「`B` が `A` に hb しない任意の store」を読めるため、`r2` が古い 0 を読む実行が許される。その結果 `min > T(O)` となって O が解放され得る（前回報告 §8.2）。**本稿の課題はこの 1 点である。**

---

## 3. 候補方式 A（seq_cst による相互排他）

### 3.1 現行の操作と memory_order

N1 に関係する操作だけを示す（全量は `evidence/atomic_ops_n1.txt`）。

| # | 操作 | 対象 | order | 場所 |
|---|---|---|---|---|
| A1 | `currentEpoch()` = load | `G` | acquire | `EpochDomain.h:262,347` |
| A2 | store `slot.epoch = R` | `y` | release | `EpochDomain.h:263` |
| A3 | fetch_add(+1) | `x` | acq_rel | `EpochDomain.h:266-268` |
| A4 | fetch_sub(-1) | `x` | acq_rel | `EpochDomain.h:288-290` |
| A5 | store `kInactiveEpoch` | `y` | release | `EpochDomain.h:326` |
| A6 | fetch_add(+1) | `G` | acq_rel | `EpochDomain.h:357-359` |
| A7 | load（初期 min） | `G` | acquire | `EpochDomain.h:374` |
| A8 | load | `x` | acquire | `EpochDomain.h:379` |
| A9 | load | `y` | acquire | `EpochDomain.h:399` |
| A10 | exchange | `P` | acq_rel | `EQProcessor.h:793-795` |
| A11 | load | `P` | acquire | `EQProcessor.h:787` |
| A12 | `currentEpoch()`（retire epoch） | `G` | acquire | `EQProcessor.Core.cpp:204,213` |
| A13 | `isOlder(entry.epoch, min)` | — | — | `DeferredDeletionQueue.h:174` |

`seq_cst` はこの経路に **0 件**。`atomic_thread_fence` も 0 件（`AudioEngine.h` の 5 件は coeff-set と learning キューの別経路）。

### 3.2 方式 A の構成（2 箇所）

```
Reader（EpochDomain::enterReader）
  store slot.epoch = R, release            (A2)  既存
  fetch_add(slot.depth, +1, acq_rel)       (A3)  既存  ← R1
+ atomic_thread_fence(seq_cst)                    ← 追加 F_R
  ... 呼出し元に戻り、loadCurrentState(acquire)  (A11)  ← r1

Reclaimer（EpochDomain::getMinReaderEpoch）
  min = currentEpoch()                     (A7)  既存  ← M0
+ atomic_thread_fence(seq_cst)                    ← 追加 F_S
  for each slot: load(depth, acquire)      (A8)  ← r2
                 load(epoch, acquire)      (A9)
  min を確定 → isOlder で解放              (A13)
```

**配置の要点**

1. `F_R` は `R1` の後・`r1` の前（`enterReader` の depth RMW の直後）。公開 → フェンス → 読み取り、の順を崩さない。
2. `F_S` は `M0` の**後**・`r2` の**前**。`M0` の後ろに置くのが重要で、これが「unlink がフェンスに hb する」経路を作る（§3.5 の場合分け 1）。
3. `F_S` は `getMinReaderEpoch` の 1 箇所でよい。D キューの `reclaim()`、Q/E/T の drain、`terminalReclaim` の同期解放はいずれも `getMinReaderEpoch()` / `minReaderEpoch()` を通るため、1 本で全経路を覆う。
4. ネスト入場（`previousDepth > 0`）は外側の `F_R` が既に効いているので、フェンスを early-return の後に置いてもよい。安全側に倒すなら RMW 直後に無条件で置く。

### 3.3 関係（SC order / coherence / sw / hb）

| 関係 | 成立条件 | 本方式で使う場面 |
|---|---|---|
| `sw`（[atomics.order]/2） | release store の値を acquire load が読む | `R1` を `r2` が読んだとき（A の HB）、`M0` が `S_E` を読んだとき |
| coherence-ordered before（[atomics.order]/3.2, /3.3） | 同一 object 上で「読んだ値の store が相手より前」 | `r1` が `S1` より前の値を読んだこと、`r2` が `R1` より前の値を読んだこと |
| S（[atomics.order]/4） | 全 seq_cst 操作（フェンス含む）の全順序 | `F_R` と `F_S` の向きを決める |
| **/4.4** | `X`（seq_cst fence）が `A` に hb、`B` が `Y`（seq_cst fence）に hb、`A` が `B` に coherence-ordered before | **補題の本体**。`X precedes Y in S` を導く |
| `hb`（[intro.races]/7） | sb / sw / 推移 | 前提の連鎖 |
| `strongly hb`（[intro.races]/8） | sb、または seq_cst 同士の sw など | 使わない（/4 の第一段落のみ） |

補足: [atomics.order]/7 Note 4 のとおり、弱い order を混ぜると「SC だから順序が付く」とは言えない。本稿の証明は一般的な SC 保証ではなく、**/4.4 を 2 回適用して矛盾を出す**形に限定している。

### 3.4 除外したい許容実行（操作列）

現行コードで許される実行を、操作列で書く（前回報告 §8.3 と同じ形）。

```
初期: G=1, P=0(null), x=0
1. W: exchange(P, P1) → 旧値 null（retire なし）          (epoch 1)
2. W: publishEpoch → G=2                                  (S_E1)
3. R: L_E=2 (A1) → store y=2 (A2) → fetch_add(x)=1 (A3) → r1=P1 (A11)
        ← O = P1。R=2。この時点では P1 は現行の公開値
4. W: exchange(P, P2) → P1 を unlink                      (S1)
5. W: retire(P1, T=2)                                     (epoch 2 で enqueue)
6. W: publishEpoch → G=3                                  (S_E2)
7. W: exchange(P, P3) → P2 unlink / retire(P2, T=3) / publishEpoch → G=4   (S_E3)
8. S: M0=4 (A7) → r2 = load(x) = 0（HB の無い load なので古い値を読める）→ min=4
9. S: isOlder(T(P1)=2, 4) = true → delete P1              ← Reader は P1 を使用中 → UAF
```

この実行が「N1 を満たさない」実行である。Reader は A も B も満たしていない（走査は Reader を見ておらず、Reader は unlink 後の publication を観測していない）。

### 3.5 補題（SC フェンスによる排除）と証明

**補題**: 次の 2 つは同時に成立しない。
(i) `r1` が `S1` より前の値（= O）を読む。
(ii) `r2` が `R1` より前の値（= 0）を読む。
ただし `F_R` は `R1` の後に `sb`、`r1` の前に `sb`、`F_S` は `S1` に `hb` され、`r2` の前に `sb` されているものとする。

**証明**

1. (i) より、`r1` は `S_O`（O を store した操作）の値を読み、`S_O` は `mo(P)` 上で `S1` より前。よって [atomics.order]/3.3 により **`r1` は `S1` に coherence-ordered before**。
2. `F_R` は `r1` に `hb`（`sb`）、`S1` は `F_S` に `hb`（仮定）。[atomics.order]/4.4 を (A=`r1`, B=`S1`, X=`F_R`, Y=`F_S`) に適用して **`F_R` precedes `F_S` in S**。
3. (ii) より、`r2` は `R1` より前の値（`R1` より前の store が書いた 0）を読む。よって [atomics.order]/3.3 により **`r2` は `R1` に coherence-ordered before**。
4. `F_S` は `r2` に `sb`（したがって `hb`）、`R1` は `F_R` に `sb`（したがって `hb`）。/4.4 を (A=`r2`, B=`R1`, X=`F_S`, Y=`F_R`) に適用して **`F_S` precedes `F_R` in S**。
5. S は全順序なので 2 と 4 は矛盾する。∎

**仮定の充足**（実コードでの根拠）

| 仮定 | 充足根拠 |
|---|---|
| `F_R` が `R1` の後・`r1` の前 | `enterReader` の depth RMW の直後に置き、呼出し元がその後に `loadCurrentState` する（§3.2） |
| `F_S` が `r2` の前 | `getMinReaderEpoch` の loop の前に置く |
| `S1 hb F_S` | §3.6 の場合分け 1 で導く（`S1 sb S_E`（W1）＋`S_E sw M0`＋`M0 sb F_S`） |

### 3.6 A が §3.4 の実行を排除することの証明

走査が O を解放する場合を考える（`isOlder(T(O), min)` が真）。`min <= M0` なので `T(O) < M0`。

**場合分け 1: `M0` が「`S1` の直後の epoch 前進 `S_E` が書いた値」以上を読んだ場合**

* `S_E sw M0`（[atomics.order]/2、`S_E` は acq_rel の RMW、`M0` は acquire load）。
* W1（publication store と epoch 前進が同一スレッド順序。既存義務 N2）により `S1 sb S_E`。よって `S1 hb M0 hb F_S`。
* 補題（§3.5）が適用でき、(i) と (ii) は同時に成立しない。
  * (ii) が偽、つまり `r2` が `R1` 以降の値を読む。この枝は 3 つに分かれる。
    - **(a) `r2` が同一区画の `R1`（またはネストの +1）を読む**: `R1 sw r2` により、続く `y` の load は `R` 以降を読む（[intro.races]/14）。よって `min <= R`。P1（`T(O) > R`）により **この走査は O を解放しない**。解放が起きないので N1 の要求は空に満たされる。
    - **(b) `r2` が退出 store（A4 / A5）を読む**: `sw` により、Reader のアクセス（退出 store より `sb` 前）が走査以降に `hb` する。この走査の解放はその後なので、A が成立する。
    - **(c) `r2` が次の区画の値を読む**: 前区画は退出済みで、受け渡し（同一スレッドの `sb`、または slot 再取得の `sw`）により前区画のアクセスが `hb` する。A が成立する。
  * (i) が偽、つまり `r1` が `S1` 以降の値を読む → Reader は O を使っていない（B）。
* どちらの枝でも N1 が成立する。

**場合分け 2: `M0` がそれより古い値を読んだ場合**

* `M0` は `S_E` の前の値を読んだので `M0 <= epoch(S1)`。`T(O)` は `S1` の後に読む（W3）ので `T(O) >= epoch(S1)`。よって `M0 <= T(O)`。
* `min <= M0 <= T(O)` なので `isOlder(T(O), min)` は偽。**この走査は O を解放しない**（そもそも危険な実行にならない）。

以上から、O を解放するどの走査でも A か B が成立する。∎

**この証明が前提とするもの**: W1（既存 N2）、W3（retire epoch を unpublish の後に読む。既存）、P1（`T(O) > R`。前回監査で証明済み）、slot 受け渡しの HB（退出 → 次の登録。§6 A-O3）。

### 3.7 コストと副作用

| 項目 | 内容 |
|---|---|
| 変更行数 | 2 行（`EpochDomain.h` の `enterReader` と `getMinReaderEpoch`）＋ コメント |
| RT 経路 | `enterReader` が 1 ブロックに 1 回。x86-64 では通常 `mfence`（または同等の locked 命令）、AArch64 では `dmb ish` 相当。**1 ブロック 10.7 ms（48 kHz / 512）に対して無視できる範囲と見積もるが、本作業は build 禁止のため未計測** |
| Reclaimer 経路 | Non-RT。100 ms に 1 回。無視できる |
| メモリオーダー | seq_cst は 2 箇所のみ。既存の acquire/release は据え置き（強化のみで弱化なし） |
| 意味の変化 | `enterReader` にフェンスが入るため、`ScopedEpochReader`（非 RT）にも同じフェンスが入る。動作は変わらない |
| ABI | 変更なし（atomic の型・レイアウトを変えない） |
| 診断 | 変更なし（`getReaderSlotDetail` / `detectStuckReaders` / `HoldWitness` はそのまま） |

### 3.8 A が閉じないもの

* **N2**（publication と epoch 前進の単一スレッド順序）: A の前提として残る。A は N2 を強制しない。
* **N6**（`AudioEngine.Snapshot.cpp:31` / `AudioEngine.h:1613` のガード無し deref）: A は影響しない。同一スレッド依存のまま。
* **N7**（`drainAllUnsafe` / `retireImmediateDuringShutdown` の epoch 迂回）: A は影響しない。shutdown の quiescence 前提は別。
* **N5（slot 再利用の torn read）**: A は**同時に閉じる**。理由は §6 の注記。ただし「scan が見落とす」一般形が消えるわけではなく、閉じるのは「区画中の Reader が pre-unlink object を保持している」場合に限る（N1 の範囲）。

---

## 4. 候補方式 B（publication state の統合）

### 4.1 設計スケッチ

slot ごとに `active` と `epoch` を 1 語に統合する。

```
struct ReaderSlot {
    std::atomic<uint64_t> state;   // (epoch << 8) | flags(active|quarantine|depth?)
};
Reader 入場:  store state = (R << 8) | ACTIVE   （または CAS）
Reader 退出:  store state = (R << 8) | 0
Reclaimer:    w = load(state); epoch = w >> 8; active = w & ACTIVE;
              active ? min = min(min, epoch) : (何もしない or 貢献させる)
```

### 4.2 各論点

| 論点 | 評価 |
|---|---|
| publication の atomicity | 1 語 load で (epoch, active) が同時に取れる。**診断を含めて torn read は消える**（現行は `depth` と `epoch` を別々に読むため、別 incarnation の値が混ざり得る）。ただし安全性については、A の補題が同じ組み合わせを排除する（§3.8） |
| inactive / active の状態表現 | 「未使用」を表す値が必要。`epoch=0` を「制約なし」にするか、`kInactiveEpoch` を sentinel にするかで挙動が変わる（§4.3） |
| Reader の入場・退場 | 入場は 1 store（または CAS）、退場は 1 store。ネスト（depth）を語内に持つならビット幅が要る。depth を別 atomic に残すと統合の意味が薄れる |
| Reclaimer の走査 | 1 load/slot で済む（現行 2 load）。読み取り回数は減るが、**同期辺は増えない** |
| slot 再利用 | 前 incarnation の語は epoch が小さいので、**寄与させるなら保守側**。sentinel で「制約なし」にするなら見落としが残る |
| epoch 前進との順序 | 語に入る epoch は入場時の値のままで、前進の順序関係は現行と同じ（W1 が必要な点も同じ） |
| ABA と世代再利用 | min の計算に対しては ABA しない（epoch は単調増加で、古い語は必ず小さい epoch）。危険なのは「制約なし」→「制約あり」の遷移を見落とすこと |
| 既存 API・ABI | `ReaderSlot` のレイアウト変更。`ReaderSlotDetail`（epoch/depth/residency/active）、`getReaderSlotDetail`、`detectStuckReaders`（`kInactiveEpoch` を skip 判定に使用）、`quarantineFlags`、`HoldWitness`、関連テストに波及。外部 C ABI は無いが、ヘッダを include する全 TU が再コンパイル対象 |

### 4.3 N1 の証明を試みる（どこで止まるか）

B の走査は `w = load(state)` の 1 回である。前回と同じ議論で、この load は「`B` が `A` に hb しない任意の store」を読める（[intro.races]/10）。

* **寄与させる設計**（inactive でも `epoch` を min に貢献させる）: 古い語を読んだ場合の epoch は小さいので保守側になり、N1 の危険は減る。しかし **未使用スロット**の語は「制約なし」でなければならず（全スロットが min を 1 に固定すると永久に reclaim できない）、その sentinel を読んだ走査はスロットを飛ばす。飛ばす判断が入る限り、「初回入場の公開を見落とす」実行が残る。
* **sentinel を epoch の現在値にする設計**（構築時に `epoch = 1` を入れ、寄与させる）: min が 1 に固定され、`T >= 1` の entry は永久に解放されない。実用にならない。
* **CAS/RMW で走査する設計**: 走査を RMW にすると [atomics.order]/11 により「modification order の直前の値」を必ず読むので、*scan が Store を取りこぼす*形は消える。しかし「走査が Reader の入場より前」の場合に `M0` と `R` の順序を保証する辺が無く、同じ穴が別の形で残る（§7 の参考 1）。

* **登録時に初期化する設計**（スレッド登録の時点で語を書く）: 初回入場より前に公開されるが、走査が登録前の sentinel を読む実行は残る。穴の位置が「初回入場」から「初回登録」へ移るだけで、閉じない。

**止まる場所**: B は「状態の読み取りを原子的にする」だけで、**Reader の公開と走査の間の同期辺を 1 本も追加しない**。N1 に必要なのは辺であり、語の統合ではない。統合で消えるのは N5 の torn read（別々の atomic を読むことによる不整合）だけで、scan visibility そのものではない。

### 4.4 判定

```
N1-CANDIDATE-B = NOT PROVED
```

B を採る場合も、N1 を閉じるには A のフェンス（または同等の辺）が必要になる。つまり B は A の代替ではなく、A に追加する変更である。

---

## 5. 比較と推奨

| 観点 | A（seq_cst フェンス） | B（state 統合） |
|---|---|---|
| N1 の証明可能性 | **可**（§3.5〜§3.6） | **不可**（§4.3） |
| 変更対象 atomic の数 | 0（既存 2 オブジェクトにフェンスを追加） | 64 語（slot ごとに統合）＋ 走査・詳細 API の読み替え |
| 変更対象ファイル | `src/core/EpochDomain.h`（2 行） | `EpochDomain.h`（ReaderSlot / enter / exit / getMinReaderEpoch / getReaderSlotDetail / detectStuckReaders / 初期化）、診断利用側、テスト |
| RT 経路への影響 | `enterReader` にフェンス 1 回/ブロック（推定 20〜40 cycle、**未計測**） | 入場が単一 store か CAS。CAS なら競合時にスピンし得る。深さ・隔離フラグのパッキングが必要 |
| memory_order の強化範囲 | seq_cst 2 箇所のみ | 強化不要（ただし効果も無い） |
| 追加コスト | 実装 2 行、レビューは「配置の不変条件」1 件 | レイアウト・API・テストの改修。N1 は閉じない |
| slot 再利用への影響 | N5 の torn read も同時に閉じる（§3.8） | torn read は消えるが、見落としは残る |
| 既存 P1 証明への影響 | 無し（順序は強化のみ） | **再証明が必要**（公開が単一語になり、P1 の「epoch store → depth RMW」の段が消える） |
| N2 への影響 | 無し（A の前提として残る） | 無し（同じく前提） |
| N6 への影響 | 無し | 無し |
| N7 への影響 | 無し | 無し |
| 新たに発生する証明義務 | A-O1〜A-O5（§6） | B-O1（sentinel の穴。解けない）＋ P1 の再証明 ＋ ビット幅・パッキングの検証 |

**推奨: A。** 2 行で N1 が閉じ、他の証明義務を増やさず、ABI も変えない。B は N1 を閉じないため、単独採用の理由が無い（診断の原子性が欲しい場合は A の後に行う別作業）。

---

## 6. 残存義務（A 採用時）

| # | 義務 | 内容 |
|---|---|---|
| A-O1 | フェンス配置の不変条件 | 「公開 → `F_R` → publication load」「`M0` → `F_S` → slot 走査」を invariant として明記し、回帰で固定する |
| A-O2 | W1 の維持（= N2） | publication store と epoch 前進が同一スレッド順序であること。A の証明はこれを前提にする |
| A-O3 | slot 受け渡しの HB | 退出 store → 次の登録（`reserveReaderThread` の CAS）の `sw` を保証する。別スレッドが同じ slot を再利用する場合に必要 |
| A-O4 | 解放経路の一元性 | 解放は必ず「フェンス済みの `minReaderEpoch`」を使う。`drainAllUnsafe` / `retireImmediateDuringShutdown` は N7 の quiescence 前提のまま（A は触らない） |
| A-O5 | RT コストの受入 | 1 ブロック 1 フェンスの実測（build 禁止のため未実施） |

N5（torn read）について: A の補題は「走査が `R1` より前の値を読む」かつ「Reader が pre-unlink 値を読む」を同時に排除する。N5 の「前 incarnation の depth と今回前の epoch」の組み合わせは、前者（`r2` が `R1` より前）に当たるため、同じ補題で排除される。ただし `getReaderSlotDetail` / `detectStuckReaders` の診断読み取りは A の対象外で、診断値の torn read は残る（安全性ではなく表示の問題）。

---

## 7. 参考事項（実装案・未実装）

**実装へ進まない。** 比較の材料としてのみ置く。

1. **走査を RMW にする案（A の代替候補として検討したが不成立）**
   `load(slot.depth, acquire)` を `fetch_add(slot.depth, 0, acquire)` に変えると、[atomics.order]/11 により走査は「直前の値」を必ず読む。取りこぼしは消えるが、走査が入場より前の場合に `M0` と `R` の順序を保証する辺が無く、「Reader が入場した直後に古い `M0` で解放する」実行が残る。**単独では N1 を閉じない。**
2. **4 操作すべてを seq_cst にする案**
   `R1`・`r2`・`S1`・`r1` を seq_cst にすると、[atomics.order]/4.1 と [intro.races]/8.1（sb ⊂ strongly hb）で同じ矛盾が出る。フェンス案と等価だが、RT 側の seq_cst load が増えるため A より不利。
3. **publication に retire epoch を埋める案（タグ付きポインタ）**
   `currentStateBits` を (epoch, pointer) にし、Reader は読んだ語の epoch で自分の保護 epoch を上書きする。走査の可視性に依存しないが、語長（48 bit ポインタ＋16 bit epoch）とラップの検討が必要で、A より大きい変更になる。N1 を閉じる別系統として将来の選択肢。
4. **B（統合）を A と併用する案**
   A で N1 を閉じたうえで、診断の原子性（`getReaderSlotDetail` / `detectStuckReaders`）のために語を統合する。安全性のためではなく診断品質のための変更。

---

## 8. 証跡

`doc/work113/20261112-2104_N1方式比較/evidence/`

| ファイル | 内容 |
|---|---|
| `source_identity.txt` | commit / ConvoPeq.md の識別・内容一致・対象 12 ファイルの SHA-256 |
| `standard_excerpts.md` | 引用した規範文面（[atomics.fences]、[atomics.order]、[intro.races]）と使い方の対応表 |
| `atomic_ops_n1.txt` | N1 に関係する atomic 操作 17 点（ファイル・行・order・役割） |

再現コマンド（読み取りのみ）:

```
python C:\Users\user\AppData\Local\Temp\opencode\snap_check.py    # ConvoPeq.md の内容一致
python C:\Users\user\AppData\Local\Temp\opencode\n1_evidence.py   # source_identity.txt と atomic_ops_n1.txt を再生成
```

---

## 9. 停止条件の確認

* production source / test / CMake を変更していない。
* build / test / debugger / replay を実行していない（フェンスのコストは未計測と明記）。
* git の変更操作を行っていない。HEAD は `173e6995`、`origin/main` と一致。
* AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。

**本稿の提出をもって停止する。** A が証明できたことは Gate 1 全体の承認でも実装 GO でもない。次段階は Owner の判断を待つ。
