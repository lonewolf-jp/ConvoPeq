# A 条件確認: W1 の invariant 化、A-O3（slot 受け渡し）、A-O4（全解放経路）

作業日時: 2026-11-12 21:28〜
基準 commit: `173e6995`（HEAD = origin/main）
性質: **読み取り専用**。production source / test / CMake / git を変更していない。build / test / debugger / replay も実行していない。

---

## 0. 結果

```
A              = PROVED（条件付き: W1 と W3。条件を満たす経路に対して）
A-O3           = CONFIRMED（slot 受け渡しの HB を証明）
A-O4           = CONFIRMED（N1 の範囲で、epoch-gated 経路は 1 箇所に収束）
B              = NOT PROVED（変更なし）
新規発見        = W1/W3 違反が 2 経路（live: completeFade / teardown: retireCurrentAndTarget）。A のフェンスでは閉じない
IMPLEMENTATION = 未承認（本稿でも付与しない）
```

前回の指摘に対して、A-O3 を証明して**義務から前提へ移し**（§3）、A-O4 を全 deleter 実行点の列挙で確認した（§4）。その結果 A の前提は W1（と W3）だけになった。

ただし、W1 を「すべての経路」に広げて点検したところ、**既存の経路が W1/W3 に違反していた**（live 1 件、teardown 1 件。§2.5）。`SnapshotCoordinator::completeFade` は epoch 前進の後に publication を交換し、retire epoch を unpublish の前に読む。この経路は A のフェンスでは閉じない（P1 の前提そのものが崩れる）。したがって「N1 がシステム全体で成立する」とはまだ言えない。A が閉じるのは「W1/W3 を満たす経路の scan visibility」である。

---

## 1. ソースの同一性と、本稿だけで検証できる形

### 1.1 識別情報

| 項目 | 値 |
|---|---|
| HEAD | `173e69957b046290f7077797782f031ed8633fd0` |
| origin/main | 同上（ahead/behind 0 0） |
| ConvoPeq.md ヘッダ stamp | `2026-11-12 00:48:16` |
| ConvoPeq.md サイズ | 7,011,745 B |
| ConvoPeq.md SHA-256 | `1801bbbacf4d65484418a1cf978acf0c09fc728028b3c8c8105ddce95f25e14f` |
| 生成器再計算との一致 | **全文一致**（7,011,745 B 同士） |
| 抜粋対象 19 ファイル | すべて HEAD と同一（`git status --porcelain` が clean） |
| 設計差分 v2.9 | 53,979 B / `c3ae5099…48427`（不変） |

### 1.2 「11-12 版の実ソースを取得できない」への対応

報告書の証跡（ハッシュと一致証明）だけではソース上の前提を追認できない、という指摘は妥当である。そこで本稿は、**証明と発見が依存する関数の逐語抜粋を証跡に含めた**。

`evidence/source_excerpts.md`（**19 ファイル / 1,863 行 / 86,248 文字**）

* 19 ファイルの SHA-256 とサイズの表
* 関数単位の逐語抜粋（行番号付き）。`EpochDomain`（register / reserve / reacquire / enter / exit / current / publish / getMin / tryReclaim / enqueue / drainAll / ReaderSlot）、`RCUReader`（enter / exit / acquireThreadSlot / Guard）、`EQProcessor`（process の gate と publication 読み / loadCurrentState / exchange / retire / flush / driver / dtor）、`DeferredDeletionQueue`（reclaim / drainAllUnsafe / isOlder）、`ISRRetireRouter`（tryReclaim / drain 群 / terminalReclaim）、`RetireQuarantineStore`（drain / drainAllUnsafe）、`AudioEngine`（dtor / releaseResources / observe / audio block / timer）、`SnapshotCoordinator`（observe / switchImmediate / startFade / completeFade / retireCurrentAndTarget）、`SnapshotSlotStore`、`ObservedRuntime`、`AtomicAccess`
* 抜粋はすべて HEAD と同一の作業ツリーから取得している（対象 19 ファイルは clean）

これで 11-12 版のスナップショットを持たない環境でも、本稿の主張を逐語で照合できる。

---

## 2. W1 の invariant 化（承認済み）

### 2.1 正式文面

> **W1 — Unlink-to-Epoch-Advance Ordering Invariant**
>
> 保護対象オブジェクト O を publication から unlink する atomic operation `S1` と、その unlink に対応する epoch advancement `S_E` は、同一スレッド上で実行され、必ず `S1` が `S_E` に sequenced-before であること。
>
> この順序は、すべての publication / retirement 経路で維持されなければならない。異なるスレッドへの処理移譲や経路変更によって、この関係が失われる構成を許容しない。

補足を 3 点加える。

* 「対応する」の意味は、`S1` の後に同じ publication 系列で実行される最初の `publishEpoch()` とする。複数の publication が同じ epoch にまとまることは許容する（`S1` ごとに 1 回の前進を要求しない）。
* `S_E` は `globalEpoch` への `fetch_add(+1, acq_rel)`（`EpochDomain.h:357-359`）、`S1` は `currentStateBits` / `bandNodeBits[i]` / `SnapshotSlotStore::m_current` への `exchange` である。
* **W1 と対になる W3**（retire epoch を unpublish の後に読む）も、A の証明の前提である。W1 だけでは P1 が立たない。§2.5 の発見は両方の違反である。

### 2.2 経路別の充足（実測）

| 経路 | publication | 順序 | 判定 |
|---|---|---|---|
| EQ パラメータ変更 | `currentStateBits`（`Parameters.cpp:31`） | `exchange` → `currentEpoch()`（`Core.cpp:204`）→ pending → 後刻 `publishEpoch`（`Core.cpp:88,122`） | **✓**（同一 Message Thread） |
| EQ バンドノード | `bandNodeBits[i]`（`Coefficients.cpp:76`、`Core.cpp:864`） | 同上。`exchangeBandNode` → `retireBandNodeDeferred`（`Core.cpp:213`） | **✓** |
| EQ リセット | `currentStateBits`（`Core.cpp:351`） | `exchange` → retire → pending → flush | **✓** |
| snapshot 即時切替 | `m_current`（`SnapshotCoordinator.h:104`） | `exchangeCurrent` → `publishEpoch`（`:106`） | **✓**（正しい順） |
| snapshot フェード開始 | `m_target`（`SnapshotCoordinator.cpp:76`） | `exchangeTarget` → `currentEpoch`（`:78`） | **✓**（target は読者非公開。epoch 前進は別経路） |
| snapshot フェード中断 | `m_target`（`SnapshotCoordinator.cpp:128`） | `exchangeTarget` → `publishEpoch`（`:131`） | **✓** |
| **snapshot フェード完了** | `m_current`（`SnapshotCoordinator.cpp:160`） | **`publishEpoch`（`:159`）→ `exchangeCurrent`（`:160`）** | **✗ 違反**（§2.5） |
| snapshot 全退避（teardown） | `m_current`（`SnapshotCoordinator.h:186`） | `publishEpoch`（`:185`）→ `exchangeCurrent`（`:186`） | **✗ 違反**（`~SnapshotCoordinator` と `finalizeShutdown` からのみ。N7 の範囲） |
| エンジン world の commit | `RuntimeWorldAuthority` → `RuntimeIntentCoordinator::commit` | 本稿の抜粋セット外（world lifetime 機構） | **未点検**（W1-O5） |

### 2.3 W1 を破る変更の例（許容しないもの）

1. `flushPendingEpochAdvance` を別スレッド（RebuildThread、専用タイマスレッド）へ移す。
2. retire の enqueue を別スレッドに委譲し、`currentEpoch()` を移譲先で読む（W3 も同時に壊れる）。
3. `updateBandNode` を非同期化し、epoch 前進の後に別スレッドで `exchangeBandNode` を実行する。
4. `publishEpoch()` を publication の交換より前に置く（今回見つかった形。§2.5）。

### 2.4 維持方法（参考・未実装）

* publication と epoch 前進を 1 つの "publisher strand" に閉じ、両者が同じスレッドで実行されることを Debug で検査する。
* `publishEpoch()` の呼出し元を許可リスト化し、新しい呼出し元の追加をレビューで止める。
* 回帰テストで「exchange → publishEpoch」の順序を固定する（テスト追加は本作業の範囲外）。

### 2.5 発見: `SnapshotCoordinator::completeFade` の W1/W3 違反

#### コード

```
SnapshotCoordinator.cpp
 150:  GlobalSnapshot* target = m_slots.exchangeTarget(nullptr, acq_rel);   // target を外す
 159:  const uint64_t retireEpoch = m_epochProvider->publishEpoch();        // ★ epoch 前進（E → E+1）
 160:  GlobalSnapshot* old = m_slots.exchangeCurrent(target, acq_rel);      // ★ current を交換（old を unlink）
 169:  enqueueWithRetry(*m_epochProvider, old, snapshotDeleter, retireEpoch);  // ★ T(old) = E
```

* `publishEpoch()` は `fetch_add(+1)` の**戻り値**（前進前の値 E）を返す（`EpochDomain.h:357-359`）。したがって `T(old) = E` だが、`old` が unlink された時点の epoch は **E+1** である。
* これは W1（`S1 sb S_E`）と W3（retire epoch を unpublish の後に読む）の両方に反する。

#### 誰が読むか

* `m_coordinator` は `m_epochDomain` に束縛される（`AudioEngine.CtorDtor.cpp:45` の `m_coordinator(m_epochDomain)`）。
* Audio Thread は `audioThreadRcuReader`（同じ `m_epochDomain`）で入場し（`AudioEngine.Processing.BlockDouble.cpp:158`）、`makeRuntimeReadHandle(audioCtx)` → `observeCurrentRuntime` → `m_slots.loadCurrent(acquire)`（`AudioEngine.h:3592` → `SnapshotCoordinator.h:81`）で **`m_current` を読む**。
* `tryCompleteFade()` は Timer（Message Thread）から呼ばれる（`AudioEngine.Timer.cpp:1113`）。

#### 窓

1. Audio Thread が 159 と 160 の間に入場し、`R = E+1` を得る。
2. その直後に `m_current` を読むと、まだ `old` が公開されている（160 の交換が起きていない）ので `old` を得る。
3. `old` は `T = E` で enqueue される。`R = E+1 > T = E`。
4. 走査の `min` は `R` 以下であり、他の Reader が居なければ `min = E+1 > T = E`。`isOlder(E, E+1)` が真になり、**使用中の `old` が解放される**。

#### なぜ `switchImmediate` は安全か（対比）

`switchImmediate` は `exchangeCurrent`（104）→ `publishEpoch`（106）の順で、`T = E` は unlink 時点の epoch（E）と一致する。この順序では、`T` を読む前進 `S_E` が publication の store より後になるため、`R = E+1` を得た Reader の `L_E` には store が happens-before し、write-read coherence（[intro.races]/14）が `old` の読みを禁じる。**順序が逆だとこの禁止が消える**のが、今回の穴の正体である。

#### 影響と限界

* 窓は 159 と 160 の間（数命令）で、fade 完了 1 回につき 1 度しか開かない。実測した crash は無いし、確率の見積りもしていない（build 禁止）。
* ただしモデル上は「毎回許される」実行であり、W1 を invariant とする以上、除外できない。
* teardown 専用の `retireCurrentAndTarget`（`SnapshotCoordinator.h:185-186`）も同じ順序で、`~SnapshotCoordinator`（`:57`）と `finalizeShutdown`（`:67`）から呼ばれる。Audio 停止後にしか走らないため N7 の範囲だが、quiescence が崩れれば同じ UAF になる。
* `ObservedRuntime` は `get()` で `ptr` を返し（`ObservedRuntime.h:51`）、`generation()` は `ptr->generation` を読む（`:57`）。したがって「解放済み snapshot を保持しても誰も deref しない」という逃げは利かない。

#### 参考（修正案・未承認・未実装）

`switchImmediate` と同じ順序に揃える。すなわち `exchangeCurrent(target)` を先に実行し、その後に `publishEpoch()` を呼んで戻り値を retire epoch にする（`T = epoch(S1) = E` となり W1/W3 を満たす）。設計判断は Owner に委ねる。

---

## 3. A-O3 の確認: slot 再利用時の HB

### 3.1 確認した事実（コード実測）

1. **`kInactiveEpoch` を書くのは 2 箇所だけ**。ctor の初期化（`EpochDomain.h:58`）と `exitReader` の最終退出（`:326`、`release`）。他に inactive を書く経路は無い（`grep` で確認。他の出現は比較・期待値・初期化子）。
2. **slot の取得は必ず `kInactiveEpoch` からの CAS**。
   * `registerReaderThread`（`:140-144`）: `compare_exchange(epoch, kInactiveEpoch → kReservedEpoch, acq_rel, acquire)`
   * `reserveReaderThread`（`:181-186`）: 同じ CAS
   * `reacquireReaderSlot`（`:233-236`）: 所有者検査の後に同じ CAS
   したがって、ある slot を新しいスレッドが使えるのは「前の所有者が `kInactiveEpoch` を書いた後」だけである。
3. **退出の順序**: `exitReader` は `depth` を `fetch_sub`（acq_rel、`:288-290`）し、`previousDepth == 1` のときだけ `epoch = kInactiveEpoch`（release、`:326`）を書く。`depth > 0` のままスレッドが死んだ場合（stuck reader）は inactive にならず、slot は転送されない。

### 3.2 証明（handoff lemma）

**補題**: 区画 I₁ が O を読み、その後 I₁ が退出して別の区画 I₂ が同じ slot を取得した場合、I₁ の O へのアクセスは I₂ の入場（およびそれ以降の走査・解放）に happens-before する。

証明:

1. I₁ のアクセスは I₁ の最終退出より `sb` 前（`process()` のスコープ終了で guard が破棄される）。
2. I₁ の最終退出は `depth` の `fetch_sub`（acq_rel）の後に `epoch = kInactiveEpoch` を `release` で書く（`:288` → `:326`、`sb`）。
3. I₂ の slot 取得は `kInactiveEpoch` からの CAS（acq_rel）。この CAS は 2. の release store が書いた値を読むので、[atomics.order]/2 により **`sw`** が成立する。
4. I₂ は CAS の後（同一スレッド）で `enterReader` を呼び、`epoch = R₂`（release）と `depth++`（acq_rel）を行う。
5. 走査の `r2` が I₂ の `depth` を読めば `sw` が成立し、I₂ の epoch store も可視になる（[intro.races]/14）。
6. 1〜5 を推移でつなぐと、I₁ のアクセス → I₁ の退出 store → I₂ の CAS → I₂ の入場 → 走査 → 解放、が `hb` の鎖になる。∎

**系（同時所有の排除）**: CAS は 1 つのスレッドだけが成功する。失敗した側は別の slot を探すか、`reacquireReaderSlot` の所有者検査で落ちる。したがって 2 つのスレッドが同じ slot を同時に所有することはない。

### 3.3 反例の検討

| 反例候補 | 成否 | 理由 |
|---|---|---|
| 2 スレッドが同じ slot を同時に確保する | 起きない | CAS の線形化点が 1 つ（§3.2 系） |
| 前の所有者が退出せずに slot を奪われる | 起きない | inactive を書くのは退出時だけ（§3.1-1） |
| stuck reader の slot が再利用される | 起きない | `depth > 0` のままでは inactive にならない（§3.1-3）。stuck は liveness の問題で `detectStuckReaders` の対象 |
| quarantine 済み slot の再利用で見落とす | 起きない | 走査は `quarantined && depth == 0` のときだけ除外する（`EpochDomain.h:392`）。転送後は depth > 0 なので寄与する |
| 転送が CAS 以外の経路で起きる | 起きない（現行） | `epoch` を書くのは ctor / enterReader / 3 つの CAS / exitReader のみ |

### 3.4 残留（A-O3'）

* 「slot の取得は `kInactiveEpoch` からの CAS である」ことが A-O3 の線形化点である。将来、slot を直接代入する経路（初期化のやり直し、強制奪取）を追加すると A-O3 は失効する。
* `quarantineFlags` は転送時にクリアされない。安全性には影響しない（§3.3）が、診断値は転送をまたぐので表示側で誤解しないこと。

---

## 4. A-O4 の確認: 全解放経路

### 4.1 deleter が実行される全箇所

`deleter(ptr)` / `retryable(ptr)` / `delete` の実行点を製品コードから列挙した（`src/tests` を除く）。

| # | 実行点 | 経路 | ゲート | 分類 |
|---|---|---|---|---|
| 1 | `DeferredDeletionQueue.h:192` | `reclaim(min)` ← `EpochDomain::tryReclaim`（`:554`）/ `reclaimRetired`（`:718`） | `isOlder(entry.epoch, min)`、`min = getMinReaderEpoch()` | **epoch-gated** |
| 2 | `RetireQuarantineStore.h:145` | `drain(min, isOlderFn)` ← `drainQuarantineStore`（`ISRRetireRouter.cpp:646`）/ `drainEmergencyAndTerminal`（`:848`） | `isOlderFn(e.epoch, min)`、`min = minReaderEpoch()` | **epoch-gated** |
| 3 | `ISRRetireRouter.cpp:214` | `TerminalReclaimAuthority::drain`（`:107`）← `tryReclaim`（`:316`）/ `drainTerminalReclaim`（`:837`） | `isOlderFn(e.epoch, min)`（`:134`） | **epoch-gated** |
| 4 | `ISRRetireRouter.cpp:167` | 同上の retryable 実行 | 同上（`:134` で選別済み） | **epoch-gated** |
| 5 | `ISRRetireRouter.cpp:772` | `terminalReclaim`（同期解放） | `isOlder(epoch, minReaderEpoch)`（`:761-762`） | **epoch-gated** |
| 6 | `ISRRetireRouter.cpp:814` | `terminalReclaimRetryable` | 同上（`:799-800`） | **epoch-gated** |
| 7 | `AudioEngine.Retire.cpp:82,399` | `m_coordinator.reclaim(m_retireRouter->getMinReaderEpoch())` | 同じ `min` | **epoch-gated**（同じ funnel） |
| 8 | `DeferredDeletionQueue.h:248` | `drainAllUnsafe` ← `EpochDomain::drainAll`（`:587`）← `~EQProcessor`（`Core.cpp:270`）、`AudioEngine` dtor（`CtorDtor.cpp:340`） | 無し（強制） | **force（N7）** |
| 9 | `RetireQuarantineStore.h:186` | `drainAllUnsafe` ← `drainAllQuarantineStore`（`ISRRetireRouter.cpp:688-694`）← `~EQProcessor`（`Core.cpp:266`）、dtor（`CtorDtor.cpp:341`）、`releaseResources`（`ReleaseResources.cpp:444,594`） | 無し（強制） | **force（N7）** |
| 10 | `ISRRetireRouter.cpp:278` | `TerminalReclaimAuthority::drainAll`（`:224`）← 同上 | 無し（強制） | **force（N7）** |
| 11 | `EQProcessor.Core.cpp:188,195` | `retireImmediateDuringShutdown` | 無し | **dead code**（呼出し 0 件） |
| 12 | `DeletionQueue.cpp:24,51` | レガシー `DeletionQueue`（`SnapshotRetireManager`） | 24 は無し、51 は `min` 引数 | **dead code**（`SnapshotRetireManager` のインスタンス 0 件） |

### 4.2 判定

1. **epoch-gated 経路（#1〜#7）は 1 つの funnel に収束している。** すべて `provider_->getMinReaderEpoch()` を通る（`ISRRetireRouter.cpp:413-417`、`EpochDomain.h:554,718,761,799`）。したがって A の `F_S` を `getMinReaderEpoch` の 1 箇所に置けば、全 epoch-gated 経路を覆える。
2. **force 経路（#8〜#10）は N1 の対象外**である。`min` を用いず、Audio 停止という外部事実に依存する（N7）。A はこれらを閉じないし、閉じたことにしない。
3. **force 経路の guard は同期しない。** `activeReaderCount() == 0`（`CtorDtor.cpp:335`、`ReleaseResources.cpp:593`）は `depth` の acquire load（`EpochDomain.h:410-422`）であり、走査と同じ可視性の性質を持つ。つまり「scan が見た 0」は quiescence の証明にならない。N7 は**外部事実（audio device の停止、device callback が戻ったこと）**で論じる必要がある。`ReleaseResources.cpp:432` のコメント自体が「※ 前提：実行時確認は未検証」と書いている。
4. **dead code（#11, #12）は N1 に影響しないが、再利用は禁止**（A-O4'）。特にレガシー `DeletionQueue::enqueue` の容量超過は `epoch` を見ずに即時解放する（`DeletionQueue.cpp:24`）。将来 `SnapshotRetireManager` を使うと、そこに穴が開く。

### 4.3 A-O4 の結論

「N1 の範囲（epoch-gated 経路）では、解放はすべて `getMinReaderEpoch()` の結果で判定されており、A の `F_S` 1 本で覆える」。force 経路と dead code は境界として明示し、N7 と A-O4' に割り当てる。

---

## 5. 条件付き証明の最終形

### 5.1 定理と前提

**定理（A 採用時の N1）**: 次の前提 H1〜H6 の下で、区画中の Reader が読んだ object O は、O を解放する走査との間で「走査が Reader を観測している（HB が成立する）」か「Reader は unlink 後の publication を観測しており O を使っていない」のいずれかが成立する。

| # | 前提 | 内容 | 状態 |
|---|---|---|---|
| H1 | フェンス配置 | Reader: `R1` の後・publication load の前に `seq_cst` フェンス。Reclaimer: `M0` の後・slot 走査の前に `seq_cst` フェンス | 未実装（設計） |
| H2 | **W1** | unlink `S1` と対応する epoch 前進 `S_E` が同一スレッドで `S1 sb S_E` | EQ は充足。**snapshot completeFade は違反**（§2.5） |
| H3 | **W3** | retire epoch を unpublish の後に読む | EQ は充足（`Core.cpp:204,213`）。**completeFade は違反** |
| H4 | P1 | Reader が読んだ O について `T(O) > R`（W1/W3 から導く。前回監査） | W1/W3 が成り立つ経路で成立 |
| H5 | handoff | slot の受け渡しは `kInactiveEpoch` からの CAS を経由し、そこで `sw` が付く | 本稿 §3 で証明 |
| H6 | funnel | 解放はすべて `getMinReaderEpoch()` の結果で判定される（force 経路は N1 の外） | 本稿 §4 で確認 |

### 5.2 成立の範囲

* **EQ ドメイン（`currentStateBits` / `bandNodeBits`）**: H2〜H6 を満たす。A のフェンスで N1 が成立する。
* **snapshot publication（`m_current`）**: H2 と H3 を満たさない（`completeFade`。teardown の `retireCurrentAndTarget` も同型）。A のフェンスを入れても N1 は成立しない。**先に順序を直す必要がある**（§2.5 の参考）。
* **エンジン world publication**: 本稿の抜粋セット外（W1-O5）。

### 5.3 証明が失効する条件

* W1 または W3 が破れる（§2.3 のいずれか、および §2.5 の既存違反が残る）。
* `F_R` の位置が「公開 → フェンス → publication load」でなくなる。
* `F_S` が `M0` の前に置かれる、または slot 走査の後に置かれる。
* 解放経路が `getMinReaderEpoch()` を経由しなくなる（新しい epoch-gated 風の経路の追加、または force 経路の常用）。
* slot の取得が `kInactiveEpoch` の CAS でなくなる（A-O3'）。

---

## 6. 残る義務

| # | 義務 | 内容 | 状態 |
|---|---|---|---|
| A-O1 | フェンス配置の不変条件 | 「公開 → `F_R` → load」「`M0` → `F_S` → 走査」を invariant として明記し、回帰で固定 | 未（設計） |
| A-O2 | W1 の維持 | §2 の invariant。既存違反（§2.5）の解消を含む | invariant 化済み。違反 2 件（live 1 / teardown 1）が残る |
| A-O3' | CAS 線形化点の維持 | slot 取得は `kInactiveEpoch` からの CAS のみ | 現行で充足 |
| A-O4' | 解放経路の境界 | epoch-gated は `getMinReaderEpoch` 経由。force は N7。dead code を再利用しない | 現行で充足（境界の明文化が残り） |
| A-O5 | RT コストの実測 | 1 ブロック 1 フェンスの実測（build 禁止のため未実施） | 未 |
| W1-O5 | エンジン world の点検 | `RuntimeIntentCoordinator::commit` の unlink と epoch 前進の順序 | 未（本稿の範囲外） |
| N7 | quiescence の根拠 | `activeReaderCount()==0` は同期しない scan。外部事実で論じる | 未（N1 の外） |
| N6 | ガード無し deref 2 箇所 | `AudioEngine.Snapshot.cpp:31`、`AudioEngine.h:1613` | 未（N1 の外） |

---

## 7. 証跡

`doc/work113/20261112-2128_A条件確認/evidence/`

| ファイル | 内容 |
|---|---|
| `source_excerpts.md` | 証明と発見が依存する関数の逐語抜粋（19 ファイル / 1,863 行 / 86,248 文字、SHA-256 付き）。11-12 版のソースを持たない環境でも本稿を照合できる |

再現コマンド（読み取りのみ）:

```
python C:\Users\user\AppData\Local\Temp\opencode\snap_check.py       # ConvoPeq.md の内容一致
python C:\Users\user\AppData\Local\Temp\opencode\a_cond_evidence.py  # source_excerpts.md を再生成
```

---

## 8. 停止条件の確認

* production source / test / CMake を変更していない。
* build / test / debugger / replay を実行していない。
* git の変更操作を行っていない。HEAD は `173e6995`、`origin/main` と一致。
* AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。
* W1 の invariant 化は**文書としてのみ**行った（AGENTS.md は変更禁止のため、本報告書に置く）。

**本稿の提出をもって停止する。** A の条件付き証明は「W1/W3 を満たす経路」に限って実装判断に使える状態になったが、既存の違反が 1 件残る。実装 GO・Gate 1 承認は付与しない。次段階は Owner の判断を待つ。
