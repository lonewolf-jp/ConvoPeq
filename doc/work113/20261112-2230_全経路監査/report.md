# 全経路監査: W1-O5（world commit）と W1/W3 充足表

作業日時: 2026-11-12 22:30〜
基準 commit: `173e6995`（HEAD = origin/main）
性質: **読み取り専用**。production source / test / CMake / git を変更していない。build / test / debugger / replay も実行していない。

---

## 0. 結果

```
W1-O5（world commit）   = W1/W3 を満たす（A の対象になる）
completeFade（修正後）   = W1/W3 を満たす（承認済みの順序に変更する前提）
retireCurrentAndTarget  = 同じ順序に変更する前提で満たす（案 1 を推奨どおり採用）
A が適用できる EBR 経路  = 4（EQ state / EQ band node / snapshot（修正後）/ world）
A では閉じない経路       = 3（convolver IRState / convolverState / DSPCore）
システム全体の N1        = NOT PROVED（A の前提外の経路が残る）
実装 GO                  = 未承認
```

world の publication は、`publishAndSwap` の後に `markRetireEpoch()` で epoch を進めて retire する順序で、**W1 と W3 を満たしていた**。epoch 前進が publication の交換より前にあるのは snapshot の 2 経路だけで、うち live なのは `completeFade` のみ（§4）。

一方、全経路を並べると、**A の前提（W1/W3 + EBR slot を持つ reader）を満たさない経路が 3 つ残る**。convolver の IRState は reader が slot を持たず、convolverState は別機構（SafeStateSwapper）、DSPCore は pin + world epoch で寿命を決めている。したがってシステム全体の N1 はまだ PROVED にできない。

---

## 1. ソースの同一性

| 項目 | 値 |
|---|---|
| HEAD | `173e69957b046290f7077797782f031ed8633fd0` |
| origin/main | 同上（ahead/behind 0 0） |
| ConvoPeq.md | 7,011,745 B / `1801bbbacf4d65484418a1cf978acf0c09fc728028b3c8c8105ddce95f25e14f`、生成器再計算と全文一致 |
| 抜粋対象 | **27 ファイル**（`evidence/source_excerpts.md`、2,445 行 / 116,319 文字、SHA-256 付き） |

抜粋は **HEAD の内容に固定**している。`src/audioengine/AudioEngine.Commit.cpp` だけは作業ツリーに未コミット変更（別作業）があるため、`git show HEAD:` の内容を使い、表にその旨を明記した。差は include 1 行と診断ブロックのみで、本稿が引用する `onRuntimeRetiredNonRt` の領域は同一である（行番号は HEAD 基準で 6 行ずれる）。

---

## 2. 監査の枠組み

「経路」を **publication を reader が load し、その object を epoch 付きで retire する組**と定義した。A が適用できる条件は次の 4 つ。

| 条件 | 内容 |
|---|---|
| C1 | W1: unlink `S1` が epoch 前進 `S_E` に sequenced-before（同一スレッド） |
| C2 | W3: retire epoch を unpublish の後に読む |
| C3 | reader が EBR slot を持ち、load の前に公開している（`protectedRead` / `active()`） |
| C4 | 解放が `getMinReaderEpoch()`（フェンス済み）の funnel を通る |

C1/C2 は §3・§4、C3/C4 は §5 の表で判定する。

---

## 3. W1-O5: world commit 経路

### 3.1 publication の実体

| 項目 | 実体 | 場所 |
|---|---|---|
| atomic | `RuntimeStore<RuntimeState, RuntimeWorldAuthority>::current` | `core/RuntimeStore.h:40-95` |
| swap | `WriteAccess::publishAndSwap(next)`（戻り値 = 旧 world） | `RuntimeWorldAuthority.h:312` |
| 書き手 | `RuntimeWorldAuthority` が所有する唯一の `WriteAccess`（INV-X4-3） | `RuntimeWorldAuthority.h:151-157, 334` |
| 読み手 | `runtimeStore_.observe()`（acquire） | `RuntimeStore.h:77-80`、`RuntimeWorldAuthority.h:211-221` |

`RuntimeIntentCoordinator::commit`（`ISRRuntimePublicationCoordinator.cpp:76-122`）は **world を swap しない**。publication metadata を bake（`pubWorld->publication = ...`）して状態を `Publishing → Ready` に遷移させるだけである。物理 swap は `RuntimeWorldAuthority::publish()`（`:312`）が行い、`commit` はその手前で呼ばれる（commit-before-swap）。

### 3.2 epoch 前進と retire epoch

```
RuntimePublishExecutor.h
  60: oldWorld = authority.publish(...)            ← S1（publishAndSwap at RuntimeWorldAuthority.h:312）
  75: bridge.willRetireRuntimeNonRt(oldWorld)      ← onRuntimeRetiredNonRt（EBR retire ではない）
  76: bridge.retirePublishedRuntimeWorldNonRt(oldWorld, false)
        → enqueueDeferredDeleteNonRt(...)         （AudioEngine.h:3979）
        → epoch = markRetireEpoch()               （AudioEngine.h:4713）
            = m_retireRouter->publishEpoch()      （AudioEngine.Publication.cpp:18）
        → enqueueWithRetry(ptr, deleter, epoch, World)   （AudioEngine.h:4716）
 110: ctx.engine.advanceRetireEpoch()             ← 追加の前進（retire の後）
```

* `markRetireEpoch()` は `fetch_add(+1)` の**戻り値**を返す。したがって `T(oldWorld)` は「前進前の値」で、unlink 直後の epoch と一致する。
* `advanceRetireEpoch()`（`:110`）は retire の後に余分な前進をするだけで、W1 を壊さない。

### 3.3 W1 / W3 の判定

| 条件 | 判定 | 根拠 |
|---|---|---|
| W1（`S1 sb S_E`） | **満たす** | `publish()` と `enqueueDeferredDeleteNonRt` は同じ executor 呼出しの中で、`sb` で並ぶ。`S_E` は `markRetireEpoch`（`AudioEngine.h:4713`） |
| W3（T を unpublish 後に読む） | **満たす** | T は `enqueueDeferredDeleteNonRt` の中で読む（swap の後） |
| 同一スレッド | **満たす** | world の publish は CoordinatorLoop の executor と startup bootstrap（`AudioEngine.Init.cpp:82-97`）のみ。`WriteAccess` が 1 つしかないため並行 publish は構造的に起きない |
| retire の変種 | **満たす** | QueueFull → Q/E/T へ同じ `epoch` で移送、quarantine sink も同じ値。shutdown 分岐は `AudioEngine.h:4693` で別に `markRetireEpoch()` を読むが、これも enqueue 時点の値で W3 を満たす |
| C4（解放の funnel） | **満たす** | engine ドメインの `getMinReaderEpoch()` 経由（前回 §4 の #1〜#7） |

### 3.4 注意（別物の混同）

`AudioEngine.Commit.cpp` の `intent.retireEpoch = world->generation`（HEAD `:501`）は **RetireIntent（slot-state System 1）のフィールド**であり、world object の EBR retire epoch ではない。EBR の retire epoch は §3.2 の `enqueueDeferredDeleteNonRt` が読む値である。この 2 つを取り違えると「world は generation で gated されている」と誤読する。

### 3.5 world の読者

| reader | guard | 場所 |
|---|---|---|
| Audio Thread | `RCUReaderGuard audioThreadRcuReader`（engine ドメイン、ブロック全体） | `AudioEngine.Processing.BlockDouble.cpp:158` |
| 非 RT（publication / orchestrator / rebuild） | `RCUReaderGuard` または `ScopedEpochReader` | `AudioEngine.Publication.cpp:59-61`、`RuntimePublicationOrchestrator.cpp:49-51`、`AudioEngine.h:3592` |

`getCurrentSnapshot()` のように guard なしで読む API も残るが、それは snapshot の話で、world は `observePublishedWorld()` / `consumeWorldHandle()` の 2 経路に限定される（INV-X4-A）。

---

## 4. completeFade / retireCurrentAndTarget の再評価

承認された順序（`exchangeCurrent` → `publishEpoch`）に変更したと仮定して、W1/W3 を再評価する。

| 経路 | 変更後の順序 | W1 | W3 | 備考 |
|---|---|---|---|---|
| `completeFade`（live） | `exchangeCurrent(target)`（old を unlink）→ `publishEpoch()`（戻り値 E）→ enqueue(old, E) | **満たす** | **満たす** | `switchImmediate`（`SnapshotCoordinator.h:104-106`）と同型。store が `S_E` より前になるため、write-read coherence が古い snapshot の読みを禁じる |
| `retireCurrentAndTarget`（teardown） | 同上 | **満たす** | **満たす** | 案 1（同じ順序に統一）を採用する前提。N7 の外部 quiescence 証明は別途必要 |
| `startFade` / `resetFadeStateAndRetireTarget` | 変更なし | 満たす | 満たす | target は reader 非公開。`resetFadeStateAndRetireTarget` は `exchangeTarget` → `publishEpoch` で W1 を満たす |

修正後は、snapshot の publication も A の対象になる（C1/C2 を満たす）。

---

## 5. W1/W3 充足表（live な publication / retirement 経路）

| # | 経路 | publication atomic | reader（スレッド / guard） | retire（ドメイン / epoch の読み方） | W1 | W3 | C3 | A 適用 |
|---|---|---|---|---|---|---|---|---|
| 1 | EQ state | `currentStateBits` | Audio（`RCUReaderGuard`、EQ slot、`protectedRead`） | EQ ドメイン / exchange 後に `currentEpoch()`（`Core.cpp:204`） | ✓ | ✓ | ✓ | **可** |
| 2 | EQ band node | `bandNodeBits[i]` | Audio（同上） | EQ ドメイン / exchange 後に `currentEpoch()`（`Core.cpp:213`） | ✓ | ✓ | ✓ | **可** |
| 3 | snapshot | `m_current` | Audio + 非 RT（engine slot） | engine ドメイン / **`publishEpoch` 後に exchange**（`completeFade`） | ✗ → 修正後 ✓ | ✗ → 修正後 ✓ | ✓ | 修正後 **可** |
| 4 | world | `RuntimeStore::current` | Audio（ブロック guard）+ 非 RT（`RCUReaderGuard` / `ScopedEpochReader`） | engine ドメイン / swap 後に `markRetireEpoch`（`AudioEngine.h:4713`） | ✓ | ✓ | ✓ | **可** |
| 5 | convolver IRState | `currentIRState` | loader / rebuild スレッド（`acquireIRState` は素の load、`releaseIRState` は no-op） | engine ドメイン / swap 後に `enqueueDeferredDeleteNonRt`（`Lifecycle.cpp:48`→`:59`） | ✓ | ✓ | **未確認** | **不可（C3 未確認）** |
| 6 | convolver state | `convolverState` | UI（`UIRcuGuard` = `enterStateReader` = `SafeStateSwapper`。`getConvolverState` の呼出しは `AudioEngine.Snapshot.cpp:28` のみ） | `DeferredFreeThread` + `SafeStateSwapper::getMinReaderEpoch` | — | — | — | **不可（別機構・自己整合）** |
| 7 | DSPCore | handle / pin + world epoch | Audio（world guard）+ pin | pin 解放 + coordinator の epoch 待ち（`invalidateHandleBeforeDestroy`） | — | — | — | **不可（別機構）** |
| 8 | legacy `DeletionQueue` | — | — | `SnapshotRetireManager` 未インスタンス化 | — | — | — | dead |
| 9 | `retireImmediateDuringShutdown` | — | — | 呼出し 0 件 | — | — | — | dead |

（#6 の reader 列は「guard の実装が別機構」であることを示す。A のフェンスは `EpochDomain` の走査にしか効かないため、#6 には適用できない。）

### 5.1 区分

* **A が適用できる（C1〜C4 を満たす）**: #1, #2, #4。**#3 は §4 の修正後**。
* **A では閉じない（別の安全性根拠が必要）**: #5（C3 未確認＝未保護 reader）、#6（SafeStateSwapper）、#7（pin + world epoch）。
* **dead**: #8, #9。

### 5.2 #5（convolver IRState）の扱い

`acquireIRState()` は `consumeAtomic(currentIRState, acquire)` を返すだけで、`releaseIRState()` は何もしない（`Lifecycle.cpp:21-29`）。reader は loader / rebuild 系の関数（`LoadPipeline.cpp:30,73,539`、`Rebuild.cpp:64`、`LoaderThread.cpp:918`）で、**EBR slot を保持していることが確認できない**。retire は engine ドメインなので、仮に reader が engine slot を持っていなければ、A のフェンスでも `minReaderEpoch` でも守れない。

本稿では「未確認」とし、次のどちらかを次段階で確定する必要がある。

1. 各 reader 呼出し点が engine ドメインの slot（`ScopedEpochReader`）を保持していることを示す。
2. 保持していないなら、writer と reader が同一スレッドであること（または別の同期）を示す。示せない場合は、IRState の読みに guard を付ける設計変更が必要になる。

---

## 6. システム全体の N1

**NOT PROVED。** 理由は 3 つ。

1. snapshot の `completeFade` が W1/W3 に違反したまま（§4 の修正が未適用）。
2. convolver IRState の reader が EBR slot を持つことが未確認（§5.2）。
3. convolverState と DSPCore は A の対象外の機構で、それぞれ別の安全性根拠（SafeStateSwapper の age ベース、pin + world epoch）に依存している。これらの N1 相当の問いは本稿では判定していない。

A の証明が届く範囲は「C1〜C4 を満たす経路」であり、#1, #2, #4、および修正後の #3 がそれに当たる。

---

## 7. 残存義務（更新）

| # | 義務 | 内容 | 状態 |
|---|---|---|---|
| A-O1 | フェンス配置の不変条件 | 「公開 → `F_R` → load」「`M0` → `F_S` → 走査」 | 未（設計） |
| A-O2 | W1/W3 の維持 | 既存違反 2 件（`completeFade` / `retireCurrentAndTarget`）の解消を含む | 修正方針は承認済み・未実装 |
| A-O3' | CAS 線形化点の維持 | slot 取得は `kInactiveEpoch` からの CAS のみ | 現行で充足 |
| A-O4' | 解放経路の境界 | epoch-gated は `getMinReaderEpoch` 経由。force は N7。dead code を再利用しない | 現行で充足 |
| A-O5 | RT コストの実測 | 1 ブロック 1 フェンスの実測 | 未 |
| W1-O5 | world commit の点検 | 本稿 §3 で完了 | **完了** |
| W1-O6 | snapshot 2 経路の修正 | §4 の順序に変更（`completeFade` / `retireCurrentAndTarget`） | 未（承認済みの設計変更） |
| W1-O7 | convolver IRState の reader | slot 保持の確認、または guard 追加 / 同一スレッド性の証明 | 未 |
| N7 | quiescence の根拠 | force drain の `activeReaderCount()==0` は同期しない scan。外部事実で論じる | 未 |
| N6 | ガード無し deref 2 箇所 | `AudioEngine.Snapshot.cpp:31`、`AudioEngine.h:1613` | 未 |
| N9 | convolverState / DSPCore の別機構 | SafeStateSwapper と pin の安全性は別監査 | 未 |

---

## 8. 証跡

`doc/work113/20261112-2230_全経路監査/evidence/`

| ファイル | 内容 |
|---|---|
| `source_excerpts.md` | 本稿が依存する関数の逐語抜粋（27 ファイル / 2,445 行 / 116,319 文字、SHA-256 付き、HEAD blob 基準）。`AudioEngine.Commit.cpp` は作業ツリーに未コミット変更があるため HEAD blob から取得した旨を明記 |

再現コマンド（読み取りのみ）:

```
python C:\Users\user\AppData\Local\Temp\opencode\snap_check.py       # ConvoPeq.md の内容一致
python C:\Users\user\AppData\Local\Temp\opencode\a_cond_evidence.py  # source_excerpts.md を再生成（HEAD blob 基準）
```

---

## 9. 停止条件の確認

* production source / test / CMake を変更していない。
* build / test / debugger / replay を実行していない。
* git の変更操作を行っていない。HEAD は `173e6995`、`origin/main` と一致。
* AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。

**本稿の提出をもって停止する。** world commit は A の対象になることを確認したが、システム全体の N1 は NOT PROVED のままである。実装 GO・Gate 1 承認は付与しない。
