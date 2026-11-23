# Portable Reader Safety 再監査（READ-ONLY）

作業日時: 2026-11-12 20:40〜
基準 commit: `173e6995`（HEAD = origin/main）
対象: Epoch-Based Reclamation（EBR）による publication の Reader 安全性を、C++ メモリモデル上で判定する
性質: **読み取り専用**。production source / test / CMake / git を変更していない。build / test / debugger / replay も実行していない。

---

## 0. 判定

```
PORTABLE-READER-SAFETY = NOT PROVED
```

理由を一行でいうと、次の 2 つを分けたときに、前半だけが証明でき、後半が未証明である。

| 命題 | 判定 | 根拠 |
|---|---|---|
| **P1**: Reader が取得した object は、その Reader が区画内にいる限り reclaim されない | **PROVED**（保護経路。§7 の W1 を前提） | §7。write-read coherence と「retire epoch は unpublish の後に読む」順序で示せる |
| **P2**: reclaimer の走査は、区画内の Reader を必ず観測する | **NOT PROVED** | §8。Reader の depth>0 公開と走査の depth 読み取りの間に同期が無く、モデル上は 0 を読む実行が排除できない |

P1 が成り立っても、P2 が成り立たないと `minReaderEpoch` が実際より大きくなり、P1 の保護対象が解放され得る。したがって全体としては **NOT PROVED**。

「実装が間違っている」という判定ではない。**現行の同期（acquire/release のみ）では、モデル水準の保証が閉じていない**という判定である。x86_64 での実測と TSan の結果は §9 に整理したとおりで、これは証明ではない。

---

## 1. 作業開始条件：ソースの最新性

| 項目 | 値 |
|---|---|
| HEAD | `173e69957b046290f7077797782f031ed8633fd0` |
| origin/main | 同上（差分 0・push 済み） |
| HEAD の subject | `feat(isr): stamp retire entries with their domain and aggregate latency per domain` |
| ConvoPeq.md ヘッダ stamp | `2026-11-12 00:48:16` |
| ConvoPeq.md ファイル mtime | `2026-11-12 01:48:29` |
| ConvoPeq.md サイズ | 7,011,745 B |
| ConvoPeq.md SHA-256 | `1801bbbacf4d65484418a1cf978acf0c09fc728028b3c8c8105ddce95f25e14f` |
| 収録ファイル見出し数 | 406 |

### 1.1 鮮度の判定方法（mtime に依存しない）

`output_sourcecode_markdown.py --check` は **STALE**（`NEWER_SRC_COUNT = 84`）を返す。ただしこれは mtime 比較であり、commit 時の checkout で mtime が更新されるため、この判定だけでは内容の新旧を決められない（AGENTS.md に既知の罠として記録がある）。

そこで、生成器の関数（`parse_generated_stamp` / `build_markdown_text` / `_normalize_stamp_lines`）をそのまま使い、**ヘッダ stamp を固定して現行ツリーから期待文書をメモリ上で再生成し、実ファイルと比較**した。

| 比較 | 結果 |
|---|---|
| 正規化後の全文一致 | **一致** |
| バイト数（実ファイル / 期待） | 7,011,745 / 7,011,745（一致） |

したがって ConvoPeq.md の内容は現行ツリーと同一である。**古いスナップショットを最新ソースとして扱っていない。**

### 1.2 HEAD との対応（重要な限定）

作業ツリーには別作業の未コミット変更が modified 34 件・untracked 222 件ある。`src/` と `CMakeLists.txt` に限ると **37 件**（modified 26 + untracked 11）。

```
M CMakeLists.txt
src/CacheManager.cpp                     src/CacheManager.h
src/ConvolverControlPanel.cpp            src/DeviceSettings.cpp
src/IRConverter.cpp                      src/IRConverter.h
src/MKLNonUniformConvolver.cpp           src/MKLNonUniformConvolver.h
src/MainApplication.cpp                  src/MainApplication.h
src/MainWindow.cpp                       src/MainWindow.h
src/NoiseShaperLearner.cpp               src/NoiseShaperLearner.h
src/ProgressiveUpgradeThread.cpp
src/audioengine/AudioEngine.Commit.cpp   src/audioengine/AudioEngine.Learning.cpp
src/audioengine/AudioEngine.Reader.cpp
src/convolver/ConvolverProcessor.LoadPipeline.cpp
src/convolver/ConvolverProcessor.LoaderThread.cpp
src/convolver/ConvolverProcessor.ResampleAndFallback.cpp
src/convolver/ConvolverProcessor.Runtime.cpp
src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp
src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp
src/tests/TailLayerIntegrityTests.cpp
?? src/InputInspection.cpp                ?? src/InputInspection.h
?? src/convolver/IRTrailingSilence.h
?? src/learner/LearnedStateMigration.cpp  ?? src/learner/LearnedStateMigration.h
?? src/learner/LearnerStateAuthority.h
?? src/tests/AudioEngineHarness/RuntimeParameterGuardTests.cpp
?? src/tests/CacheManagerHeaderValidationTests.cpp
?? src/tests/InputInspectionTests.cpp
?? src/tests/LearnedStateMigrationTests.cpp
?? src/tests/LearnerStateAuthorityTests.cpp
```

**この一覧に本監査の引用ファイルは 1 つも含まれない。** 監査対象 18 ファイルはすべて HEAD と同一（`git status --porcelain` が clean）である。ConvoPeq.md が HEAD と一致するのは、この 18 ファイルについてのみ、という限定付きで読むこと。

監査対象の同定情報（サイズ / SHA-256 先頭 16 桁）は `evidence/source_identity.txt` §4 にある。

### 1.3 参照した受理済み設計

`doc/work113/20261111-1512_domain設計/design_diff.md`（53,979 B / SHA-256 `c3ae5099…48427`）。**改変していない。**

---

## 2. 監査範囲

| 対象 | 扱い |
|---|---|
| EQ ドメイン（`EQProcessor` が所有する `EpochDomain`） | **本監査の主対象**（Gate 1 の対象そのもの） |
| エンジン ISR ドメイン（`AudioEngine::m_epochDomain`） | `EpochDomain` / `DeferredDeletionQueue` / `RCUReader` が同一実装なので **P2 の結論（未証明）は同一**。P1 は W1（publication と epoch 前進の単一スレッド順序）を要するため、本稿で確認したのは EQ についてのみ |
| `ConvolverProcessor`（`SafeStateSwapper` 系） | 別実装のため対象外（同じ弱さを持つ可能性はあるが、本稿では判定しない） |
| queue の MPSC 整合性、shutdown 順序、world lifetime | §10 の残存義務として列挙するのみ |

前提として、Owner 指摘の「producer ordering 14/14」は**証拠として採用していない**。リポジトリ内に `14/14` という producer ordering の記録は見当たらず（`doc/` 全体を検索した結果、一致は work59 / work70 の別文脈と Decision-Pending 文書のみ）、再現できる形の証跡が無いためである。本稿は §4〜§7 でソースから関係を導き直した。既存の Gate 1 解析 `doc/work113/20261111-0700_Gate1解析/gate1_reader_safety.md` の §2（保護経路）と §3（保護されない経路）は再確認し、§7 で差分を述べる。

---

## 3. 経路の再構成

### 3.1 Reader 側（EQ ドメイン / Audio Thread）

```
EQProcessor::process()                       EQProcessor.Processing.cpp:486
  ├ RCUReaderGuard guard(rcuReader);         :488 → RCUReader.h:225 → RCUReader::enter() RCUReader.h:40
  │   ├ ownerThreadToken CAS（所有権の線形化点）        RCUReader.h:57-68
  │   ├ acquireThreadSlot()                            RCUReader.h:178-209
  │   │   ├ reserveReaderThread()  epoch: inactive→reserved（CAS acq_rel）EpochDomain.h:141-164
  │   │   └ registerReaderThread()                     EpochDomain.h:116-139
  │   └ enterReader(tid)                               EpochDomain.h:253-277
  │       ├ L_E: currentEpoch() = load(globalEpoch, acquire)   :262 / :344-348
  │       ├ store slot.epoch = L_E（release）                  :263
  │       └ fetch_add(slot.depth, +1, acq_rel)                 :266-268
  ├ const bool protectedRead = guard.rootEnterSucceeded();  :493
  ├ stateSnapshot = protectedRead ? loadCurrentState(acquire) : nullptr;  :497-499
  │   loadCurrentState → load(currentStateBits, acquire)     EQProcessor.h:785-788
  ├ …EQState のメンバーを参照（:636 以降）
  ├ BandNode: protectedRead のときだけ loadBandNode(acquire) :671-682
  └ スコープ終了で guard 破棄 → RCUReader::exit()（depth-- → epoch=inactive）RCUReader.h:227 / EpochDomain.h:280-342
```

* **flags / depth / epoch の読み取り順序**（reclaimer 側）は `getMinReaderEpoch()` の `depth → quarantineFlags → epoch` の順（`EpochDomain.h:379,391,399`）。Reader 側は `epoch → depth` の順に書く（`EpochDomain.h:263,266`）。この向きは正しい（読む側が depth を先に読むので、depth>0 を観測した時点で epoch の store が HB で見える）。
* acquire load の対象は `globalEpoch`（区画の開始）と `currentStateBits`（publication）の 2 つ。取得条件は「Reader が slot を取得して `enterReader` を完了したこと」で、`protectedRead` がその判定である。
* 保護が無いとき（slot 枯渇・登録クローズ・別スレッド所有）は publication を読まずに `nullptr` 経路へ落ちる。診断は `unprotectedReadCount_`（`EQProcessor.h:651-655`）。

非 RT 経路は `ScopedEpochReader`（`EQProcessor.h:505-511` → `ScopedEpochReader.h:56-68`）で、`active()` が false のときは dereference しない契約（`EQResponse.cpp:92-93`、`RebuildDispatch.cpp:706-707`、`EQProcessor.Core.cpp:386-395` ほか）。

**ガードを持たずに EQState を dereference する箇所が 2 つ残っている**（§10 の残存義務 N6）。

* `AudioEngine.Snapshot.cpp:31-32`（`getEQStateSnapshot()->toEQParameters()`）
* `AudioEngine.h:1613-1614`（`getEQWorldProjectionForBuild()`）

### 3.2 Writer 側

```
EQProcessor::setBandFrequency() ほか          EQProcessor.Parameters.cpp:19-38
  ├ loadCurrentState(acquire)                 :26  ← 旧状態をコピーするため
  ├ new EQState(*oldState); 変更              :28-29
  ├ S_B: exchangeCurrentState(new, acq_rel)   :31  → EQProcessor.h:790-796（atomic_exchange_explicit）
  │     ※「unlink」と「publication 更新」は同一 RMW。別操作ではない
  ├ retireEQStateDeferred(prev)               :33   → EQProcessor.Core.cpp:199-206
  │     └ T = currentEpoch()（acquire, exchange の後）  :204
  │       → enqueueDeferredDeleteWithFallback(ptr, deleter, T)  :26-80
  │         → coordinator → router → entry.epoch = T, entry.domainId 刻印
  ├ store m_epochAdvancePending = true（release）      :36  ← epoch 前進は遅延
  └ updateBandNode(band) → exchangeBandNode + retireBandNodeDeferred  Coefficients.cpp:73-83
```

* `currentEpoch()` の取得点は **unpublish（exchange）の後**。`retireEQStateDeferred` / `retireBandNodeDeferred` の中で読む（`Core.cpp:204,213`）。
* epoch の前進は `flushPendingEpochAdvance()`（`Core.cpp:83-97`）または 100 ms ドライバ（`Core.cpp:107-141`）で、**1 回の前進で複数の publication をまとめて跨ぐ**（同じ epoch のまま複数の retire が積まれる）。
* epoch の前進は `publishEpoch()` = `fetch_add(globalEpoch, +1, acq_rel)`（`EpochDomain.h:351-360`）。
* ドライバの呼出し元は `AudioEngine::timerCallback`（100 ms・Message Thread・Non-RT）。`flushPendingEpochAdvance` の呼出し元は `releaseResources`（`PublishReason::Release`）と `prepareToPlay`（`PublishReason::Prepare`）。

> 前稿 `gate1_reader_safety.md` §1 の「`Core.cpp:266` → flush」は、現在の行番号では `Core.cpp:351`（`resetToDefaults`）/ `Parameters.cpp:31` が該当する。構造（exchange → retire → pending → 後刻 publish）は同じ。

### 3.3 Reclamation 側

```
EQProcessor::driveRetireReclaimNonRt()        EQProcessor.Core.cpp:107-141
  └ m_ownedRetireRouter.tryReclaim()          ISRRetireRouter.cpp:630-638
      ├ provider_->tryReclaim()               EpochDomain.h:546-556
      │   └ deferredDeletionQueue.reclaim(getMinReaderEpoch())   :554
      ├ drainQuarantineStore()                ISRRetireRouter.cpp:643-649
      └ drainEmergencyAndTerminal()           ISRRetireRouter.cpp:842-854
```

```
getMinReaderEpoch()                           EpochDomain.h:372-408
  min = load(globalEpoch, acquire)            :374
  for each slot (64):
    depth = load(slot.depth, acquire)         :379
    flags = load(slot.quarantineFlags, acq)   :391
    if (quarantined && depth == 0) continue   :392
    if (depth == 0) continue                  :395
    epoch = load(slot.epoch, acquire)         :399
    if (epoch == inactive || epoch == reserved) continue  :400
    min = min(min, epoch)                     :403
  return min

DeferredDeletionQueue::reclaim(min)           DeferredDeletionQueue.h:148-225
  if (isOlder(entry.epoch, min)) canDelete = true          :174
  isOlder(a,b) = (int64_t)(a-b) < 0  →  a < b（wrap 安全）:308-311
  FIFO 先頭が削除不可なら即 break（後続は解放しない）        :219-221
  entry.deleter(entry.ptr)                                  :192
```

object の destroy は `deleter`（`deleteEQStatePtr` / `deleteBandNodePtr`、`Core.cpp:22-23`）で、reclaim のスレッド上で実行される。

**reclaim の実行スレッドは 1 つではない。** EQ ドメインは Message Thread（timer / releaseResources）だが、エンジンドメインの Q/E/T は `CoordinatorLoop`（背景 Non-RT スレッド）が drain する（`ISRRetireRouter.h:446-461`、`AudioEngine.Threading.cpp:475-485`）。後述の P2 はどちらのスレッドでも同じ形で問題になる。

---

## 4. atomic 操作と memory_order の一覧

出現数の実測は `evidence/memory_order_counts.txt`（対象 19 ファイル＝監査対象 18＋AGENTS.md、コメント含む）。

| order | 件数 |
|---|---|
| `memory_order_seq_cst` | **0** |
| `memory_order_acq_rel` | 131 |
| `memory_order_acquire` | 382 |
| `memory_order_release` | 183 |
| `memory_order_relaxed` | 130 |

`seq_cst` は対象 19 ファイルに **0 件**。`atomic_thread_fence` は 5 件あるが、いずれも `AudioEngine.h` の別経路（`getActiveCoeffSet` の acquire fence :4647、learning コマンドの enqueue/dequeue :5789,5803、learner dispatch の enqueue/dequeue :5827,5843）で、**EQ の `currentStateBits` と EBR の走査には関与しない**。

**EBR の経路に `seq_cst` の辺が 1 本も無いことが、P2 の未証明の直接の原因**である（§8）。

### 4.1 安全性に関与する操作（これだけを見ればよい）

| # | 対象 | 操作 | order | 場所 |
|---|---|---|---|---|
| A1 | `globalEpoch` | load | acquire | `EpochDomain.h:347`（`currentEpoch`） |
| A2 | `globalEpoch` | fetch_add(+1) | acq_rel | `EpochDomain.h:357-359`（`publishEpoch`） |
| A3 | `slot.epoch` | store R | release | `EpochDomain.h:263`（`enterReader`） |
| A4 | `slot.depth` | fetch_add(+1) | acq_rel | `EpochDomain.h:266-268`（`enterReader`） |
| A5 | `slot.depth` | fetch_sub(-1) | acq_rel | `EpochDomain.h:288-290`（`exitReader`） |
| A6 | `slot.epoch` | store inactive | release | `EpochDomain.h:326`（`exitReader`） |
| A7 | `slot.depth` | load | acquire | `EpochDomain.h:379`（`getMinReaderEpoch`） |
| A8 | `slot.epoch` | load | acquire | `EpochDomain.h:399`（`getMinReaderEpoch`） |
| A9 | `currentStateBits` | exchange(ptr) | acq_rel | `EQProcessor.h:793-795`（`exchangeCurrentState`） |
| A10 | `currentStateBits` | load | acquire | `EQProcessor.h:787`（`loadCurrentState`） |
| A11 | `slot.epoch` CAS | inactive→reserved | acq_rel / acquire | `EpochDomain.h:149-154` |
| A12 | queue `sequences` | store | release | `DeferredDeletionQueue.h:208` |
| A13 | queue `sequences` | load | acquire | `DeferredDeletionQueue.h:160` |
| A14 | queue `dequeuePos` | CAS | release / acquire | `DeferredDeletionQueue.h:180-184` |

補助（既定 order の確認）: `convo::publishAtomic` = release、`consumeAtomic` = acquire、`exchangeAtomic` = acq_rel（`AtomicAccess.h:52-72`）。

---

## 5. synchronizes-with / happens-before の関係

記号: `sb` = sequenced-before、`sw` = synchronizes-with、`HB` = happens-before（推移閉包）。`S_B` = A9（publication の exchange）、`S_E` = A2（epoch 前進）、`L_E` = A1（Reader の区画開始時の epoch 読み）、`L_B` = A10（Reader の publication 読み）、`R` = `L_E` が返した値、`T(O)` = object O を enqueue したときの entry.epoch。

```
Writer thread (control)                          Reader thread (audio)
─────────────────────────                        ─────────────────────
S_B  exchange(currentStateBits)  acq_rel
  │ sb
S_E  fetch_add(globalEpoch,+1)   acq_rel  ──sw──▶  L_E  load(globalEpoch) acquire   [L_E が S_E の書いた値を読んだ場合のみ]
  │ sb                                                  │ sb
T = currentEpoch() acquire                              │
  │ sb                                                  │
enqueue(entry.epoch = T)                                │
  │ sb                                                  ▼
sequences store release  ──sw──▶  load sequences acquire（reclaimer）      L_B  load(currentStateBits) acquire
                                          │ sb                                │ sw（L_B が S_B 系の release を読んだ場合）
                                          ▼ sb                                ▼
                                  entry 書き込みが見える                  new EQState の初期化が見える

Reader announcement:
  slot.epoch store release (A3)  ──sb──▶  slot.depth fetch_add acq_rel (A4)  ──sw──▶  slot.depth load acquire (A7)   [reclaimer が A4 の値を読んだ場合のみ]
                                                                                        │ sb
                                                                                        ▼
                                                                                  slot.epoch load acquire (A8)  → R が見える
```

### 5.1 成立する HB（証明に使える辺）

* **R1**: `L_E` が `S_E` の書いた値 R を読めば `S_E —sw→ L_E`。`S_E` より前に sb で並ぶ `S_B` は `S_B HB L_E`。
* **R2**: `L_E —sb→ L_B`。よって R1 の `S_B HB L_B`。
* **R3**: `L_B` が読んだ値の store（release 側）と `L_B` が synchronize するため、その object の初期化は `HB L_B`（copy-on-write した `EQState` の中身が見える）。
* **R4**: `A3 (release) —sb→ A4 (acq_rel RMW)`。reclaimer の `A7` が A4 の値を読めば `A3 HB A7`、続く A8 は R 以降の値を読む（write-read coherence）。
* **R5**: `enqueue` の entry 書き込み —sb→ `A12 (release)`、reclaimer の `A13 (acquire)` がその値を読めば `HB`。entry.epoch は正しく見える。
* **R6**: `S_B —sb→`（`retireEQStateDeferred` の）`currentEpoch()`。**同一スレッド内の write-read coherence により、この読みは直前の `S_E` が書いた値以上を返す。**

### 5.2 成立しない辺（ここが要点）

* **Reader の区画開始 → reclaimer の走査** の HB は無い。A4 と A7 の間に `sw` が成立するのは「A7 が A4 の書いた値を読んだ場合」だけで、その保証は無い。**モデルは A7 が 0 を読む実行を許す。**
* **Writer の `S_B` → Reader の `L_B`** の HB は、`L_B` がその `S_B` 自身の値を読んだ場合、または `L_E` が対応する `S_E` を読んだ場合に限り成立する。どちらでもない場合、`L_B` と `S_B` の間には HB が無く、`L_B` は「unlink 済みの値」を読める（ただし §7 のとおり安全）。
* **`seq_cst` フェンスは無い。** C++20 の fence-to-fence synchronization（[atomics.fences]）に相当する辺が 1 本も無い。

---

## 6. 中心的な検証事項への回答

### Q1. Writer の acq_rel exchange と Reader の acquire load の間に、どのような synchronizes-with / happens-before が成立するか

publication の atomic は `currentStateBits` 1 個で、unlink と publish は同一の `exchange`（A9）。`synchronizes-with` が成立するのは、**`L_B` が A9（またはその後続の release store）が書いた値を読んだとき**だけである。このとき `S_B —sw→ L_B` となり、`S_B` より前の Writer の処理（新しい `EQState` の初期化）が `HB L_B` になる。

区画の保護は別の atomic（`globalEpoch`）で取る。`L_E` が `S_E` の値を読めば `S_E —sw→ L_E`、推移で `S_B HB L_E HB L_B` になる（§5.1 の R1〜R2）。つまり **2 つの atomic が、Reader の 2 回の読み（epoch → pointer）の順序で鎖になる**。この鎖が P1 の全体である。

### Q2. Reader が unlink 前の publication value を取得する実行は許されるか

**許される（排除できない）。ただし安全である。**

`L_B` は、`S_E` の直前の store（`S*`）が書いた値を読める。その値は `L_B` の時点で既に置き換えられている（unlink 済み）かもしれない。write-read coherence が禁じるのは「`HB L_B` の store より前の値を読むこと」だけで、**`HB` の無い store（`S_E` より後の store）と競合して古い値を読むことは禁じない**。

重要なのは、その古い値の `T(O)` が Reader の `R` より大きいこと（§7）。unlink 済みを読むことはあっても、**reclaim 済みを読むことは無い**、というのが P1 の内容である。

### Q3. release sequence、atomic coherence、fence、別の同期経路がその実行を排除できるか

| 手段 | 排除できるか | 理由 |
|---|---|---|
| release sequence | **不要**（すでに成立） | 全ての store が release 以上の RMW か release store。release sequence の拡張を待たずに `sw` が取れる |
| atomic coherence（write-read） | **Q2 の「古い値を読む」は排除できない**。ただし「`HB` 済みの値より古い値を読む」ことは排除できる。P1 はこれを使う | [intro.races] の write-read coherence |
| fence | **無い**（`seq_cst` 0 件） | §4 |
| 別の同期経路（queue の seq / dequeuePos） | **Reader には効かない** | queue の同期は reclaimer と enqueue 側の間だけ |
| slot の depth/epoch | **片方向のみ** | A3→A4→A7 は成立するが、A4→A7 は「A7 が A4 を読んだ場合」だけ |

結論: **Reader の load 側は coherence で説明できる。reclaimer の走査側は説明できない。**

### Q4. retire epoch と reader epoch の関係から UAF が発生し得るか

**発生し得る。条件は §8.2。** ただし原因は「Reader が古い値を読むこと」ではなく「**reclaimer が Reader を見落とすこと**」である。この 2 つを混同すると判定を誤る。

### Q5. 特定アーキテクチャでの実測と portable な証明を混同していないか

混同していない（§9）。リポジトリにあるのは機能テスト（ctest 61/61）、TSan（3 ターゲットで警告 0）、12 秒のランタイム計測であり、いずれも**モデル水準の証明ではない**。TSan は data race を検出する道具で、「atomic の古い値読み」は検出対象にしない。

### 6.5 許容実行の一覧（何が許され、何が安全か）

| # | 実行 | モデル上 | 安全か | 根拠 |
|---|---|---|---|---|
| E1 | `L_B` が unlink 済みの値 O を読む | **許される** | 安全 | `T(O) > R`（§7）。reclaim 条件が偽 |
| E2 | `L_B` が「今の」値を読む | 許される | 安全 | `T(O) >= R+1`（§7 場合 (b)） |
| E3 | `L_E` が古い epoch 値（現在より小さい）を読む | **許される** | 安全（保守側） | `R` が小さいほど保護が強い。`minReaderEpoch <= R` |
| E4 | `L_B` が「`HB` する store より前」の値を読む | **許されない** | — | write-read coherence |
| E5 | Reader が読んだ object O の retire epoch が `R` 以下になる | **許されない** | — | §7 の場合 (a)(b)。O の retire は `S_E` の後で、同一スレッドの write-read coherence（W1）により `R+1` 以上 |
| E6 | reclaimer の走査が `depth == 0` を読む（Reader は区画内） | **許される** | **安全でない**（§8） | A4→A7 に同期辺が無い |
| E7 | ガード無しで `currentStateBits` を読む（Gate 1 以前の形） | 許される | 安全でない（修正済み） | `Processing.cpp:493-499` で遮断済み |

E1〜E3 が「Reader が古い値を読む」話で、E6 が「reclaimer が見落とす」話である。**現行コードが閉じているのは E1〜E5 までで、E6 が残る。**

---

## 7. P1（保護経路の安全）の証明

前提を明示する。

* **W1**: あるドメインの `currentStateBits` への store と `globalEpoch` の前進は、同一スレッドの program order で並ぶ（現行の主経路は Message Thread の setter と timer。`releaseResources` / `prepareToPlay` は Non-RT だが、setter と同一スレッドであることはコードで強制していない → §10 N2）。
* **W2**: `EQState` / `BandNode` は publication 後に変更されない（copy-on-write）。
* **W3**: `T(O)` は O を unpublish する exchange の**後**に `currentEpoch()` を読んで決まる（`Core.cpp:204,213`）。

**定理（P1）**: Reader が成功裏に入場し、`L_E` が R を返し、`L_B` が object O を返したならば、`T(O) > R` である。

証明:

1. `L_E` が R を読んだのは、`globalEpoch` への store（初期値 1 か、`S_E` の RMW が書いた R）による。`S_E` の場合、`S_E —sw→ L_E`（acquire load が release 側の RMW の値を読む）。
2. `S*` を「`S_E` より前に sb で並ぶ最後の `currentStateBits` store」とする（無ければ 3 は飛ばす）。`S* HB L_E HB L_B`。
3. write-read coherence により、`L_B` が読む値は `S*` の値か、それより後の modification order の値である。
   * **場合 (a)**: `L_B` が `S*` の値 O を読む。O を置き換える次の store `S_next` は、`S*` が「`S_E` より前の最後の store」であることから `S_E` より後に sb で並ぶ。よって O の retire（W3）は `S_E` の後に行われ、`currentEpoch()` は W1 の同一スレッド write-read coherence により `R+1` 以上を返す。すなわち `T(O) >= R+1 > R`。
   * **場合 (b)**: `L_B` が `S*` より後の store の値を読む。その store は `S_E` より後（同じ理由）。以後の retire は `R+1` 以上を返す。すなわち `T(O) >= R+1 > R`。
4. `L_E` が初期値 1 を読んだ場合（`S_E` が無い場合）は `R = 1`。`T(O) >= 1 = R` であり、`isOlder(T, min)` は `T < min <= R` とならないため安全（最小 epoch による保守側）。

reclaim の発火条件は `isOlder(T(O), minReaderEpoch)` = `T(O) < minReaderEpoch`（`DeferredDeletionQueue.h:174,308-311`）。Reader の slot が数えられていれば `minReaderEpoch <= R`。`T(O) > R >= minReaderEpoch` なので条件は偽。**O は解放されない。** ∎

**この証明が依存する外部条件**: W1（単一スレッド順序）と「slot が数えられること」。前者は §10 N2、後者が §8。

> 前稿 `gate1_reader_safety.md` §2 との差: 前稿は「`S_B HB L_E` だから coherence で新しい値を読む」で止めており、**retire epoch が必ず `R+1` 以上になる**という段（本稿の 3-a/3-b）を明示していない。この段が無いと「unlink 済みだが未 reclaim」を安全と説明できない。結論は同じで、根拠が 1 段増えた。

---

## 8. P2（reclaimer の走査）が未証明であること

### 8.1 何が足りないか

`getMinReaderEpoch()` は各 slot の `depth` を acquire load する（A7）。Reader は `depth` を acq_rel の RMW で +1 する（A4）。しかし **A4 と A7 の間に、A7 が A4 の値を読むことを強制する辺が無い**。`sw` は「読んだ場合」にしか成立しない。

C++ のメモリモデルには「いつかは最新値が見える」という保証が無い。`HB` の無い load は、modification order 上の任意の古い値を返してよい（write-read coherence は「`HB` する store より前を読むこと」だけを禁じる）。したがって、**Reader が区画内にいるのに走査が `depth == 0` を読む実行が、モデル上は許される**。

同じ穴は slot の再利用にもある。走査が「前の入場の `depth == 1`」を読み、`epoch` は今回の入場前の `kInactiveEpoch` を読むと、`EpochDomain.h:400` のスキップに当たり、その Reader は数えられない。

さらに、`getMinReaderEpoch()` は初期値を `currentEpoch()` にする（`EpochDomain.h:374`）。Reader を 1 つも観測できない走査では `minReaderEpoch` が「現在の epoch」そのものになり、`entry.epoch < 現在` の entry をすべて解放対象にする。見落としの代償が大きい構造である。

### 8.2 UAF の成立条件

すべて満たすと use-after-free になる。

1. Audio Reader が slot s を取得し、`R` を公開して `depth = 1` にする（A3, A4）。
2. Reader が `L_B` で object O を読む。P1 により `T(O) >= R+1`。
3. reclaimer の走査が slot s の `depth` を **0（古い値）と読む**。モデル上許される（8.1）。
4. 走査時点の `globalEpoch` が `R+2` 以上に進んでいる（1 回以上の追加 publication と flush）。すると `minReaderEpoch >= R+2 > T(O)`。
5. `reclaim()` が `isOlder(T(O), minReaderEpoch)` を真と判定し、`deleter` を実行する（`DeferredDeletionQueue.h:174,192`）。
6. Reader はまだ O を dereference している。→ UAF。

**最小の窓**: Reader の区画が「自分の publication より後にもう 1 回 epoch が進む」長さ続き、その間の走査で見落とされること。48 kHz / 512 sample なら 1 ブロック約 10.7 ms、timer は 100 ms 周期なので、区画が 1 ブロックなら窓は狭い。ただし**モデル上の排除はできない**。

### 8.3 除外できない理由（反例の骨格）

```
G: 1 → 2 (S_E1) → 3 (S_E2)
Reader: L_E=2, depth=1, L_B=O(T=3)
Scan  : load(depth)=0        ← A4 の前の値を読む（HB 無し。モデル上許される）
        min = currentEpoch = 3   （他に Reader 無し）
        isOlder(3, 3) = false → まだ解放しない
G: 4 (S_E3)
Scan  : load(depth)=0（再び見落とし）, min = 4
        isOlder(3, 4) = true → delete O   ← Reader はまだ O を使用中
```

この実行を禁止する規則が、現行コードには無い。

---

## 9. 実測と portable 証明の区別

| 種別 | 実績 | 何を言えるか | 何を言えないか |
|---|---|---|---|
| 機能テスト | ctest 61/61（`DomainTagTests` を含む） | 通常実行で壊れない | モデル上の許容実行の排除 |
| TSan | WSL clang 21 で 3 ターゲット警告 0（`MpscBoundedRingTests` / `ScopedEpochReaderStructuralTests` / `ISRRuntimeIdentityTests`、`doc/work113/20261112-1250_CIとTSan/report.md`） | data race が観測されない | **atomic の古い値読み**（race ではないので TSan の対象外） |
| ランタイム計測 | 12 秒 × 3 シナリオ、`stampMissing=0` ほか | 帰属と遅延の分布 | メモリモデルの性質 |
| x86_64 の性質 | `exchange` は locked RMW、`load(acquire)` は mov | 実装上は「1 publication 古い値」までしか観測しにくい。RMW 完了後の走査は同一 cache line の最新値を見る | モデル上の「無期限に古い値」を排除しない |

**x86_64 で観測されないことは、portable な証明ではない。** 本稿の NOT PROVED はその区別の上での判定である。逆に、8.2 の実行が実機で起きる証拠も無い（起きたとは主張しない）。

---

## 10. 残存証明義務

P2 を閉じるために必要なもの。**実装は行っていない。**

| # | 義務 | 現状 | 閉じ方の候補（参考・未実装） |
|---|---|---|---|
| N1 | **走査が区画内 Reader を観測すること**の保証 | 無い（§8） | seq_cst フェンスを Reader 側（depth++ の後・pointer load の前）と reclaimer 側（走査の後・解放の前）に置き、C++20 の fence-to-fence 同期で辺を作る。または publication に retire epoch を埋めて Reader 側で検証する |
| N2 | publication store と epoch 前進が同一スレッド順序であること（W1） | 現行の呼出し元では満たされるが、コードで強制していない | 呼出し元の制約を invariant として明記し、テストで固定する |
| N3 | retire epoch を unpublish の後に読むこと（W3） | 満たす（`Core.cpp:204,213`） | 将来の変更で崩さないよう回帰で固定 |
| N4 | publication 後の object 不変（W2） | 満たす（copy-on-write） | 同上 |
| N5 | slot 再利用時の「前の入場の depth」を読む実行の排除 | 無い（8.1 の後半） | N1 と同じ手段 |
| N6 | ガード無しの dereference 2 箇所（`AudioEngine.Snapshot.cpp:31`、`AudioEngine.h:1613`） | reclaim と同一スレッドであることに依存 | スレッド制約の明記、または `ScopedEpochReader` の付与 |
| N7 | `drainAllUnsafe` / `retireImmediateDuringShutdown` の epoch 迂回 | 外部 quiescence 前提（コメントあり） | 前提の証跡（audio 停止の観測点）と結び付ける |
| N8 | EQ 以外のドメイン（エンジン ISR / Convolver）への同判定の適用 | エンジンは同一実装、Convolver は別実装 | 別途監査 |

N1 が閉じない限り、P2 は証明できない。N2〜N8 は N1 と独立に残る。

---

## 11. 参考事項（実装案・分離）

**実装へ進まない。** 判断材料としてのみ置く。

1. **seq_cst フェンス案（N1 の直接解）**
   Reader: `slot.epoch` store → `depth` RMW → `atomic_thread_fence(seq_cst)` → `loadCurrentState`。
   Reclaimer: `getMinReaderEpoch()` → `atomic_thread_fence(seq_cst)` → `reclaim`。
   これで「走査が見落とし、かつ Reader が解放済み object を読む」実行を禁じられる、というのが hazard pointer で使われる論法。ただし C++20 の fence-to-fence 規則の条件（対象 atomic と読み値の対応）を満たす形に落とす必要があり、素朴に 2 本置くだけでは閉じない可能性がある。
2. **publication に epoch を埋める案**
   `currentStateBits` を (epoch, pointer) のタグ付き語にして、Reader は読んだ語の epoch で自分の保護 epoch を上書きする。retire 側も同じ語から epoch を取る。走査の可視性に依存しない。
3. **Reader 側の検証ループ案**
   `loadCurrentState` の後に `currentEpoch()` を再読し、変わっていれば区画を取り直す。**N1 は閉じない**（Reader の公開が見えない問題は残る）。P1 の追加防御としては有効。
4. **`getEQStateSnapshot()` の 2 箇所に guard を付ける案（N6）**
   低コスト。ただし N1 とは無関係。
5. **`seq_cst` の全面採用は不要**。閉じるべき辺は N1 の 1 本だけで、他の経路は acquire/release で足りている（§7）。

---

## 12. 証跡

`doc/work113/20261112-2040_Reader安全/evidence/`

| ファイル | 内容 |
|---|---|
| `source_identity.txt` | commit / ConvoPeq.md の識別・内容一致・対象ファイルの SHA-256（AGENTS.md を含む 19 行）・未コミット一覧 |
| `atomic_inventory.txt` | 製品コードの atomic 操作 1,106 行の棚卸し（ファイル・行・抜粋） |
| `snippets.md` | 監査対象コードの抜粋（20 箇所・行番号付き） |
| `memory_order_counts.txt` | memory_order 別の出現数 |

再現コマンド（読み取りのみ）:

```
python C:\Users\user\AppData\Local\Temp\opencode\snap_check.py     # ConvoPeq.md の内容一致
python C:\Users\user\AppData\Local\Temp\opencode\psr_evidence.py    # 本節の証跡を再生成
```

---

## 13. 停止条件の確認

* production source / test / CMake を変更していない（§1.2 に挙げた既存の未コミット変更は別作業であり、本監査は触れていない）。
* build / test / debugger / replay を実行していない。
* git の変更操作（add / commit / push / merge / reset / checkout）を行っていない。HEAD は `173e6995`、`origin/main` と一致したまま。
* AGENTS.md の既存未コミット変更に触れていない。
* 設計差分 v2.9 を変更していない（53,979 B / SHA-256 不変）。

**本稿の提出をもって停止する。** Gate 1 の承認、実装 GO、実行 GO は付与しない。次段階は Owner の判断を待つ。
