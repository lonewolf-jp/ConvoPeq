# P3-5-FPM-C2 — S7 Producer / Reader Temporal-Correlation Capture Preparation

## 1. Gate Identity

```text
Gate                         = P3-5-FPM-C2
Mode                         = read-only / runtime-capture preparation
C1                           = CLOSED
Target                       = S7 producer identity + S7 reader identity
C3                           = runtime capture; not started
CDB                          = NOT EXECUTED
M0 / M1 / M2                 = 0
Dr. Memory                   = 0
Build                        = NOT RUN
Production source change     = 0
Test change                  = 0
CMake change                 = 0
```

C2は、既存のRCA-6/RCA-7/RCA-9/RCA-10/RCA-11の証拠を使い、C3で実行するcapture procedureを確定する。CDBスクリプトの作成・実行、runtime observation、`waitForDrain()`、`publishEpoch()`、shutdown transition、およびproduction/test/CMakeの変更は行わない。

C1でterminal contract、`T3a`/`T3b`、`ShutdownComplete`、`Unknown`の意味は閉じた。C2で再検討する対象は、S7 head entryの生産者関数と、同時刻のEngine `EpochDomain` ReaderSlot状態だけである。

## 2. Frozen S7 Target

RCA-7/RCA-9から凍結するS7 targetは次のとおりである。

```text
S7 DQueue head:
  dequeuePos       = 4
  enqueuePos       = 5
  head sequence    = 5
  expectedSequence = 5
  head epoch       = 9
  globalEpoch      = 9
  minReaderEpoch   = 9
  epoch gate       = isOlder(9, 9) = false
  dequeue CAS      = not reached
```

これはentryのproducerまたはreaderを証明するデータではない。C2でT0/T1/T2を一つの実行時系列に結合するまで、`minReaderEpoch=9`からreader identityを導出してはならない。

## 3. Source / Binary Layout Reconciliation

### 3.1 Source authority

| contract | source |
|---|---|
| DQueue entry fields | `src/DeferredDeletionQueue.h:21-38` |
| DQueue enqueue and ticket CAS | `src/DeferredDeletionQueue.h:65-107` |
| DQueue reclaim head logic | `src/DeferredDeletionQueue.h:109-177` |
| DQueue queue layout | `src/DeferredDeletionQueue.h:262-272` |
| globalEpoch, enterReader, exitReader | `src/core/EpochDomain.h:117-189` |
| getMinReaderEpoch algorithm | `src/core/EpochDomain.h:214-248` |
| tryReclaim and enqueue delegation | `src/core/EpochDomain.h:385-410` |
| ReaderSlot layout | `src/core/EpochDomain.h:567-588` |
| router enqueue delegation | `src/audioengine/ISRRetireRouter.cpp:239-273` |
| S7 graceful/final drain boundary | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:280-376,608-745` |
| waitForDrain timeout marker | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:615-633` |

### 3.2 Current binary / PDB identity

| artifact | SHA-256 | size / timestamp |
|---|---|---|
| `build/Release/AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | 41,216,512 bytes; `2026-09-23T16:00:47.6227315Z` |
| `build/Release/AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | 59,355,136 bytes; `2026-09-22T12:50:18.5624865Z` |
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | 5,535,334 bytes; C1 identity |

C3開始時に同じhash、size、timestamp、module base、field offsetsを再確認する。1つでも一致しない場合は `STOP-C2-4` とする。

### 3.3 Static disassembly points

現行Releaseの静的disassemblyで次のRVAを固定した。VAは現在のimage base `0x141000000` を足した値である。

| label | RVA | VA | purpose |
|---|---:|---:|---|
| `S7_TIMEOUT` | `0x1fa80b3` | `0x141fa80b3` | `waitForDrain()` timeout decision; existing S7 marker |
| `WAIT_RETURN` | `0x1fa80b2` | `0x141fa80b2` | waitForDrain return boundary |
| `ENQ_ENTRY` | `0x1f9c550` | `0x141f9c550` | DQueue enqueue body; reads `enqueuePos`, captures arguments and caller return address |
| `ENQ_CAS_PRE` | `0x1f9c57d` | `0x141f9c57d` | immediately before `enqueuePos` CAS |
| `ENQ_CAS_POST` | `0x1f9c59a` | `0x141f9c59a` | successful CAS branch, before entry fields are written |
| `ENQ_WRITE_DONE` | `0x1f9c5da` | `0x141f9c5da` | entry fields and sequence are written; after-state capture |
| `RECLAIM_ENTRY` | `0x1f9cd00` | `0x141f9cd00` | DQueue reclaim body entry |
| `RECLAIM_SEQ_READ` | `0x1f9cd3c` | `0x141f9cd3c` | head sequence read |
| `RECLAIM_EPOCH_READ` | `0x1f9cd59` | `0x141f9cd59` | head entry epoch read |
| `RECLAIM_EPOCH_GATE` | `0x1f9cd66` | `0x141f9cd66` | epoch comparison gate |
| `RECLAIM_CAS_PRE` | `0x1f9cd72` | `0x141f9cd72` | dequeue CAS before |
| `RECLAIM_CAS_SUCCESS` | `0x1f9cd7b` | `0x141f9cd7b` | dequeue CAS success |
| `RECLAIM_RETURN` | `0x1f9ce04` | `0x141f9ce04` | reclaim return |
| `GETMIN_CALL_PRE` | `0x1f9cfe2` | `0x141f9cfe2` | `getMinReaderEpoch()` virtual call |
| `GETMIN_RETURN` | `0x1f9cfe5` | `0x141f9cfe5` | minReaderEpoch return in `rax` |
| `TRYRECLAIM_ENTRY` | `0x1f9cfb0` | `0x141f9cfb0` | Engine EpochDomain `tryReclaim()` entry |

`ENQ_ENTRY`から`ENQ_WRITE_DONE`までは、現行disassembly上で次のWin64 ABI mappingと一致する。

```text
rcx       = DQueue this
rdx       = entry.ptr
r8        = entry.deleter
r9        = entry.epoch
[rsp+0x28] = entry.type
[rsp+0x30] = entry.publicationSequenceId
[rsp+0x38] = entry.generation
[rsp]      = caller return address at ENQ_ENTRY
```

`[rsp]`のreturn-address mappingは、現行関数が先頭のpushを伴わずcaller shadow spaceを使う静的配置から導かれる。C3前に同じimageで再確認し、変化 있으면推測せず `STOP-C2-4` とする。

## 4. Address Calculation

S7 markerで停止した時点のregistersから、C3内部で次の固定offsetを使う。offsetsはRVAではなくruntime object addressのoffsetである。

```text
engineThis = rbx at S7_TIMEOUT
epochBase  = engineThis + 0x10b76c0
dqueueBase = epochBase  + 0x1440

globalEpochAddress = epochBase  + 0x18
readerBase         = epochBase  + 0x20
readerStride       = 0x50
readerCount        = 64

sequenceBase       = dqueueBase + 0x30000
enqueuePosAddress  = dqueueBase + 0x34000
dequeuePosAddress  = dqueueBase + 0x34040
entryStride        = 0x30
```

S7 target ticketは4であるため、C2で読むentryとsequenceの固定位置は次になる。

```text
entryIndex          = 4
sequenceAddress     = dqueueBase + 0x30000 + 4 * 4 = dqueueBase + 0x30010
entryAddress        = dqueueBase + 4 * 0x30        = dqueueBase + 0xc0
entry.ptr           = entryAddress + 0x00
entry.deleter       = entryAddress + 0x08
entry.epoch         = entryAddress + 0x10
entry.type          = entryAddress + 0x18
entry.pubSequenceId = entryAddress + 0x20
entry.generation    = entryAddress + 0x28
```

`DeletionEntry::objectBytes`は`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`で追加される任意fieldであり、現行entry stride `0x30`のcaptured entryでは読まない。strideまたはfield offsetsが変わった場合は推測で補完せず `STOP-C2-4` とする。

## 5. Capture Architecture

### 5.1 T0 — direct enqueue event

C3はprocess開始時から`ENQ_ENTRY`にsoftware breakpointを置く。S7 markerで待ち始めてからenqueue watchpointを後で置く方法は採用しない。S7 headのentryは既にenqueue済みであるため、それではT0を捕捉できない。

`ENQ_ENTRY`では、単純な条件とmemory readだけで候補を絞る。

```text
candidate condition:
  entry epoch (r9)       = 9
  enqueuePos before     = 4
  DQueue address        = S7 dqueueBase after post-run join
```

候補が複数でもexceptionなく全て記録する。事後joinでS7の`dqueueBase`、ticket 4、sequence 5、entry epoch 9に一致するものだけを選出する。条件に一致する候補が0件、または1件に確定できない場合は `STOP-C2-1` または `STOP-C2-5` とする。

T0で停止したprocessの全threadを停止したまま、次の値を同じbreakpoint command内で読む。memory readの合間に`gc`を挟まない。

```text
event label
thread id / current thread
caller return address = poi(@rsp)
caller symbol resolution = ln(poi(@rsp)) and stack trace
DQueue address
EpochDomain address
globalEpoch before
enqueuePos before
sequence[4] before
entry pointer argument
entry deleter argument
entry epoch argument
entry type argument
entry publicationSequenceId argument
entry generation argument
all ReaderSlot fields
```

`ENQ_CAS_PRE`ではCAS直前の`enqueuePos`とsequenceを再読する。`ENQ_CAS_POST`ではCAS成功直後の`enqueuePos`を読む。`ENQ_WRITE_DONE`では下列を同じ停止状態で読む。

```text
enqueuePos after       = 5
sequence[4] after      = 5
entry.ptr
entry.deleter
entry.epoch            = 9
entry.type
entry.publicationSequenceId
entry.generation
globalEpoch after
```

`ENQ_WRITE_DONE`のreadでentry fieldsとsequence publish orderingを確認できない場合は、enqueue event itselfを採用せず `STOP-C2-1` とする。

### 5.2 T1 — first post-S7 tryReclaim

S7 markerのcommandで`S7_SEEN`とS7 object basesを保存する。`TRYRECLAIM_ENTRY`、`GETMIN_CALL_PRE`、`GETMIN_RETURN`、`RECLAIM_ENTRY`以下は、S7 object conditionと`S7_SEEN`で絞り、S7後の最初のtryReclaimだけをT1とする。

T1は次の二点を読む。

1. `GETMIN_CALL_PRE`直前のreader snapshotとqueue snapshot。
2. `RECLAIM_EPOCH_READ`でhead entry.Fieldsを確定した直後のreader snapshotとqueue snapshot。

`GETMIN_RETURN`では`rax`の`minReaderEpoch`を直接記録する。`minReaderEpoch=9`という結果だけではreader slotを補完しない。

T1の必須fieldは次のとおりである。

```text
globalEpoch
enqueuePos
dequeuePos
head sequence
expectedSequence
head entry pointer
head deleter
head epoch
head type
head publicationSequenceId
head generation
minReaderEpoch from GETMIN_RETURN
ReaderSlot[0..63] full fields
```

`RECLAIM_EPOCH_READ`では、現行disassemblyのregister mappingを利用する。

```text
rbp = current dequeuePos
r15 = dqueueBase + 0x30000 + 4 * (rbp & 0xfff)
rsi = dqueueBase + 0x30 * (rbp & 0xfff)
r13 = minReaderEpoch passed to reclaim
```

S7 targetでは`rbp=4`、`r15`はsequence 5のaddress、`rsi`はentry 4のaddressになる。実際のreadでは`rbp`、`r15`、`rsi`のregister値もlogに残し、offset式を推測で補わない。

### 5.3 T2 — final wait / timeout

`S7_TIMEOUT`をT2 timeout decisionとして使い、`WAIT_RETURN`をreturn boundaryとして読む。T2ではT0/T1と同じS7 `dqueueBase`と`epochBase`から、次を読む。

```text
globalEpoch
enqueuePos
dequeuePos
current head relation
ReaderSlot[0..63] full fields
waitForDrain return/timeout marker
```

T2のreader snapshotがT1のreadと同一process、同一S7 queue、同一epoch keyで結びつかなければ、時間系列の関連性として採用しない。

### 5.4 ReaderSlot literal-read method

RCA-9の失敗原因であるCDB `.for` address expressionをC2で再使用しない。C3のscriptは、64個のslotをhost側で事前展開し、各slotについてliteral addressの単純なread commandを実行する。

```text
slotBase(i) = epochBase + 0x20 + i * 0x50

slotBase(i)+0x00 = epoch                 dq
slotBase(i)+0x08 = depth                 dd
slotBase(i)+0x10 = enterCount             dq
slotBase(i)+0x18 = residencyStartUs       dq
slotBase(i)+0x20 = ownerThreadId          dq
slotBase(i)+0x28 = ownerTag[32]           db
slotBase(i)+0x48 = quarantineFlags        db
```

実際のC3 scriptでは、`i=0`から`i=63`までの各groupを事前展開したliteral commandとして持つ。CDB上の`.for`、`.while`、複雑なaddress expression、slot indexの算術式は禁止する。

```text
C2_T0_SLOT_00
  dq <literal slot 0 address> L1
  dd <literal slot 0 address+0x08> L1
  dq <literal slot 0 address+0x10> L1
  dq <literal slot 0 address+0x18> L1
  dq <literal slot 0 address+0x20> L1
  db <literal slot 0 address+0x28> L20
  db <literal slot 0 address+0x48> L1

C2_T0_SLOT_01
  dq <literal slot 1 address> L1
  ...
```

C3開始前にscriptをreviewし、64 groupsが0から63まで欠落なく重複なく存在することを確認する。`ReaderSlot`の値が一つでも取得できない場合は推測で補わず `STOP-C2-3` とする。

### 5.5 enterReader / exitReader optional events

T0/T1/T2のfull slot snapshotがC2の必須条件である。`enterReader()`/`exitReader()`のevent logは追加観測であり、C2 completion条件には含めない。

追加する場合は、現行sourceと同一imageの静的symbol/line mappingを再確認してから、C3 scriptにcode breakpointを追加する。production/test codeへのhook、getter追加、source instrumentationは禁止する。C2時点では、slot epoch/depth/flag/owner fieldのT0/T1/T2差分だけでCase C/Dを判定できる。

## 6. Required Timeline

C3のlogは、少なくとも次のevent keyでjoinする。

```text
[T0] DQueue enqueue
     caller return address = ?
     caller function = ?
     queue address = ?
     entry pointer = ?
     entry deleter = ?
     entry epoch = ?
     entry type/metadata = ?
     enqueuePos before = ?
     enqueuePos after = ?
     globalEpoch before/after = ?
     ReaderSlot[0..63] = ...

[T1] first post-S7 tryReclaim
     globalEpoch = ?
     enqueuePos = ?
     dequeuePos = ?
     expectedSequence = ?
     head.sequence = ?
     head.epoch = ?
     minReaderEpoch = ?
     ReaderSlot[0..63] = ...

[T2] final wait / timeout
     globalEpoch = ?
     enqueuePos = ?
     dequeuePos = ?
     reader state = ?
     ReaderSlot[0..63] = ...
```

必須join keyは次である。

```text
same process execution
same S7 engineThis
same S7 dqueueBase
same S7 epochBase
ticket/enqueue transition = 4 -> 5
head sequence = 5
head epoch = 9
```

このkeyが一致しないT0/T1/T2を、異なるrun、異なるqueue、異なるentry、または後続shutdown entryの観測として扱う。

## 7. Case Separation

Case判定はT0/T1/T2のfull slot snapshotとqueue identityが揃った後だけ行う。

| Case | required evidence | classification |
|---|---|---|
| A | T0/T1/T2で `depth>0`、非quarantine、epoch!=inactive/reserved のeligible readerが無く、minReaderEpoch=9がcurrentEpochから残る | no eligible reader at 9 |
| B | 同一slotがT0とT1で `epoch=9`、`depth>0`、quarantine flagなしで安定し、ownerThreadId/ownerTagも一致 | reader at epoch 9 |
| C | T0/T1/T2の境界でepoch/depth/flagが変化し、snapshotだけではlifecycle途中か安定状態を区別できない | lifecycle-transition observation |
| D | T0のenqueue後にglobalEpochまたはread-side stateが変化し、S7 epoch-equal blockがその変化をまたぐ | post-enqueue state change |

主判定のprecedenceは、`STOP > C > D > B > A` とする。CとDはraw factsを併記し、単独でA/Bを否定しない。`minReaderEpoch=9`だけではAともBとも判定しない。

## 8. STOP Matrix

```text
STOP-C2-1  enqueue callerを直接取得できない
STOP-C2-2  S7 enqueueと別時刻のreader stateしか取得できない
STOP-C2-3  ReaderSlot runtime valueが取得できない
STOP-C2-4  binary/source layout driftが再発
STOP-C2-5  S7 headと別のDQueue entryを観測している可能性
STOP-C2-6  captureのためproduction/test source modificationが必要
STOP-C2-7  CDB expression failureを回避するため推測値が必要
```

C2 preparationではruntime captureをしていないため、現時点の状態は `ARMED` であり、C3の実captureで各条件を評価する。推測値、欠落field、別queueのeventを補って続行してはならない。

## 9. C2 Completion Matrix

```text
C2_CAPTURE_METHOD          = READY
S7_ENQUEUE_CAPTURE_POINT   = DEFINED
S7_READER_CAPTURE_METHOD   = DEFINED
CDB_REGISTER_DEPENDENCY    = MINIMAL
SOURCE/BINARY_LAYOUT       = RECONCILED
NO_SOURCE_CHANGE           = PROVEN
NO_TEST_CHANGE             = PROVEN
NO_CMAKE_CHANGE            = PROVEN
NO_BUILD                    = PROVEN
CDB                         = NOT_EXECUTED
M0/M1/M2                    = 0
DrMemory                    = 0
```

C2はcapture procedureの確定で完了する。producer/reader identityそのもの被判定了らC2はclosedではない。C3 runtime captureのentry conditionは、C3で同じbinary/source identityと全STOP条件を再確認した後に限る。

## 10. Evidence Index

| evidence | location |
|---|---|
| C1 terminal-contract closure | `doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C1_ShutdownDrain_terminal_contract_reconciliation.md` |
| S7 head and epoch gate | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-7_DQueue_head_block_exact.md` |
| DQueue entry field/provenance boundary | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-6_DQueue5_entry_provenance.md` |
| failed `.for` expression and STOP evidence | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-9_EpochProvenance_reader_lifecycle_exact.md` |
| reader lifecycle and fixed-slot boundary | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-10_ReaderLifecycle_epoch_provenance_static_closure.md` |
| producer/liveness temporal boundary | `doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-RCA-11_ShutdownDrain_epoch_liveness_static_closure.md` |
| DQueue source layout | `src/DeferredDeletionQueue.h:21-107,262-272` |
| ReaderSlot source layout | `src/core/EpochDomain.h:117-248,567-588` |
| current binary static disassembly | `llvm-objdump` read-only over `AudioEngineHarness.exe`; RVAs in section 3.3 |
| existing failed script | `C:\Users\user\AppData\Local\Temp\opencode\rca9-cdb-capture.txt`; read only, not modified or rerun |

## 11. C3 Handoff

C3はC2 reportを読み、次の順序で開始する。

```text
1. binary/source identityとstatic offsetsを再確認
2. C2のliteral 64-slot command群を生成・レビュー
3. ENQ_ENTRYからT0を捕捉
4. S7 marker後にT1/T2を捕捉
5. queue/ticket/sequence/epochでT0-T1-T2をjoin
6. Case A/B/C/Dを分離
7. STOP条件に該当しない場合だけproducer/reader attributionを確定
```

C3のruntime capture commands、CDB invocation、M0/M1/M2 authorizationは、C2 reportのreview後に別途ゲートとして発行する。
