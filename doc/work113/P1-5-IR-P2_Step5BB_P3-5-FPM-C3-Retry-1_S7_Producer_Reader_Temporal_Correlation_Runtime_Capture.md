# P3-5-FPM-C3-Retry-1 — S7 Producer / Reader Temporal Correlation Runtime Capture

## 1. Gate and final result

```text
Gate                         = P3-5-FPM-C3-Retry-1
Mode                         = runtime capture / read-only
Stimulus                     = AudioEngineHarness.exe --fpm-m0
Approved retry count         = 1
Observed retry count         = 1
CDB exit                     = 0
Source/test/CMake change     = 0
Build                        = 0
M1/M2                        = 0
Dr. Memory                   = 0
T0                           = CAPTURED
T0 CAS                       = CAPTURED
S7                           = CAPTURED
T1                           = NOT CAPTURED
T2                           = CAPTURED
T0/S7 join                   = PROVEN
T0/S7/T1/T2 complete join    = NOT PROVEN
Case A/B/C/D                 = NOT PROVEN
S7_PRODUCER                  = PARTIALLY RESOLVED
S7_READER                    = UNRESOLVED
Final verdict                = C3-Retry-1 = STOPPED (STOP-6 / STOP-7 / STOP-8)
IMPLEMENTATION               = FORBIDDEN
```

CDB pseudo-register/expression errorは0件で、T0/CAS/S7/T2の直接runtime evidenceは取得できた。しかしS7後のT1対象は8つのT1 breakpointすべて0 hitで、`minReaderEpoch`、T1 ReaderSlot snapshot、reclaim head identityを取得できなかった。またT0 caller function symbolは未取得で、保存されたstackは`dq @rsp L1`のみだった。したがってretry全体を成功扱いにはせず、STOP-6、STOP-7、STOP-8を適用したpartial-success/STOP verdictとした。

## 2. Preflight identity

実行直前に次のidentityを再確認した。全件一致した場合にのみ1回実行した。

| artifact | SHA-256 | size | result |
|---|---|---:|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | 5,535,334 | PASS |
| `AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | 41,216,512 | PASS |
| `AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | 59,355,136 | PASS |
| retry script | `AFCC0017DA95D36F23D497123547AEE0B0B8D7572F43231835A25AE532EF0F18` | 11,150 | PASS |
| CDB binary | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | 178,016 | PASS |

CDB versionは`10.0.29617.1000 (WinBuild.160101.0800)`。script fingerprintは`bp=14 / ReaderSlot L80 commands=384 / .for=0 / .while=0 / custom register=0 / obsolete register=0`であった。process residue beforeは0件。

## 3. Invocation and evidence files

```text
C:\VSC_Project\ConvoPeq\tmp\cdb.exe
  -logo C:\Users\user\AppData\Local\Temp\opencode\c3-retry-1-cdb.log
  -y C:\VSC_Project\ConvoPeq\build\Release
  -cf C:\VSC_Project\ConvoPeq\doc\work113\P1-5-IR-P2_Step5BB_P3-5-FPM-C3-RP_retry_not_executed.cdb
  C:\VSC_Project\ConvoPeq\build\Release\AudioEngineHarness.exe --fpm-m0
```

```text
log       = C:\Users\user\AppData\Local\Temp\opencode\c3-retry-1-cdb.log
SHA-256   = 1F5109C6113F718AD5E50704056D108B5184042BEFA93AB0EEA3967E1B98BBE2
bytes     = 178875
lines     = 1989
exit      = 0
CDB residue after = 0
Harness residue after = 0
```

通常`[FPM]`出力はattribution evidenceとして採用していない。CDB commandの初回`break instruction exception`はWindows user-mode debuggerのinitial breakpointであり、CDB式errorではない。

CDB startup warningは以下2種で、runtime式・register errorではない。

```text
Unable to verify checksum for AudioEngineHarness.exe
Unable to add extension DLL: ntsdexts / uext / exts
```

identityは外部`Get-FileHash`で完全一致済みである。

## 4. Event marker matrix

| event | runtime marker | count | evidence lines | result |
|---|---|---:|---|---|
| T0 entry | `C3_T0_ENTRY` | 1 | 292 | PASS |
| T0 CAS pre | `C3_T0_CAS_PRE` | 1 | 828 | PASS |
| T0 CAS post | `C3_T0_CAS_POST` | 1 | 841 | PASS |
| T0 write done | `C3_T0_WRITE_DONE` | 1 | 854 | PASS |
| S7 marker | `C3_S7_MARKER` | 1 | 872 | PASS |
| T1 tryReclaim | `C3_T1_TRYRECLAIM` | 0 | - | MISS |
| T1 getMin pre | `C3_T1_GETMIN_CALL_PRE` | 0 | - | MISS |
| T1 getMin return | `C3_T1_GETMIN_RETURN` | 0 | - | MISS |
| T1 reclaim epoch | `C3_T1_RECLAIM_EPOCH_READ` | 0 | - | MISS |
| T2 wait return | `C3_T2_WAIT_RETURN` | 1 | 1405 | PASS |
| script end | `C3RP_RETRY_SCRIPT_END` | 1 | 1987 | PASS |

CDB expression error count:

```text
Bad register          = 0
Couldn't resolve      = 0
Unable to resolve     = 0
Syntax error          = 0
Expression syntax err = 0
Access violation      = 0
```

したがってSTOP-9はtriggerされていない。

## 5. T0 direct evidence

Log lines 292-314:

```text
caller return address = 0x00007ff759a8c735
relative RVA           = AudioEngineHarness+0x1f9c735
DQueue this ($t2)     = 0x000001764737ab80
epochBase ($t1)       = 0x0000017647379740
EpochDomain ($t3)     = 0x0000017647379750
entry ptr (rdx)       = 0x0000017655a01080
entry deleter (r8)    = AudioEngineHarness+0x1f9a79c0
entry epoch (r9)      = 9
entry type            = 0
publicationSequenceId = 0
generation            = 0
globalEpoch           = 9
enqueuePos            = 4
sequence[4]           = 4
```

caller return address `0x141f9c735` はDQueue enqueue関数を呼ぶtyped wrapperのreturn addressだった。binary disassemblyでは`0x141f9c710`から`0x141f9c735`へ`:

```text
rcx += 0x1430
stack metadata = 0
type byte = caller stack +0x70
call DeferredDeletionQueue.enqueue@0x141f9c550
```

`ln poi(@rsp)`と直接caller stack unwindはlog上で関数名を返さなかった。したがってproducer関数名そのものは確定せず、`S7_PRODUCER = DSPCore destroy-deleter candidate`としてのみ記録する。entry ptr/deleter/epochのruntime identityはprovenだが、呼び出し関数名と完全なstack traceはunresolvedである。保存されたstack-related outputは`@rsp`の1 qwordと`@rsp+0x28/+0x30`のentry metadataであり、full stackではない。

T0 ReaderSlot blockはlines 315-827、各64 slotの`L80` windowを出力し、再構築結果は`8192 bytes (64 x 0x80)`、64 slot recordを完全取得。block SHA-256はS7/T2と同一の`94E0E1861BF2F8BEE7B798741BF68FC919E91FDF9BE35377E46046F36189805B`。depth>0またはenterCount>0のslotは0件。T0からS7へのactive reader state変化はない。

## 6. T0 CAS and write ordering

| phase | log line | enqueuePos | sequence[4] | entry ptr/deleter/epoch | result |
|---|---:|---:|---:|---|---|
| T0 entry | 312-313 | 4 | 4 | epoch argument 9 | PASS |
| CAS pre | 828-840 | 4 | 4 | `rcx=5` expected CAS value | PASS |
| CAS post | 841-853 | 5 | 4 | ticket 4 still being written | PASS |
| write done | 854-871 | 5 | 5 | ptr=`0x17655a01080`, deleter=`...9a979c0`, epoch=9, type/pubseq/generation=0 | PASS |

T0 relationは厳密に`enqueuePos 4 -> 5`、sequence `4 -> 5`、ticket/entry 4、entry epoch 9であった。STOP-4はtriggerされていない。

## 7. S7 direct evidence

Log lines 872-891:

```text
engineThis             = 0x00000176462c2080
epochBase              = engineThis + 0x10b76c0
dqueueBase             = epochBase + 0x1440
globalEpoch            = 9
enqueuePos             = 5
dequeuePos             = 4
head sequence          = 5
head entry ptr         = 0x0000017655a01080
head entry deleter     = AudioEngineHarness+0x1f9a79c0
head entry epoch       = 9
head type              = 0
publicationSequenceId  = 0
generation             = 0
```

S7 ReaderSlot blockはlines 892-1404、64 slots完全取得。block hashはT0と同一で、eligible reader 0件。

## 8. T0 / S7 join

| join key | T0 | S7 | result |
|---|---|---|---|
| process | same CDB process | same CDB process | PASS |
| epochBase | `0x17647379740` | engine-derived same | PASS |
| dqueueBase | `0x1764737ab80` | engine-derived same | PASS |
| ticket | 4 | enqueuePos 5 / slot 4 | PASS |
| sequence | 4 before -> 5 after | head sequence 5 | PASS |
| entry ptr | `0x17655a01080` | same | PASS |
| entry deleter | `AudioEngineHarness+0x1f9a79c0` | same | PASS |
| entry epoch | 9 | 9 | PASS |

このためT0のDSPCore destroy-deleter entryとS7 head entryの同一execution joinはprovenである。producer関数名はcaller symbol欠落のためpartial unresolvedとしたが、entry identityとqueue transitionは直接runtime evidenceで閉じた。STOP-5およびSTOP-10はtriggerされていない。

## 9. T1 direct evidence

T1 breakpoint addressとruntime stop/hit 수는以下0件である。

| breakpoint | RVA | observed stop | marker |
|---|---:|---:|---|
| Engine tryReclaim | `0x1f9cfb0` | 0 | 0 |
| getMin call pre | `0x1f9cfe2` | 0 | 0 |
| getMin return | `0x1f9cfe5` | 0 | 0 |
| DQueue reclaim entry | `0x1f9cd00` | 0 | 0 |
| reclaim epoch read | `0x1f9cd59` | 0 | 0 |
| epoch gate | `0x1f9cd66` | 0 | 0 |
| dequeue CAS pre | `0x1f9cd72` | 0 | 0 |
| reclaim return | `0x1f9ce04` | 0 | 0 |

したがって以下は取得不能である。

```text
T1 minReaderEpoch              = NOT_CAPTURED
T1 ReaderSlot[0..63]           = NOT_CAPTURED
T1 reclaim head identity       = NOT_CAPTURED
T1 minReader vs head same call = NOT_PROVEN
```

CDB errorではなく、指定T1 pathがrun中に観測されなかった。T1 ReaderSlot全件欠落でSTOP-6、`minReaderEpoch`欠落でSTOP-7、完全timeline欠落でSTOP-8をtriggerした。script修正・追加run・M1/M2起動は行っていない。

## 10. T2 direct evidence

Log lines 1405-1424:

```text
globalEpoch     = 9
enqueuePos      = 5
dequeuePos      = 4
head sequence   = 5
head entry ptr  = 0x0000017655a01080
head deleter    = AudioEngineHarness+0x1f9a79c0
head epoch      = 9
head type/pubseq/generation = 0
```

T2 ReaderSlot blockはlines 1425-1937、64 slots完全取得。block hashはT0/S7と同一で、eligible reader 0件。T0 -> S7 -> T2ではglobalEpoch、enqueuePos、dequeuePos、head entry/sequence/epoch、ReaderSlot bytesが不変である。

## 11. Reader attribution and Case A-D

T0、S7、T2はいずれも64-slot完全snapshotを取得し、3 blockのbytes/hashが同一で、eligible readerは0件だった。これは「観測した3時点でactive/eligible readerはいない」というraw factである。

しかしSTOP precedenceに従いCase Aを確定してはならない。T1 `getMinReaderEpoch`とreclaim call localsが存在しないため、

```text
Case A = NOT_PROVEN
Case B = NOT_PROVEN / no B evidence
Case C = NOT_PROVEN
Case D = NOT_PROVEN
S7_READER = UNRESOLVED
```

`minReaderEpoch=9`は本次logのT1 evidenceとして得られていない。RCA-7や既存ログから補完しない。

## 12. STOP matrix

| stop | result | reason |
|---|---|---|
| STOP-1 identity/layout drift | NOT TRIGGERED | hashes一致 |
| STOP-2 T0 candidate 0 | NOT TRIGGERED | T0 1件 |
| STOP-3 T0 multiple/unresolved | NOT TRIGGERED | candidate 1件、entry identity join成立 |
| STOP-4 CAS/write mismatch | NOT TRIGGERED | 4 -> 5, sequence 4 -> 5 |
| STOP-5 T0/S7 join failure | NOT TRIGGERED | object/entry join proven |
| STOP-6 ReaderSlot partial failure | TRIGGERED | T0/S7/T2 are 64/64, but T1 ReaderSlot 0..63 all missing |
| STOP-7 minReaderEpoch missing | TRIGGERED | T1 8 breakpoints all 0 hit |
| STOP-8 T0/T1/T2 join failure | TRIGGERED | T1 absent |
| STOP-9 CDB expression/register error | NOT TRIGGERED | errors 0 |
| STOP-10 separate S7 entry | NOT TRIGGERED | ptr/deleter/epoch/sequence match |
| STOP-11 source change needed | NOT TRIGGERED | source change 0 |

## 13. Process and change boundary

```text
M0 debuggees launched  = 1
M1 runs                = 0
M2 runs                = 0
Dr.Memory              = 0
Build                  = 0
Production source      = 0
Test source            = 0
CMake                  = 0
retry script edits     = 0
CDB process residue    = 0
Harness residue        = 0
```

既存のworktree変更は保持し、本gateでrollback/resetしていない。

## 14. Attribution boundary and next authorization

```text
T0 ENTRY IDENTITY      = PROVEN
T0 CAS/WRITE ORDER     = PROVEN
T0/S7 SAME ENTRY       = PROVEN
T0/S7/T2 READER STATE  = PROVEN AS IDENTICAL EMPTY/INACTIVE SNAPSHOTS
T1 RECLAIM PROVENANCE  = NOT CAPTURED
MIN_READER_EPOCH_9     = NOT OBSERVED IN THIS RETRY
S7_PRODUCER            = DSPCore destroy-deleter candidate proven; caller function unresolved
S7_READER              = UNRESOLVED
CAUSE                  = UNRESOLVED
IMPLEMENTATION         = FORBIDDEN
```

次gateをauthorizationする場合は、本retryの成功部分とSTOP-7/8failureを利用し、未実行scriptを別gateでそのまま再実行してはならない。T1 filter/capture pointの観測不能理由をread-onlyで調査する段階から開始する必要がある。

## 15. Evidence index

- preserved retry log: `doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C3-Retry-1_cdb.log`
- original retry log: `C:\Users\user\AppData\Local\Temp\opencode\c3-retry-1-cdb.log`
- retry log SHA-256: `1F5109C6113F718AD5E50704056D108B5184042BEFA93AB0EEA3967E1B98BBE2`
- retry script: `doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C3-RP_retry_not_executed.cdb`
- retry script SHA-256: `AFCC0017DA95D36F23D497123547AEE0B0B8D7572F43231835A25AE532EF0F18`
- C2 method: `doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C2_S7_Producer_Reader_Temporal_Correlation_Capture_Preparation.md`
- C3-RP gate: `doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C3-RP_CDB_Syntax_Retry_Preparation.md`

```text
C3-Retry-1 = STOPPED (T0/S7/T2 captured; T1 missing; STOP-6 / STOP-7 / STOP-8)
IMPLEMENTATION = FORBIDDEN
```
