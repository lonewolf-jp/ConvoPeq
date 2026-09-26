# P3-5-FPM-C3-RP — CDB Syntax Retry Preparation

## 1. Gate identity and final verdict

```text
Gate                         = P3-5-FPM-C3-RP
Mode                         = read-only / debugger-tool validation
C3                           = STOPPED; preserved and not rerun
CDB version                  = 10.0.29617.1000 (WinBuild.160101.0800)
CDB probe target             = cmd.exe /d /c exit
ConvoPeqHarness              = NOT LAUNCHED
C3 runtime capture           = NOT EXECUTED
Production source change     = 0
Test source change           = 0
CMake change                 = 0
Build                        = 0
FPM M0/M1/M2                 = 0
Dr. Memory                   = 0
IMPLEMENTATION               = FORBIDDEN
Final verdict                = C3-RP = READY FOR ONE RETRY
```

READYはretry実行の承認条件を満たす意味であり、C3の実測完了、producer/reader attribution、またはimplementation authorizationを意味しない。実作業は、別途承認されたC3 retry 1回だけである。

## 2. RP-1 — C3 evidence and identity freeze

C3失敗の入力は変更せず、現行ファイルから再hashした。

| input | SHA-256 | bytes | result |
|---|---|---:|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | 5,535,334 | PASS |
| `build/Release/AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | 41,216,512 | PASS |
| `build/Release/AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | 59,355,136 | PASS |
| C3 script | `5902AAB8FA8FE68208CB2B2A4D1F2EB708AD29AA37AFA727D749D05720DBAE88` | 10,857 | PASS |
| C3 CDB log | `D6D160AA3788C247B79608EDEEF8DB5B129AB682ADE68EB4437D7B399BFCB7BA` | 32,443 | PASS |
| C3 report | `B91669448C83B356E096D1E75E6030BD76769E63914D12EB8B7CFABA8152B279` | 9,098 | PASS |
| CDB binary | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | 178,016 | PASS |

現行C2/C3 baselineと一致した。CDBの静的layoutも再確認した。

| point | frozen RVA | static disassembly |
|---|---:|---|
| `T0_ENTRY` / `ENQ_ENTRY` | `0x1f9c550` | PASS |
| `T0_CAS_PRE` | `0x1f9c57d` | PASS |
| `T0_CAS_POST` | `0x1f9c59a` | PASS |
| `T0_WRITE_DONE` | `0x1f9c5da` | PASS |
| `S7_TIMEOUT` | `0x1fa80b3` | PASS |
| `T1_GETMIN_CALL_PRE` | `0x1f9cfe2` | PASS |
| `T1_GETMIN_RETURN` | `0x1f9cfe5` | PASS |
| `T1_RECLAIM_EPOCH_READ` | `0x1f9cd59` | PASS |
| `T2_WAIT_RETURN` | `0x1fa80b2` | PASS |

## 3. C3 failure evidence

凍結C3 logの実際の失敗は次のとおりである。

```text
Bad register error in 'r $s7 = 0'
Bad register error in 'r $t1done = 0'
Bad register error in 'r $t2done = 0'
Bad register error in 'r $s7e = 0'
Bad register error in 'r $s7q = 0'
Bad register error in 'r $s7rbx = 0'
Couldn't resolve error at '$s7 == 1) { .if (@rcx == $s7rbx) { ...'
```

これは値の取得失敗ではなく、CDBのpseudo-register parserがcustom nameをresolveできなかった失敗である。T0/S7/T1/T2 event labelはいずれもC3 logに観測されず、C3 process exit markerだけが残った。したがってC3のruntime値、`minReaderEpoch=9`、既存RCA-7のreader/stateはretryのevidenceに補完していない。

## 4. RP-2 — pseudo-register minimal probe

CDB公式仕様ではuser-defined pseudo-registerは`$t0`〜`$t19`であり、`r` commandでassignmentし、expressionではMASM/C++に応じて`@$tN`を使う。今回の現行CDBに対して、targetをConvoPeqにせず `cmd.exe /d /c exit` のみをdebuggeeとして起動した。

ConvoPeq processの前後値はともに0件である。

| probe | assignment | readback | equality `.if` | error | verdict |
|---|---:|---:|---:|---:|---|
| `$t0` | 1 | 1 | 1 | 0 | PASS |
| `$t1` | 1 | 1 | 1 | 0 | PASS |
| `$t2` | 1 | 1 | 1 | 0 | PASS |
| `$t3` | 1 | 1 | 1 | 0 | PASS |
| `$t4` | 1 | 1 | 1 | 0 | PASS |
| `$t5` | 1 | 1 | 1 | 0 | PASS |
| `$t6` | 1 | 1 | 1 | 0 | PASS |
| `$t7` | 1 | 1 | 1 | 0 | PASS |
| `$t8` | 1 | 1 | 1 | 0 | PASS |
| `$t9` | 1 | 1 | 1 | 0 | PASS |
| `$t10`〜`$t19` | 10/10 | 10/10 | 10/10 | 0 | PASS |
| **合計** | **20/20** | **20/20** | **20/20** | **0** | **PASS** |

`Bad register` = 0、`Couldn't resolve` = 0、readback mismatch = 0、conditional failure = 0。CDB version/command syntaxは不明でなかった。

なお、targetを一切指定しない`cdb.exe -cf`はCDBがkernel debuggerとして`\\.\com1`を開こうとして`Win32 error 2`で終了した。この診断はpseudo-register判定に採用せず、ConvoPeqを起動しないbenign target probeに切り替えた。

## 5. RP-3 — state machine separation

retry stateは独自named pseudo-registerを使わず、CDBが認識する`$t0`〜`$t5`だけに分離した。

| register | role | assignment point |
|---|---|---|
| `$t0` | semantic state | `0=not joined`, `1=T0 captured`, `2=S7 captured`, `3=T1 active`, `4=T2 captured` |
| `$t1` | `epochBase` | T0 candidateまたはS7 markerで固定 |
| `$t2` | `dqueueBase` | T0 candidateまたはS7 markerで固定 |
| `$t3` | EpochDomain pointer | `epochBase + 0x10` |
| `$t4` | `engineThis` | S7 markerで固定 |
| `$t5` | S7 reached latch | `0=S7未到達`, `1=S7到達` |

T0の候補評価では`$t4`を確定せず0に留め、S7 markerで`engineThis`を確定する。T1以降は`$t3`のEpochDomain pointerをidentity filterとして使用する。`$t5`はS7到達後に後続enqueueをT0候補として誤取得しないlatchであり、S7前の候補は全件記録して事後joinする。T2はT1到達に依存せず、S7到達後かつ未取得の最初の`WAIT_RETURN`を捕捉する。

## 6. RP-4 — `.if` conditional probe

RP-4はtarget memory/register validationと分離し、pseudo-registerのみをcondition operandにした。

```text
r $tN = value
r $tN
.if (@$tN == value) { .echo PASS } .else { .echo FAIL }
```

20候補すべてで`PASS` markerが出た。custom register assignmentのtrialで`$s7`や`$s7e`へ置換するのではなく、probeで確定した`$t0`〜`$t19`の仕様だけをretryへ反映した。

さらにretryの14 breakpoint payloadをCDBへ`bu`登録し、payload parserを確認するのみ行った。`bu` commandは14件、syntax/register/resolve errorは0件で、payloadの実行は行っていない。

## 7. RP-5 — ReaderSlot preservation

C2で確定したlayoutを変更していない。

```text
ReaderSlot stride = 0x50
readerCount       = 64
slot base offsets = 0x20 + i * 0x50, i=0..63
```

retry static validationは、旧C3 scriptと新scriptの6 blockについてoffset列を完全比較した。

| block | old literal reads | new literal reads | old/new offset identity | verdict |
|---|---:|---:|---:|---|
| T0 | 64 | 64 | identical | PASS |
| S7 | 64 | 64 | identical | PASS |
| T1 entry | 64 | 64 | identical | PASS |
| T1 getMin return | 64 | 64 | identical | PASS |
| T1 reclaim epoch read | 64 | 64 | identical | PASS |
| T2 wait return | 64 | 64 | identical | PASS |
| **total** | **384** | **384** | **identical** | **PASS** |

`ReaderSlot` offset、stride、readerCountを変更していない。`.for`、`.while`、dynamic slot-index address expressionは0である。固定offsetの単純な`db`/`dq`/`dd`/`poi` memory expressionはC2 evidenceとして維持した。

## 8. RP-6 — retry script static validation

生成した未実行script:

```text
doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C3-RP_retry_not_executed.cdb
SHA-256 = AFCC0017DA95D36F23D497123547AEE0B0B8D7572F43231835A25AE532EF0F18
bytes   = 11150
```

static validator結果:

```text
checks = 78
pass   = 78
fail   = 0
```

| gate | result | evidence |
|---|---|---|
| custom pseudo-register name = 0 | PASS | `$t0`〜`$t5`のみ |
| invalid register assignment = 0 | PASS | `$s7`, `$s7e`, `$s7q`, `$s7rbx`, `$t0q`, `$t1done`, `$t2done`なし |
| invalid conditional reference = 0 | PASS | expression参照は`@$tN`、assignmentのみ`r $tN` |
| `.for` / `.while` | PASS | 0 |
| complex address expression | PASS | dynamic slot-index/loop expression 0 |
| 64 literal ReaderSlot reads | PASS | 6 block × 64 = 384 |
| T0/T1/T2 capture points | PASS | 9 required breakpoint labels present once each |
| quote/brace balance | PASS | 全14 breakpoint line |
| retry execution | PASS | script `g` commandは未実行 |
| C3 original script unchanged | PASS | frozen hash一致 |

## 9. Capture points retained

```text
T0_ENTRY
T0_CAS_PRE
T0_CAS_POST
T0_WRITE_DONE
S7_MARKER
T1_GETMIN_CALL_PRE
T1_GETMIN_RETURN
T1_RECLAIM_EPOCH_READ
T2_WAIT_RETURN
```

T0のretry commandはcaller return address、caller symbol、stack、entry pointer/deleter/epoch/type/publicationSequenceId/generation、globalEpoch、enqueuePos、sequence[4]、ReaderSlot 0..63を読む。S7/T1/T2はC2で定義したqueue/epoch/ReaderSlot snapshotとjoin keysを維持する。runtime値をこの報告書で補完していない。

## 10. RP-7 / RP-8 — retry evidence and join contract

retry実行時に取得すべき値は変更しない。最低限、次を一つのexecution内でjoinする。

```text
process
engineThis
epochBase
dqueueBase
ticket = 4
sequence = 5
entry epoch = 9
```

T0 producer候補、S7 producer、reader at epoch 9、Case A/B/C/Dは、retryの実測dataが揃うまで`UNRESOLVED`のまま維持する。`minReaderEpoch=9`はreader identityの代替値として使わない。

## 11. Prohibitions and execution boundary

| prohibited item | result | evidence |
|---|---:|---|
| C3 same-script rerun | PASS | C3 original hash preserved; no C3 target invocation |
| `$s7` rename-and-rerun | PASS | no ConvoPeq runtime retry |
| speculative `$s7 -> $t7` | PASS | probe後の`$t0`〜`$t5`のみを使用 |
| ReaderSlot offset/stride change | PASS | 384/384 identity |
| S7 reader supplementation | PASS | unresolved status retained |
| `minReaderEpoch=9` as reader evidence | PASS | not used |
| T0 producer inference | PASS | no inference |
| T1/T2 supplementation | PASS | no inference |
| M0/M1/M2 | PASS | 0 |
| production/test/CMake source change | PASS | 0 gate-originated changes |
| getter/counter/instrumentation | PASS | 0 |
| build | PASS | 0 |
| Dr. Memory | PASS | 0 |
| ConvoPeqHarness process | PASS | 0 before/after every benign CDB probe |
| CDB process residue | PASS | 0 after probe |

既存の作業tree変更はC3-RP開始前から存在し、C3-RPでrollback・上書き・resetしていない。CDB parser probeで起動したprocessは終了済みで、portを占有していない。

## 12. Authorization status

```text
C3-RP                         = READY FOR ONE RETRY
C3 retry execution            = NOT EXECUTED
C3 runtime capture            = NOT EXECUTED
T0 producer                    = UNRESOLVED
S7 producer                    = UNRESOLVED
S7 reader                      = UNRESOLVED
Same-execution join           = NOT PROVEN
Case A/B/C/D                   = NOT PROVEN
IMPLEMENTATION                 = FORBIDDEN
```

次のgateは、承認済みのC3 retry 1回を`AudioEngineHarness.exe --fpm-m0`で実行し、生成logをT0/S7/T1/T2 direct evidenceとして評価するだけに限定する。retryが失敗した場合は推測で値を進めず，再度STOPとして停止する。

## 13. Evidence index

- C2 procedure: `doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C2_S7_Producer_Reader_Temporal_Correlation_Capture_Preparation.md`
- C3 stop report: `doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C3_S7_Producer_Reader_Temporal_Correlation_Runtime_Capture.md`
- C3 frozen script: `C:\Users\user\AppData\Local\Temp\opencode\c3-s7-capture.txt`
- C3 frozen log: `C:\Users\user\AppData\Local\Temp\opencode\c3-s7-capture.log`
- New unexecuted retry script: `doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C3-RP_retry_not_executed.cdb`
- Microsoft pseudo-register specification: `https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/pseudo-register-syntax`
- Microsoft `.if` specification: `https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/-if`

```text
C3-RP = READY FOR ONE RETRY
```
