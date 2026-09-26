# P3-5-FPM-C3 — S7 Producer / Reader Temporal Correlation Runtime Capture

## 1. Result

```text
Gate                         = P3-5-FPM-C3
Mode                         = runtime capture
C2                            = CLOSED / method READY
C3                            = STOPPED
CDB process exit              = 0
T0 producer                    = NOT_CAPTURED
T1 reader state                = NOT_CAPTURED
T2 reader state                = NOT_CAPTURED
Case A/B/C/D                  = UNCLASSIFIED
S7_PRODUCER                   = UNRESOLVED
S7_READER                     = UNRESOLVED
IMPLEMENTATION                = FORBIDDEN
```

CDBが一度でもexpression/register errorを出力したため、C3のSTOP条件に該当して再実行・script修正・offset補正・値の補完を行わない。

## 2. Preflight

C3開始前にC2のidentityとstatic layoutを再確認した。

| artifact | expected | observed | result |
|---|---|---|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609`, 5,535,334 bytes | same | PASS |
| `AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | same | PASS |
| `AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | same | PASS |
| PE image base | `0x140000000` | `0x140000000` | PASS |
| `ENQ_ENTRY` | RVA `0x1f9c550` | static disassembly match | PASS |
| `ENQ_CAS_PRE` | RVA `0x1f9c57d` | static disassembly match | PASS |
| `ENQ_CAS_POST` | RVA `0x1f9c59a` | static disassembly match | PASS |
| `ENQ_WRITE_DONE` | RVA `0x1f9c5da` | static disassembly match | PASS |
| `S7_TIMEOUT` | RVA `0x1fa80b3` | static disassembly match | PASS |
| ReaderSlot stride | `0x50` | static source/PDB evidence match | PASS |

Preflightで`STOP-C2-4`に該当するlayout driftはなかった。Driftは今回CDB failureの原因ではない。

## 3. Workload interpretation

既存harnessのS7再現経路は`runFpmM0()`であり、`--fpm-m0`をC3のS7 stimulusとして一度だけ起動した。M1/M2は起動していない。`--fpm-m0`のT3b測定結果をC3のattribution evidenceとして採用せず、C3結果のstimulus起動としてのみ記録する。

## 4. Capture script preflight

C3 scriptはC2どおり、`.for`、`.while`、複雑なCDB address expressionを使用せず、各観測blockに64個のliteral `db` readを配置した。

| block | literal slot reads |
|---|---:|
| T0 enqueue | 64 |
| S7 marker | 64 |
| T1 tryReclaim entry | 64 |
| T1 getMin return | 64 |
| T1 head epoch read | 64 |
| T2 wait return | 64 |

Script path:

```text
C:\Users\user\AppData\Local\Temp\opencode\c3-s7-capture.txt
SHA-256 = 5902AAB8FA8FE68208CB2B2A4D1F2EB708AD29AA37AFA727D749D05720DBAE88
bytes   = 10857
```

静的script validationでは6blockすべてのslot countが64で、debugger loopは0だった。ただしCDB pseudo-registerの初期化構文は静的validationでは検出されず、runtimeで失敗した。

## 5. CDB invocation

```text
C:\VSC_Project\ConvoPeq\tmp\cdb.exe
  -logo C:\Users\user\AppData\Local\Temp\opencode\c3-s7-capture.log
  -y C:\VSC_Project\ConvoPeq\build\Release
  -cf C:\Users\user\AppData\Local\Temp\opencode\c3-s7-capture.txt
  AudioEngineHarness.exe --fpm-m0
```

Log path:

```text
C:\Users\user\AppData\Local\Temp\opencode\c3-s7-capture.log
SHA-256 = D6D160AA3788C247B79608EDEEF8DB5B129AB682ADE68EB4437D7B399BFCB7BA
bytes   = 32443
```

CDB process exit codeは`0`であった。しかしexit code 0はcapture成功を意味しない。log内で必須のdebugger errorを確認した。

## 6. Failure capture

### First errors

```text
Bad register error in 'r $s7 = 0'
Bad register error in 'r $t1done = 0'
Bad register error in 'r $t2done = 0'
Bad register error in 'r $s7e = 0'
Bad register error in 'r $s7q = 0'
Bad register error in 'r $s7rbx = 0'
Couldn't resolve error at '$s7 == 1) { .if (@rcx == $s7rbx) { ...'
```

### Actual event labels

```text
C3_SCRIPT_BEGIN  = present
C3_T0_ENTRY      = absent
C3_T0_CAS_PRE    = absent
C3_T0_CAS_POST   = absent
C3_T0_WRITE_DONE = absent
C3_S7_MARKER     = absent
C3_T1_*          = absent
C3_T2_*          = absent
C3_PROCESS_EXIT  = present
```

したがって、process終了までのCDB logにT0/S7/T1/T2のdirect evidenceは存在しない。`[FPM]`の有効な結果もC3 attribution evidenceとして採用していない。

## 7. Root cause

C3 scriptはpseudo-register名として`$s7`、`$s7e`、`$s7q`、`$s7rbx`、`$t1done`、`$t2done`を使用した。現行CDBはこれらのnameのassignmentをvalid pseudo-register assignmentとして解決できず、`Bad register error`を出力した。

その後のbreakpoint commandは未定義の`$s7`/`$s7rbx`を`.if`で参照したため、`Couldn't resolve error`まで連鎖した。これはRCA-9の`.for` failureとは別の、C3 scriptのregister naming/parser errorである。

Root cause classification:

```text
FAILURE_CLASS = CDB_PSEUDO_REGISTER_EXPRESSION
BINARY_LAYOUT_DRIFT = NO
RUNTIME_VALUE_MISSING = STOPPED_AT_SCRIPT_INITIALIZATION
SOURCE_CAUSE = CAPTURE_SCRIPT_ERROR
```

## 8. STOP matrix

```text
STOP-C3-1  T0 enqueue callerを一意に特定できない       = TRIGGERED / NOT_CAPTURED
STOP-C3-2  T0 entryとS7 headを一意にjoinできない       = TRIGGERED / NOT_CAPTURED
STOP-C3-3  64 ReaderSlotの一部でも読めない             = TRIGGERED / NOT_CAPTURED
STOP-C3-4  binary/source layout drift                 = NOT_TRIGGERED
STOP-C3-5  T0/T1/T2が同一executionとしてjoinできない  = TRIGGERED / NOT_CAPTURED
STOP-C3-6  CDB expression failureを推測で回避          = TRIGGERED
STOP-C3-7  S7とは別DQueue entryを観測している可能性     = NOT_EVALUATED
STOP-C3-8  runtime captureのためsource変更が必要         = NOT_TRIGGERED
```

`STOP-C3-6`を優先適用し、C3を終了する。T0/T1/T2の欠落を、`minReaderEpoch=9`、事後のreader count、または既存RCA-7結果で補完しない。

## 9. Agent failure diagnosis

```text
Failure pattern  = deterministic debugger-script initialization error
Last success     = identity/static-layout preflight and literal-slot script validation
Failed step      = first pseudo-register assignments in CDB script
Repeated pattern = none; no retry was performed
Recovery action  = stop, preserve script/log, clean processes, write STOP report
Result           = partial / blocked at capture start
Preventive rule  = validate pseudo-register names with a non-runtime CDB syntax gate
```

Recovery後に同じscriptを起動していない。scriptの自動修正も行っていない。

## 10. Tree and process state

```text
production source change = 0
test source change       = 0
CMake change             = 0
getter/counter/telemetry  = 0
build                    = NOT RUN
Dr. Memory               = 0
M1/M2                    = 0
CDB process residue      = 0
Harness process residue  = 0
```

Tempのscript/logはC3失敗のevidenceとして保存した。repositoryのsource/test/CMakeは変更していない。

## 11. Completion status

```text
C3_CAPTURE                    = STOPPED
T0_PRODUCER                   = UNRESOLVED
T1_READER_STATE               = UNRESOLVED
T2_READER_STATE               = UNRESOLVED
SAME_EXECUTION_JOIN           = NOT_PROVEN
CASE_A                        = NOT_PROVEN
CASE_B                        = NOT_PROVEN
CASE_C                        = NOT_PROVEN
CASE_D                        = NOT_PROVEN
TERMINAL_CONTRACT_REOPENED    = NO
IMPLEMENTATION               = FORBIDDEN
```

C1のcontract/liveness closureは変更しない。C3でterminal contractを再判定しない。

## 12. Next authorization boundary

次のretryは今回のC3の延長ではなく、別gateで明示的なauthorizationを取得する。次回のretry preparationでvalidなCDB pseudo-register namesを事前検証し、source/test/CMakeを変更せずにT0/T1/T2を実行する。次回のretryは別途承認された1回だけとする。

本次C3ではそのretryを開始しない。推定でpseudo-register namesを補完せず、次のrunも許可しない。

## 13. Evidence index

| evidence | location |
|---|---|
| C2 procedure | `doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C2_S7_Producer_Reader_Temporal_Correlation_Capture_Preparation.md` |
| C3 script | `C:\Users\user\AppData\Local\Temp\opencode\c3-s7-capture.txt` |
| C3 CDB log | `C:\Users\user\AppData\Local\Temp\opencode\c3-s7-capture.log` |
| C3 script hash | `5902AAB8FA8FE68208CB2B2A4D1F2EB708AD29AA37AFA727D749D05720DBAE88` |
| C3 log hash | `D6D160AA3788C247B79608EDEEF8DB5B129AB682ADE68EB4437D7B399BFCB7BA` |
| identity | `ConvoPeq.md`, `build/Release/AudioEngineHarness.exe`, `build/Release/AudioEngineHarness.pdb` |
| static RVA verification | `llvm-objdump` read-only over current `AudioEngineHarness.exe` |
| prior RCA-9 failure boundary | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-9_EpochProvenance_reader_lifecycle_exact.md` |

C3はruntime attributionを完了していない。確定していない値を確定済みとして引き継ぐことは禁止する。
