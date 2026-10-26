# v3.78.64 — PersistentCache Implementation Design Closure Audit

- Date: 2026-10-05
- Authority: `ConvoPeq.md`（Owner 指定 = コーダーAI が参照する `C:\VSC_Project\ConvoPeq\ConvoPeq.md`）
- Type: **read-only implementation design closure audit**
- production source / test / CMake / build / version / ConvoPeq 再生成 / commit / push: **すべて 0**

---

## 1. Authority Reconciliation

### 1.1 Owner 指定ファイルの実在確認

Owner 指示の `ConvoPeq(20261002-142258).md` は**実在しない**。

```
検索対象: C:\VSC_Project\ConvoPeq / C:\Users\user\AppData\Local\Temp\opencode / ...\opencode\worktree
ファイル名一致 "142258"                    → 0 件
doc\ConvoPeqMD\ の末尾 6 ディレクトリ:
    20261001184006 / 20261001192535 / 20261002200923 / 20261002205611 / 20261002231921 / 20261003005500
    → 20261002142258 は存在しない
```

したがって Owner の standing instruction（`ConvoPeq.md` と `ConvoPeq(20261002-142258).md` は同一ファイル）に基づき、**Owner 指定 authority = コーダーAI が参照する `ConvoPeq.md`** とする。

### 1.2 検出した `ConvoPeq.md` 全候補と SHA-256

| # | path | bytes | SHA-256 (先頭24) | 位置づけ |
| --- | --- | --- | --- | --- |
| **A** | `C:\VSC_Project\ConvoPeq\ConvoPeq.md` | 5,907,375 | `80e3e9459d4b8308de771f2d` | **Owner 指定**（CODER AI の `ConvoPeq.md`） |
| **B** | `C:\Users\user\AppData\Local\Temp\opencode\stg11-d1-d18\ConvoPeq.md` | 5,942,475 | `f73e61942904c939686ed222` | **v3.78.63 が effective authority として使用した snapshot** |
| C | `...\.local\share\opencode\worktree\...\bright-falcon\ConvoPeq.md` | 5,489,292 | `025ee1e63f853257a81e214c` | 本作業ディレクトリ（未使用） |

**A と B は同一ファイルではない**（SHA-256 が異なる）。v3.78.63 は B を effective authority として用いた。Owner 指示に従い、同一性を仮定せず比較を実施した。

### 1.3 必須比較 — `MixedPhasePersistentCache.h/.cpp`

```
--- src/MixedPhasePersistentCache.h
  A: sha256=0c3fd1f5861e0da51e5a1cecc38df7d81ba85105a499f37069a05014193935f9  bytes=3063  lines=97
  B: sha256=0c3fd1f5861e0da51e5a1cecc38df7d81ba85105a499f37069a05014193935f9  bytes=3063  lines=97
  BYTE_IDENTICAL = True

--- src/MixedPhasePersistentCache.cpp
  A: sha256=df0160458cedb4f9cf32d39591666c0060e0fc430ddc94e839f165ebd34ddf85  bytes=14461 lines=352
  B: sha256=df0160458cedb4f9cf32d39591666c0060e0fc430ddc94e839f165ebd34ddf85  bytes=14461 lines=352
  BYTE_IDENTICAL = True
```

**audit 対象 2 ファイルは Owner 指定 authority に対して byte-identical。**

### 1.4 snapshot 間の全差異（影響範囲の確定）

セクション数 A=364 / B=365、共通 363 セクションのうち:

```
identical sections : 353
differing sections: 10
   CMakeLists.txt
   src/DeviceSettings.cpp
   src/audioengine/AudioEngine.Parameters.cpp
   src/audioengine/AudioEngine.h
   src/audioengine/DSPTransition.h
   src/audioengine/ISRDSPHandle.cpp
   src/audioengine/OwnerChannel.h
   src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp
   src/tests/NormalRetireDSPHandleCompareTests.cpp
   src/tests/OwnerChannelTests.cpp
A のみ: src/tests/AudioEngineHarness/STG11D18CrossfadeTests.cpp
B のみ: src/tests/AudioEngineHarness/STG11D12NoiseShaperEnumGuardTests.cpp
       src/tests/AudioEngineHarness/STG11D12StateIntegerGuardTests.cpp
```

v3.78.63 が**証拠として引用した**差異セクション 3 個の中心的差分:

**`CMakeLists.txt`**（A:2279 行 / B:2291 行、changed=18）

```
-# CMakeLists.txt  ── v0.6.10 (JUCE 8.0.12 / ...)      ← A
+# CMakeLists.txt  ── v0.6.11 (JUCE 8.0.12 / ...)      ← B
-  target_link_libraries(STG11EQRetireTests ... MKL::MKL)          ← A（無条件 link）
+  if(MSVC AND NOT ...IntelLLVM) / if(CONVOPEQ_HAS_MKL) ...       ← B（icx guard 追加）
+  target_compile_options(STG11EQRetireTests PRIVATE /Qmkl:sequential)
-        src/tests/.../STG11D18CrossfadeTests.cpp                 ← A
+        src/tests/.../STG11D12StateIntegerGuardTests.cpp          ← B
+        src/tests/.../STG11D12NoiseShaperEnumGuardTests.cpp
+  target_compile_options(PolyphaseGainFidelityTests PRIVATE /Qmkl:sequential)
```

`CMAKE_CXX_STANDARD 20` / `CMAKE_CXX_STANDARD_REQUIRED ON` / `CONVOPEQ_ENABLE_CONVOLVER_SPLIT_MIXED_PHASE=1` / `JUCE 8.0.12` / `/arch:AVX2` / `MSVC_RUNTIME_LIBRARY` 設定は**両 snapshot で同一**（行番号のみシフト）。

**`src/audioengine/AudioEngine.Parameters.cpp`**（A:795 / B:839、changed=44）
差分は **STG-11-D12-1（`bitDepth` setter 境界 guard）と STG-11-D12-2（`NoiseShaperType` enum domain guard）の追加のみ**。
v3.78.63 が引用した `hasFiniteDouble("mixedF1Hz", MIXED_F1_MIN_HZ, MIXED_F1_MAX_HZ)` / `hasFiniteDouble("mixedF2Hz", ...)` の 19 件は**完全一致**（`IDENTICAL_SET = True`）。

**`src/audioengine/AudioEngine.h`**（A:5450 / B:5488、changed=52）
差分は **STG-11-D15-1（`ownerChannelProducerMutex_` 追加による `ownerChannel().enqueue` / `take-back` の直列化）のみ**。
v3.78.63 が引用した `rebuildThreadLoop()` 宣言 / `std::thread rebuildThread` / `rebuildThreadShouldExit` は**内容同一**（行番号 A:2917 → B:2930 のシフトのみ）。

残り 7 セクション（`DeviceSettings.cpp` / `DSPTransition.h` / `ISRDSPHandle.cpp` / `OwnerChannel.h` / 3 test files）は v3.78.63 の証拠に一切使用していない。

### 1.5 判定

```
Authority reconciliation = PASS
判定の保留              = しない
```

`MixedPhasePersistentCache.h/.cpp` は Owner 指定 authority に対して byte-identical であり、他の証拠セクションも v3.78.63 が引用した行が完全一致する。**v3.78.63 の全結論は Owner 指定 authority に対して有効**であり、本 v3.78.64 の以降すべての設計は **Owner 指定 authority を唯一の出所**として構築する。

---

## 2. v3.78.63 契約の維持（implementation design の入力仕様）

以下は変更せず採用する。

| # | 契約 |
| --- | --- |
| 1 | `fileHash == 0` は invalid |
| 2 | SR は finite かつ `44100.0 <= SR <= 768000.0`（inclusive） |
| 3 | PhaseMode は既存 authority（`ConvolverProcessor.h:117-122 enum class PhaseMode : int { AsIs=0, Mixed=1, Minimum=2 }`）を使用し PersistentCache 側で再定義しない |
| 4 | F1/F2 は既存 authority（`MIXED_F1_MIN_HZ=100.0f` / `MIXED_F1_MAX_HZ=400.0f` / `MIXED_F2_MIN_HZ=700.0f` / `MIXED_F2_MAX_HZ=1300.0f`、`F2 >= F1 + 10.0f`）を使用し duplicate constants を作らない |
| 5 | `targetLength` は `1 .. MAX_IR_LATENCY`（`ConvolverProcessor.h:250 = 2097152`） |
| 6 | `numSamples` は `1 .. MAX_IR_LATENCY` |
| 7 | `numChannels` は `1 .. kMaxIRLoadChannels`（`IRLoadAdmission.h:43 = 8u` を再利用） |
| 8 | `rho.size() == theta.size()` |
| 9 | `sections` は `0 .. 20`（`sections == 0` は許容） |
| 10 | v3 DiskHeader（`sizeof == 76` / `lastUsedTime offset == 48` / `totalFileSize offset == 68` / `is_trivially_copyable_v` / runtime `headerReserved == 0`） |
| 11 | canonical filename = `64` lowercase hex + `".mph"` = 68 文字 |
| 12 | filename は hash ではなく **identity の fixed-width bit serialization**（`computeKeyHash()` は authority ではない） |
| 13 | destructive operation の対象は **current v3 canonical artifact のみ** |
| 14 | future / obsolete / malformed / noncanonical artifact は **preserve** |
| 15 | `evictLRU(0)` は **no-op**（`clear()` を alias にしてはならない） |
| 16 | `save` failure 後は **eviction しない** |
| 17 | `eviction failure` は `save` success を **rollback しない** |
| 18 | identity 比較は **bit-pattern equality**（epsilon 比較禁止） |
| 19 | timestamp は `juce::Time::currentTimeMillis()`、`0 = invalid` / `1..INT64_MAX = valid` |
| 20 | CRC / checksum は追加しない。same-size payload corruption の検出は保証しない |

---

## 3. C / F の implementation alternatives 比較

### 3.1 比較対象

```
A. native ReplaceFileW + explicit backup + rollback
B. 二段階 commit（staging 書込 → rename）
C. その他 Windows x64 で同等の ownership / transactional guarantee を満たす方式
   C1. SetFileInformationByHandle(FileRenameInfo, ReplaceIfExists=TRUE)
   C2. TxF (Transactional NTFS)
   C3. 同一 HANDLE への in-place 全面書込 + SetFileSizeEx
   C4. FILE_RENAME_FLAG_POSIX_SEMANTICS (SetFileInformationByHandleEx)
   C5. JUCE TemporaryFile (現行)
```

### 3.2 証拠（Windows x64 exact semantics）

**`ReplaceFileW`**（Microsoft Learn）

- `lpBackupFileName` に **`[in, optional] LPCWSTR`**
- `lpReplacedFileName` は「opened with the **GENERIC_READ, DELETE, and SYNCHRONIZE** access rights. The sharing mode is **FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE**. The caller must have write access to the file to be replaced.」
- 成功時、元のファイルの **Creation time / Short file name / Object identifier / DACLs / Security resource attributes / Encryption / Compression / Named streams not already in the replacement file** を保存する
- failure:
  - `1176 ERROR_UNABLE_TO_MOVE_REPLACEMENT`「If `lpBackupFileName` was specified, the replaced and replacement files retain their original file names. **Otherwise, the replaced file no longer exists** and the replacement file exists under its original name.」
  - `1177 ERROR_UNABLE_TO_MOVE_REPLACEMENT_2`「…The file to be replaced still exists **with a different name**. If `lpBackupFileName` is specified, it will be the name of the replaced file.」
  - `1175 ERROR_UNABLE_TO_REMOVE_REPLACED`「The replaced and replacement files retain their original file names.」
  - その他（「such as `ERROR_INVALID_PARAMETER`」）→「the replaced and replacement files will retain their original file names. In this scenario, a backup file does not exist and it is not guaranteed that the replacement file will have inherited all of the attributes and streams of the replaced file.」

**決定的不変条件（`lpBackupFileName != nullptr` のとき）**

> **failure 发生后、旧 artifact のバイト列は `lpReplacedFileName`（target）か `lpBackupFileName`（backup）のいずれかに必ず存在する。**

`1175` とその他すべてのエラーは「元の名前を維持」、`1176` は「backup 指定時は元の名前を維持」、`1177` は「backup 指定時は backup が旧 artifact の名前になる」。したがって **backup を必ず指定することで、target が消滅する経路（JUCE が踏む 1176 の `nullptr` 分岐）が構造的に消える。**

**`MoveFileExW`**（Microsoft Learn）

- `MOVEFILE_WRITE_THROUGH`「Setting this value guarantees that a move performed as **a copy and delete operation** is flushed to disk before the function returns. The flush occurs at the end of the copy operation. This value has no effect if `MOVEFILE_DELAY_UNTIL_REBOOT` is set.」→ **同一 volume の rename では write-through の追加保証にならない**
- `MOVEFILE_COPY_ALLOWED`「If the file is to be moved to a different volume, the function simulates the move by using the CopyFile and DeleteFile functions.」→ **cross-volume では copy+delete になり atomic 性を失う**
- `MOVEFILE_REPLACE_EXISTING` は destination を置換するが、backup 機構は無く **failure 時の target 保存を保証しない**

**`SetFileInformationByHandle`**（Microsoft Learn）

- `FileRenameInfo`（3）→ `FILE_RENAME_INFO`。backup 相当の概念は無い。
- 「You must specify appropriate access flags when creating the file handle for use with `SetFileInformationByHandle`.」→ rename には `DELETE` access が必要。

**`FlushFileBuffers`**（Microsoft Learn）

- 「The **FlushFileBuffers** function writes all the buffered information for a specified file to the device or pipe.」
- 「If an application is performing multiple writes to disk and also needs to ensure critical data is written to persistent media, the application should use **unbuffered I/O** instead of frequently calling `FlushFileBuffers`.」

### 3.3 10 軸比較表

| 軸 | A. `ReplaceFileW` + backup + rollback | B. 二段階 commit | C1. `FileRenameInfo` | C2. TxF | C3. in-place 全面書込 | C4. POSIX_SEMANTICS | C5. JUCE TemporaryFile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **protected artifact preservation** | **成立**。backup 指定により旧 artifact は target か backup に必ず存在 | **条件付きで成立**。target→backup rename と staging→target rename の間に **target 不在の窓**ができる。窓中は「absent」判定となり artifact を失ったように見える（バイト列は backup に存在） | 条件付きで成立（backup なし） | 不明（TxF の isolated な代替ではない） | **不成立**。書き込み途中で旧内容消失 | 保留（後述） | **不成立**。現行は backup なし |
| **replacement failure preservation** | **成立**。MS 実文由来の不変条件。rollback は `MoveFileExW(backup→target, MOVEFILE_REPLACE_EXISTING)` | 成立するが窓あり | 保証なし | 不明 | **不成立** | 保留 | **不成立**（1176 で target 消滅） |
| **same-HANDLE load/touch との干渉** | target の `DELETE` access が必要。load HANDLE は `FILE_SHARE_DELETE` 禁止のため **同一 process では mutex が直列化**、cross-process では `save` が failure → preserve（正しい方向） | rename も `DELETE` が必要。同様 | 同様 | 同様 | **干渉大**。load の読み取り中に上書きされる | ** antagonistic**。open 中の置換を許すため「open 中は改変不可」の保証を壊す | 同様 |
| **Windows share semantics** | `ReplaceFileW` は target を `GENERIC_READ\|DELETE\|SYNCHRONIZE` / share 全部指定で開く。`FILE_SHARE_DELETE` 禁止の load HANDLE 存在下では **OS が拒否**する → failure → preserve | 同左 | 同左 | — | — | POSIX semantics は `FILE_SHARE_DELETE` 禁止契約と**直接矛盾** | 同左 |
| **crash / process termination 中の状態** | 単一 metadata 操作のため、**power loss は「旧名」か「新名」のいずれか**。両者とも current クラス。**破損した中間態は発生しない** | rename と rename の間なら **old 名・new 名の双方**。target 不在は可能性あり | 同左 | — | **途中破棄で header/payload 不整合** | — | 同左 |
| **temporary artifact** | staging（`CREATE_NEW`）と backup の 2 つ。どちらも `*.mph` ではない命名にする | staging 1 個 | staging 1 個 | — | 無し | staging 1 個 | **JUCE は `<canonical>_temp<hex>.mph` と同ディレクトリに生成するため `*.mph` glob に合致する（v3.78.63 §2.3(a) で確定）** |
| **rollback failure** | backup が残る。**backup を削除しない**。save=false + 診断ログ。artifact は失われない | rollback 失敗時は backup が残る（同様） | — | — | rollback 不可 | — | rollback 機構が無い |
| **mutex 保持時間** | staging 書込（≤134,218,124 B）+ `FlushFileBuffers` + `ReplaceFileW` + rollback。**sleep/retry を排除した設計**（現行の 5×100ms は不使用） | 同左 + rename 2 回 | rename 1 回 | — | 書込のみ（最小）だが契約違反 | — | **5 × (`ReplaceFileW` + `Thread::sleep(100)`) + `~TemporaryFile` 5 × 50ms = v3.78.63 §13 で実測 約 600–750 ms** |
| **非 RT blocking** | I/O bound。人工的な下限なし | 同左 | 同左 | — | 同左 | — | 上記の sleep 由来の下限あり |
| **implementation complexity** | 中。RAII HANDLE wrapper / backup 経路生成 / error 分類 / rollback の 4 要素 | 中〜高。rename 窓の回復ロジック|Juliet が要る | 中。RAII wrapper のみ | 高。影響範囲が広い | 低（だが契約違反） | 中〜高。Windows 10 1607+ 限定 | 極低（だが契約違反） |

### 3.4 判定

```
採用 = A（native ReplaceFileW + explicit backup + rollback）
ただし target absent の場合は A の退化工形として
    staging 書込 → MoveFileExW(staging → target, MOVEFILE_WRITE_THROUGH)
    （置換対象が無いので backup は不要。失敗しても何も破壊されない）
棄却 = C2 (TxF は Microsoft が deprecated / 一般利用を推奨しない)
       C3 (in-place 全面書込は failure 時に旧 artifact を失う)
       C4 (POSIX_SEMANTICS は FILE_SHARE_DELETE 禁止契約と矛盾)
       C5 (JUCE TemporaryFile は backup 無し。v3.78.63 §4 で充足不能を証明済み)
保留 = B は A の部分集合（target→backup rename を A の backup に置き換えたもの）であり、
       独立案としては A に劣るため採用しない。ただし A の rollback 段は
       実質 B の第 1 rename と同一であり、この点で A は B を内包する。
```

**A の採用根拠（1 行）**: `lpBackupFileName` を必ず指定することで、Microsoft Learn の `ReplaceFileW` failure 表が示す唯一の「replaced file no longer exists」経路（1176 かつ `lpBackupFileName == nullptr`）が構造的に消滅し、**「failure 後は旧 artifact のバイト列が target か backup に必ず存在する」**という不変条件が得られる。

---

## 4. same-HANDLE design（内部限定 Windows native HANDLE abstraction）

### 4.1 契約シーケンス（Owner 指定）

```
Open H
-> GetFileSizeEx(H)
-> ReadFile(header)
-> canonical validation
-> ReadFile(payload)
-> timestamp update on H
-> FlushFileBuffers(H)
-> CloseHandle(H)
```

### 4.2 Open 仕様

```cpp
HANDLE h = ::CreateFileW (path,
                          GENERIC_READ | GENERIC_WRITE,
                          FILE_SHARE_READ | FILE_SHARE_WRITE,   // FILE_SHARE_DELETE は禁止
                          nullptr,
                          OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL,
                          nullptr);
```

| パラメータ | 値 | 根拠 |
| --- | --- | --- |
| `dwDesiredAccess` | `GENERIC_READ \| GENERIC_WRITE` | header 読込 + offset 48 の 8 byte 書込の両方が必要 |
| `dwShareMode` | `FILE_SHARE_READ \| FILE_SHARE_WRITE` | Owner 契約。`FILE_SHARE_DELETE` を外すことで、同一パスへの `DeleteFile` / `MoveFile` / `ReplaceFile` を OS が拒否する（`CreateFileW` dwShareMode「FILE_SHARE_DELETE — Enables subsequent open operations on a file or device to request delete access」/「The sharing options for each open handle remain in effect until that handle is closed」） |
| `dwCreationDisposition` | `OPEN_EXISTING` | 存在しないファイルを開かない（存在確認は別段） |
| `dwFlagsAndAttributes` | `FILE_ATTRIBUTE_NORMAL` | `FILE_FLAG_DELETE_ON_CLOSE` は禁止（他 path を破壊しうる）。`FILE_FLAG_OPEN_REPARSE_POINT` は **validator 側で reparse tag を検出した場合のみ**使う |

**副作用の明示**: `GENERIC_WRITE` を含むため `load()` は `current` artifact に対して write 権限を持つ。write は offset 48..55 の 8 byte に限定し、**canonical validation が `current` と判定した後にのみ実行する**（`noncanonical` / `future` / `obsolete` / `malformed` / `indeterminate` は write 段に進まない）。

### 4.3 各段の failure mapping（Owner contract を変更せず明示）

| 段 | Win32 失敗 | 扱い | logical `load()` result |
| --- | --- | --- | --- |
| Open | `INVALID_HANDLE_VALUE` | `OpenFailed` | `false` |
| `GetFileSizeEx` | 戻り 0 | `SizeQueryFailed` → **`indeterminate`** | `false` |
| `ReadFile(header)` | `bytesRead != 76`（`ERROR_HANDLE_EOF` 含む） | `ShortRead` → **`malformed`** | `false` |
| canonical validation | verdict != `current` | `Rejected` | `false` |
| `ReadFile(payload)` | `bytesRead != expected - 76` | `ShortRead` → `malformed` | `false` |
| `outIr.setSize` / `outRho.resize` / `outTheta.resize` | `std::bad_alloc` | 伝播（現行 `convertToMixedPhaseAllpass` の `catch(...)` が `setMixedPhaseState(0)` + rethrow で処理） | 例外 |
| **timestamp update** | `WriteFile` 戻り != 8 | **`LogicalSuccessPreserved`（Owner 契約 §4 参照）** | **`true` のまま** |
| **`FlushFileBuffers`** | 戻り 0 | **`LogicalSuccessPreserved`** | **`true` のまま** |
| `CloseHandle` | 戻り 0 | **`LogicalSuccessPreserved`** | **`true` のまま** |

### 4.4 timestamp write / `FlushFileBuffers` failure の扱い（Owner contract の明示）

v3.78.62 で Owner が確定した契約:

> `load()` の logical success = header validation + payload 完全読込 + outIr/rho/theta 構築。**timestamp write / `FlushFileBuffers` / `CloseHandle` failure は logical result を `false` に戻さない。**

本設計はこれを変更しない。帰結と境界を以下のように確定する。

**確定事項**

1. **logical result に影響するのは `current` 判定と payload 完全読込まで**。timestamp は LRU ordering の **hint** であり、IR の内容・shape・identity の正解性には関与しない。
2. **`lastUsedTime`（offset 48, `uint64_t`）だけを書き換える**。magic / version / headerReserved / fileHash / sampleRate / phaseMode / freqStartHz / freqEndHz / targetLength / numChannels / numSamples / numAllpassSections / totalFileSize には**触れない**。したがって **write 成功・失敗・途中破棄のいずれでも artifact の canonicality は構造的に保存される**。
3. **`ReadFile` は「requested bytes より少ない」場合に `bytesRead` を返す**。8 byte 書きで `bytesRead < 8` の場合、`lastUsedTime` は torn value になる可能性がある。

**torn timestamp の帰結（残余曖昧性ではなく、確定した bounded 劣化）**

| torn 結果 | `evictLRU` での挙動 | 影響範囲 |
| --- | --- | --- |
| `> 0` かつ `1..INT64_MAX` | valid timestamp として比較される | eviction 順序が意図と前後し得る（**LRU hint の劣化のみ**） |
| `== 0` | invalid → LAST に配置 | 当該 artifact がその LRU 実行では eviction されない（**容量管理 hint の劣化のみ**） |
| `> INT64_MAX` | invalid → LAST に配置 | 同上 |

**いずれも payload corruption ではなく、identity 亦くは canonicality の破壊でもない。** したがって Owner contract（logical result を `false` にしない）と矛盾しない。設計は **8 byte を 1 回の `WriteFile` で書き、戻り != 8 を失敗として記録する**ことで、torn を最小化しつつ contract を維持する。**残余曖昧性 0。**

### 4.5 cross-process 干渉

`FILE_SHARE_WRITE` を許すため、他 process は in-place write が可能（`ReplaceFile` / `DeleteFile` は `DELETE` access を要求し `FILE_SHARE_DELETE` 禁止のため OS 拒否）。他 process の in-place write は timestamp の lost update にとどまり、bounded。cross-process synchronization は v3.78.62 §13 で**契約外**と確定済み。**残余曖昧性 0。**

### 4.6 RAII abstraction

```cpp
namespace convo { namespace detail {

class UniqueWinHandle final
{
public:
    UniqueWinHandle() = default;
    explicit UniqueWinHandle (HANDLE h) noexcept : handle_ (h) {}
    ~UniqueWinHandle() noexcept { close(); }

    UniqueWinHandle (const UniqueWinHandle&) = delete;
    UniqueWinHandle& operator= (const UniqueWinHandle&) = delete;
    UniqueWinHandle (UniqueWinHandle&&) noexcept;
    UniqueWinHandle& operator= (UniqueWinHandle&&) noexcept;

    HANDLE get() const noexcept { return handle_; }
    explicit operator bool() const noexcept { return handle_ != INVALID_HANDLE_VALUE; }
    void close() noexcept;

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};

enum class ReadOutcome { ok, openFailed, sizeQueryFailed, shortRead };
ReadOutcome readExact (HANDLE h, uint64_t offset, void* dst, uint64_t bytes) noexcept;

enum class WriteOutcome { ok, seekFailed, shortWrite };
WriteOutcome writeExact (HANDLE h, uint64_t offset, const void* src, uint64_t bytes) noexcept;

bool flush (HANDLE h) noexcept;
bool querySizeEx (HANDLE h, std::uint64_t& out) noexcept;

}} // namespace convo::detail
```

`UniqueWinHandle` は move-only。所有権は `HANDLE` 値 1 個のみ。**`load()` / `touch()` は RAII のスコープ结束时で必ず `CloseHandle` する**（current source のように handle を跨ぎ越す経路を作らない）。

---

## 5. mutex boundary の設計

### 5.1 対象と非対象

```
mutex 対象（filesystem に触れる public API）:
    load, save, touch, evictLRU, remove, clear, getEntryCount

mutex 非対象（pure path construction）:
    getCacheDirectory(), getCacheFile(valid), getCacheFile(invalid) -> empty File
```

`getCacheDirectory()` は filesystem forcing を持たない。directory の存在確認・生成は **private `ensureCacheDirectoryUnlocked()`** に集約し、public 経路からは `pure path construction -> lock -> (必要なら) ensure -> filesystem 操作` の順に並べる。

### 5.2 public → public nested locking の禁止と分解

**禁止される現在の構造（Owner authority の `MixedPhasePersistentCache.cpp`）**

```
:141  load()  -> touch()          public -> public
:277  evictLRU() -> clear()        public -> public
```

**分解後の構造**

```cpp
// ============ public: pure path construction（mutex 対象外） ============
static juce::File getCacheDirectory();                                  // pure
static juce::File getCacheFile (const Identity& id);                    // pure / invalid -> empty File
static juce::String canonicalFileName (const Identity& id);              // pure / invalid -> empty String

// ============ public: filesystem-touching（1 個の mutex を取得） ============
static bool   load          (const Identity&, LoadOutputs&);            // { lock; loadUnlocked(); }
static bool   save          (const Identity&, const SaveInputs&);       // { lock; saveUnlocked(); }
static void   touch         (const Identity&);                         // { lock; touchUnlocked(); }
static void   evictLRU      (std::size_t maxCount);                     // { lock; evictLRUUnlocked(); }
static void   remove        (const Identity&);                         // { lock; removeUnlocked(); }
static void   clear         ();                                        // { lock; deleteAllCurrentCanonicalUnlocked(); }
static std::size_t getEntryCount();                                     // { lock; countCurrentCanonicalUnlocked(); }

// ============ private: unlocked helpers（呼び出し側が lock を保持） ============
static bool   loadUnlocked           (const Identity&, LoadOutputs&);
static bool   saveUnlocked           (const Identity&, const SaveInputs&);
static void   touchUnlocked          (const Identity&);
static void   touchOnHandleUnlocked  (UniqueWinHandle&, const DiskHeader& /*shape only*/);
static void   evictLRUUnlocked       (std::size_t maxCount);
static void   removeUnlocked         (const Identity&);
static void   deleteAllCurrentCanonicalUnlocked();
static std::size_t countCurrentCanonicalUnlocked();
static juce::Result ensureCacheDirectoryUnlocked();
```

**nested locking の禁止が構造的に担保される理由**

| 呼び出し | 経路 | public → public か |
| --- | --- | --- |
| `load` の timestamp 更新 | `loadUnlocked` → `touchOnHandleUnlocked`（既に開いた HANDLE を再利用） | **private → private** |
| `load` からの standalone touch 呼出し | **廃止**。`load` は自分の HANDLE 上でのみ touch する | 該当なし |
| `evictLRU(0)` | `evictLRUUnlocked` は `if (maxCount == 0) return;`（`deleteAllCurrentCanonicalUnlocked` を**呼ばない**） | 該当なし |
| `clear` | `clear` → `deleteAllCurrentCanonicalUnlocked` | **public → private** |
| `evictLRU` の delete | `evictLRUUnlocked` が自前で `deleteCurrentCanonicalUnlocked` を呼ぶ（`clear` を経由しない） | **private → private** |
| `save` の rollback | `saveUnlocked` → `rollbackFromBackupUnlocked` | **private → private** |

**`load` が `touch` を public 経由で呼ばないことの帰結**: v3.78.63 §11.1 で確定した「`load()` が毎回 `touch()` を呼ぶためファイルサイズが `2^n` 倍に増える」破損は、同一 HANDLE への 8 byte in-place write に置換することで**構造的に消滅**する。

### 5.3 mutex の保持範囲と最大 blocking duration

| API | lock 保持範囲 | 含まれる I/O |
| --- | --- | --- |
| `getCacheDirectory` / `getCacheFile` | **lock しない** | なし |
| `load` | `ensure` → `Open` → `GetFileSizeEx` → `ReadFile(header)` → validation → `ReadFile(payload)` → `SetFilePointerEx(48)` → `WriteFile(8)` → `FlushFileBuffers` → `CloseHandle` | 1 file read（≤ 134,218,124 B）+ 8 B write + flush |
| `save` | `ensure` → staging `CreateFileW(CREATE_NEW)` → write ≤134,218,124 B → `FlushFileBuffers` → `CloseHandle` → target 分類 → `ReplaceFileW` / `MoveFileExW` → rollback **または** backup delete | 1 file write（最大）+ 1 metadata op |
| `touch` | `ensure` → `Open` → `GetFileSizeEx` → `ReadFile(header)` → validation → seek/write 8 B → `FlushFileBuffers` → `CloseHandle` | 8 B write + flush |
| `evictLRU` | `ensure` → directory 列挙 → 候補 header 読込（N 件） → 並べ替え → delete（M 件） | N × 76 B read + M delete |
| `remove` | `ensure` → 分類 → delete（1 回） | 76 B read + 1 delete |
| `clear` | `ensure` → directory 列挙 → 分類 → delete | N × 76 B read + N delete |
| `getEntryCount` | `ensure` → directory 列挙 → 分類 → count | N × 76 B read |

**最大 blocking duration**

```
人工的な下限 = 0
```

JUCE `TemporaryFile::overwriteTargetFileWithTemporary()` の `5 × Thread::sleep(100)` と `~TemporaryFile::deleteTemporaryFile()` の `5 × Thread::sleep(50)` は**設計から排除**する。v3.78.63 §13 で実測した **約 600–750 ms の retry 下限は消滅**する。

残る blocking は **I/O bound** である。数値上界は以下のように規定する。

| ケース | 上界を構成する要素 |
| --- | --- |
| `load` / `touch` | 1 file read ≤ `kMaxPersistentCacheFileBytes`（§8） |
| `save` | 1 file write ≤ `kMaxPersistentCacheFileBytes` + 1 metadata operation |
| `evictLRU` / `clear` | (`kMaxPersistentCacheEntries` + 1) × 76 B read + `kMaxPersistentCacheEntries` × delete |
| `getEntryCount` | (`kMaxPersistentCacheEntries` + 1) 程度 × 76 B read |

**wall-clock 数値上界は source から導出できない**（デバイス性能依存）。これは契約の曖昧性ではなく**測定項目**であり、target device での測定により確定させる。**残余曖昧性 0（契約として閉じ、測定は実装後の受入条件）**。

### 5.4 ISR 適合

`doc/Practical Stable ISR Bridge Runtime.md` の原則に対する位置づけ（v3.78.63 §13 の caller graph を継承）。

| 観点 | 判定 |
| --- | --- |
| Audio Thread 到達 | **なし**（T1 = `LoaderThread`, T2 = `rebuildThread` のみ） |
| RT lock | なし（Audio Thread が mutex に触れない） |
| RT 待機 | なし（sleep は非 RT 側のみ、設計から排除） |
| RT 判断 | なし |
| verify empty | `convertToMixedPhaseAllpass` の `catch(...)` → `setMixedPhaseState(0)` + rethrow（`MixedPhase.cpp:713-718`） |
| mutex の再入可能性 | process-local 1 個。non-recursive。所有者是 **非 RT の 2 スレッドのみ** |

**適合。**

---

## 6. touch の全面再設計

### 6.1 棄却する現行方式

```
:242-243  // read-modify-write: ファイル全体を読み込み、ヘッダのタイムスタンプのみ更新して書き戻す
          // (FileOutputStream で開くとファイルが切り詰められるため、read-modify-write で行う)
:244-257  buffer.resize(fileSize); inStream.read(buffer.data(), buffer.size());
:260-265  lastUsedTimeOffset = offsetof(DiskHeader, lastUsedTime); memcpy(&buffer[off], &now, 8);
:267      juce::FileOutputStream outStream(file);
:270      outStream.write(buffer.data(), buffer.size());
```

**棄却根拠**

1. source コメントが **JUCE 8.0.12 に対して誤り**。`juce_Files_windows.cpp:429-450` `FileOutputStream::openHandle` は `OPEN_ALWAYS` + `SetFilePointer(FILE_END)` で**必ず追記**する。
2. `truncate()` は `SetEndOfFile` を呼ぶが `juce_FileOutputStream.h:105-109`「Attempts to truncate the file to the current write position. To truncate a file to a specific size, first use setPosition() to seek to the appropriate location, and then call this method.」のとおり**明示呼び出し時のみ**。
3. したがって `write()` は追記となり、ファイルサイズが **2 倍**になる。
4. `load()` は `:141` で毎回 `touch()` を呼ぶため、**n 回の load で `2^n` 倍**に増加する（v3.78.63 §11.1 で確定）。
5. `header.totalFileSize` は元のままなので、v3 canonical validation（`actual file size == expected`）で**恒久的に malformed** 化するが、`load()` は offset 0 から読むため**破損後も成功し静かに進行**する。

### 6.2 採用方式

```cpp
// private touchOnHandleUnlocked — 既存の HANDLE を再利用し、header の 8 byte だけを書き換える
void MixedPhasePersistentCache::touchOnHandleUnlocked (detail::UniqueWinHandle& h,
                                                       std::int64_t         expectedSize) noexcept
{
    // 前提: h は canonical validation を current と判定した後の GENERIC_READ|GENERIC_WRITE HANDLE。
    // 書き換えは lastUsedTime (offset 48) の 8 byte のみ。他 field には触れない。

    const auto nowMs = juce::Time::currentTimeMillis();      // int64 epoch millis
    const std::uint64_t stamp =
        (nowMs >= 1 && nowMs <= INT64_MAX) ? static_cast<std::uint64_t>(nowMs) : std::uint64_t { 0 };

    const std::uint64_t offset = offsetof (DiskHeader, lastUsedTime);
    if (offset + sizeof (std::uint64_t) > static_cast<std::uint64_t> (expectedSize))
        return;                                              // 構造的に到達しない（validator が保証）

    if (detail::writeExact (h.get(), offset, &stamp, sizeof (stamp)) != detail::WriteOutcome::ok)
    {
        // Owner contract: logical load result は false に戻さない。LRU hint の劣化のみ。
        juce::Logger::writeToLog ("MixedPhasePersistentCache: lastUsedTime write failed");
        return;                                              // FlushFileBuffers は不要（何も書けていない）
    }

    detail::flush (h.get());                                 // 戻り値は logical result に影響させない
}
```

### 6.3 禁止事項（Owner 指定）

```
× FileOutputStream による EOF append
× whole-file read-modify-write
× setPosition(0) + write による上書き（truncate() を呼ばない限り truncate されない）
× lastUsedTime 以外の byte への書込
× 書込失敗時に load() の logical result を false にする（Owner contract 違反）
```

### 6.4 public `touch()` の経路

```
touch(id):
    lock
    touchUnlocked(id):
        ensureCacheDirectoryUnlocked()                      // 触らないが存在確認はする
        path = purePathFor (id)                            // mutex なしの pure construction を再利用する
        if (! path.existsAsFile()) return                   // target absent -> no-op
        h = CreateFileW (GENERIC_READ|GENERIC_WRITE, FILE_SHARE_READ|FILE_SHARE_WRITE, OPEN_EXISTING)
        if (! h) return
        size = GetFileSizeEx(h)
        if (! size) return                                  // indeterminate -> no-op
        header = readExact(h, 0, sizeof(DiskHeader))
        if (! header) return
        if (classify (path, header, size, id) != current) return   // future/obsolete/malformed/noncanonical/indeterminate -> preserve
        touchOnHandleUnlocked (h, size)                     // 8 byte のみ
        h.close()                                          // RAII
```

**`touch()` は `current` 以外の artifact を一切変更しない**（Owner 契約 §2 の 14 と同旨）。

---

## 7. canonical validator の API 設計

### 7.1 verdict 定義

```cpp
enum class CanonicalVerdict
{
    current,          // current v3 canonical
    future,           // version >= 4
    obsolete,         // version < 3  (0 / 1 / 2 / 読取不能な値を除く)
    malformed,        // ours だが構造的に不正
    noncanonical,     // filename shape 不正（cache artifact ではない）
    indeterminate     // I/O failure / 判定不能
};
```

**行動と verdict の分離**: `current` 以外の全 verdict の行動は **`preserve` で同一**である。verdict の順序は**診断ラベル（label）のみ**を左右し、行動を左右しない。この性質により validator の判定順が行動の誤りを生まない。

### 7.2 Identity 型（filename serialization の単一の出所）

```cpp
struct Identity
{
    std::uint64_t fileHash   { 0 };
    double        sampleRate { 0.0 };
    std::int32_t  phaseMode  { 0 };      // ConvolverProcessor::PhaseMode の生値
    float         freqStartHz { 0.0f };
    float         freqEndHz   { 0.0f };
    std::int32_t  targetLength { 0 };
};

// bit-pattern equality（epsilon 比較を一切使わない）
bool sameIdentity (const Identity& a, const Identity& b) noexcept;
```

`sameIdentity` は `std::memcmp` 相当の byte 比較で、`sampleRate` は `double` の 8 byte、`freqStartHz` / `freqEndHz` は `float` の 4 byte、`fileHash` は `uint64_t` の 8 byte、`phaseMode` / `targetLength` は `int32_t` の 4 byte を**そのまま**比較する。

### 7.3 canonical filename serialization

```
fileHash        16 lowercase hex   (uint64_t, unsigned bit pattern)
sampleRateBits  16 lowercase hex   (double の 8 byte を uint64_t として見た bit pattern)
phaseMode        8 lowercase hex   (uint32_t, 2 の補数表現ではなく "負なら上位 F のまま 8 桁")
freqStartBits    8 lowercase hex   (float の 4 byte を uint32_t として見た bit pattern)
freqEndBits      8 lowercase hex
targetLength     8 lowercase hex   (uint32_t)
合計            64 hex + ".mph" = 68 文字
```

実装要件:

- 専用 fixed-width serializer を使う。`juce::String::toHexString` は `juce_String.cpp:1920-1937 hexToString` の `do {...} while (v != 0)` により**可変幅**（`juce_String.cpp:2625-2633` の JUCE 自身が `toHexString(0x1234abcd) == "1234abcd"` と `toHexString((int64)-1) == "ffffffffffffffff"` を実証）であり**使用禁止**。
- 出力は `[0-9a-f]` のみ（小文字固定）。
- `sampleRate` は `double` の object representation を `std::bit_cast<std::uint64_t>`（C++20）で取り出す。`static_cast` は不可。
- `phaseMode` / `targetLength` は **raw integer を 8 桁にゼロ詰め**する。`phaseMode` は `int32_t` なので `-1` は `ffffffff` になるが、domain validation で排除されるため出力されない。

**逆変換**: filename 64 hex → 6 field を復元し `Identity` を構成する。復元後に §7.4 の domain validation を通す（不正なら `noncanonical`）。

### 7.4 validator アルゴリズム（判定順が仕様）

```
classifyArtifact(path, requested /*optional*/) -> CanonicalReport
```

| # | 検査 | 失敗時の verdict |
| --- | --- | --- |
| 1 | filename が 68 文字 かつ `[0-9a-f]{64}` + `.mph` | `noncanonical` |
| 2 | filename の 64 hex から identity を復元できる | `noncanonical` |
| 3 | 復元 identity が domain を満たす（§2 の 1,2,3,4,5） | `noncanonical` |
| 4 | `GetFileAttributesEx` / `GetFileInformationByHandleEx(FileAttributeTagInfo)` 成功 | `indeterminate` |
| 5 | `FILE_ATTRIBUTE_DIRECTORY` が立っている | `malformed` |
| 6 | reparse tag != 0 | **`indeterminate`**（§8.2 の根拠） |
| 7 | `GetFileSizeEx` 成功 | `indeterminate` |
| 8 | `GENERIC_READ` で `Open` 成功（`FILE_SHARE_READ\|FILE_SHARE_WRITE\|FILE_SHARE_DELETE`, `OPEN_EXISTING`） | `indeterminate` |
| 9 | `ReadFile` で **76 byte ちょうど** 読めた | `malformed` |
| 10 | `header.magic == kMagic` | `malformed` |
| 11 | `header.version >= 4` | **`future`**（以降を検査しない） |
| 12 | `header.version < 3` | **`obsolete`**（以降を検査しない） |
| 13 | `header.headerReserved == 0` | `malformed` |
| 14 | `header.fileHash != 0` | `malformed` |
| 15 | header identity の domain（PhaseMode / SR / F1-F2 / targetLength） | `malformed` |
| 16 | header identity の domain（numChannels / numSamples / numAllpassSections） | `malformed` |
| 17 | **filename identity と header identity が bit-pattern で一致** | `malformed` |
| 18 | `requested` が supplied なら **requested identity と header identity が bit-pattern で一致** | `malformed` |
| 19 | `expectedSize = checkedArithmetic(...)` が overflow しない | `malformed` |
| 20 | `header.totalFileSize == expectedSize` | `malformed` |
| 21 | **actual file size == expectedSize** | `malformed` |
| 22 | `expectedSize <= kMaxPersistentCacheFileBytes` | `malformed` |
| — | すべて通過 | **`current`** |

**判定順の設計根拠**

- **`magic` を `version` より先に検査する**: 他 tool が作ったファイルの `version` field は意味を持たないため、`malformed`（foreign）として扱うのが正しい。
- **`version` を `headerReserved` より先に検査する**: v4 以降は header layout を我々が知らない。**構造を検査せず `future` として返す**ことで、future artifact を誤って `malformed` に倒さない。
- **`headerReserved` を `version == 3` の後に検査する**: `headerReserved` の意味は v3 layout にしか定義されない。
- **`filename identity == header identity` を必須にする**: これが「filename を collision authority とする」契約の担保点。hash filename の現行実装では原理的に検証不能だった（v3.78.63 §9.4）。
- **`requested` 検査は supplied のときだけ**: `evictLRU` / `clear` / `getEntryCount` の directory scan では requested が無いので判定 18 を飛ばす。`load` / `save` / `touch` / `remove` では requested が必ずある。

### 7.5 canonicality と payload validity の分離

```cpp
// canonicality（上記 1..22）
CanonicalVerdict classifyArtifact (const juce::File& path, const Identity* requested) noexcept;

// B: payload structural readability（canonicality が current の後にのみ呼ぶ）
bool readPayloadStructurally (const DiskHeader& h, std::uint64_t availableBytes,
                              juce::AudioBuffer<double>& ir,
                              std::vector<double>& rho, std::vector<double>& theta);

// C: payload semantic validity（optional / 非 canonicality）
struct SemanticIssue { std::uint32_t section; std::uint64_t index; Kind kind; };
std::vector<SemanticIssue> inspectPayloadSemantics (const DiskHeader& h, ...);
```

- **A（canonicality）**: filename / version / magic / headerReserved / identity 三者一致 / shape / size。
- **B（payload structural readability）**: 「期待する byte 数が実在するか」だけ。NaN / Inf / 値域 / checksum は見ない。
- **C（semantic）**: NaN / Inf / `rho` の値域 / `theta` の値域など。**canonicality 判定には一切関与させない**。
- **CRC / checksum は追加しない**（Owner 契約 §20）。したがって **same-size payload corruption の検出は保証しない**既知 limitation として維持する。

### 7.6 `rho.size() == theta.size()` の検査位置

**ファイル側からは検証できない**（ファイルは `numAllpassSections` と `2 * numAllpassSections` 個の double を持ち、`rho` / `theta` の区別を持たない）。したがって:

| 検査 | 位置 |
| --- | --- |
| `rho.size() == theta.size()` | **`save()` の入力 validation**（Owner 契約 §2 の 8） |
| `numAllpassSections` の domain `0..20` | validator 検査 16 |
| `2 * numAllpassSections * sizeof(double)` 分の存在 | validator 検査 19-21（expectedSize に含まれる） |

---

## 8. 定数

### 8.1 Owner-approved constants（採用候補）

```cpp
namespace convo {

constexpr double kPersistentCacheMinSampleRate = 44100.0;
constexpr double kPersistentCacheMaxSampleRate = 768000.0;

constexpr int kMinPersistentCacheEntries = 1;
constexpr int kDefaultPersistentCacheEntries = 8;
constexpr int kMaxPersistentCacheEntries = 64;

constexpr int kMaxPersistentCacheSections = 20;

} // namespace convo
```

SR は **inclusive**（`44100.0 <= SR <= 768000.0`）かつ `finite`。

### 8.2 既存 authority を再利用し、重複定義しない

| 値 | authority |
| --- | --- |
| PhaseMode domain | `ConvolverProcessor.h:117-122` `enum class PhaseMode : int { AsIs=0, Mixed=1, Minimum=2 }` |
| F1 domain | `ConvolverProcessor.h:225-226` `MIXED_F1_MIN_HZ=100.0f` / `MIXED_F1_MAX_HZ=400.0f` |
| F2 domain | `ConvolverProcessor.h:228-229` `MIXED_F2_MIN_HZ=700.0f` / `MIXED_F2_MAX_HZ=1300.0f` |
| F2 >= F1 + 10 Hz | `ConvolverProcessor.Runtime.cpp:1057 / :1090-1091` の `jlimit((std::max)(MIXED_F2_MIN_HZ, clamped + 10.0f), ...)` |
| IR sample 上限 | `ConvolverProcessor.h:250` `MAX_IR_LATENCY = 2097152` |
| channel 上限 | `IRLoadAdmission.h:43` `kMaxIRLoadChannels = 8u` |
| sections 上限の根拠 | `ConvolverProcessor.MixedPhase.cpp:477` `designer_config.numSections = liveReconfigure ? (highRateLive ? 2 : 8) : 20;` |

**PersistentCache 側で duplicate constants を作らない**。`kPersistentCacheMaxSampleRate` に対する **`F2 <= Nyquist`（= `SR/2`）は追加しない**（v3.78.62 で Owner が明示的に「追加しない」と決定）。

### 8.3 derived constant（magic number 禁止）

```cpp
// 134,218,124 を直接書かない。checked arithmetic で導出する。
constexpr std::size_t deriveMaxPersistentCacheFileBytes() noexcept
{
    constexpr std::size_t headerBytes = 76;                                  // = sizeof(DiskHeader)
    constexpr std::size_t channelBound  = convo::irload::kMaxIRLoadChannels;   // 8
    constexpr std::size_t sampleBound   = static_cast<std::size_t> (ConvolverProcessor::MAX_IR_LATENCY);
    constexpr std::size_t sectionBound  = 20;

    // 段階ごとに上限を確認し、multiplication overflow を早期検出する。
    std::size_t total = headerBytes;
    total = checkedAdd (total, checkedMul (checkedMul (channelBound, sampleBound), sizeof (double)));
    total = checkedAdd (total, checkedMul (sectionBound, sizeof (double)));   // rho
    total = checkedAdd (total, checkedMul (sectionBound, sizeof (double)));   // theta
    return total;
}

constexpr std::size_t kMaxPersistentCacheFileBytes = deriveMaxPersistentCacheFileBytes();
```

導出値:

```
76
+ 8 * 2,097,152 * 8  = 134,217,728
+ 20 * 8             = 160
+ 20 * 8             = 160
= 134,218,124 bytes
```

**representability の確認対象**

| 型 | 確認 |
| --- | --- |
| `std::size_t`（Windows x64 = 64 bit） | `134,218,124 << 2^63` ⇒ **representable**。ただし `checkedAdd` / `checkedMul` で全段階を検査する |
| `std::uint64_t` | 同上 |
| `std::int64_t` | 同上（`header.totalFileSize` は `int64_t` として保存されるため **符号ビットをまたがない**） |
| `std::uint32_t` / `std::int32_t` | **非 representable**。`totalFileSize` は `int64_t` のままであり `int32_t` へ narrowing してはならない |
| `DWORD`（`WriteFile` の `nNumberOfBytesToWrite`） | `134,218,124 < 2^32` ⇒ representable。ただし API は `size_t` を取るため明示的に確認する |

### 8.4 DiskHeader（v3）

```cpp
static constexpr uint64_t kMagic   = 0x4D69786564506800ULL;
static constexpr uint32_t kVersion = 3;

#pragma pack(push, 1)
struct DiskHeader
{
    std::uint64_t magic;             // offset  0
    std::uint32_t version;           // offset  8
    std::uint32_t headerReserved;    // offset 12   runtime: == 0 必須
    std::uint64_t fileHash;          // offset 16
    double        sampleRate;        // offset 24
    std::int32_t  phaseMode;         // offset 32
    float         freqStartHz;       // offset 36
    float         freqEndHz;         // offset 40
    std::int32_t  targetLength;      // offset 44
    std::uint64_t lastUsedTime;      // offset 48
    std::int32_t  numChannels;       // offset 56
    std::int32_t  numSamples;        // offset 60
    std::int32_t  numAllpassSections;// offset 64
    std::int64_t  totalFileSize;     // offset 68
};
#pragma pack(pop)

static_assert(sizeof (DiskHeader) == 76);
static_assert(offsetof (DiskHeader, lastUsedTime)  == 48);
static_assert(offsetof (DiskHeader, totalFileSize) == 68);
static_assert(std::is_trivially_copyable_v<DiskHeader>);
```

**訂正（v3.78.63 §10.2 の発見）**: 現行 v2 layout でも `lastUsedTime == 48` / `totalFileSize == 68` / `sizeof == 76` は**既に成立**している。追加で必要なのは (a) `static_assert` の表明（現行 `:87` は `<= 128`）、(b) `is_trivially_copyable_v` の static_assert、(c) runtime `headerReserved == 0` 検証、(d) `kVersion = 2 -> 3`。

version lifecycle:

```
2      -> obsolete   preserve / reject / no auto-conversion / no auto-delete
3      -> current
>= 4   -> future     preserve / reject / no auto-conversion / no auto-delete
```

---

## 9. maxCacheEntries の ValueTree 境界

### 9.1 禁止事項

```
x  juce::var -> size_t の直接 narrowing
x  static_cast<size_t>(static_cast<int>(v.getProperty("maxCacheEntries")))
```

現行（`ConvolverProcessor.StateAndUI.cpp:382`）は `int -> size_t` の無検査変換により negative が `2^64` 相当に wrap し、`jlimit<size_t>(1, 64, ·)` で **64** になる（v3.78.63 §7.4 で実測）。

### 9.2 仕様（順序と写像）

```
type classification -> finite -> integral -> range classification -> narrowing
```

| 入力 | 期待 |
| --- | --- |
| non-numeric（void / undefined / bool / String / Array / Binary / Object / Method） | `8` |
| NaN | `8` |
| +Inf / -Inf | `8` |
| fractional | `8` |
| integer < 1 | `8` |
| `1 .. 64` | unchanged |
| `> 64` | `64` |

### 9.3 実装案

```cpp
namespace convo {

static int normalizeMaxCacheEntriesFromDouble (double d) noexcept
{
    if (! std::isfinite (d))                       return kDefaultPersistentCacheEntries;  // NaN / +-Inf
    if (std::trunc (d) != d)                       return kDefaultPersistentCacheEntries;  // fractional
    if (d < static_cast<double> (kMinPersistentCacheEntries)) return kDefaultPersistentCacheEntries;
    if (d > static_cast<double> (kMaxPersistentCacheEntries)) return kMaxPersistentCacheEntries;
    return static_cast<int> (d);                    // この時点で 1..64 が保証される narrowing
}

static int normalizeMaxCacheEntries (const juce::var& v) noexcept
{
    // (1) type classification
    if (v.isDouble())  return normalizeMaxCacheEntriesFromDouble (static_cast<double> (v));

    if (v.isInt())
    {
        const auto raw = static_cast<int> (v);
        if (raw < kMinPersistentCacheEntries) return kDefaultPersistentCacheEntries;
        if (raw > kMaxPersistentCacheEntries) return kMaxPersistentCacheEntries;
        return raw;                                 // (4) narrowing は range check 後
    }

    if (v.isInt64())
    {
        const auto raw = static_cast<std::int64_t> (v);
        if (raw < static_cast<std::int64_t> (kMinPersistentCacheEntries)) return kDefaultPersistentCacheEntries;
        if (raw > static_cast<std::int64_t> (kMaxPersistentCacheEntries)) return kMaxPersistentCacheEntries;
        return static_cast<int> (raw);              // range check 後なので narrowing は安全
    }

    return kDefaultPersistentCacheEntries;          // bool / String / void / undefined / その他
}

// 呼び出し側
if (v.hasProperty ("maxCacheEntries"))
    setMaxCacheEntries (static_cast<std::size_t> (normalizeMaxCacheEntries (v.getProperty ("maxCacheEntries"))));
```

`juce::var` の全変換経路を source で閉じた根拠（`JUCE/modules/juce_core/containers/juce_Variant.cpp`）:

```
:553 isVoid   :555 isInt    :556 isInt64   :557 isBool   :558 isDouble   :559 isString
:80  (*toInt) の default = defaultToInt
:96  defaultToInt  -> 0                     （void / undefined / array / binary / object / method）
:146 intToInt      -> data.intValue
:179 int64ToInt    -> (int) data.int64Value  （unchecked narrowing — 本設計では int64 経路を独立に処理して回避）
:212 doubleToInt   -> (int) data.doubleValue（unchecked — 本設計では isDouble() で先に分岐して回避）
:242 boolToInt     -> 0 or 1                （bool は non-numeric として 8 に写像）
:274 stringToInt   -> getString(data)->getIntValue()（"abc" は 0 → 本設計では String を non-numeric として先に弾く）
```

**本設計は `v.isDouble()` / `v.isInt()` / `v.isInt64()` で型分類してから、それぞれ専用の handler へ渡す**ため、`doubleToInt` / `int64ToInt` / `stringToInt` を**一切経由しない**。`:212` の unchecked narrowing が回避される。

### 9.4 `getMaxCacheEntries()` 側

`ConvolverProcessor.LoadPipeline.cpp:182-196` は `setMaxCacheEntries(size_t)` → `jlimit<size_t>(1, 64, maxEntries)` → `static_cast<int>(clamped)`。normalization 済み `int`（`1..64`）が `size_t` に渡るため、**この段階は narrowing 的に安全**。`jlimit` の下限 `1` は `kMinPersistentCacheEntries` と一致し、上限 `64` は `kMaxPersistentCacheEntries` と一致する。**既存の `jlimit<size_t>(1, 64, ·)` は正しいため、`kMinPersistentCacheEntries` / `kMaxPersistentCacheEntries` の参照に置き換えるだけでよい**（Owner 契約 §8 の duplicate 禁止に従い、名前付き定数に昇格する）。

---

## 10. Save collision state machine

### 10.1 state machine

```
                         ┌──────────────────────────────────────────┐
                         │  save(identity, ir, rho, theta)          │
                         └────────────────────┬─────────────────────┘
                                              ▼
                    ┌───────────────────────────────────────────┐
              S0    │  (1) input validation                     │
                    │  fileHash != 0 / SR finite & in range     │
                    │  PhaseMode domain / F1-F2 domain          │
                    │  targetLength 1..MAX_IR_LATENCY            │
                    │  numChannels 1..kMaxIRLoadChannels          │
                    │  numSamples 1..MAX_IR_LATENCY              │
                    │  rho.size() == theta.size()                │
                    │  sections 0..20 / sections == rho.size()   │
                    └───────────────┬───────────────────────────┘
                     fail │         pass
                          ▼          ▼
                    ┌─────────┐   ┌───────────────────────────────────────────┐
                    │ return  │   │  (2) pure path construction              │
                    │  false  │   │  canonical = 68 hex + ".mph"              │
                    │ (no I/O)│   │  staging  = stagingDir/<canonical>.stage  │
                    └─────────┘   │  backup   = stagingDir/<canonical>.bak    │
                                └──┬───────────────────────┬──────────────────┘
                     ensure dir fail│                       │ok
                                  ▼                       ▼
                    ┌──────────────────┐   ┌──────────────────────────────────────┐
                    │ return false     │   │  (3) target classification            │
                    │ (no mutation)    │   │  classifyArtifact(canonical,          │
                    └──────────────────┘   │                 /*requested=*/&id)  │
                                           └───────┬──────────────┬───────────────┘
                                     current        │              │ future / obsolete /
                        ┌──────────────────────────┘              │ malformed / noncanonical /
                        ▼                                         │ indeterminate
    ┌──────────────────────────────┐                              ▼
    │ (4a) target absent           │              ┌──────────────────────────────┐
    │  CREATE_NEW staging          │              │ preserve + return false      │
    │  write -> FlushFileBuffers   │              │ staging は作らない / 触らない  │
    │  MoveFileExW(stage->canonical│              │ target は 1 byte も変更しない   │
    │        MOVEFILE_WRITE_THROUGH)│              └──────────────────────────────┘
    │  fail -> staging delete       │
    │        return false           │
    └──────────────┬────────────────┘
                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ (4b) target exists & current v3 canonical                    │
    │  CREATE_NEW staging (backup パスは未作成)                     │
    │  write -> FlushFileBuffers -> CloseHandle(staging)           │
    │  ReplaceFileW(canonical, staging, backup,                    │
    │               REPLACEFILE_IGNORE_MERGE_ERRORS                 │
    │             | REPLACEFILE_IGNORE_ACL_ERRORS, NULL, NULL)      │
    │      success  -> DeleteFileW(backup)  -> return true         │
    │      failure  -> GetLastError() 分類 -> rollback             │
    └──────────────────────────────────────────────────────────────┘
```

### 10.2 failure 分類と rollback

| `GetLastError()` | rollback | 最終結果 |
| --- | --- | --- |
| `1176 ERROR_UNABLE_TO_MOVE_REPLACEMENT` | backup 指定により「replaced and replacement files retain their original file names」。target は旧 artifact のまま。staging を `DeleteFileW`、**backup は存在しないので触らない** | `false` / target は **intact** |
| `1177 ERROR_UNABLE_TO_MOVE_REPLACEMENT_2` | 「If `lpBackupFileName` is specified, it will be the name of the replaced file」⇒ **backup に旧 artifact**。`MoveFileExW(backup -> canonical, MOVEFILE_REPLACE_EXISTING \| MOVEFILE_WRITE_THROUGH)` | rollback 成功 -> `false` / target は **intact**。rollback 失敗 -> `false` + **`backup` を削除しない**（旧 artifact は backup に存在） |
| `1175 ERROR_UNABLE_TO_REMOVE_REPLACED` | 両者とも元の名前を維持。target は旧 artifact のまま。staging を `DeleteFileW` | `false` / target は **intact** |
| `ERROR_ACCESS_DENIED` / `ERROR_SHARING_VIOLATION` | target は無変更。staging を `DeleteFileW`（失敗しても **target は無変更**） | `false` / target は **intact** |
| その他（`ERROR_INVALID_PARAMETER` ほか） | 「the replaced and replacement files will retain their original file names」。target は旧 artifact のまま | `false` / target は **intact** |

**不変条件**: `ReplaceFileW` が失敗した場合、**旧 artifact のバイト列は必ず canonical か backup のどちらかに存在する**。`save()` は canonical 以外を削除しないため、artifact が失われることはない。

**rollback failure の扱い**: `backup` を**削除しない**。canonical が存在しなければ canonical は「absent」判定となり、次回 `save()` は `target absent` 分岐（4a）で新規作成する。旧 artifact は `backup` に**preserve** される。これは §12 の D-1 決定事項の対象。

### 10.3 「確認できない場合は安全側で preserve」の明示

```
cannot prove current-v3 canonical
    -> do not overwrite
    -> do not delete
    -> return false
```

以下のすべてが `indeterminate` に落ち、`save()` は `false` を返し **canonical target に 1 byte も書き込まない**:

| 状況 | 分類 |
| --- | --- |
| `GetFileAttributesEx` 失敗 | `indeterminate` |
| reparse tag != 0 | `indeterminate` |
| `GetFileSizeEx` 失敗 | `indeterminate` |
| classification 用 `Open` 失敗（timeout / sharing violation / access denied） | `indeterminate` |
| `ReadFile(header)` が 76 byte 未満 | `malformed` |
| `ReplaceFileW` が失敗 | `false`（target は §10.2 の不変条件により intact） |

**destructive operation の安全側デフォルトは常に `preserve`**。

### 10.4 caller 側の変更（scope 通知）

Owner 契約 §2 の 16/17（`save` false → eviction しない / `eviction failure` は `save` success を rollback しない）は、`MixedPhasePersistentCache` 内部ではなく **呼び出し側** `ConvolverProcessor.MixedPhase.cpp:695-701` で修正が必要である。

```
現状:
    convo::MixedPhasePersistentCache::save (...);        // 戻り値捨て
    convo::MixedPhasePersistentCache::evictLRU (owner->getMaxCacheEntries());

設計後:
    if (convo::MixedPhasePersistentCache::save (...))
        convo::MixedPhasePersistentCache::evictLRU (owner->getMaxCacheEntries());
```

**これは `MixedPhasePersistentCache` 以外（`ConvolverProcessor.MixedPhase.cpp`）の変更を伴う。** Owner の明示承認が必要（§13 の D-2）。

### 10.5 eviction / clear / remove / evictLRU(0)

```
evictLRU(maxCount):
    lock
    evictLRUUnlocked(maxCount):
        if (maxCount == 0) return;                          // no-op（clear を呼ばない）
        if (maxCount > kMaxPersistentCacheEntries) maxCount = kMaxPersistentCacheEntries;
        candidates = []
        for each *.mph under cacheDir:
            r = classifyArtifact (path, /*requested=*/nullptr)
            if (r.verdict != current) continue;              // future/obsolete/malformed/noncanonical/indeterminate は候補外
            candidates.push_back ({ r.header.lastUsedTime, r.path })
        if (candidates.size() <= maxCount) return;
        sort by (validTimestamp ASC, invalidTimestamp LAST, canonicalFilename ASC)
        delete candidates[0 .. (size - maxCount)]           // delete attempt のみ

clear():
    lock
    deleteAllCurrentCanonicalUnlocked():                     // clear() と同じ関数を public clear() が直接呼ぶ
        for each *.mph under cacheDir:
            if (classifyArtifact (path, nullptr) == current)
                path.deleteFile()                           // delete attempt のみ

remove(id):
    lock
    removeUnlocked(id):
        if (! domainValid(id)) return;                       // invalid identity -> no-op、filesystem access なし
        p = purePathFor(id)
        if (! p.existsAsFile()) return;                     // target absent -> no-op
        if (classifyArtifact (p, &id) != current) return;   // future/obsolete/malformed/noncanonical/indeterminate -> preserve
        p.deleteFile()

getEntryCount():
    lock
    countCurrentCanonicalUnlocked():
        n = 0
        for each *.mph under cacheDir:
            if (classifyArtifact (path, nullptr) == current) ++n
        return n
```

- **`remove()` は directory scan を行わない**。requested identity から導出した canonical target のみを対象とする。
- **`evictLRU(0)` は `clear()` を呼ばない**（`return` のみ）。`clear()` と `deleteAllCurrentCanonicalUnlocked()` は同一実装だが、**`evictLRUUnlocked` は `clear()` も `deleteAllCurrentCanonicalUnlocked()` も呼ばない**。
- `evictLRU` の sort key は v3.78.63 §11.1 の契約に従う:

```
key = ( timestamp == 0 || timestamp > INT64_MAX ? 1 : 0,     // 0 = valid, 1 = invalid
        timestamp,                                            // valid のとき ASC
        canonicalFilename )                                   // tie-break ASC
```

---

## 11. L の status 分類の修正

### 11.1 分離の根拠

v3.78.63 は **caller graph（記述的項目）** と **mutex 未実装（実装欠落）** を 1 つの `L` 行に同居させていた。v3.78.63 §7.1 の「call graph 自体は CLOSED、VIOLATION は mutex 未実装が根拠」という記述も、1 つの status 列に 2 種類の意味を詰めていたため、**分離する**。

### 11.2 修正後

| Item | 内容 | Contract | Evidence | Implementation | Residual |
| --- | --- | --- | --- | --- | --- |
| **L** | 9 public API の全 production call site 列挙 | PROVEN | **CLOSED** | **GO**（production code 変更を要求しない） | 0 |
| **J** | single process-local mutex | PROVEN | **CLOSED** | **NOT IMPLEMENTED** | 0 |

`L = GO` の根拠: L の契約は「列挙する」こと自体であり、列挙は完了している。`L` は production code に一切変更を要求しない。`remove` / `clear` / `getEntryCount` / `getCacheDirectory` / `getCacheFile` の production caller 0 件は**契約違反ではなく dead public API** であり、`§5.2` の分解で private helper 化された `deleteAllCurrentCanonicalUnlocked` / `countCurrentCanonicalUnlocked` は `clear()` / `getEntryCount()` から到達する（public caller 0 件は変わらない）。

**mutex 未実装は J に帰属**させる。

---

## 12. Owner  ratification が必要な決定事項

以下 2 件は、Owner の既存契約だけでは一意に定まらない。**他の全設計項目はこの 2 件に依存せず閉じている。**

### D-1: staging / backup artifact の孤児ポリシー

**問題**: 設計は `<cacheDir>/staging/`（同一 volume 内に限定、§8.3 の same-volume 要件を満たす）に `<canonical64hex>.stage` と `<canonical64hex>.bak` を作る。以下の 3 過程で孤児が残る。

```
(a) staging 書込後、ReplaceFileW 前に process crash
(b) ReplaceFileW 成功後、backup delete 前に process crash
(c) rollback の MoveFileExW が失敗した（§10.2）
```

これらは `*.mph` glob の**外**（拡張子 `.stage` / `.bak`）なので、`evictLRU` / `clear` / `getEntryCount` の canonical scan には現れない。v3.78.63 の契約「noncanonical → preserve, never automatically delete」は **cacheDir 内の `.mph` 名前空間**について是如何されたものであり、**staging ディレクトリには及んでいない**。

**選択肢**

| | 内容 | 評価 |
| --- | --- | --- |
| **D-1-a（推奨）** | staging ディレクトリを **本実装が所有する予約名前空間**と宣言し、`save()` の失敗 cleanup、`~UniqueWinHandle` scope、以及次回 `save()` 開始時に **同一 canonical 名に対応する孤児のみ**を削除する。cacheDir 直下の `.mph` には一切触れない | 孤児は無制限に増えない。**cacheDir の canonical 名前空間は完全に不変**。contract 14（noncanonical は cacheDir では preserve）を侵犯しない |
| D-1-b | 孤児をそのまま preserve する | 契約文言的には最も単純だが、**容量が無制限に増加**する |
| D-1-c | staging を別 volume の temp dir に置く | `ReplaceFileW` / `MoveFileExW` の same-volume 要件に **反する**（`MOVEFILE_COPY_ALLOWED` なしの cross-volume rename は失敗）。**不可** |

**推奨**: **D-1-a**。D-1-a を採用する場合、Owner 契約 14 の文言を「**cacheDir 直下の `.mph` 名前空間**に対する規則」として明確化する 1 文の追記が必要。

**残余曖昧性**: 本項目は **Owner ratification 待ち**（他項目に影響しない）。

### D-2: E 修正の実装 scope

**問題**: 契約 16/17 の実装は `ConvolverProcessor.MixedPhase.cpp:695-701` の変更を要する（§10.4）。`MixedPhasePersistentCache` 単体の変更では充足しない。

**選択肢**: D-2-a 承認（`ConvolverProcessor.MixedPhase.cpp` の 2 行変更を scope に含める）/ D-2-b `MixedPhasePersistentCache` 内部で `save()` が成功時のみ `evictLRU` を起動する（外部 caller を変えない。ただし `maxCacheEntries` を cache 側才知道しないため **不可能**）。

**推奨**: **D-2-a**。

**残余曖昧性**: **Owner ratification 待ち**。

---

## 13. 設計 closure の判定

### 13.1 閉鎖した項目

| 領域 | 残余曖昧性 |
| --- | --- |
| C / same-HANDLE load・touch sequence | **0**（§4。10 段の failure mapping を全確定。timestamp / `FlushFileBuffers` / `CloseHandle` failure は Owner contract により logical result を落とさず、帰結は LRU hint の bounded 劣化のみ） |
| F / replacement failure preservation | **0**（§3.2 の MS 実文-derived 不変条件。backup 指定により 1176 の target 消滅経路が消滅。rollback failure でも backup に旧 artifact が残る） |
| J / mutex boundary | **0**（§5。public→public nested は構造的に 0 件。保持範囲と上界構成要素を確定。人工 sleep は排除） |
| touch redesign | **0**（§6。8 byte in-place のみ。他 field 非接触。whole-file RMW と append は禁止） |
| Save collision state machine | **0**（§10。6 分岐全部。indeterminate の全経路を列挙。preserve が安全側デフォルト） |
| canonical validator API | **0**（§7。22 段の判定順。verdict は行動に影響しない性質。payload validity と分離。checksum 不追加） |
| constants | **0**（§8。6 定数 + derived。representability を `size_t` / `uint64_t` / `int64_t` / `DWORD` で確認。`int32_t` は非 representable と明記） |
| maxCacheEntries boundary | **0**（§9。5 段順序。`juce::var` の全変換経路を source で回避） |
| L / caller graph | **0**（§11。`L = GO` / `J = NOT IMPLEMENTED` に分離） |
| ISR 適合 | **0**（§5.4。Audio Thread 到達なし、RT lock / 待機 / 判断なし） |

### 13.2 未閉鎖（Owner ratification 待ち）

| ID | 内容 | 他設計への影響 |
| --- | --- | --- |
| **D-1** | staging / backup 孤児ポリシー | **なし**（§7 / §10 の判定順・state machine は D-1 の選択に依存しない） |
| **D-2** | E 修正の実装 scope（`ConvolverProcessor.MixedPhase.cpp` の変更可否） | **なし**（§10.4 の caller 側設計は確定済み。scope 承認のみ要る） |

### 13.3 Gate

```
Contract Gate                = PASS   （全項目 PROVEN）
Evidence Gate                = PASS   （C / F の JUCE 8.0.12 exact source と Win32 exact semantics を取得済み）
Design Closure Gate          = PASS with 2 ratification items
    C / F / J / Touch / Save-Collision : residual ambiguity = 0
    Owner ratification required          : D-1, D-2（各 1 文の判断）
Implementation status       = NO-GO   （production source は未変更。v3.78.63 のとおり）
Authority reconciliation     = PASS    （Owner 指定 authority に対して byte-identical）
```

### 13.4 停止条件の遵守

```
production source 変更   0      test 変更        0      CMake 変更      0
build                   0      version 変更     0      ConvoPeq 再生成  0
commit                  0      push             0
git reset / clean / rebase / amend / squash / force push   0
```

本成果物は **implementation design closure audit のみ**。production implementation plan は、Owner が D-1 / D-2 を裁定した後にのみ作成する。