# v3.78.63 — PersistentCache Save Collision / Protected Artifact Ownership Closure Audit

- Date: 2026-10-05
- Authority: `ConvoPeq.md`
- Type: read-only audit（production source / test / CMake / build / version / ConvoPeq 再生成 / commit / push すべて 0）

---

## 0. 作業制限と authority の確定

### 0.1 実施した操作

read-only のみ。`src/` `test` `CMakeLists.txt` `build.bat` に一切書き込んでいない。`git status` は `doc/work113/**` の未追跡ファイル群のみ。

### 0.2 authority ファイルの確定

Owner 指示の `ConvoPeq(20261002-142258).md` は、実ファイルシステム上に存在しない。

```
C:\VSC_Project\ConvoPeq\doc\ConvoPeqMD\  に 20261002142258 ディレクトリは存在しない。
最も新しい snapshot は doc\ConvoPeqMD\20261003005500\ConvoPeq.md（5,942,475 bytes / 2026-10-03T00:55:03）。
```

grounding worktree `C:\Users\user\AppData\Local\Temp\opencode\stg11-d1-d18\ConvoPeq.md` は 5,942,475 bytes で同一サイズ・同一 mtime のため、これが effective authority として採用した。

### 0.3 authority と source の同一性（重要）

`ConvoPeq.md` から `### 📄 src\MixedPhasePersistentCache.cpp` / `.h` のコードフェンスを抽出し、worktree / main checkout の実ファイルと byte 比較した。

| 対象 | md 本文 | stg11-d1-d18 | main (bda43034) | 一致 |
| --- | --- | --- | --- | --- |
| `src\MixedPhasePersistentCache.cpp` | 352 行 / sha256 `df0160458cedb4f9…` | 352 行 | 352 行 | **identical** |
| `src\MixedPhasePersistentCache.h` | 97 行 / sha256 `0c3fd1f5861e0da5…` | 97 行 | 97 行 | **identical** |

したがって本監査の全結論は authority と矛盾しない。worktree (HEAD `3bc9c32e`) と main checkout (HEAD `bda43034`) のどちらを根拠にしても同一結果となる。

### 0.4 参照した外部・ライブラリ version

| 対象 | 値 | evidence |
| --- | --- | --- |
| JUCE | **8.0.12** | `JUCE/modules/juce_core/juce_core.h:47` (`version: 8.0.12`) |
| JUCE の取得形態 | vendored（`JUCE/.git` なし） | ディレクトリ実査 |
| C++ standard | **C++20** | `CMakeLists.txt:1152 set(CMAKE_CXX_STANDARD 20)` / `:1153 REQUIRED ON` |
| MixedPhase 分割 | 有効 | `CMakeLists.txt:493 / :1472 / :2111 CONVOPEQ_ENABLE_CONVOLVER_SPLIT_MIXED_PHASE=1` |
| 対象 platform | Windows x64（`/arch:AVX2`, MSVC 19.44+ / icx） | `CMakeLists.txt:2, :7, :13` |

---

## 1. Status 定義（v3.78.62 採用・変更なし）

```
Contract status      = PROVEN | OPEN
Evidence status      = CLOSED | OPEN
Implementation status = GO | NOT IMPLEMENTED | VIOLATION | NO-GO
Residual ambiguity   = 0 | >0
```

`Contract = VIOLATION` は本レポートでは使用しない。「契約違反」は implementation の状態である。

---

## 2. 【新規最重要項目】Save Collision / Protected Artifact Ownership

### 2.1 Claim

requested identity から導出される canonical v3 target に対して、`save()` が既存ファイルを何をするか、が所有権の核心である。

```
Existing target                        Save action
--------------------------------------------------------------------------
absent                                 create
current v3 canonical                   replace allowed（削除ではない）
future (version >= 4)                  MUST NOT overwrite / MUST NOT delete / preserve → save failure
obsolete (version < 3)                 MUST NOT overwrite / MUST NOT delete / preserve → save failure
malformed                              MUST NOT overwrite / MUST NOT delete / preserve → save failure
canonical filename だが current として不正 MUST NOT overwrite / MUST NOT delete / preserve → save failure
indeterminate / I/O error              MUST NOT overwrite / MUST NOT delete / preserve → save failure
```

destructive operation の安全側デフォルトは `preserve`。

### 2.2 Current ConvoPeq source evidence

`src/MixedPhasePersistentCache.cpp:150-225`（`save()`）の全文構造:

```
:159  const auto file = getCacheFile(...);
:161  const auto dir  = file.getParentDirectory();
:162-167  if (!dir.exists()) { dir.createDirectory(); }        ← directory mutation
:169  juce::TemporaryFile tempFile(file);
:171  juce::FileOutputStream stream(tempFile.getFile());
:172-173  if (!stream.openedOk()) return false;
:180-194  header を memset→field 設定（totalFileSize 未設定）
:195-213  header / payload / rho / theta を write
:216-219  header.totalFileSize = stream.getPosition(); setPosition(0); 再 write
:221  stream.flush();
:224  return tempFile.overwriteTargetFileWithTemporary();
```

**既存 target に対する分類処理（read / magic / version / shape / size の確認）は 1 行も存在しない。**
`file.existsAsFile()` も、`header.version != kVersion` も、`totalFileSize` 照合も、`DeleteFile` も現れない。`save()` は `getCacheFile()`（`computeKeyHash()` 由来の hash ファイル名）のパスに対して無条件に `TemporaryFile` → `overwriteTargetFileWithTemporary()` を実行する。

また header には `headerReserved` を検証・設定するコードが存在しない（`:181` の `std::memset(&header, 0, sizeof(header))` により write 時は 0 だが、`load()` 側は `:100-108` で `headerReserved` を読まない）。

### 2.3 External / library evidence

**(a) `TemporaryFile` が生成する temp artifact の形状**

`JUCE/modules/juce_core/files/juce_TemporaryFile.cpp:61-70`

```cpp
TemporaryFile::TemporaryFile (const File& target, const int optionFlags)
    : temporaryFile (createTempFile (target.getParentDirectory(),
                                     target.getFileNameWithoutExtension()
                                       + "_temp" + String::toHexString (Random::getSystemRandom().nextInt()),
                                     target.getFileExtension(), optionFlags)),
      targetFile (target)
```

target が `<64hex>.mph` なら temp は

```
<target dir>/<64hex>_temp<random-hex>.mph
```

**同一ディレクトリ・同一 `.mph` 拡張子**で生成される。

**(b) 現行 `clear()` / `evictLRU()` / `getEntryCount()` は `*.mph` glob で走査する**

```
:282-283  dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");   // evictLRU
:334-335  dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");   // clear
:346-347  dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");   // getEntryCount
```

したがって **in-flight の temp artifact は上記 3 API の走査対象になる**。canonical filename shape（固定幅 64 hex + `.mph`）で判定する validator を置けば temp は `noncanonical` に分類され、protected される。

**(c) 現行 source に mutex が無いこと的后果**

`MixedPhasePersistentCache` は mutex を持たない（`src/MixedPhasePersistentCache.h` に `juce::CriticalSection` / `std::mutex` の宣言なし。`.h:1-96` 全文確認）。§6.5 の caller graph は 2 本の非 RT スレッドが `save()` を実行し得ることを示すため、`clear()` / `evictLRU()` / `getEntryCount()` が `save()` の write window と重なり得る。

### 2.4 Reasoning

`save()` は既存 target の状態を一切観測しないため、次が確定する。

1. **future artifact**（version >= 4 の canonical filename を持つファイル）は、`save()` によって無条件に置換される。Owner contract の「MUST NOT overwrite」に違反。
2. **obsolete artifact**（version 2 = 現行実装が書き込む version）も同様に無条件置換される。v3 binary が v2 artifact を自動置換する形になり、「no auto-conversion」の精神と衝突する。
3. **malformed artifact**（壊れた / 別 tool が作った同じ canonical 名）も無条件置換される。
4. **indeterminate / I/O error** により「current v3 canonical か判定できない」場合でも、判定を行わないので `preserve` に倒れない。

**`deleteFile()` を直接呼んでいなくても `ReplaceFileW` による置換が destructive である**という.Owner の指摘は、§4 の JUCE exact source で裏付け済みである（`File::replaceInternal` は `ReplaceFileW` を呼ぶ。`MixedPhasePersistentCache.cpp` 内の `deleteFile()` 出現は `:313`（evictLRU）と `:325`（remove）と `:337`（clear）のみ）。

### 2.5 Residual ambiguity

0。上記 4 点は current source の行構造と JUCE exact source の両方から一意に導かれる。

### 2.6 Contract status

```
Save Collision / Protected Artifact Ownership
    Contract = PROVEN        （v3.78.63 §3/§4/§9/§10 で Owner が確定）
    Evidence = CLOSED
    Implementation = VIOLATION
    Residual ambiguity = 0
```

### 2.7 Evidence status

CLOSED。JUCE 8.0.12 exact source と current source の双方から判定可能。

### 2.8 Implementation status

**VIOLATION**。`save()` に target classification が存在しない。

---

## 3. C — same-HANDLE semantics

### 3.1 Claim

```
Open H (GENERIC_READ|GENERIC_WRITE, FILE_SHARE_READ|FILE_SHARE_WRITE, OPEN_EXISTING)
→ GetFileSizeEx(H)
→ ReadFile(H, header)
→ Validate
→ ReadFile(H, payload)
→ touchOnHandle(H)
→ FlushFileBuffers(H)
→ CloseHandle(H)
```

`FILE_SHARE_DELETE` は prohibited。

### 3.2 Current ConvoPeq source evidence

`src/MixedPhasePersistentCache.cpp:76-144`（`load()`）:

```
:87-88   if (!file.existsAsFile()) return false;
:90      juce::FileInputStream stream(file);          ← read-only HANDLE 別オブジェクト
:91-92   if (!stream.openedOk()) return false;
:96      stream.read(&header, sizeof(DiskHeader))
:100-108 header 検証
:111-124 outIr.setSize(...) + per-channel stream.read(...)
:127-138 rho / theta の stream.read(...)
:141     touch(fileHash, sampleRate, phaseMode, freqStartHz, freqEndHz, targetLength);
:143     return true;
```

契約シーケンス違反:

| 契約 step | current source |
| --- | --- |
| 単一 HANDLE | 2 個以上の別 HANDLE（`FileInputStream` の H と `touch()` 内の H） |
| `GetFileSizeEx(H)` | 呼ばない |
| `touchOnHandle(H)` | 別 public API `touch()` を `:141` で呼び、**ファイルパスを再度解決して別 HANDLE で開く** |
| `FlushFileBuffers` / `CloseHandle` | 明示なし（stream dtor に委譲） |

また `:113` の `if (numChannels <= 0 || numSamples <= 0) return false;` は `:116` の `outIr.setSize(numChannels, numSamples)` の**前**にあるが、`:111-112` で `static_cast<int>(header.numChannels)` 了我が既に signed overflow している（`header.numChannels` は `int32_t` なので overflow はしないが、上限チェックが無い）。`:120` `static_cast<size_t>(numSamples) * sizeof(double)` は checked arithmetic ではない。

### 3.3 External / library evidence（JUCE 8.0.12 exact source）

**(a) `FileInputStream::openHandle` — `JUCE/modules/juce_core/native/juce_Files_windows.cpp:396-411`**

```cpp
void FileInputStream::openHandle()
{
    auto h = CreateFile (file.getFullPathName().toWideCharPointer(),
                         GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h != INVALID_HANDLE_VALUE) fileHandle = (void*) h;
    else status = WindowsFileHelpers::getResultForLastError();
}

FileInputStream::~FileInputStream() { CloseHandle ((HANDLE) fileHandle); }
```

`readInternal`（`:413-426`）は `ReadFile` を呼ぶ。**書き込みはできない**。`GENERIC_READ` のみ。

**(b) `FileOutputStream::openHandle` — 同 `:429-450`**

```cpp
void FileOutputStream::openHandle()
{
    auto h = CreateFile (file.getFullPathName().toWideCharPointer(),
                         GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                         OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE)
    {
        LARGE_INTEGER li;  li.QuadPart = 0;
        li.LowPart = SetFilePointer (h, 0, &li.HighPart, FILE_END);   // ← 常に EOF
        if (li.LowPart != INVALID_SET_FILE_POINTER) { fileHandle = (void*) h; currentPosition = li.QuadPart; return; }
    }
    status = WindowsFileHelpers::getResultForLastError();
}
```

`GENERIC_WRITE` のみ。**読み込みはできない**。share mode は `FILE_SHARE_READ` のみで `FILE_SHARE_WRITE` / `FILE_SHARE_DELETE` を含まない。`OPEN_ALWAYS` + `FILE_END` により**必ず追記位置で始まる**。

`truncate()`（同 `:475-483`）は `flush()` 後に `SetEndOfFile` を呼ぶが、**明示呼び出し時のみ**発動する。`flush()` だけを呼び出しても切り詰めは起きない。

**(c) JUCE Windows で `GENERIC_READ|GENERIC_WRITE` を開く唯一の箇所 — 同 `:486-511`**

`MemoryMappedFile::openInternal`（`readWrite` 時のみ `accessMode = GENERIC_READ | GENERIC_WRITE; createType = OPEN_ALWAYS;`、share は `FILE_SHARE_READ | FILE_SHARE_DELETE | FILE_SHARE_WRITE`）。ただしこれは `CreateFileMapping` + `MapViewOfFile` を要求し、`GetFileSizeEx` / `FlushFileBuffers` / `ReadFile` のシーケンス API を提供しない。

**(d) ファイルサイズ取得は `GetFileSizeEx` ではない — 同 `:544-552`**

```cpp
int64 File::getSize() const
{
    WIN32_FILE_ATTRIBUTE_DATA attributes;
    if (GetFileAttributesEx (fullPath.toWideCharPointer(), GetFileExInfoStandard, &attributes))
        return (((int64) attributes.nFileSizeHigh) << 32) | attributes.nFileSizeLow;
    return 0;
}
```

契約が要求する「同一 HANDLE 上の `GetFileSizeEx(H)`」は JUCE では表現できない。

**(e) HANDLE ownership / lifetime**

`FileInputStream` は dtor で `CloseHandle`、`FileOutputStream` は `closeHandle()`（同 `:452-455`）で `CloseHandle`。両 handle は private メンバ `detail::NativeFileHandle fileHandle` であり、**公開 accessor はない**（`juce_FileInputStream.h:79-87`, `juce_FileOutputStream.h:103-115` は `openedOk/ getPosition/ setPosition/ read/write/ truncate/ flush` のみ）。

**(f) failure mapping**

`WindowsFileHelpers::getResultForLastError()` が `juce::Result` を生成し、`status` に保持（`getStatus()` / `openedOk()` で取得可能）。内部限定 native HANDLE を用いる場合、同等の `GetLastError()` → 契約 failure 分類を自前で実装する必要がある。

**(g) Windows x64 primitive（Microsoft Learn 実文）**

- `CreateFileW` / `dwShareMode`: 「**FILE_SHARE_DELETE** (0x4) — Enables subsequent open operations on a file or device to request delete access.」／「The sharing options for each open handle remain in effect until that handle is closed, regardless of process context.」／「If this parameter is zero and CreateFile succeeds, the file or device cannot be shared and cannot be opened again until the handle to the file or device is closed.」
- `CreateFileW` / `dwCreationDisposition`: `OPEN_ALWAYS`（4）「Opens a file, always. If the specified file exists, the function succeeds and the last-error code is set to **ERROR_ALREADY_EXISTS**」／`CREATE_ALWAYS`（2）「If the specified file exists and is writable, the function **truncates the file**, the function succeeds, and last-error code is set to ERROR_ALREADY_EXISTS (183)」。
- `ReadFile`：「If a read operation on a file begins at or beyond the end of the file, then the read operation **fails with the error ERROR_HANDLE_EOF**. If a read operation on a file begins before the end of the file, but the read operation extends past the end of the file, then the read operation **succeeds**, and the number of bytes read is the number of bytes that were read before the end of file was reached.」／「lpNumberOfBytesRead ... **ReadFile sets this value to zero before doing any work or error checking.**」
- `SetFilePointerEx`：「The file handle must have been created with the **GENERIC_READ or GENERIC_WRITE** access right.」／`FILE_BEGIN`（0）「The starting point is zero or the beginning of the file.」
- `GetFileSizeEx`：「If the function succeeds, the return value is nonzero. If the function fails, the return value is zero.」
- `FlushFileBuffers`：「The file handle must have the **GENERIC_WRITE** access right.」

### 3.4 Reasoning

Owner が提示した 10 項目の必要証拠はすべて取得済みである。証拠から導かれる結論は 3 点。

1. **JUCE の公開 stream API では契約シーケンスを表現できない。** `FileInputStream` は `GENERIC_READ` のみ、`FileOutputStream` は `GENERIC_WRITE` のみ。単一 HANDLE で header 読み → payload 読み → timestamp 書き杀菌需要一个 HANDLE，`GENERIC_READ|GENERIC_WRITE` で `OPEN_EXISTING` により開く必要がある。JUCE にはそのペアを作る公開 API がない。
2. **`FILE_SHARE_DELETE` を除外すれば、同一パスに対する `DeleteFile` / `MoveFile` / `ReplaceFileW` は OS で拒否される。** これは「load 中の artifact を他スレッドが改変・削除できない」という Owner 意図と整合する。`FileOutputStream` が `FILE_SHARE_READ` しか持たないため、`touch()` が `FileOutputStream` を開いている間、そのファイルへの delete/rename は既に OS 拒否される。
3. **`FlushFileBuffers(H)` を 「失敗しても logical result を false に戻さない」 契約は OS  primitive と矛盾しない。** `FlushFileBuffers` は GENERIC_WRITE を要求し、`GetLastError()` を返すだけである。

**副作用（Owner が明示していないが記録すべき制約）**: `FILE_SHARE_DELETE` を許さない load HANDLE は、その間同一 identity に対する `ReplaceFileW`（save の replacement）を OS が拒否する。§6.5 の caller graph では `load()` と `save()` が **別スレッド**（LoaderThread と rebuildThread）で走る可能性があるため、この拒绝は mutex が無ければ実害に出る。Owner contract の「7 filesystem-touching public API を単一 mutex で直列化する」ことが、本項の前提条件である。

### 3.5 Residual ambiguity

0。JUCE 8.0.12 の exact source と Windows primitive の exact semantics の両方が確定した。

### 3.6 Contract status

```
C  Contract = PROVEN
   Evidence = CLOSED
   Implementation = NO-GO
   Residual ambiguity = 0
```

**Owner の期待表では `Evidence = OPEN` だったが、本版で JUCE 8.0.12 exact source（`juce_Files_windows.cpp` / `juce_FileInputStream.h` / `juce_FileOutputStream.h`）と Microsoft Learn の Win32 primitive 実文を 10 項目すべて取得したため CLOSED に更新する。** CLOSED の内容は「JUCE 公開 API では不可能。内部限定 Windows native HANDLE で実装する」であり、これは Owner の想定（v3.78.62 §7「JUCE wrapper で不足なら内部限定 Windows native HANDLE」）と一致する。

### 3.7 Evidence status

CLOSED。

### 3.8 Implementation status

**NO-GO**。current source は `FileInputStream` + public `touch()` であり、契約シーケンスを満たさない。

---

## 4. F — Replacement Failure Preservation（+ Save Collision の destructive 実体）

### 4.1 Claim

```
existing current-v3 canonical artifact
+ replacement failure
→ existing artifact remains intact
```

### 4.2 Current ConvoPeq source evidence

```
src/MixedPhasePersistentCache.cpp:169  juce::TemporaryFile tempFile(file);
src/MixedPhasePersistentCache.cpp:224  return tempFile.overwriteTargetFileWithTemporary();
```

`save()` は return value 之外の救済処理を持たない（`:224` がそのまま `save()` の return）。`:115-119` が `write()` 失敗時の early return だが、既存 target は既に未変更である。

### 4.3 External / library evidence（JUCE 8.0.12 exact source + Win32 exact semantics）

**(a) 呼び出し連鎖**

```
MixedPhasePersistentCache.cpp:224  tempFile.overwriteTargetFileWithTemporary()
  → JUCE/modules/juce_core/files/juce_TemporaryFile.cpp:94-119
        bool TemporaryFile::overwriteTargetFileWithTemporary() const {
            jassert (targetFile != File());
            if (temporaryFile.exists()) {
                for (int i = 5; --i >= 0;) {
                    if (temporaryFile.replaceFileIn (targetFile)) return true;
                    Thread::sleep (100);
                }
            } else { jassertfalse; }
            return false;
        }
  → JUCE/modules/juce_core/files/juce_File.cpp:323-336
        bool File::replaceFileIn (const File& newFile) const {
            if (newFile.fullPath == fullPath) return true;
            if (! newFile.exists()) return moveFileTo (newFile);
            if (! replaceInternal (newFile)) return false;
            deleteFile();
            return true;
        }
  → JUCE/modules/juce_core/files/juce_File.cpp:300-315
        bool File::moveFileTo (const File& newFile) const {
            if (newFile.fullPath == fullPath) return true;
            if (! exists()) return false;
            if (! newFile.deleteFile()) return false;
            return moveInternal (newFile);
        }
  → JUCE/modules/juce_core/native/juce_Files_windows.cpp:372-378
        bool File::replaceInternal (const File& dest) const {
            return ReplaceFile (dest.getFullPathName().toWideCharPointer(),
                                fullPath.toWideCharPointer(),
                                nullptr,                       // ← lpBackupFileName = nullptr
                                REPLACEFILE_IGNORE_MERGE_ERRORS | 4,   // REPLACEFILE_IGNORE_ACL_ERRORS
                                nullptr, nullptr) != 0;
        }
  → JUCE/modules/juce_core/native/juce_Files_windows.cpp:366-370
        bool File::moveInternal (const File& dest) const {
            return MoveFile (fullPath.toWideCharPointer(), dest.getFullPathName().toWideCharPointer()) != 0;
        }
```

**(b) Win32 `ReplaceFileW` の failure semantics（Microsoft Learn 実文）**

`ReplaceFileW` — `dwReplaceFlags` の値:

| Value | Meaning |
| --- | --- |
| `REPLACEFILE_WRITE_THROUGH` 0x1 | This value is not supported. |
| `REPLACEFILE_IGNORE_MERGE_ERRORS` 0x2 | Ignores errors that occur while merging information (such as attributes and ACLs) from the replaced file to the replacement file. |
| `REPLACEFILE_IGNORE_ACL_ERRORS` 0x4 | Ignores errors that occur while merging ACL information from the replaced file to the replacement file. |

Return value 表（原文）:

| Return code/value | Description |
| --- | --- |
| `ERROR_UNABLE_TO_MOVE_REPLACEMENT` 1176 (0x498) | The replacement file could not be renamed. **If `lpBackupFileName` was specified, the replaced and replacement files retain their original file names. Otherwise, the replaced file no longer exists and the replacement file exists under its original name.** |
| `ERROR_UNABLE_TO_MOVE_REPLACEMENT_2` 1177 (0x499) | The replacement file could not be moved. The replacement file still exists under its original name; however, it has inherited the file streams and attributes from the file it is replacing. **The file to be replaced still exists with a different name.** If `lpBackupFileName` is specified, it will be the name of the replaced file. |
| `ERROR_UNABLE_TO_REMOVE_REPLACED` 1175 (0x497) | The replaced file could not be deleted. The replaced and replacement files retain their original file names. |

> 「If any other error is returned, such as **ERROR_INVALID_PARAMETER**, the replaced and replacement files will retain their original file names. In this scenario, a backup file does not exist and it is not guaranteed that the replacement file will have inherited all of the attributes and streams of the replaced file.」

`lpReplacedFileName` の扱い（原文）:

> 「This file is opened with the **GENERIC_READ, DELETE, and SYNCHRONIZE** access rights. The sharing mode is **FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE**. The caller must have write access to the file to be replaced.」

**(c) 失敗時の後始末**

`File::replaceFileIn`（`juce_File.cpp:331-332`）は `replaceInternal` 失敗時に `return false` するのみで、temporary を削除しない。`TemporaryFile::overwriteTargetFileWithTemporary()`（`:103-109`）は最大 5 回 `Thread::sleep(100)` を挟んで retry し、5 回失敗で `false`。その後 `~TemporaryFile`（`:77-91`）が `deleteTemporaryFile()`（`:121-133`、5 回 × `Thread::sleep(50)` retry）を呼ぶ。

### 4.4 Reasoning

**JUCE 8.0.12 は `ReplaceFileW` に `lpBackupFileName = nullptr` を渡す。**
Microsoft Learn の表が明示しているとおり、`lpBackupFileName` が指定されない場合:

- **1176（`ERROR_UNABLE_TO_MOVE_REPLACEMENT`）**: 「the replaced file **no longer exists**」→ **canonical target の既存 artifact が消える。**
- **1177（`ERROR_UNABLE_TO_MOVE_REPLACEMENT_2`）**: 「The file to be replaced still exists **with a different name**」→ **canonical target の既存 artifact が別名に退避する（canonical path には残らない）。**

したがって **F の契約「replacement failure → existing artifact intact」は JUCE 8.0.12 の `TemporaryFile::overwriteTargetFileWithTemporary()` では充足不可能である。** これは「TemporaryFile だから atomic」という誤った根拠ではなく、**具体的な failure code による反証**である。

さらに、1177 発生時の retry は次の状態で行うことになる。

```
attempt1: ReplaceFileW(target, temp, nullptr, ...) → 1177
         状態: target は別名に変化、temp は元名に存在
attempt2: replaceFileIn → newFile(target) は存在しない → moveFileTo(target)
         → newFile.deleteFile() は !exists() で true
         → MoveFile(temp, target)
```

これは「既存 artifact の保全」ではない。

**結論**: F を満たすには、内部限定 Windows native `ReplaceFileW` を `lpBackupFileName` 明示付きで呼び、失敗時は backup から `MoveFileW` で rollback する（あるいは二段階 commit）設計が必要。JUCE `TemporaryFile` はこの契約の達成手段にできない。

**副次的な帰結（§2 との接続）**: `ReplaceFileW` は `lpReplacedFileName` を `DELETE` access で開く。したがって §3 の load HANDLE が `FILE_SHARE_DELETE` を持たない場合、`save()` の replacement は OS で拒否される。§13 の単一 mutex がこの直列化の前提となる。

### 4.5 Residual ambiguity

0。「JUCE `TemporaryFile` 経路では F を満たせない」ことは `juce_Files_windows.cpp:374-377`（`nullptr` ypass）と Microsoft Learn `ReplaceFileW` Return value 表（1176 / 1177 の `lpBackupFileName` 未指定時記述）から一意に導かれる。

### 4.6 Contract status

```
F  Contract = PROVEN
   Evidence = CLOSED
   Implementation = NOT IMPLEMENTED
   Residual ambiguity = 0
```

### 4.7 Evidence status

CLOSED。**ただし CLOSED の結論は「現行手段（JUCE `TemporaryFile`）では満たせない」である。** implementation plan 側で手段の再設計が必要である点を明記する。

### 4.8 Implementation status

**NOT IMPLEMENTED**。current source は `TemporaryFile::overwriteTargetFileWithTemporary()` を使用しており、F を満たす rollback を一切持たない。

---

## 5. Save Collision と F の分離（§6 の指示への対応）

| | Save Collision / Protected Artifact Ownership | F — Replacement Failure |
| --- | --- | --- |
| 問い | **何を上書きしてよいか**（ownership） | **上書きしてよい existing artifact に対する replacement が失敗したとき、旧 artifact が intact か**（transactional replacement） |
| 分類対象 | 既存 target の canonical 状態（current / future / obsolete / malformed / indeterminate） | replacement 試行の結果（成功 / 失敗コード） |
| 判定場所 | `save()` の入口、directory scan 無し | `ReplaceFileW` / `MoveFileW` の結果ハンドリング |
| current source | **存在しない** → 常に proceed | **存在しない** → `return tempFile.overwriteTargetFileWithTemporary();` |
| Contract | PROVEN | PROVEN |
| Evidence | CLOSED | CLOSED |
| Implementation | **VIOLATION** | **NOT IMPLEMENTED** |

両者は独立項目として扱う。`save()` は Case A（current v3 canonical）で replacement が失敗したことを観測し、Case B〜F では replacement を試行してはならない。current source は Case A の replacement failure 観測も、Case B〜F の classification も持っていない。

---

## 6. L — Complete Caller Graph（9 public API 全 production call site）

### 6.1 検索範囲と方法

```
走査対象: <root>/src/**/*.cpp, *.h, *.hpp   362 files
検索 1  : r"MixedPhasePersistentCache\s*::\s*(getCacheDirectory|getCacheFile|load|save|touch|evictLRU|remove|clear|getEntryCount)\b"
検索 2  : r"MixedPhasePersistentCache"          ← トークン自体の全出現
検索 3  : 各 API 名を無修飾で呼び出す内部 site（MixedPhasePersistentCache.cpp 内の自己呼び出し）
```

`MixedPhasePersistentCache` トークンの全出現は **18 件**。そのうち定義・include・コメントを除くと **3 件のみが呼び出し**である。

```
src/MixedPhasePersistentCache.cpp:1    #include "MixedPhasePersistentCache.h"      ← include
src/MixedPhasePersistentCache.h:13     /* MixedPhasePersistentCache: 最適化済み ... */  ← doc comment
src/MixedPhasePersistentCache.h:17     class MixedPhasePersistentCache            ← 宣言
src/convolver/ConvolverProcessor.MixedPhase.cpp:7   #include "MixedPhasePersistentCache.h"  ← include
src/convolver/ConvolverProcessor.MixedPhase.cpp:124 convo::MixedPhasePersistentCache::load(   ← 呼び出し 1
src/convolver/ConvolverProcessor.MixedPhase.cpp:695 convo::MixedPhasePersistentCache::save(   ← 呼び出し 2
src/convolver/ConvolverProcessor.MixedPhase.cpp:701 convo::MixedPhasePersistentCache::evictLRU(...)  ← 呼び出し 3
src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp:2201  //   MixedPhasePersistentCacheが温まっていれば...  ← コメントのみ
```

### 6.2 Call site 表（Owner 指定の 7 欄）

| API | caller function | caller file:line | caller thread | caller entry point | thread transition | Audio Thread reachability |
| --- | --- | --- | --- | --- | --- | --- |
| `getCacheDirectory` | — | — | — | — | — | **Production caller = none** |
| `getCacheFile` | — | — | — | — | — | **Production caller = none** |
| `load` | `ConvolverProcessor::convertToMixedPhaseAllpass` | `src/convolver/ConvolverProcessor.MixedPhase.cpp:124` | T1 = `LoaderThread`（`juce::Thread`） | T1: `ConvolverProcessor::loadImpulseResponse`（`LoadPipeline.cpp:17-99`）→ `activeLoader->startThread()`（`LoadPipeline.cpp:96`） | Message Thread → LoaderThread（`startThread()` による非同期 hop） | **No** |
| `load` | 同上 | 同上 | T2 = `AudioEngine::rebuildThread`（`std::thread`） | T2: `AudioEngine::rebuildThreadLoop`（`RebuildDispatch.cpp:845`）→ `rebuildAllIRsSynchronous`（`RebuildDispatch.cpp:1269`）→ `LoaderThread::runSynchronously()`（`Rebuild.cpp:95`）→ `performLoad`（`LoaderThread.cpp:314-337`）→ `stepOnce` → `doTransformStep`（`LoaderThread.cpp:822`） | rebuildThread 内で同期（hop なし） | **No** |
| `save` | `ConvolverProcessor::convertToMixedPhaseAllpass` | `MixedPhase.cpp:695` | T1 / T2（`save()` は `optimizedByCmaes` かつ `owner && fileHash != 0` の場合のみ、`MixedPhase.cpp:687`） | 同上 | 同上 | **No** |
| `evictLRU` | 同上 | `MixedPhase.cpp:701` | T1 / T2（`save()` 直後、同じ if-block 内） | 同上 | 同上 | **No** |
| `touch` | — | — | — | — | — | **Production caller = none**（内部呼出しは `MixedPhasePersistentCache.cpp:141` の `load()` 内のみ。`load` の thread を継承） |
| `evictLRU`（内部 `clear()` 呼出し） | `MixedPhasePersistentCache::evictLRU` | `MixedPhasePersistentCache.cpp:277` | `evictLRU` の thread を継承 | — | — | No |
| `remove` | — | — | — | — | — | **Production caller = none** |
| `clear` | — | — | — | — | — | **Production caller = none**（内部呼出しは `MixedPhasePersistentCache.cpp:277` の `evictLRU(0)` のみ） |
| `getEntryCount` | — | — | — | — | — | **Production caller = none** |

### 6.3 T1 / T2 の entry point 確定 evidence

**T1 の生成元**

```
src/convolver/ConvolverProcessor.LoadPipeline.cpp:17   bool ConvolverProcessor::loadImpulseResponse(const juce::File& irFile, bool optimizeForRealTime)
src/convolver/ConvolverProcessor.LoadPipeline.cpp:96       activeLoader->startThread();
src/convolver/ConvolverProcessor.LoaderThread.cpp:15   LoaderThread::LoaderThread(ConvolverProcessor&, const juce::File&, double, int, PhaseMode, ...)   ← juce::Thread 派生
src/convolver/ConvolverProcessor.LoaderThreadInline.h:3   class LoaderThread : public juce::Thread
src/convolver/ConvolverProcessor.LoaderThread.cpp:41   void ConvolverProcessor::LoaderThread::run()
src/convolver/ConvolverProcessor.LoaderThread.cpp:87   LoadResult result = performLoad(this);
```

`loadImpulseResponse` の production 呼び出し元（`src/` 全走査）:

```
src/convolver/ConvolverProcessor.Lifecycle.cpp:219   loadImpulseResponse(irFile, false);   ← ConvolverProcessor::timerCallback()（juce::Timer → Message Thread）
src/convolver/ConvolverProcessor.Rebuild.cpp:17       loadImpulseResponse(juce::File(), false);   ← ConvolverProcessor::rebuildAllIRs()
```

`ConvolverProcessor::timerCallback()` は `ConvolverProcessor.h:708 void timerCallback() override;`（`juce::Timer`）。`juce::Timer` は Message Thread で駆動される。`Lifecycle.cpp:205` のコメント「リングバッファオーバーフローによるリビルド要求を処理 (Audio Thread からは呼ばれない)」と整合する。

**T2 の生成元**

```
src/audioengine/AudioEngine.Init.cpp:33                  rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);
src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp:104  rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);
src/audioengine/AudioEngine.h:2914                      std::thread rebuildThread;
src/audioengine/AudioEngine.h:1753                      [[nodiscard]] std::thread::id rebuildThreadId() const noexcept { return rebuildThread.get_id(); }
src/audioengine/AudioEngine.RebuildDispatch.cpp:845      void AudioEngine::rebuildThreadLoop()
src/audioengine/AudioEngine.RebuildDispatch.cpp:1269     newDSP->convolverRt().rebuildAllIRsSynchronous(isObsolete);
src/audioengine/AudioEngine.RebuildDispatch.cpp:1047     recoveryDSP->convolverRt().rebuildAllIRsSynchronous(isRecoveryAborted);
src/audioengine/AudioEngine.RebuildDispatch.cpp:1134     recoveryDSP->convolverRt().rebuildAllIRsSynchronous(isRecoveryAborted);
src/convolver/ConvolverProcessor.Rebuild.cpp:44          void ConvolverProcessor::rebuildAllIRsSynchronous(std::function<bool()> shouldCancel)
src/convolver/ConvolverProcessor.Rebuild.cpp:95              loader.runSynchronously();
```

**Audio Thread 到達性の判定**: `convertToMixedPhaseAllpass` への production 呼び出しは `MixedPhase.cpp:35`（`convertToMixedPhase` 内）と `LoaderThread.cpp:863`（`doTransformStep` 内）の 2 経路のみ。`doTransformStep` は `stepOnce`（`LoaderThread.cpp:339-368`）からのみ呼ばれ、`stepOnce` は `performLoad` からのみ呼ばれる。`performLoad` は `LoaderThread::run()`（T1）または `LoaderThread::runSynchronously()`（T2）の 2 経路のみ。いずれも非 RT。**Audio Thread 到達なし。**

**注意（Practical Stable ISR との整合）**: T1 と T2 は別スレッドであり、両方が同時に `save()` を実行し得る。current source に mutex が無い（§2.3(c)）ため、同一 identity に対する read-modify-write（`touch()`）と `ReplaceFileW`（`save()`）が競合する可能性がある。

### 6.4 Caller graph の帰結

1. `remove` / `clear` / `getEntryCount` / `getCacheDirectory` / `getCacheFile` は **production caller が存在しない**。したがって `clear()` の「全 *.mph 削除」は現状 **production から到達不能** であり、実害は `evictLRU(0)` 経由（`MixedPhase.cpp:701` の `owner->getMaxCacheEntries()` が 0 を返した場合）のみ。
2. `evictLRU` は production caller が存在し、`save()` の直後に **同一スレッドから同一 identity 群に対して** 呼ばれる。`save()` の return value は無視されている（`MixedPhase.cpp:695-698` で戻り値が捨てられ、`:701` が無条件に実行される）。
3. `getMaxCacheEntries()` の初期値は `ConvolverProcessor.h:89 / :1083 int maxCacheEntries = 0;` であり、`copySnapshotToPendingUnlocked`（`StateAndUI.cpp:201`）が `jlimit(0, 64, ...)` で 0 を許す。したがって **`evictLRU(0)` → `clear()` → 全 *.mph 削除** は production から到達し得る。

### 6.5 Residual ambiguity

0。9 API すべての production call site を列挙し、production caller 0 件は `Production caller = none` として明示した。「caller dependent」のような曖昧表現は使用していない。

### 6.6 Contract status

```
L  Contract = PROVEN
   Evidence = CLOSED
   Implementation = VIOLATION（mutex 未実装のため T1/T2 間の競合が無防備）
   Residual ambiguity = 0
```

### 6.7 Evidence status

CLOSED。

### 6.8 Implementation status

call graph 自体は CLOSED である。ただし §13（single mutex）が NOT IMPLEMENTED であるため、実装状態は **VIOLATION**。

---

## 7. A — maxCacheEntries 数値境界

### 7.1 Claim

```
non-numeric -> 8
NaN         -> 8
+Inf        -> 8
-Inf        -> 8
fractional  -> 8
integer < 1 -> 8
1..64       -> unchanged
>64         -> 64
```

処理順序: `type classification → finite → integral → range classification → narrowing`

### 7.2 Current ConvoPeq source evidence

```
src/convolver/ConvolverProcessor.StateAndUI.cpp:250  v.setProperty ("maxCacheEntries", static_cast<int>(getMaxCacheEntries()), nullptr);
src/convolver/ConvolverProcessor.StateAndUI.cpp:382  if (v.hasProperty ("maxCacheEntries")) setMaxCacheEntries (static_cast<size_t>(static_cast<int>(v.getProperty("maxCacheEntries"))));
src/convolver/ConvolverProcessor.LoadPipeline.cpp:182-191
    void ConvolverProcessor::setMaxCacheEntries(size_t maxEntries) {
        const size_t clamped = juce::jlimit<size_t>(1, 64, maxEntries);
        { const juce::ScopedLock lock(pendingOverrideLock); pendingOverride.maxCacheEntries = static_cast<int>(clamped); }
        if (cacheManager) cacheManager->evictLRU(clamped);
    }
src/convolver/ConvolverProcessor.LoadPipeline.cpp:193-196
    size_t ConvolverProcessor::getMaxCacheEntries() const { ...; return static_cast<size_t>(pendingOverride.maxCacheEntries); }
src/convolver/ConvolverProcessor.StateAndUI.cpp:201  pendingOverride.maxCacheEntries = juce::jlimit(0, 64, snapshot.maxCacheEntries);
src/ConvolverProcessor.h:89 / :1083                    int maxCacheEntries = 0;
```

`setState` は `static_cast<int>(var)` → `static_cast<size_t>(int)` → `jlimit<size_t>(1, 64, ...)` の 2 段変換。**型分類・finite 判定・integral 判定・range 分類のいずれも無い。**

### 7.3 External / library evidence（JUCE 8.0.12 exact source）

`getProperty(name)` は `const var&` を返す（`JUCE/modules/juce_data_structures/values/juce_ValueTree.cpp:751-754`）。

`var` の型判別と変換（`JUCE/modules/juce_core/containers/juce_Variant.cpp`）:

```cpp
:80    int    (*toInt)   (const ValueUnion&) = defaultToInt;
:96    static int   defaultToInt (const ValueUnion&) { return 0; }        // void / undefined / array / binary / object / method 経路
:146   static int   intToInt     (const ValueUnion& data) noexcept { return data.intValue; }
:179   static int   int64ToInt   (const ValueUnion& data) noexcept { return (int) data.int64Value; }   // narrowing, 非 checked
:212   static int   doubleToInt  (const ValueUnion& data) noexcept { return (int) data.doubleValue; }  // ← 想定外値に対し unchecked
:242   static int   boolToInt    (const ValueUnion& data) noexcept { return data.boolValue ? 1 : 0; }
:274   static int   stringToInt  (const ValueUnion& data) noexcept { return getString (data)->getIntValue(); }  // 数値でない文字列は 0
:553   bool var::isVoid()   const noexcept { return type->isVoid; }
:555   bool var::isInt()    const noexcept { return type->isInt; }
:556   bool var::isInt64()  const noexcept { return type->isInt64; }
:557   bool var::isBool()   const noexcept { return type->isBool; }
:558   bool var::isDouble() const noexcept { return type->isDouble; }
:559   bool var::isString() const noexcept { return type->isString; }
:562   bool var::isBinaryData() const noexcept { return type->isBinary; }
:565   var::operator int() const noexcept  { return type->toInt (value); }
:519   var::var (const int v)     noexcept : type (&Instance::attributesInt)    { value.intValue = v; }
:522   var::var (const double v)   noexcept : type (&Instance::attributesDouble) { value.doubleValue = v; }
```

**`doubleToInt` は `(int) data.doubleValue` という無検査 narrowing であり、`isfinite` も `trunc(x)==x` も検査しない。** JUCE は `double` var を `int` に変換する際に NaN / ±Inf / 範囲外を拒否しない。

プロジェクトは C++20（`CMakeLists.txt:1152`）。C++20 では floating→integral の範囲外変換は UB ではなく「`2^N` を法とする一意の値」になるが、`NaN` / `Inf` に対する値は事前の保証がなく、コンパイラ・命令依存である。**いずれにせよ Owner contract の `→ 8` には到達しない。**

### 7.4 Reasoning — 現行実装の観測可能な結果（決定的ケースのみ）

`v.getProperty("maxCacheEntries")` が返す `var` の型ごとの結果:

| 入力 var | `var::operator int()` の経路 | 得られる `int` | `size_t` 化 | `jlimit<size_t>(1,64,·)` | Owner contract | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 存在しない（`hasProperty` false） | 経路を通らない | — | — | — | — | `setState` が呼ばれない |
| void / undefined | `defaultToInt` (`:96`) | `0` | `0` | **`1`** | `8` | **VIOLATION** |
| String `"abc"` | `stringToInt` (`:274`) | `0` | `0` | **`1`** | `8` | **VIOLATION** |
| bool `true` | `boolToInt` (`:242`) | `1` | `1` | **`1`** | `8` | **VIOLATION** |
| int `-5` | `intToInt` (`:146`) | `-5` | `18446744073709551611` | **`64`** | `8` | **VIOLATION** |
| int `0` | `intToInt` | `0` | `0` | **`1`** | `8` | **VIOLATION** |
| double `3.7` | `doubleToInt` (`:212`) | `3` | `3` | **`3`** | `8` | **VIOLATION** |
| double `NaN` / `±Inf` / 範囲外 | `doubleToInt` (`:212`) unchecked | 実装依存 | 実装依存 | 実装依存（`8` にはならない） | `8` | **VIOLATION**（結果非決定的） |
| int64 `2^32+5` | `int64ToInt` (`:179`) unchecked narrowing | `5` | `5` | `5` | 判定不能（`1..64` なら unchanged だが、正当な入力でない） | **VIOLATION**（narrowing unchecked） |
| int `5` | `intToInt` | `5` | `5` | `5` | `5` | 適合 |
| int `100` | `intToInt` | `100` | `100` | `64` | `64` | 適合 |

`int` → `size_t` の無検査変換が negative を `2^64` 相当に回し、`jlimit(1, 64, ...)` により **64** になる点が、Owner が指摘した「negative 値を wrap させ得る」の具体的帰結である。

### 7.5 Residual ambiguity

0。10 通りの入力分類について観測結果が一意に決まる（double の NaN/Inf については値が実装依存であり、**契約値 `8` には到達しない**ことのみが確定する）。

### 7.6 Contract status

```
A  Contract = PROVEN
   Evidence = CLOSED
   Implementation = VIOLATION
   Residual ambiguity = 0
```

### 7.7 Evidence status

CLOSED。`juce::var` の全変換経路を source で閉じた。

### 7.8 Implementation status

**VIOLATION**。分類が無く、negative が 64 に化ける。

---

## 8. I — path purity / J — mutex / K — canonical validator

### 8.1 I（path purity）

**Claim**: `getCacheDirectory()` は pure、`getCacheFile(valid)` は pure、`getCacheFile(invalid)` は empty。directory absent で `getEntryCount→0` / `evictLRU→no-op` / `clear→no-op`。

**Current source**: `:46-58 getCacheDirectory()` は `if (!dir.exists()) { auto result = dir.createDirectory(); ... }` を含む。`:69 getCacheFile` は `getCacheDirectory().getChildFile(filename)` を呼ぶため副作用が伝播する。`:342-344 / :331-332 / :281` の `getCacheDirectory()` 呼び出しも全部が副作用する。

**External evidence**: `juce::File::getSpecialLocation` / `getChildFile` は pure path construction（`juce_Files_windows.cpp:372-383` の `createDirectoryInternal` は `CreateDirectory` を呼ぶが、`getChildFile` は呼まない）。

**Residual ambiguity**: 0。

```
I  Contract = PROVEN / Evidence = CLOSED / Implementation = VIOLATION / Residual = 0
```

### 8.2 J（single process-local mutex）

**Claim**: 1 個の process-local mutex が `load / save / touch / evictLRU / remove / clear / getEntryCount` を保護。pure path construction は非対象。public → public の nested lock 禁止。

**Current source**: `MixedPhasePersistentCache.h:1-96` に mutex 宣言なし。`.cpp:150-225 save()` から `:277 clear()`、`:341 touch()` への **public → public nested call** が 2 箇所存在する（`:141 load→touch`、`:277 evictLRU→clear`）。

**External evidence**: §6.3 の caller graph により T1/T2 の 2 スレッドが `save()` を実行し得る。`ReplaceFileW` は target を `DELETE` access で開くため、§3 の結論（load HANDLE に `FILE_SHARE_DELETE` を付けない）と組み合わさると mutex 無しで OS レベルの失敗が発生する。

**Residual ambiguity**: 0。

```
J  Contract = PROVEN / Evidence = CLOSED / Implementation = NOT IMPLEMENTED / Residual = 0
```

（private helper への分解で nested public call を解消する必要があり、関数の物理分割自体が未実装なので NOT IMPLEMENTED）

### 8.3 K（canonical validator）

**Claim**:

```
canonical filename
version == 3
magic valid
headerReserved == 0
filename identity valid
header identity valid
filename identity == header identity
requested identity == header identity
shape/domain valid
expected size valid
header.totalFileSize == expected
actual file size == expected
expected <= max file size
```

CRC/checksum は要求しない。payload semantic validity は canonicality から分離する。

**Current source**: canonical validator 関数が存在しない。`load()` は `:100-108` で `magic / version / 6 項目 identity` のみを見るが、

- `headerReserved` 検証なし
- `header.numChannels` / `numSamples` / `numAllpassSections` の上限検証なし（`:111-113` で `<= 0` のみ）
- `rho.size() == theta.size()` の検証なし（`save()` は `:177` で `rho.size()` のみを見、`theta.size()` を見ない）
- `header.totalFileSize` 検証なし（`load()` は `:216` で書くだけで `:96-144` で読まない）
- 実ファイルサイズ検証なし
- **filename identity の検証が原理的に不能** — §9 参照

**External evidence**: `File::getSize()` は `GetFileAttributesEx`（`juce_Files_windows.cpp:544-552`）であり、同一 HANDLE 上のサイズ取得ではない。契約どおりにやるなら `GetFileSizeEx(H)` が必要（§3）。

**Residual ambiguity**: 0。

```
K  Contract = PROVEN / Evidence = CLOSED / Implementation = NOT IMPLEMENTED / Residual = 0
```

---

## 9. v3 filename / B — collision authority

### 9.1 Claim

```
fileHash        16 lowercase hex
sampleRateBits  16 lowercase hex
phaseMode        8 lowercase hex
freqStartBits    8 lowercase hex
freqEndBits      8 lowercase hex
targetLength     8 lowercase hex
= 64 hex + ".mph" = 68 characters
```

IEEE bit pattern の固定幅 hex serialization。decimal serialization 禁止。`computeKeyHash()` は v3 canonical filename authority ではない。

### 9.2 Current ConvoPeq source evidence

```
src/MixedPhasePersistentCache.cpp:20-40   computeKeyHash(...)  ← hashCombine 6 段
src/MixedPhasePersistentCache.cpp:66-69
    const auto hash = computeKeyHash(fileHash, sampleRate, phaseMode, freqStartHz, freqEndHz, targetLength);
    const auto filename = juce::String::toHexString(static_cast<int64_t>(hash)) + ".mph";
    return getCacheDirectory().getChildFile(filename);
```

- filename は **hash** であり、identity そのものではない
- `toHexString` は可変幅（§9.3）
- 固定幅 64 hex ではない
- identity validation は行われない（`computeKeyHash` は無条件に走る。`fileHash == 0` でも、SR が範囲外でも、非有限でも）

### 9.3 External / library evidence（JUCE 8.0.12 exact source）

```cpp
// JUCE/modules/juce_core/text/juce_String.h:1122-1124
/** Returns a string representing this numeric value in hexadecimal. */
template <typename IntegerType>
static String toHexString (IntegerType number)      { return createHex (number); }

// JUCE/modules/juce_core/text/juce_String.h:1396-1402
static String createHex (uint8);
static String createHex (uint16);
static String createHex (uint32);
static String createHex (uint64);
template <typename Type>
static String createHex (Type n)  { return createHex (static_cast<...UnsignedTypeWithSize<(int) sizeof (n)>::type> (n)); }

// JUCE/modules/juce_core/text/juce_String.cpp:1920-1942
static const char hexDigits[] = "0123456789abcdef";
template <typename Type>
static String hexToString (Type v)
{
    String::CharPointerType::CharType buffer[32];
    auto* end = buffer + numElementsInArray (buffer) - 1;
    auto* t = end;
    *t = 0;
    do {
        *--t = hexDigits [(int) (v & 15)];
        v = static_cast<Type> (v >> 4);
    } while (v != 0);            // ← v == 0 で終了 = 先頭 0 が削られる
    return String (String::CharPointerType (t), String::CharPointerType (end));
}
String String::createHex (uint64 n)   { return hexToString (n); }
```

JUCE 自体の unit test（`juce_String.cpp:2625-2638`）:

```cpp
expect (String::toHexString (0x1234abcd).equalsIgnoreCase ("1234abcd"));
expect (String::toHexString ((int32) -1).equalsIgnoreCase ("ffffffff"));
expect (String::toHexString ((int64) -1).equalsIgnoreCase ("ffffffffffffffff"));
```

`hexToString` は先頭ゼロを削るため、`String::toHexString` は **固定幅ではない**。戻り値が lowercase であることは `hexDigits[]`（`:1918`）の宣言から確定するが、幅は確定しない。

### 9.4 Reasoning

1. `toHexString(static_cast<int64_t>(hash))` は hash の bit pattern を正しく lowercase hex 化する（`createHex` が `uint64_t` の template 特殊化を経由するため sign は消える）。幅は可変であり、先頭 0 を削るので**異なる hash は異なる文字列になる**（単射）。したがって hash 間の衝突は起きない。
2. しかし filename は **identity ではなく hash** であるため、K の要求する `filename identity valid` および `filename identity == header identity` は **原理的に検証不能**。これが B（collision authority）を filename に置く意味である。
3. `computeKeyHash` は `hashCombine` チェーンであり、`fileHash` を含む。然而 `fileHash == 0` を拒否する gate が無いため、`getCacheFile()` は invalid identity に対しても path を返す（§15）。
4. したがって v3 contract（68 文字固定幅 identity serialization）は current source に存在しない。

### 9.5 Residual ambiguity

0。

```
v3 filename  Contract = PROVEN / Evidence = CLOSED / Implementation = NOT IMPLEMENTED / Residual = 0
B             Contract = PROVEN / Evidence = CLOSED / Implementation = NOT IMPLEMENTED / Residual = 0
```

---

## 10. D — save validation / G — checked arithmetic / H — constants

### 10.1 D（save validation）

**Claim**: `fileHash != 0`、`finite(SR)`、`44100 <= SR <= 768000`、valid PhaseMode、valid F1/F2、`1 <= targetLength <= MAX_IR_LATENCY`、`1 <= numChannels <= kMaxIRLoadChannels`、`1 <= numSamples <= MAX_IR_LATENCY`、`rho.size() == theta.size()`、`sections <= 20`（`sections == 0` は許容）。整数は `raw signed → range validation → narrowing` の順。

**Current source**: `:150-225 save()` に**上記の検証がいずれも存在しない**。特に

- `:175-177` `const int numSec = static_cast<int>(rho.size());` — `theta.size()` を見ない（`rho.size() != theta.size()` を許す）
- `:181-193` header への field 設定は unchecked
- `:190` `header.lastUsedTime = static_cast<uint64_t>(juce::Time::getMillisecondCounter());`
- `:211-213` `theta` 書きは `numSec > 0` ガードのみで、サイズ一致を要求しない
- `fileHash == 0` を拒否しない
- `sampleRate` の有限性・範囲を検証しない
- `numChannels` / `numSamples` の上限を検証しない
- `:196`/`:203`/`:209`/`:211` の `static_cast<size_t>(numSamples) * sizeof(double)` は checked arithmetic ではない

**External evidence**: `ConvolverProcessor.h:117-122 enum class PhaseMode : int { AsIs=0, Mixed=1, Minimum=2 }`、`:225-229 MIXED_F1_MIN_HZ=100.0f / MIXED_F1_MAX_HZ=400.0f / MIXED_F2_MIN_HZ=700.0f / MIXED_F2_MAX_HZ=1300.0f`、`:250 MAX_IR_LATENCY=2097152`、`IRLoadAdmission.h:43 kMaxIRLoadChannels=8u`、`ConvolverProcessor.MixedPhase.cpp:477 designer_config.numSections = liveReconfigure ? (highRateLive ? 2 : 8) : 20;`。これらは §17 の authority と整合する。

**Residual ambiguity**: 0。

```
D  Contract = PROVEN / Evidence = CLOSED / Implementation = NOT IMPLEMENTED / Residual = 0
```

### 10.2 G（checked arithmetic / maximum file size）

**Claim**: `sizeof(DiskHeader) + 8 * MAX_IR_LATENCY * sizeof(double) + 20*sizeof(double) + 20*sizeof(double)` = `134,218,124`。magic number 禁止。`uint64_t / size_t / int64_t` への representability を個別に checked arithmetic で確認。

**算出した値**（現行 v2 `DiskHeader` を Owner 指定の v3 layout に置き換えた場合）:

```
76
+ 8 * 2,097,152 * 8        = 134,217,728
+ 20 * 8                   = 160
+ 20 * 8                   = 160
= 134,218,124 bytes
```

**Current source**: `DiskHeader` の現行 layout（`MixedPhasePersistentCache.h:68-85`, `#pragma pack(push,1)`）:

```
offset  0  uint64_t magic            (8)
offset  8  uint32_t version          (4)
offset 12  uint32_t headerReserved   (4)
offset 16  uint64_t fileHash         (8)
offset 24  double   sampleRate       (8)
offset 32  int32_t  phaseMode        (4)
offset 36  float    freqStartHz      (4)
offset 40  float    freqEndHz        (4)
offset 44  int32_t  targetLength     (4)
offset 48  uint64_t lastUsedTime     (8)   ← offset 48 は既に一致
offset 56  int32_t  numChannels      (4)
offset 60  int32_t  numSamples       (4)
offset 64  int32_t  numAllpassSections (4)
offset 68  int64_t  totalFileSize    (8)   ← offset 68 は既に一致
sizeof = 76                                    ← 既に一致
```

**重要な訂正**: `lastUsedTime offset == 48` と `totalFileSize offset == 68` と `sizeof == 76` は **現行 v2 layout で既に満たされている**。Owner が「未実装」とした `static_assert` 3 本の**表明自体**だけが未実装である。

```
src/MixedPhasePersistentCache.h:87   static_assert(sizeof(DiskHeader) <= 128, "DiskHeader is too large");   ← 表明は `<= 128` のみ
```

したがって `sizeof == 76` / `offsetof == 48` / `offsetof == 68` は **値としては既に production で成立**している。追加で必要なのは (a) `static_assert` の強化（`<= 128` を `== 76` に）、(b) `std::is_trivially_copyable_v<DiskHeader>` の static_assert、(c) **runtime `headerReserved == 0` の検証**（`load()` に無い）。

`kVersion` は `:66 static constexpr uint32_t kVersion = 2;` であり、`kVersion = 3` への更新が未実施。

**Residual ambiguity**: 0。

```
G  Contract = PROVEN / Evidence = CLOSED / Implementation = NOT IMPLEMENTED / Residual = 0
DiskHeader/version  Contract = PROVEN / Evidence = CLOSED / Implementation = NOT IMPLEMENTED / Residual = 0
```

### 10.3 H（constants）

**Owner-approved constants**（v3.78.63 §17 で確定）:

```cpp
constexpr double kPersistentCacheMinSampleRate = 44100.0;
constexpr double kPersistentCacheMaxSampleRate = 768000.0;
constexpr int kMinPersistentCacheEntries = 1;
constexpr int kDefaultPersistentCacheEntries = 8;
constexpr int kMaxPersistentCacheEntries = 64;
constexpr int kMaxPersistentCacheSections = 20;
```

**Current source**: これらの 6 定数は `MixedPhasePersistentCache.h` に**存在しない**。SR 上限・下限、entries の min/max/default、sections 上限のいずれも PersistentCache 側に無い。

**F1/F2 は duplicate を新設しない**（`ConvolverProcessor.h:225-229` が authority）。**PhaseMode も `ConvolverProcessor.h:117-122` が authority**。`kMaxIRLoadChannels` は `IRLoadAdmission.h:43` の既存値（`8u`）を再利用。

**Residual ambiguity**: 0。

```
H  Contract = PROVEN / Evidence = CLOSED / Implementation = NOT IMPLEMENTED / Residual = 0
```

---

## 11. Timestamp / M / clear / remove / E

### 11.1 Timestamp

**Claim**: `juce::Time::currentTimeMillis()`。domain `0 = invalid` / `1..INT64_MAX = valid` / `> INT64_MAX = invalid`。future timestamp / clock rollback は単独では invalid 理由にしない。LRU は `valid timestamp ascending → invalid LAST → canonical filename ascending`。

**Current source**: `:190 header.lastUsedTime = static_cast<uint64_t>(juce::Time::getMillisecondCounter());` と `:264 const uint64_t now = static_cast<uint64_t>(juce::Time::getMillisecondCounter());` の **2 箇所**が `getMillisecondCounter()` を使用する。

**External evidence**（JUCE 8.0.12 exact source, `JUCE/modules/juce_core/time/juce_Time.cpp`）:

```cpp
:239   uint32 juce_millisecondsSinceStartup() noexcept;      // ← 戻り値は uint32
:241   uint32 Time::getMillisecondCounter() noexcept
       {
           auto now = juce_millisecondsSinceStartup();
           if (now < TimeHelpers::lastMSCounterValue.get()) { /* 単調化 */ }
           ...
           return now;                                       // uint32
       }

:220   int64 Time::currentTimeMillis() noexcept
       {
       #if JUCE_WINDOWS
           struct _timeb t;
           _ftime_s (&t);
           return ((int64) t.time) * 1000 + t.millitm;       // ← int64 epoch millis
       #else
           ...
       #endif
       }
```

`getMillisecondCounter()` は **起動からの uint32 ミリ秒**であり、`2^32 ms` ≈ **49.7 日**で wrap する。`uint64_t` への cast は wrap を防げない。`currentTimeMillis()` は `_ftime_s` による epoch ミリ秒（`int64`）。

**さらに重要な欠陥（本版で新たに確定）**: `touch()` は **ファイルを破損する**。

```
src/MixedPhasePersistentCache.cpp:242-243
    // read-modify-write: ファイル全体を読み込み、ヘッダのタイムスタンプのみ更新して書き戻す
    // (FileOutputStream で開くとファイルが切り詰められるため、read-modify-write で行う)
...
:267   juce::FileOutputStream outStream(file);
:270   outStream.write(buffer.data(), buffer.size());
```

この source コメントは **JUCE 8.0.12 に対して誤り**である。`juce_Files_windows.cpp:429-450` の `FileOutputStream::openHandle()` は

```cpp
CreateFile (..., GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
li.LowPart = SetFilePointer (h, 0, &li.HighPart, FILE_END);   // ← 常に EOF
currentPosition = li.QuadPart;
```

であり、`truncate()` は `SetEndOfFile` を呼ぶが**明示呼び出し時のみ**（`juce_FileOutputStream.h:105-109`: 「Attempts to truncate the file to the current write position. To truncate a file to a specific size, first use setPosition() to seek to the appropriate location, and then call this method.」）。したがって `outStream.write(buffer.data(), buffer.size())` は **追記**となり、ファイルサイズは **2 倍**になる。

帰結:

1. `load()` は `:141` で**毎回** `touch()` を呼ぶ。n 回の load ごとにファイルサイズは `2^n` 倍に増える（1 → 2 → 4 → 8 …）。
2. `header.totalFileSize` は元の値のままなので、v3 canonical validator（`actual file size == expected`）では**恒久的に malformed 判定**となる。
3. `touch()` は v2 header（76 bytes）で `lastUsedTime offset == 48` を使うため header 部分自体は正しいが、payload が二重化された artifact は `load()` で header と payload を-offset 0 から読むため**偶然ロード成功し得る**。したがって破損が静かに進行する。

```
Timestamp  Contract = PROVEN / Evidence = CLOSED / Implementation = VIOLATION / Residual = 0
```

### 11.2 M — `evictLRU(0)`

**Claim**: `evictLRU(0) -> no-op`。`clear()` を alias にしてはならない。

**Current source**: `:273-279`

```cpp
void MixedPhasePersistentCache::evictLRU(size_t maxCount)
{
    if (maxCount == 0)
    {
        clear();
        return;
    }
    ...
```

**VIOLATION**（明確な alias）。§6.4 のとおり production 到達可能（`getMaxCacheEntries()` の初期値 `ConvolverProcessor.h:89/:1083 int maxCacheEntries = 0;` + `StateAndUI.cpp:201` の `jlimit(0, 64, ·)` が 0 を許す）。

```
M  Contract = PROVEN / Evidence = CLOSED / Implementation = VIOLATION / Residual = 0
```

### 11.3 clear

**Claim**: `clear() -> current canonical v3 artifacts only`。「full clear」という表現は禁止。protected: future / obsolete / malformed / noncanonical。

**Current source**: `:328-338`

```cpp
void MixedPhasePersistentCache::clear()
{
    auto dir = getCacheDirectory();
    if (!dir.exists()) return;
    juce::Array<juce::File> files;
    dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");
    for (auto& f : files) f.deleteFile();
}
```

`*.mph` を**無条件に削除**する。canonical 判定が無く、in-flight の `TemporaryFile` temp（`<64hex>_temp<hex>.mph`, §2.3(a)）も削除対象になる。

```
clear  Contract = PROVEN / Evidence = CLOSED / Implementation = VIOLATION / Residual = 0
```

### 11.4 remove

**Claim**:

```
target absent            -> no-op
current v3 canonical     -> delete attempt
future >= 4              -> preserve
obsolete                 -> preserve
malformed                -> preserve
```

`remove()` は directory scan を行わず、requested identity から導出した canonical target のみを対象とする。削除前に current-v3 canonical validation を通す。

**Current source**: `:316-326`

```cpp
void MixedPhasePersistentCache::remove(uint64_t fileHash, double sampleRate, int phaseMode,
                                       float freqStartHz, float freqEndHz, int targetLength)
{
    const auto file = getCacheFile(fileHash, sampleRate, phaseMode, freqStartHz, freqEndHz, targetLength);
    if (file.existsAsFile())
        file.deleteFile();
}
```

directory scan は行わない，符合。ただし canonical validation が**無い**ため、canonical target に future / obsolete / malformed artifact があれば削除する。

**Owner 指示 §6 の補足が有効であることを source が裏付ける**: `computeKeyHash()` は `fileHash` を含むが、filename は hash であるため「同じ canonical filename を持つ artifact が version >= 4 または malformed header である可能性」があり得る。現状実装はその検証を欠く。

```
remove  Contract = PROVEN / Evidence = CLOSED / Implementation = VIOLATION / Residual = 0
```

### 11.5 E — save → eviction coupling

**Claim**:

```
save false -> no eviction
save true  -> eviction attempt allowed
eviction failure -> save success remains success
```

**Current source**: `src/convolver/ConvolverProcessor.MixedPhase.cpp:695-701`

```cpp
695  convo::MixedPhasePersistentCache::save(
696      key.fileHash, key.sampleRate, static_cast<int>(key.phaseMode),
697      key.f1, key.f2, key.targetLength,
698      mixedIR, rho, theta);
699
700  // ディスクキャッシュのLRUエビクション
701  convo::MixedPhasePersistentCache::evictLRU(owner->getMaxCacheEntries());
```

`:695` の戻り値は捨てられ、`:701` は無条件に実行される（`:687 if (optimizedByCmaes)` ガードのみ）。`save()` が `false` を返しても `evictLRU()` が走る。

```
E  Contract = PROVEN / Evidence = CLOSED / Implementation = VIOLATION / Residual = 0
```

---

## 12. Identity validation と filename serialization の順序

### 12.1 Claim

`getCacheFile()` は `identity validation → canonical serialization → path construction` の順。invalid identity について

```
fileHash == 0
non-finite sampleRate
sampleRate outside [44100, 768000]
invalid PhaseMode
invalid F1/F2
invalid targetLength
```

は **empty `juce::File`** を返し、**filesystem access / directory creation / filesystem mutation を一切発生させない**。

### 12.2 Current ConvoPeq source evidence

`:60-70 getCacheFile()` は無条件に `computeKeyHash()` → `toHexString()` → `getCacheDirectory()`（副作用）を実行する。identity validation は 0。

`:151-167 evictLRU()` は `:282-283` で directory scan し、`:293-306` で各ファイルの header を読むが、`header.magic != kMagic` で `continue` するのみで、`version` / `headerReserved` / shape / size を検証しない。`:308-309` の sort は `a.first < b.first` のみで、invalid timestamp を LAST に置かないし tie-break に filename を使わない。

`:328-338 clear()` と `:340-349 getEntryCount()` は同じ無条件 glob。

### 12.3 Reasoning

- `getCacheFile()` は invalid identity でも path を返し、`:69` 経由で `getCacheDirectory()` を呼ぶため **directory を新規作成しうる**。これは I（path purity）と同じ欠陥の伝播である。
- `evictLRU()` は `version` を見ないため、**future / obsolete artifact を LRU 候補にする**。`:305 timedFiles.emplace_back(header.lastUsedTime, f);` の後に `:313 timedFiles[i].second.deleteFile();` で削除する。
- `clear()` / `getEntryCount()` も保護条件が無い。

### 12.4 Residual ambiguity

0。

```
B/K/M/remove integration  Contract = PROVEN / Evidence = CLOSED / Implementation = NOT IMPLEMENTED / Residual = 0
```

---

## 13. Practical Stable ISR Bridge Runtime との整合

`doc/Practical Stable ISR Bridge Runtime.md` の原則（RT は待たない / lock しない / 判断しない / verify empty）に対する current source の位置づけ。

| 観点 | current source | 判定 |
| --- | --- | --- |
| Audio Thread 到達 | §6.3 により **到達なし**（T1 = LoaderThread, T2 = rebuildThread） | **適合** |
| RT での lock | Audio Thread に無いため RT lock は無い | **適合** |
| RT での待機 | `Thread::sleep(100)` × 5 と `Thread::sleep(50)` × 5 は **非 RT スレッド上**（T1/T2）で起きる | **適合**（ただし非 RT レイテンシ上限として記録） |
| verify empty | `convertToMixedPhaseAllpass` は `catch(...)` で `setMixedPhaseState(0)` + rethrow（`MixedPhase.cpp:713-718`） | **適合** |

**非 RT ブロッキング上界（新規記録）**: §4.3(c) の retry より、`save()` 失敗時に mutex を保持する時間は

```
overwriteTargetFileWithTemporary: 最大 5 回 × (ReplaceFileW + Thread::sleep(100)) ≈ 400-500 ms
~TemporaryFile::deleteTemporaryFile : 最大 5 回 × Thread::sleep(50)          ≈ 200-250 ms
合計                                                                        ≈ 600-750 ms
```

Owner contract の単一 mutex は `save()` の**全期間**を直列化する。T1 と T2 のどちらかが `save()` 中なら他方は最大約 750 ms ブロックされる。これは deadlock ではないが、**非 RT スレッド間のスケジューリング上の負荷**として記録する。RT には影響しない。

---

## 14. Ownership 最終表（§10 の要求形）

```
                  current v3     future     obsolete     malformed
------------------------------------------------------------------
save              replace        preserve   preserve     preserve
evictLRU          eligible       preserve   preserve     preserve
clear             delete         preserve   preserve     preserve
remove            delete         preserve   preserve     preserve
```

- `save` の `replace` は「削除」ではなく `same-identity current-v3 artifact → newer same-identity current-v3 artifact` への正当な更新。
- `evictLRU` の `eligible` は current-v3 canonical validation を通過し、かつ LRU target として选中されたもの。
- `clear` / `remove` の `delete` は delete **attempt** であり、成功保証ではない（§4 の 1176/1177 問題）。
- noncanonical ファイル（§2.3(a) の in-flight temp を含む）は 4 API のいずれにも delete 対象とならない。

### 14.1 Existing target × Save action（§9 の要求形）

| Existing target | Save action |
| --- | --- |
| absent | create |
| current v3 canonical | replace allowed |
| future >= 4 | preserve / save failure |
| obsolete < 3 | preserve / save failure |
| malformed | preserve / save failure |
| canonical filename but invalid current artifact | preserve / save failure |
| indeterminate / I/O error | preserve / save failure |

noncanonical file は requested canonical target と別名であるため、save は触れない（§2.3(a) の temp は `TemporaryFile` の所有物であり、`~TemporaryFile` が回収する）。

---

## 15. Final Matrix（§23 の要求形）

| Item | Contract | Evidence | Implementation | Residual ambiguity |
| --- | --- | --- | --- | --- |
| A — maxCacheEntries numeric boundary | PROVEN | CLOSED | VIOLATION | 0 |
| B — canonical collision authority | PROVEN | CLOSED | NOT IMPLEMENTED | 0 |
| C — same-HANDLE semantics | PROVEN | CLOSED | NO-GO | 0 |
| D — save validation | PROVEN | CLOSED | NOT IMPLEMENTED | 0 |
| E — save → eviction coupling | PROVEN | CLOSED | VIOLATION | 0 |
| F — replacement failure preservation | PROVEN | CLOSED | NOT IMPLEMENTED | 0 |
| G — checked arithmetic / max file size | PROVEN | CLOSED | NOT IMPLEMENTED | 0 |
| H — owner constants | PROVEN | CLOSED | NOT IMPLEMENTED | 0 |
| I — path purity | PROVEN | CLOSED | VIOLATION | 0 |
| J — single process-local mutex | PROVEN | CLOSED | NOT IMPLEMENTED | 0 |
| K — canonical validator | PROVEN | CLOSED | NOT IMPLEMENTED | 0 |
| DiskHeader / version lifecycle | PROVEN | CLOSED | NOT IMPLEMENTED | 0 |
| Timestamp | PROVEN | CLOSED | VIOLATION | 0 |
| M — evictLRU(0) | PROVEN | CLOSED | VIOLATION | 0 |
| clear | PROVEN | CLOSED | VIOLATION | 0 |
| remove | PROVEN | CLOSED | VIOLATION | 0 |
| v3 filename (68-char fixed-width) | PROVEN | CLOSED | NOT IMPLEMENTED | 0 |
| **Save Collision / Protected Artifact Ownership** | **PROVEN** | **CLOSED** | **VIOLATION** | **0** |
| B/K/M/remove integration | PROVEN | CLOSED | NOT IMPLEMENTED | 0 |
| L — complete caller graph | PROVEN | CLOSED | VIOLATION | 0 |

`Contract = VIOLATION` は使用していない。`Contract = OPEN` は 0 件。

### 15.1 v3.78.62 期待表との差分（明示）

| Item | v3.78.62 期待 | v3.78.63 判定 | 差分の理由 |
| --- | --- | --- | --- |
| C Evidence | OPEN | **CLOSED** | JUCE 8.0.12 exact source（`juce_Files_windows.cpp:396-450 / :486-511 / :544-552`, `juce_FileInputStream.h`, `juce_FileOutputStream.h`）と Microsoft Learn の `CreateFileW / ReadFile / SetFilePointerEx / GetFileSizeEx / FlushFileBuffers` 実文を 10 項目すべて取得。結論は「JUCE 公開 API では不可能 → 内部限定 native HANDLE」。 |
| F Evidence | OPEN | **CLOSED** | `juce_Files_windows.cpp:374-377` で `lpBackupFileName = nullptr` であることを確認し、`ReplaceFileW` の 1176 / 1177 failure semantics（Microsoft Learn 実文）と照合。結論は「JUCE `TemporaryFile` 経路では充足不能」。 |
| L Evidence | OPEN | **CLOSED** | 9 API すべての production call site を完全列挙。5 API は `Production caller = none`。 |
| L Implementation | NO-GO | **VIOLATION** | caller graph は確定したが、§13 の single mutex が未実装のため 2 スレッド間の競合が無防備。call graph 自体に未確定要素は無い。 |
| Save Collision | （未定義） | **PROVEN / CLOSED / VIOLATION** | v3.78.63 §2-§5 で新規項目として追加。 |
| DiskHeader offsets | 「未実装」 | **値 76 / 48 / 68 は既に成立** | `MixedPhasePersistentCache.h:68-85` の `#pragma pack(push,1)` layout を算出した結果、`lastUsedTime` は offset 48、`totalFileSize` は offset 68、`sizeof` は 76 で**既に契約値と一致**。未実装なのは `static_assert` の表明（`:87` は `<= 128` のまま）と runtime `headerReserved == 0` 検証のみ。 |

---

## 16. Final Gate

### 16.1 Contract Gate

```
全項目 Contract = PROVEN
```

**PASS**（20 項目すべて PROVEN、`Contract = OPEN` は 0 件）。

### 16.2 Evidence Gate

```
全項目 Evidence = CLOSED
Residual ambiguity = 0
```

**PASS**（20 項目すべて CLOSED、`Residual ambiguity = 0`）。

### 16.3 Implementation Gate

```
current source = v2 implementation
Implementation = NO-GO
```

**NO-GO**（前提どおり）。内訳は `VIOLATION` 7 項目 / `NOT IMPLEMENTED` 10 項目 / `NO-GO` 1 項目（= `Implementation = GO` 0 項目）。

### 16.4 v3.78.63 成功条件

```
Contract Gate       = PASS
Evidence Gate       = PASS
Implementation Gate = NO-GO
```

**達成。** これは「実装 GO」ではなく、「実装へ進んでよい契約・証拠基盤が完成した」ことを意味する。

---

## 17. 実装移行時の必須前提（Owner 承認が必要）

以下は本監査で evidence が閉じた結果、**現行手段では達成できない**と確定した項目である。implementation plan はこれらを前提として設計しなければならない。

1. **F は JUCE `TemporaryFile` では達成不能。** `ReplaceFileW` に `lpBackupFileName` を明示した内部限定 native 呼び出し + 失敗時 rollback、または二段階 commit が必要。
2. **C は JUCE 公開 stream API では達成不能。** 内部限定 Windows native `HANDLE` が必要。`GetFileSizeEx` / `ReadFile` / `SetFilePointerEx` / `FlushFileBuffers` / `CloseHandle` を直接呼ぶ。
3. **Touch は既存 whole-file RMW + `FileOutputStream` では達成不能（破損）。** same-HANDLE in-place header write に置き換える必要がある（`FileOutputStream` は `OPEN_ALWAYS` + `FILE_END` で追記）。
4. **`getCacheFile()` の temp artifact が `*.mph` glob に合致する。** canonical validator は filename shape（68 文字固定幅 hex + `.mph`）で判定し、`_temp` 入りは `noncanonical` に分類する必要がある。
5. **mutex は 7 filesystem-touching API を直列化するが、非 RT ブロッキング上界が約 750 ms（§13）。** RT には影響しないが、Owner が許容するか判断を要する。

---

## 18. 停止条件

v3.78.63 では以下を**一切実施していない**。

```
production source 変更      0
test 変更                    0
CMake 変更                  0
build                       0
version 変更                0
ConvoPeq.md 再生成          0
commit                      0
push                        0
git reset / clean / rebase / amend / squash / force push  0
既存 commit SHA 変更        0
```

Owner の明示的承認なく production source を変更してはならない。§17 の 5 項目について Owner が手段を承認した後にのみ implementation plan を作成する。