# D-1R-2I — Integrated Modification Plan Validation / Pre-Implementation Closure Audit

- 位置づけ: D-1R-2H の訂正要求 + N1〜N5 Owner 裁定の反映 + 改修計画全体の最終妥当性確認（read-only）
- 作成: 2026-10-06
- 本監査で変更していないもの: `src/**` / `tests/**` / `CMakeLists.txt` / `build/**` / `ConvoPeq.md` / commit / push（すべて 0）

> **本報告書の判定語は 3 種類のみを用いる**
>
> | 語 | 意味 |
> | --- | --- |
> | **SOURCE-PROVEN** | 現行 source（authority snapshot）の行から直接読み取れる事実。実行プローブ・JUCE 実装・MS Learn 等の一次資料で裏付け済み |
> | **DESIGN-VALIDATED** | 改修後 contract として妥当であると本監査が論証したもの。**現行コードでは成立していない** |
> | **NOT PROVEN** | 上記いずれでもない。Owner decision または production 実証が必要 |
>
> 「この監査で設計が妥当である」ことと「現行 source ですでに成立している」ことを本報告書内で混同しない。

---

## §1. Authority

| 項目 | 値 | Owner 指定値との一致 |
| --- | --- | --- |
| ファイル | `C:\VSC_Project\ConvoPeq\doc\ConvoPeqMD\20261002231921\ConvoPeq.md` | — |
| size | 5,941,824 bytes | **一致** |
| actual lines | 131,019 | **一致** |
| SHA-256 | `3a51ab3414dba90005b99540376a93d6ab76ae1a96a754ce9e7050e2e0593045` | **一致** |
| `split('\n')` 要素数 | 131,020 | （末尾 empty segment。line count と混同しない） |

本報告書の `:NNNNN` はすべてこのファイルの実 line 番号（1-based, CRLF 基準）。**古い snapshot の行番号は使用していない。**

本セッションで 3 項目（size / lines / SHA-256）を再計測し、Owner 指定値と完全一致することを確認した（SOURCE-PROVEN）。

---

## §2. read-only 遵守

| 対象 | 変更数 |
| --- | --- |
| `src/**` | 0 |
| `tests/**` | 0 |
| `CMakeLists.txt` | 0 |
| `build/**` | 0 |
| CTest 実行 | 0 |
| `ConvoPeq.md` 再生成 | 0 |
| `git commit` | 0 |
| `git push` | 0 |
| 作成物 | `doc/work113/P1-5-IR-P2_MIXEDPHASE-BG-CMAES_D1R2I_7864_20261006.md` のみ |

HEAD は `b0b46941` のまま。`git status --porcelain -- src tests CMakeLists.txt build` は空。

---

## §3. D-1R-2H の訂正一覧

### 3.1 訂正 C1: `touch()` の破壊機序（Owner 指示 §2-1）

#### 3.1.1 D-1R-2H の誤った主張

D-1R-2H §22 N4 および付録 B は以下と記載した:

> `FileOutputStream(file)` が canonical target を **truncate** する非 atomic read-modify-write

**この主張は誤りである。**

#### 3.1.2 一次資料による反証

JUCE `juce_Files_windows.cpp:429-450`:

```cpp
void FileOutputStream::openHandle()
{
    auto h = CreateFile (file.getFullPathName().toWideCharPointer(),
                         GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                         OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);   // ← OPEN_ALWAYS
    if (h != INVALID_HANDLE_VALUE)
    {
        LARGE_INTEGER li;
        li.QuadPart = 0;
        li.LowPart = SetFilePointer (h, 0, &li.HighPart, FILE_END);     // ← FILE_END
        if (li.LowPart != INVALID_SET_FILE_POINTER)
        {
            fileHandle = (void*) h;
            currentPosition = li.QuadPart;                               // ← 既存サイズ
            return;
        }
    }
    status = WindowsFileHelpers::getResultForLastError();
}
```

JUCE `juce_Files_windows.cpp:475-483`:
```cpp
Result FileOutputStream::truncate()
{
    if (fileHandle == nullptr) return status;
    flush();
    return SetEndOfFile ((HANDLE) fileHandle) ? Result::ok()
                                              : WindowsFileHelpers::getResultForLastError();
}
```

JUCE `juce_FileOutputStream.cpp:39-45, 53-56, 88-125`:
```cpp
FileOutputStream::FileOutputStream (const File& f, const size_t bufferSizeToUse)
    : file (f), bufferSize (bufferSizeToUse), buffer (jmax (bufferSizeToUse, (size_t) 16))
{
    openHandle();            // ← 位置は EOF
}
int64 FileOutputStream::getPosition() { return currentPosition; }
bool FileOutputStream::write (const void* const src, const size_t numBytes) { /* currentPosition を前進 */ }
```

**3 つの事実（SOURCE-PROVEN）**:
1. `CreateFile` は `CREATE_ALWAYS` でも `TRUNCATE_EXISTING` でもなく **`OPEN_ALWAYS`**。既存ファイルを切り詰めない
2. `SetFilePointer(..., FILE_END)` により **write position は既存ファイルの末尾**になる
3. `truncate()`（`SetEndOfFile`）は **`touch()` から一度も呼ばれていない**

#### 3.1.3 訂正後の正しい机理

現行 `touch()`（`:77383-77423`）:

```cpp
const auto fileSize = file.getSize();                 // :77402  → N
buffer.resize (static_cast<size_t> (fileSize));        // :77406  → buffer.size() == N
...
const uint64_t now = static_cast<uint64_t> (juce::Time::getMillisecondCounter());  // :77416
std::memcpy (&buffer[lastUsedTimeOffset], &now, sizeof (now));                      // :77417
juce::FileOutputStream outStream (file);               // :77419  ← 位置 = N（末尾）
if (! outStream.openedOk()) return;                    // :77420
outStream.write (buffer.data(), buffer.size());        // :77422  ← N バイトを**追記**
```

```
read    → N bytes を buffer に取得
modify  → buffer[lastUsedTimeOffset] のみ差替
open    → OPEN_ALWAYS + FILE_END（位置 = N）
write   → buffer の N バイトを位置 N から追記
結果    → ファイルサイズ = N + N = 2N
```

**訂正後の定義: `touch()` は canonical target を truncate するのではなく、既存ファイルの**後ろに内容を追記**するため、artifact のサイズ整合性を破壊する。**

#### 3.1.4 v3 採用条件（`actualFileSize == expectedFileSize`）との相互作用

| 項目 | 値 |
| --- | --- |
| `touch()` 直前の `actualFileSize` | `N = expectedSize` |
| `touch()` 直後の `actualFileSize` | `2N` |
| `touch()` 直後の `header.totalFileSize` | `N`（buffer 内は書き換えられていない。`:77417` は `lastUsedTime` のみ） |
| v3 canonical 条件 `actualFileSize == expectedSize` | **不成立** |

→ **`touch()` 後の artifact は v3 contract では必ず `Malformed` になる。**（DESIGN-VALIDATED として §12.4 に契約化）

#### 3.1.5 追加の SOURCE-PROVEN 事実（本次新規）

| # | 事実 | 出典 |
| --- | --- | --- |
| C1-a | **truncate ではないため、ファイルは越来越大**。`touch()` が N 回呼ばれると `actualFileSize` は `(N+1) × expectedSize` になる | `OPEN_ALWAYS` + `FILE_END` の累積 |
| C1-b | **`load()` は依然として成功する**。`header` は buffer 先頭 76 bytes、IR/rho/theta は続く `N-76` bytes にあるため、`:77248`/`:77269-77276`/`:77283-77290` は最初の N bytes を正しく読む。→ **サイズ異常は `load()` から不可視** | `:77248`, `:77269-77290` |
| C1-c | **`evictLRU` も異常を検出しない**。`:77452` は最初の 76 bytes だけ読むので `magic` は一致し `timedFiles` に入る | `:77445-77458` |
| C1-d | **`save()` だけがサイズを修復する**。`ReplaceFileW` は新しい staging file で丸ごと置き換えるため、`save()` 成功時に限り `actualFileSize == expectedSize` に戻る | `:77321`, `:77376` |
| C1-e | **sharing mode は `FILE_SHARE_READ` のみ**。`touch()` の `FileOutputStream` は同時に別の `GENERIC_WRITE` open を許さない。→ `touch()` と `save()` は同一ファイル上で競合しうる（`save` 失敗 or `touch` 失敗） | `:432` `FILE_SHARE_READ` |
| C1-f | **`outStream.write(...)` の戻り値は無視**（`:77422`）。部分書き込みが検知されない | `:77422` |
| C1-g | `:77394-77395` のソースコメント「FileOutputStream で開くとファイルが切り詰められるため、read-modify-write で行う」は**その根拠が誤り**。実際の誤 Description は「追記になる」こと。**コメントと実装の不一致**（D-1R-2H が指摘した実測上の問題との関連） | `:77394-77395` |

#### 3.1.6 訂正後の判定

| 項目 | 判定 |
| --- | --- |
| `touch()` が canonical target を truncate する | **取り下げ**（SOURCE-PROVEN により反証） |
| `touch()` がファイルサイズを 2 倍にする（append 機構） | **SOURCE-PROVEN** |
| `touch()` が v3 の `actualFileSize == expectedSize` を破壊する | **DESIGN-VALIDATED**（v3 contract を前提とした帰結） |
| `touch()` が `load()` から不可視である | **SOURCE-PROVEN** |
| `CacheManager::touch` にも同種の破壊がある | **NO**（`:46425-46452` は in-memory `cacheMap`/`lruList` のみ更新し、ファイル I/O を行わない。**無害**。SOURCE-PROVEN） |

### 3.2 訂正 C2: maxCacheEntries の data flow（Owner 指示 §3）

#### 3.2.1 D-1R-2H の誤った主张

D-1R-2H §13.5 は以下と記載した:

> `:49062` `const size_t cacheLimit = getMaxCacheEntries();` が**メモリ cache** の eviction limit にも使われる。`cacheLimit == 0` のときメモリ cache 側の挙動は本監査の範囲外（`evictOldestCacheEntry` の実装未読）。→ NOT PROVEN

**この主張は誤りである。** `:49062` は `getMaxCacheEntries()` の呼び出し部位のように見えるが、`cacheLimit` は `CacheManager`（C2）系に消費されるものであり、`ConvolverProcessor::irCache`（C3）の上限ではない。

#### 3.2.2 一次資料による反証

`ConvolverProcessor` の cache メンバ（`:57160-57169`）:
```cpp
struct CacheEntry
{
    std::unique_ptr<juce::AudioBuffer<double>> ir;
    std::vector<convo::SecondOrderAllpass> allpassSections;
    uint32_t lastUsedTime;
};

std::map<IRCacheKey, CacheEntry> irCache;
juce::CriticalSection cacheMutex;
static constexpr size_t MAX_CACHE_ENTRIES = 8;   // ← 独立定数。maxCacheEntries ではない
void evictOldestCacheEntry();
```

`evictOldestCacheEntry()`（`:49709-49727`）:
```cpp
void ConvolverProcessor::evictOldestCacheEntry()
{
    const juce::ScopedLock sl(cacheMutex);
    if (irCache.size() <= MAX_CACHE_ENTRIES) return;    // ← MAX_CACHE_ENTRIES
    auto oldest = irCache.begin();
    uint32_t minTime = std::numeric_limits<uint32_t>::max();
    ...
}
```

→ **`irCache` の上限は `MAX_CACHE_ENTRIES = 8` であり、`maxCacheEntries` とは無関係。**（SOURCE-PROVEN）

#### 3.2.3 訂正後の完全分離（§5 に詳述）

3 系統は **完全に独立**。Owner の指示 §3 と一致する。

#### 3.2.3b 追加の SOURCE-PROVEN 事実（本次新規）

| # | 事実 | 出典 |
| --- | --- | --- |
| C2-a | **`ConvolverProcessor::clearCache()` は C2 のみ clear**。`cacheManager->clear()` を呼ぶだけで、**C1（MixedPhase disk cache）を消さない** | `:49003-49008` |
| C2-b | **C1 を clear できる公開経路は `MixedPhasePersistentCache::clear()` のみ**。`evictLRU(0)`（`:77429`）経由以外には到達しない | `:77427-77431`, `:77480-77490` |
| C2-c | **`ConvolverProcessor` に C1 の `remove()` 経路は存在しない** | `MixedPhasePersistentCache::remove` の外部参照 0 件（§5.3） |
| C2-d | **`getMaxCacheEntries()` は C1 と C2 の両方に影響するが clamp が異なる**。setter は `jlimit<size_t>(1, 64, ...)`、snapshot 復元は `jlimit(0, 64, ...)` | `:48988` vs `:52927` |
| C2-e | **`CacheManager::evictLRU` の default 引数は 10**（`maxCacheEntries` とは別既定） | `:46570` |

### 3.3 訂正 C3: PROVEN / DESIGN-VALIDATED の分離（Owner 指示 §4）

D-1R-2H は設計判断を `PROVEN` と表記していた。本監査で以下のように**再分類**する。

#### 3.3.1 SOURCE-PROVEN（現行 source から直接証明できる事実）

| # | 事実 | 出典 |
| --- | --- | --- |
| S-01 | 現在の persistent cache version = 2 | `:77575` |
| S-02 | 現在の filename 生成は `toHexString(int64)` = **variable-length** hash | `:77220` + `juce_String.cpp:2625-2633` |
| S-03 | `load()` は header upper-bound validation をしていない | `:77263-77266`（`<= 0` のみ） |
| S-04 | `load()` は payload（IR / rho / theta）の finite 検証をしていない | `:77228-77296` 全体 |
| S-05 | `load()` は `actualFileSize` / `totalFileSize` を検証していない | 同上 |
| S-06 | `load()` は identity を **tolerance 比較**（`> 1.0e-6`）する | `:77255`, `:77257`, `:77258` |
| S-07 | `touch()` が別 handle で書き戻している（read handle → close → write handle） | `:77396-77409`, `:77419` |
| S-08 | `touch()` は追記によりファイルサイズを 2 倍にする | `OPEN_ALWAYS` + `FILE_END`（§3.1.2） |
| S-09 | `touch()` の `write()` 戻り値を無視する | `:77422` |
| S-10 | `evictLRU(0)` が `clear()` を呼ぶ | `:77427-77431` |
| S-11 | `maxCacheEntries` の既定値が 0（snapshot / override 両方） | `:55888`, `:56882` |
| S-12 | snapshot 復元の `jlimit` 下限が 0（setter は 1） | `:52927` vs `:48988` |
| S-13 | `setMaxCacheEntries` を受ける前に `static_cast<int>` / `static_cast<size_t>` が評価される | `:53108`, `:13610` |
| S-14 | `save()` 後に `save` の return value を無視して `evictLRU` している | `:50440-50446` |
| S-15 | `save()` は input validation を 全く行っていない | `:77302-77377` 全体 |
| S-16 | `clear()` は `*.mph` を無条件削除する（version / identity 検査なし） | `:77486-77489` |
| S-17 | `evictLRU` は `magic` のみ検査し version / identity を問わない | `:77454` |
| S-18 | `remove()` は分類検査なしでパス削除する | `:77474-77477` |
| S-19 | `getEntryCount()` は `*.mph` 総数を返す（分類なし） | `:77499-77500` |
| S-20 | 現行 `.bak` は**存在しない**（`lpBackupFileName = nullptr`） | `juce_Files_windows.cpp:376` |
| S-21 | staging file は `<hash>_temp<randhex>.mph`（cache directory 内、`*.mph` namespace） | `juce_TemporaryFile.cpp:61-70` + `:77220` |
| S-22 | producer は `rho`/`theta` を同一ループで `push_back` する | `:50434-50439` |
| S-23 | producer は `mixedIR` の全 channel × 全 sample に `isfinite` を検査する（save より前） | `:50346-50357` |
| S-24 | `MixedPhasePersistentCache::remove/clear/getEntryCount/touch` の**外部呼び出し元は 0 件** | §5.3 |
| S-25 | `MixedPhasePersistentCache` への参照は test 側 1 ファイル 1 箇所のみ。専用 unit test が存在しない | §14.1 |
| S-26 | `producer` は `f2 >= f1 + 10` を `jlimit(max(MIXED_F2_MIN_HZ, f1 + 10.0f), ...)` で強制する | `:50704-50706`, `:50795-50797`, `:52906-52908` |
| S-27 | `F1 ∈ [100, 400]`, `F2 ∈ [700, 1300]` の定数が存在する | `:56024-56028` |
| S-28 | `PhaseMode ∈ {AsIs=0, Mixed=1, Minimum=2}` | `:55916-55921` |
| S-29 | in-memory `CacheEntry::lastUsedTime` は `uint32_t` | `:57163` |
| S-30 | C1 / C2 は directory と拡張子が異なる | `:77202` vs `:46123`、`:77220` vs `:46138` |

#### 3.3.2 DESIGN-VALIDATED（改修後 contract として妥当。**現行では成立していない**）

| # | contract | 現行状態 |
| --- | --- | --- |
| D-01 | version = 3 | 未成立（S-01） |
| D-02 | 64-hex identity filename（Option A: numeric fixed-width hex） | 未成立（S-02） |
| D-03 | `Canonical` / `Usable` separation | 未成立（S-03〜S-06） |
| D-04 | `evictLRU(0) = no-op` | 未成立（S-10） |
| D-05 | `if (saved) evictLRU(...)` | 未成立（S-14） |
| D-06 | power-loss durability を out-of-contract | **運用上の宣言のみ。現行コードに明示表現なし** |
| D-07 | corrupt artifact を never accept | 未成立（S-04, S-05） |
| D-08 | `.mph.stage` staging namespace 分離 | 未成立（S-21） |
| D-09 | `touch()` を load lifecycle に統合 | 未成立（S-07〜S-09） |
| D-10 | residue classification（`SafeToStartNewTransaction` 等） | 未成立（`save()` に判定ロジックなし、S-15） |
| D-11 | normal save を `MoveFileExW` 系 transaction にする | 未成立（`ReplaceFileW` 使用、S-20） |
| D-12 | `lastUsedTime` を same-handle 更新にする | 未成立（S-07） |
| D-13 | `GetSystemTimeAsFileTime` を wall-clock timestamp にする | 未成立（`getMillisecondCounter()` 32-bit） |
| D-14 | destructive API = `CurrentV3Canonical` only | 未成立（S-16〜S-19） |
| D-15 | header upper-bound validation（channels / samples / sections） | 未成立（S-03） |
| D-16 | `expectedSize` の exact-size 3 条件 + checked arithmetic | 未成立（S-05） |

#### 3.3.3 混同してはならない点（Owner 指示の核心）

> **`load()` が `actualFileSize` を検証しない（S-05）は SOURCE-PROVEN。
> `actualFileSize == expectedSize` を canonical 条件にすることは DESIGN-VALIDATED。
> 前者は「既に検証している」ことを根拠としてよいと判断してはならない。**

D-1R-2H の一部の記述（「`actualFileSize == expectedSize` 条件が機能する」）は、この区別が不完全であったため本節で再分類する。

---

## §4. 最新 source との再照合（D-1R-2H の主張の行番号・内容の再検証）

| D-1R-2H の主張 | 再照合結果 | 判定 |
| --- | --- | --- |
| `kVersion = 2` が `:77575` | 一致 | **SOURCE-PROVEN** |
| `struct DiskHeader` が `:77578-:77593`（`#pragma pack(1)`、14 field） | 一致 | **SOURCE-PROVEN** |
| `sizeof(DiskHeader) == 76` | D-1R-2A の MSVC 実行プローブと一致。source 側の `static_assert(sizeof(DiskHeader) <= 128)` は `:77596` | **SOURCE-PROVEN**（値 76 は probe） |
| `MixedPhasePersistentCache::touch` が `:77383` | 一致 | **SOURCE-PROVEN** |
| `load` の identity 検証が `:77252-:77260` | 一致（`:77252` magic/version、`:77254-77260` identity） | **SOURCE-PROVEN** |
| `:77268` `outIr.setSize(numChannels, numSamples)` に上限検査なし | 一致 | **SOURCE-PROVEN** |
| `evictLRU` の magic のみ検査が `:77454` | 一致 | **SOURCE-PROVEN** |
| `clear`/`getEntryCount` の `*.mph` 列挙が `:77487`/`:77499` | 一致 | **SOURCE-PROVEN** |
| `save()` の I/O failure 経路 E1〜E8 | 一致（`:77317`, `:77324`, `:77347`, `:77354`, `:77361`, `:77363`, `:77370`, `:77376`） | **SOURCE-PROVEN** |
| producer finite sweep が `:50346-:50357` | 一致 | **SOURCE-PROVEN** |
| `maxCacheEntries` F1〜F10 | 一致。ただし `:49062` の帰属が誤り（§3.2） | **訂正 C2** |
| `touch()` が truncate する | **不一致（誤り）** | **訂正 C1** |
| `MIXED_F1_MIN/MAX_HZ` の値 | 新規取得：`:56024` = 100.0f、`:56025` = 400.0f | **SOURCE-PROVEN** |
| `MIXED_F2_MIN/MAX_HZ` の値 | 新規取得：`:56027` = 700.0f、`:56028` = 1300.0f | **SOURCE-PROVEN** |

---

## §5. persistent disk cache と他 cache 系統の責務分離（Owner 指示 §3）

### 5.1 3 系統の完全分離表

| # | 系統 | class | directory | 拡張子 | 上限の出典 | LRU 機構 | clear 経路 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **C1** | **MixedPhase disk persistent cache（今回の改修対象）** | `convo::MixedPhasePersistentCache`（`:77526`） | `%APPDATA%/ConvoPeq/MixedPhaseCache`（`:77200-77202`） | `.mph`（`:77220`） | `ConvolverProcessor::getMaxCacheEntries()` → `:50446` | **ファイル内 `header.lastUsedTime` + 起動時ソート**（`:77442-77465`） | `MixedPhasePersistentCache::clear()`（`:77480`）のみ。**外部呼び出し元 0 件** |
| **C2** | PreparedIR disk cache | `convo::CacheManager`（`:46553`） | `%APPDATA%/ConvoPeq/IRCache`（`:46121-46123`） | `.bin`（`:46138`） | `evictLRU(size_t maxEntries = 10)`（`:46570`、**default 10**） | **in-memory `lruList` / `cacheMap`**（`:46425-46452`） | `CacheManager::clear()` = `dir.deleteRecursively()`（`:46460-46474`）。`ConvolverProcessor::clearCache()`（`:49007`）から到達 |
| **C3** | in-memory IR cache | `ConvolverProcessor::irCache`（`:57166`） | （メモリ） | — | **`MAX_CACHE_ENTRIES = 8`**（`:57168`、**`maxCacheEntries` と無関係**） | `evictOldestCacheEntry()`（`:49709-49727`） | なし（`clearCache()` は C2 のみ。C2-c） |

### 5.2 directory / 拡張子の非衝突（SOURCE-PROVEN）

```
C1: %APPDATA%/ConvoPeq/MixedPhaseCache/<hash>.mph
C2: %APPDATA%/ConvoPeq/IRCache/<key>_<fftSize>.bin
```

→ **C1 と C2 は directory と拡張子が共に異なる**。C1 の `*.mph` 列挙は C2 の `.bin` に影響しない。C2 の `deleteRecursively()` は `%APPDATA%/ConvoPeq/IRCache` のみを対象とし、C1 を消さない。

### 5.3 `MixedPhasePersistentCache` の外部呼び出し元（authority 全走査結果）

| member | 宣言 | 定義 | 外部呼び出し元 | 件数 |
| --- | --- | --- | --- | --- |
| `getCacheDirectory()` | `:77529` | `:77198` | 内部のみ | **0** |
| `getCacheFile(...)` | `:77531` | `:77212` | 内部のみ（`:77311`, `:77389`） | **0** |
| `load(...)` | `:77537` | `:77228` | `:49869` | **1** |
| `save(...)` | `:77546` | `:77302` | `:50440` | **1** |
| `touch(...)` | `:77555` | `:77383` | `:77293`（**`load()` 内部**のみ） | 内部のみ |
| `evictLRU(size_t)` | `:77561` | `:77425` | `:50446` | **1** |
| `remove(...)` | `:77563` | `:77468` | — | **0** |
| `clear()` | `:77569` | `:77480` | —（`evictLRU(0)` 経由 `:77429` のみ） | **0 直接** |
| `getEntryCount()` | `:77571` | `:77492` | — | **0** |
| `computeKeyHash(...)` | `:77598`（private） | `:77172` | 内部のみ（`getCacheFile` `:77218`） | **0** |

**帰結（SOURCE-PROVEN）**:
- `remove()` / `clear()` / `getEntryCount()` / `touch()` は **外部呼び出し元 0 件**。N4 の「`touch()` public API 自体は不要になる可能性が高い」に対し、**PROVEN で外部 caller ゼロが確認されたため、削除可能**。
- `evictLRU(0) → clear()` は `:50446` からのみ到達可能。
- `ConvolverProcessor` には C1 を操作する公開 API がない（C2-c）。

### 5.4 改修範囲の明示（Owner 指示 §3 末尾）

> **本改修の作用域は C1（`convo::MixedPhasePersistentCache`、`%APPDATA%/ConvoPeq/MixedPhaseCache/*.mph`）に限定する。**

C2（`CacheManager` / `IRCache/*.bin`）と C3（`ConvolverProcessor::irCache` / `MAX_CACHE_ENTRIES = 8`）は本計画の変更対象外。ただし §5.1 の分離を明記することで、`getMaxCacheEntries()` が C1 と C2 の双方に効くという**副作用（C2-d）**だけを明示的に扱う。

---

## §6. N1〜N5 の裁定結果（Owner 裁定の反映）

### 6.1 N1 — `CurrentV3NonCanonical` を独立 state として保持

**Owner 裁定**: 独立 state として保持。ただし enum の実装方法（8 値 enum そのものか、`ArtifactState + VersionClass + Canonicality` の分解か）は固定しない。**重要なのは判定結果として区別を失わないこと。**

本監査は Owner's 分解方向を妥当と判断し、**3 軸分解**を採用する（DESIGN-VALIDATED）。

```text
VersionClass:
    Obsolete   (version <  3)     ← v2
    Current    (version == 3)
    Future     (version >  3)

Canonicality:
    Canonical       (filename == lowercaseEncode(headerIdentity))
    NonCanonical    (Decodeable だが canonical でない：大文字小文字 / 桁数 / prefix 不一致)
    Unparseable     (64-hex として解釈不能)

Integrity:
    Complete     (actualFileSize == expectedSize && header.totalFileSize == expectedSize)
    Incomplete   (サイズ不一致)
    Unreadable   (header 読取不能 / magic 不一致)

RecoveryCandidateFlag:
    (ValidRecoveryCandidate = Current ∧ RecoveryNamespace ∧ selfConsistent)
```

**合成された ArtifactState（Owner の 8 値と 1 対 1）**:

| ArtifactState | VersionClass | Canonicality | Integrity | recovery ns |
| --- | --- | --- | --- | --- |
| `Absent` | — | — | — | — |
| `CurrentV3Canonical` | Current | Canonical | Complete | no |
| `CurrentV3NonCanonical` | Current | NonCanonical / Unparseable | Complete | no |
| `Future` | Future | 任意 | 任意 | no |
| `Obsolete` | Obsolete | 任意 | 任意 | no |
| `Malformed` | Current | 任意 | Incomplete / Unreadable | no |
| `NonCanonical` | 任意 | NonCanonical / Unparseable | Complete | no |
| `ValidRecoveryCandidate` | Current | Canonical（self-consistency） | Complete | **yes** |

**合成規則（決定性）**:
1. ファイルが存在しなければ `Absent`
2. `Integrity == Unreadable` または `magic != kMagic` なら `Malformed`
3. `VersionClass == Future` なら `Future`
4. `VersionClass == Obsolete` なら `Obsolete`
5. recovery namespace（`.stage` / `.bak`）なら `ValidRecoveryCandidate`（filename↔header self-consistency 成立時）または `Malformed`
6. `Integrity == Incomplete` なら `Malformed`
7. `Canonicality == Canonical` なら `CurrentV3Canonical`
8. `VersionClass == Current` なら `CurrentV3NonCanonical`
9. それ以外は `NonCanonical`

**判定**: 合成規則は **total**（9 ステップで必ず 1 つの値を返す）かつ **disjoint**（ステップ 2〜9 は排他的に順序適用）。**DESIGN-VALIDATED**。

### 6.2 N2 — attributes / streams を canonicality に含めない

**Owner 裁定**: 論理 identity は `filename identity + header + payload + expected size` で定義。Windows attributes / ADS / ACL を DSP cache の canonicality 判定に入れない。ただし 1177 recovery は通常 save transaction と混在させない。**1177 recovery を新 v3 normal save transaction の基本経路にしない**。normal save は `MoveFileExW` 系として設計する。

本監査は Owner's 裁定を**完全に採用**し、以下を確定する:

| 項目 | 確定内容 |
| --- | --- |
| canonicality の判定入力 | `filename identity` / `header` / `payload` / `expected size` のみ |
| attributes / ADS / ACL | **判定対象外**。PROVEN 相当の根拠なし（attributes は DSP 出力に影響しない） |
| 1177 の位置づけ | **SM-B（legacy/recovery-specific）**。SM-A（normal save）の failure state として設計しない |
| normal v3 save | **`MoveFileExW` 系**。`ReplaceFileW` の 1175/1176/1177 を基本機構にしない |
| 1177 の扱い | 「legacy/recovery-specific analysis」として文書化のみ。**production code path に入れない** |

**判定**: **DESIGN-VALIDATED**（Owner 裁定の採用）。現行 S-20 は `ReplaceFileW` を使用しているため、**normal save の API 変更が必然**（D-11）。

### 6.3 N3 — `save()` の bool 契約維持

**Owner 裁定**: `save() == true ⇔ 新しい usable canonical target が成立した`。residue 状態を理由に `false` にするのは避ける。ただし今回は複雑な diagnostic API の追加までは要求しない。まず `None / PendingCleanup / PendingRecovery` を**内部 transaction state** として成立させられるかを監査する。外部 API を増やす必要があるかは別途提案。

本監査は**内部 state のみ**での実現が成立すると判断した（DESIGN-VALIDATED）。

```cpp
// 内部状態（外部 API には出さない）
struct SaveTransactionOutcome
{
    bool                 saved;              // Owner の bool 契約そのもの
    ResidueDisposition   residue;            // None / PendingCleanup / PendingRecovery
};

// save() 公開シグネチャは不変
static bool save(uint64_t fileHash, double sampleRate, int phaseMode,
                 float freqStartHz, float freqEndHz, int targetLength,
                 const juce::AudioBuffer<double>& ir,
                 const std::vector<double>& rho,
                 const std::vector<double>& theta);
```

**外部 API を増やさない根拠**:
- `ResidueDisposition` は `saveImpl` のローカル変数として生成・破棄される
- `residue != None` の場合は次回 `save` の入口（residue classification）で必ず消費される（§10）
- **したがって caller に伝える必要がない**（§10 の T は total かつ冪等）
- diagnostics に出したければ次回 GO 後に別途提案する（§15 Phase 5）

**判定**: **DESIGN-VALIDATED**。**外部 API 追加は現時点で不要**。

### 6.4 N4 — `touch()` を load lifecycle に統合

**Owner 裁定**: 永続 LRU を維持する方向で検証。public / 独立 `touch()` は残さず、`load()` lifecycle 内に統合する。caller 全探索を行い廃止可能性を確認。LRU timestamp 更新失敗は cache load failure と**同一視しない**（payload が usable なら load 成功、timestamp 更新失敗は diagnostic）。

本監査は Owner's 方針を**完全採用**し、**caller 全探索の結果 PROVEN で廃止可能性を確定**した。

#### 6.4.1 caller 全探索結果（SOURCE-PROVEN）

| member | 外部呼び出し元 | 件数 |
| --- | --- | --- |
| `MixedPhasePersistentCache::touch` | `:77293`（`load()` 内部の 1 箇所のみ） | 外部 **0** |

→ **`touch()` は外部 caller ゼロ**。`load()` 内部化すれば **API 破壊的変更は発生しない**（PROVEN）。

#### 6.4.2 第一候補の load lifecycle（Owner 指定）

```text
load()
  Open H                                  ← read/write 両方のアクセス権で 1 つの handle
    ↓
  header validation                       ← 76 bytes を H から読む
    ↓
  payload validation                      ← IR / rho / theta を H から読む
    ↓
  same H の lastUsedTime 更新             ← offsetof へ in-place write（H は閉じない）
    ↓
  FlushFileBuffers(H)                     ← flushInternal 相当
    ↓
  Close H
    ↓
  return true
```

#### 6.4.3 解決される 5 つの問題（Owner 列挙）

| 問題 | 解決根拠 |
| --- | --- |
| read handle / write handle 分離 | 単一 handle。`FILE_SHARE_READ` による他 open の拒否（SOURCE-PROVEN `:432`）も回避 |
| load → touch 間 race | handle を開いたまま更新するため、TOCTOU 窓が消える |
| touch による別 transaction | 別 `FileOutputStream` が無くなるため、**§3.1 の追記による 2 倍化機構が構造的に消える** |
| touch 単独の破壊 | `touch()` 公開 API が消える（外部 caller 0 件、PROVEN） |
| mutex lifecycle 分断 | `load()` 内の単一区間に閉ぢる |

#### 6.4.4 LRU timestamp 更新失敗の扱い（Owner 明示要求）

```text
payload が usable なら load 成功（save/load の戻り値は true）
timestamp 更新失敗は diagnostic（戻り値には影響しない）
```

**DESIGN-VALIDATED**。理由:
- payload が usable であれば cache は機能する。LRU metadata の失敗で cache を miss 扱いすると「再計算 → 同じ cache を再 save」という不必要な churn が発生する
- `header.lastUsedTime` は §6.5 のとおり **mutable LRU metadata** であり、logical canonicality とは別軸

#### 6.4.5 残る residual

`FlushFileBuffers(H)` の失敗、および in-place write の失敗は diagnostic として扱う。ただし **in-place write 失敗がファイル整合性を損なう可能性**（write 位置のずれ）があるため、その場合は §10 の residue classification により次回 `save` 時に再構築される。これは DESIGN-VALIDATED の範囲内。

### 6.5 N5 — evictLRU ordering と timestamp validity の位置づけ

**Owner 裁定（ordering）**:
```text
1. valid timestamp ascending
2. invalid timestamp last
3. timestamp が同じ場合は canonical filename ascending
```

**Owner 第一候補（validity の位置づけ）**: `timestamp` は mutable LRU metadata → logical canonicality とは分離。`timestamp == 0` でも artifact 自体を即 `Malformed`/`NonCanonical` 扱いせず、eviction ordering では `invalid timestamp` として扱う。

本監査は**完全に採用**（DESIGN-VALIDATED）。

```text
eviction ordering key :=
    ( isInvalidTimestamp ? 1 : 0,      /* invalid は末尾 */
      isInvalidTimestamp ? 0 : ts,     /* valid は昇順 */
      canonicalFilename )              /* 同一 ts のとき昇順 */
```

**現行との差（SOURCE-PROVEN）**: 現行の `std::sort`（`:77460-77461`）は `(lastUsedTime, File)` ペアの辞書順で、**invalid を末尾に置く規則がない**。また `File::operator<` は path 比較であり「canonical filename ascending」と同義ではない（`*.mph` 以外的ファイルが混在しうるため）。

**重要**: `timestamp == 0` の artifact は **canonical として保持される**（eviction 対象になるが、削除されるとは限らない）。したがって `timestamp validity` を `ArtifactState` の入力に**入れない**。§6.1 の合成規則（9 ステップ）に timestamp 判定は存在しない。**この分離は Owner の裁定どおり。**

### 6.6 N1〜N5 裁定の適用結果サマリ

| 裁定 | 反映先 | 現行状態 |
| --- | --- | --- |
| N1 独立 state | §6.1 の 3 軸分解 + 9 ステップ合成規則 | 新規（現行に state 分類なし） |
| N2 attributes 除外 + 1177 分離 | §6.2 + §13 | **normal save の API 変更が必然**（S-20） |
| N3 bool 契約維持 + 内部 residue | §6.3 | 外部 API 追加不要 |
| N4 touch 統合 | §6.4 + §12.4 | caller 0 件で**廃止可能（PROVEN）** |
| N5 ordering + validity 分離 | §6.5 | 現行 sort は規則に不適合（SOURCE-PROVEN） |

---

## §7. v3 定数一覧（Owner 指定候補の検証）

### 7.1 Owner 指定定数と検証結果

```cpp
constexpr uint32_t kPersistentCacheVersion              = 3;
constexpr double   kPersistentCacheMinSampleRate        = 44100.0;
constexpr double   kPersistentCacheMaxSampleRate        = 768000.0;
constexpr int      kMinPersistentCacheChannels          = 1;
constexpr int      kMaxPersistentCacheChannels          = 8;
constexpr int      kMinPersistentCacheSamples           = 1;
constexpr int      kMaxPersistentCacheSamples           = 2097152;
constexpr int      kMinPersistentCacheTargetLength      = 1;
constexpr int      kMaxPersistentCacheTargetLength      = 2097152;
constexpr int      kMaxPersistentCacheSections          = 20;
constexpr int      kMinPersistentCacheEntries           = 1;
constexpr int      kDefaultPersistentCacheEntries       = 8;
constexpr int      kMaxPersistentCacheEntries           = 64;
```

### 7.2 各定数の source 裏付け

| 定数 | Owner 指定値 | SOURCE-PROVEN 裏付け | 判定 |
| --- | --- | --- | --- |
| `kPersistentCacheVersion = 3` | 3 | 現行は 2（`:77575`）。3 は Owner specification | **DESIGN-VALIDATED**（Owner spec） |
| `kPersistentCacheMaxSampleRate = 768000.0` | 768000 | `:45872` `static constexpr int kMaxSampleRate = 768000;` | **SOURCE-PROVEN**（上限側） |
| `kPersistentCacheMinSampleRate = 44100.0` | 44100 | **source 内に 44100 の既存定数は存在しない**。Owner が明示的に「今回の Owner specification であり、source 内の既存定数から PROVEN されたものとは扱わないでください」と指示 | **DESIGN-VALIDATED**（Owner spec のみ。SOURCE-PROVEN ではない） |
| `kMaxPersistentCacheChannels = 8` | 8 | `:53874` `inline constexpr unsigned kMaxIRLoadChannels = 8u;` | **SOURCE-PROVEN** |
| `kMinPersistentCacheChannels = 1` | 1 | producer は `numChannels >= 1`（`:49901-49902` で `<= 0` を return） | **DESIGN-VALIDATED** |
| `kMaxPersistentCacheSamples = 2097152` | 2097152 | `:56049` `static constexpr int MAX_IR_LATENCY = 2097152;` | **SOURCE-PROVEN** |
| `kMinPersistentCacheSamples = 1` | 1 | 同上（`> 0` のみ意味を持つ） | **DESIGN-VALIDATED** |
| `kMinPersistentCacheTargetLength = 1` | 1 | producer は `linearIR.getNumSamples()`（`:50417`）で `> 0` が保証される（`:49902`） | **DESIGN-VALIDATED** |
| `kMaxPersistentCacheTargetLength = 2097152` | 2097152 | `:56049` と同源 | **DESIGN-VALIDATED**（`targetLength == samples` により間接的に PROVEN） |
| `kMaxPersistentCacheSections = 20` | 20 | `:50222` で `numSections ∈ {2, 8, 20}`。最大 20 | **SOURCE-PROVEN**（producer）。cache contract 側は DESIGN-VALIDATED |
| `kMinPersistentCacheEntries = 1` | 1 | 現行 setter は `jlimit<size_t>(1, 64, ...)`（`:48988`） | **SOURCE-PROVEN**（setter 側） |
| `kDefaultPersistentCacheEntries = 8` | 8 | **現行既定値は 0**（`:55888`, `:56882`）であり snapshot 復元も下限 0（`:52927`）。8 は Owner specification。`MAX_CACHE_ENTRIES = 8`（`:57168`）は C3 の別系統であり C1 の既定値根拠にはならない | **DESIGN-VALIDATED**（Owner spec のみ） |
| `kMaxPersistentCacheEntries = 64` | 64 | 現行 setter は `jlimit<size_t>(1, 64, ...)`（`:48988`） | **SOURCE-PROVEN**（setter 側） |

### 7.3 定数配置（D-4 裁定の反映）

F1/F2 は **共通 domain 定数定義**（`ConvolverProcessor::MIXED_F1_MIN_HZ` 等、`:56024-56028`）を single source とし、`PersistentCache` に複製しない。D-4 裁定どおり「定義を 1 箇所にする」。v3 contract はこれらを参照する。

### 7.4 `phaseMode` の domain

Owner の定数一覧に `phaseMode` はない。v3 identity は `phaseModeBits` として 8 hex を使うため、**domain を明示する必要がある**。

| 項目 | 値 | 出典 |
| --- | --- | --- |
| `PhaseMode : int { AsIs = 0, Mixed = 1, Minimum = 2 }` | `:55916-55921` | **SOURCE-PROVEN** |
| C1 の producer が渡す値 | `PhaseMode::Mixed`（`:50414` → `:50441`） | **SOURCE-PROVEN** |
| v3 domain 候補 | `0 <= phaseMode <= 2`（または `{0,1,2}` の集合） | **DESIGN-VALIDATED** |

**注意**: `phaseMode` は **identity field**（filename に含まれる）であるため、domain 検証は canonicality の必須項目。SOURCE-PROVEN の enum 値 3 個に一致する集合とする。

### 7.5 max file size（Owner 指示 §7）

```
sizeof(DiskHeader)
+ 8 * 2097152 * sizeof(double)
+ 20 * sizeof(double)
+ 20 * sizeof(double)
= 76 + 134,217,728 + 160 + 160
= 134,218,124 bytes
```

**checked arithmetic による導出**（magic number 禁止）:

```cpp
constexpr uint64_t kHeaderBytes      = sizeof(DiskHeader);
constexpr uint64_t kMaxIrBytes       = static_cast<uint64_t>(kMaxPersistentCacheChannels)
                                    * static_cast<uint64_t>(kMaxPersistentCacheSamples)
                                    * sizeof(double);
constexpr uint64_t kMaxRhoBytes      = static_cast<uint64_t>(kMaxPersistentCacheSections) * sizeof(double);
constexpr uint64_t kMaxThetaBytes    = static_cast<uint64_t>(kMaxPersistentCacheSections) * sizeof(double);

// すべての積は uint64_t に昇格してから計算する
constexpr uint64_t kMaxPersistentCacheFileBytes =
    kHeaderBytes + kMaxIrBytes + kMaxRhoBytes + kMaxThetaBytes;   // == 134,218,124
```

**前提条件（Owner 明示要求）**:
```text
expectedSize <= INT64_MAX        ← header.totalFileSize（int64_t）への格納前提
```

検証: `expectedSize <= 134,218,124 << INT64_MAX (9,223,372,036,854,775,807)`。
ただし**実装では `expectedSize` を計算した後、明示的に `expectedSize <= INT64_MAX` を検査してから `header.totalFileSize` へ格納する**（型変換前に検査する）。

**判定**: **DESIGN-VALIDATED**（算術値は D-1R-2A で PROVEN、`INT64_MAX` 検査は新規契約）。

---

## §8. v3 identity serialization（Owner 指示 §8 の採用）

### 8.1 Option A: numeric fixed-width hexadecimal

```text
fileHash          16 hex   (uint64_t,           zero-padded lowercase)
sampleRateBits    16 hex   (bit_cast<uint64_t>,zero-padded lowercase)
phaseModeBits      8 hex   (static_cast<uint32_t>(int32_t))
freqStartBits      8 hex   (bit_cast<uint32_t>(float))
freqEndBits        8 hex   (bit_cast<uint32_t>(float))
targetLengthBits   8 hex   (static_cast<uint32_t>(int32_t))
----------------------------------------------------------------
                   64 hex
```

**「little-endian」という表現を使用しない**（Owner 明示指示）。本 serialization は byte 並びの表現ではなく**数値の 16 進表記**であるため、endian の概念を含まない。

### 8.2 canonical / decodeable の分離（Owner 指示 §8 末尾）

```text
decodeable : filename が 64 桁 [0-9a-fA-F] + 既定 extension として解釈可能
canonical  : decodeable かつ actual filename == lowercaseEncode(decoded identity)
```

| ファイル名 | decodeable | canonical | load target | evict/clear/remove | 自動 rename |
| --- | --- | --- | --- | --- | --- |
| `<64hex lower>.mph` | yes | **yes** | 採用 | 対象 | しない |
| `<64HEX UPPER>.mph` | yes | **no** | 不採用 | 対象外 | **しない** |
| `<63hex>.mph` / `<65hex>.mph` | no | no | 不採用 | 対象外 | しない |
| `<64hex>.mph.stage` | yes | no | recovery candidate | 対象外 | しない |
| `<64hex>.mph.bak` | yes | no | recovery candidate | 対象外 | しない |

### 8.3 判定

| 項目 | 判定 |
| --- | --- |
| 64-hex Option A の妥当性 | **DESIGN-VALIDATED**（round-trip deterministic。§8.4） |
| 現行 v2 との非互換 | **SOURCE-PROVEN**（`:77220` の可変長 hash。JuCE unit test `juce_String.cpp:2625-2633` が zero-pad なしを証明） |
| 全 v2 artifact が v3 で保護される | **DESIGN-VALIDATED**（可変長 → `NonCanonical` かつ `version == 2` → `Obsolete`。二重保護） |

### 8.4 round-trip determinism

`decode → encode → decode` は不動点到達。各 field は固定幅であり、`bit_cast` は bit pattern を platform 依存なく保存・復元し、`static_cast<uint32_t>(int32_t)` は two's complement を保持する。**DESIGN-VALIDATED**。

### 8.5 NaN / Inf identity の扱い

domain validator が `isfinite(sampleRate)` / `isfinite(freqStart)` / `isfinite(freqEnd)` を要求するため、NaN/Inf bit pattern を持つ identity は構造的に canonical になり得ない。→ **`NaN/Inf identity = invalid artifact`**（§10 の `Malformed`）。

**判定**: **DESIGN-VALIDATED**。
---

## §9. Canonical / Usable / Recovery の責務境界（Owner 指示 §9）

### 9.1 3 概念の定義

```text
Canonical
    = filename / header / identity / shape / size が正しい
    （filename == lowercaseEncode(headerIdentity)、headerReserved == 0、
      version == 3、channels/samples/targetLength/sections が domain 内、
      samples == targetLength、freqStart/freqEnd/sampleRate が domain 内、
      actualFileSize == expectedFileSize、header.totalFileSize == expectedFileSize）

Usable
    = Canonical
    + payload semantic validation
    （IR 全要素 finite、rho 全要素 finite、theta 全要素 finite、
      rho.size() == theta.size()）

ValidRecoveryCandidate
    = recovery namespace にある（.mph.stage / .mph.bak）
    + filename ↔ header の self-consistency 成立
    + Canonical 条件のうち identity / size を満たす
```

### 9.2 API ごとの受理要件

| API | 受理要件 | 理由 |
| --- | --- | --- |
| `load(key)` | **Usable** | Canonical のみでは payload NaN が DSP へ渡る（S-04） |
| startup `recovery` | **ValidRecoveryCandidate** → restore → **Usable re-validation** | recovery は「候補」であって「受理」ではない |
| `evictLRU()` 対象 | **CurrentV3Canonical** のみ | Future / Obsolete / Malformed / NonCanonical / `.stage` / `.bak` は保護 |
| `remove()` 対象 | **CurrentV3Canonical** のみ | 同上 |
| `clear()` 対象 | **CurrentV3Canonical** のみ | 同上 |
| `getEntryCount()` 対象 | **CurrentV3Canonical** のみ | Owner 指示 §9 に基づく（現行 S-19 は分類なし） |

### 9.3 recovery の 2 段構え（Owner 明示要求）

```text
recovery:
    candidate = classifyRecoveryCandidate(kind, filename, header, actualFileSize)
    if (candidate.state != ValidRecoveryCandidate) return NotRecovered
    restore candidate → target                  ← 同一 identity の target へ
    usable = classifyUsable(target, actualFileSize)
    if (usable.state != CurrentV3Canonical)    ← structural + payload の再検証
    {
        target を invalid 扱い
        return NotRecovered
    }
    return Recovered
```

**判定**: **DESIGN-VALIDATED**。現状は `load` に payload 検証がゼロ（S-04）かつ `actualFileSize` 検証がゼロ（S-05）なので、この 2 段構えは新規契約。

### 9.4 「save failure で usable target 成立済みなのに false を返さない」（N3 との整合）

`save()` の `true` 条件は §11 の S1〜S8 の連鎖。その連鎖の終端で **target の Usable 検証まで含む**ため、`true` を返した時点で「新しい usable canonical target が成立した」ことになる。residue の有無は `true`/`false` に影響しない。→ N3 の Owner 裁定（§6.3）と整合（DESIGN-VALIDATED）。

---

## §10. State machine と residue decision table

### 10.1 Artifact state の判定順序（§6.1 の 9 ステップ）

省略せず再掲（decision table の入力）。

| step | 条件 | 結果 |
| --- | --- | --- |
| 1 | ファイル不在 | `Absent` |
| 2 | header 読取不能 or `magic != kMagic` | `Malformed` |
| 3 | `version > 3` | `Future` |
| 4 | `version < 3` | `Obsolete` |
| 5 | recovery namespace（`.stage` / `.bak`） | self-consistency 成立 → `ValidRecoveryCandidate` / 不成立 → `Malformed` |
| 6 | `actualFileSize != expectedSize` or `header.totalFileSize != expectedSize` | `Malformed` |
| 7 | `filename == lowercaseEncode(headerIdentity)` | `CurrentV3Canonical` |
| 8 | `version == 3` | `CurrentV3NonCanonical` |
| 9 | それ以外 | `NonCanonical` |

**total / disjoint**（9 ステップで必ず 1 値、ステップ 2〜9 は排他的）。timestamp validity は**含まれない**（N5、§6.5）。

### 10.2 Residue classification（save 入口、Owner 指示 §2-1）

```text
classifyResidue(identity) -> ResidueVerdict {
    SafeToStartNewTransaction,
    RecoverableResidue,
    UnsafeResidue,
}
```

`save()` の入口順序（§11 の先頭 3 段と同一）:

```text
1. input validation
2. classifyResidue(identity)
3. if (RecoverableResidue)  -> runRecovery(identity) ; re-classify
4. if (UnsafeResidue)       -> return false（既存 artifact を一切触らない）
5. if (SafeToStartNewTransaction) -> proceed to transaction
```

**核心規則**: 「`save` が stale residue を見つけたからといって既存 artifact を勝手に削除しない」。destructive cleanup は `SafeToStartNewTransaction` かつ cleanup 対象が**同一 identity** の stale residue に限定される。

### 10.3 identity membership 条件（Owner 指示 §12）

```text
membership := filename の 64-hex identity prefix が完全一致すること
```

**filename suffix（`.stage` / `.bak`）だけでは membership を証明しない。**

```text
identity A : A.mph, A.mph.stage, A.mph.bak      → transaction A
identity B : B.mph, B.mph.bak                   → transaction B（別）

A.mph.stage と B.mph.bak は同一 transaction ではない
→ 異なる identity の residue を自動削除・混合・復元しない
```

**判定**: **DESIGN-VALIDATED**。現行は hash 方式（S-02）のため identity を filename から復元できず、membership 判定ができない（**CONTRADICTED 相当**）。

### 10.4 Table A: stage の disposition（同一 identity に限定）

| stage state | disposition | 根拠 |
| --- | --- | --- |
| `Absent` | 何もしない | — |
| `CurrentV3Canonical` | **削除**（stale residue。同一 identity の新 stage を作成してよい） | target が valid なら stage は前回の残骸 |
| `CurrentV3NonCanonical` | **削除**（同一 identity の stale residue） | §10.2 核心規則の例外（同一 identity 限定） |
| `Future` | **削除しない**（preserve 契約） | Future は自動削除しない |
| `Obsolete` | **削除しない**（preserve 契約） | v2 は preserve |
| `Malformed` | **削除**（同一 identity の破損 residue。identity は prefix で確定済み） | §10.2 核心規則の例外 |
| `NonCanonical`（stem 以外 / 桁数不正） | **削除しない**（identity を確定できない） | membership 不明 |
| `ValidRecoveryCandidate` | §10.5 Table B の target/bak 側ルールに従う | — |

**注**: `Future` / `Obsolete` は **version が stem からしか判定できない**（filename の 64-hex identity は version を含まない）ため、stem を `<64hex>.stage` として version を読む。読み取れない場合は `Malformed` として stage のみ削除（canonical `*.mph` namespace には属さないため destructive API の保護外）。

### 10.5 Table B: target × bak の decision table（49 セル、同一 identity 限定）

凡例: **KEEP** = 保持 / **DEL** = 削除 / **REC** = recovery 試行 / **NOOP** = 何もしない

| target \ bak | Absent | CurrentV3Canonical | CurrentV3NonCanonical | Future | Obsolete | Malformed | NonCanonical | ValidRecoveryCandidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Absent** | NOOP（save 新規作成） | REC(bak→target) | NOOP（bak 無効） | NOOP（preserve） | NOOP（preserve） | NOOP | NOOP（identity 不明） | REC(bak→target) |
| **CurrentV3Canonical** | NOOP（save 置換） | REC(bak→target) | NOOP（bak 無効） | NOOP（preserve） | NOOP（preserve） | NOOP | NOOP | NOOP（target 有効。bak は不要） |
| **CurrentV3NonCanonical** | NOOP（save 置換） | REC(bak→target) | NOOP | NOOP（preserve） | NOOP（preserve） | NOOP | NOOP | REC(bak→target) |
| **Future** | NOOP（save 失敗・UnsafeResidue） | NOOP（**target 保全**。bak は候補だが target を壊さない） | NOOP | NOOP（保全） | NOOP（保全） | NOOP | NOOP | NOOP（保全） |
| **Obsolete** | NOOP（save 置換可。Obsolete は load 対象外だが置換は可） | REC(bak→target) | NOOP | NOOP（保全） | NOOP（保全） | NOOP | NOOP | REC(bak→target) |
| **Malformed** | NOOP | REC(bak→target) | NOOP | NOOP（Future 保全優先） | NOOP | NOOP | NOOP | REC(bak→target) |
| **NonCanonical** | NOOP | REC(bak→target) | NOOP | NOOP | NOOP | NOOP | NOOP | REC(bak→target) |

### 10.6 Table B の適用規則（49 セルを生成する total な規則）

Table B は上記 49 セルを網羅するが、実装は**順序付き first-match 規則**で表す（判定の一意性を保証するため）。

```text
R1. target.state == Future
      -> NOOP（UnsafeResidue）。save は false。既存 target を一切触らない。
         bak が ValidRecoveryCandidate でも target を置換しない
         （target の保全が最優先。Owner D-1「future は preserve」）

R2. target.state == CurrentV3Canonical
      -> NOOP（save 置換）。bak は ValidRecoveryCandidate であっても
         target が有効なので recovery を実行しない

R3. target.state == Absent
   R3a. bak.state == ValidRecoveryCandidate      -> REC(bak→target)
   R3b. bak.state ∈ {Absent, CurrentV3NonCanonical, Future, Obsolete,
                     Malformed, NonCanonical}     -> NOOP

R4. target.state ∈ {CurrentV3NonCanonical, Obsolete, Malformed, NonCanonical}
   R4a. bak.state == ValidRecoveryCandidate      -> REC(bak→target)
   R4b. otherwise                               -> NOOP
```

**Totality**: R1〜R4 は target.state の全 7 値を disjoint に被覆し、各分岐で bak.state の全 7 値を disjoint に被覆する。→ **必ず 1 つの action に到達**（DESIGN-VALIDATED）。

**Disjointness**: R1 と R2 が target.state を先に確定するため、R3/R4 は残りの 5 state のみを見る。**規則間に曖昧さはない**。

### 10.7 REC（recovery）の内部 2 段構え

```text
REC(bak→target):
    1. bak.state == ValidRecoveryCandidate を再確認（§10.1 step 5）
    2. identity prefix == target identity であることを再確認（membership）
    3. rename bak → target
    4. classifyUsable(target)                    ← §9.3 の 2 段構え
    5. classifyUsable != CurrentV3Canonical なら invalid 扱い
```

**重要**: REC は **stage を restore しない**。`target absent + bak valid + stage valid` では **bak を採用する**（D-1R-2H §3.2 の PreserveOld を、Owner N2 の「1177 を normal save に入れない」方針と整合する形で採用）。**判定**: DESIGN-VALIDATED。

### 10.8 destructive eligibility

| API | 対象 state |
| --- | --- |
| `remove(key)` | `CurrentV3Canonical` のみ |
| `clear()` | `CurrentV3Canonical` のみ（`Future` / `Obsolete` / `Malformed` / `NonCanonical` / `.stage` / `.bak` を残す） |
| `evictLRU()` | `CurrentV3Canonical` のみ（ordering は §6.5） |

**判定**: **DESIGN-VALIDATED**。現行は `clear()` が無条件削除（S-16）、`evictLRU` が magic のみ検査（S-17）、`remove()` が検査なし（S-18）。

---

## §11. Save transaction と save success contract（Owner 指示 §11）

### 11.1 Transaction 順序（Owner 指定の 11 段）

```text
input validation
    ↓
residue classification
    ↓
recovery if and only if safe
    ↓
stage creation
    ↓
stage write
    ↓
stage flush
    ↓
existing target transition
    ↓
stage → target
    ↓
target structural validation
    ↓
target usable validation
    ↓
save = true
```

### 11.2 禁止事項（Owner 明示）

| 禁止 | 理由 |
| --- | --- |
| **target を truncate してから新 IR を書き込む方式** | v3 の `MoveFileExW` 系 transaction では発生しない。§3.1 で現行 `touch()` がこの系統の**<em>不同な</em>害**（append）を起こしていた |
| **`delete old target` → `create new target`** | target 不在窓が不用意に広がる。§3 の T1/T3 |
| `.stage` / `.bak` を canonical `*.mph` namespace に置くこと | §10 の destructive eligibility が成立しなくなる。現行 S-21 がこの違反 |

### 11.3 stage 命名（destructive eligibility との整合）

```text
canonical : <64hex lower>.mph          ← *.mph に一致
stage     : <64hex lower>.mph.stage    ← *.mph に不一致
backup    : <64hex lower>.mph.bak      ← *.mph に不一致
```

JUCE `WildCardMatcher` は full match を要求する（D-1R-2F probe3 で 16/16 一致、`juce_String.cpp:1135-1169`）。したがって `*.mph` 列挙は `.stage` / `.bak` を含まない。**DESIGN-VALIDATED**。

**現行との対比**: 現行の staging は `<hash>_temp<randhex>.mph`（S-21、`juce_TemporaryFile.cpp:61-70`）で `*.mph` に**一致する**。これが §10 の destructive eligibility を現行で成立させない主因。

### 11.4 save success contract（N3、Owner の bool 契約）

```text
save() == true   ⇔ 新しい usable canonical target が成立した
save() == false  ⇔ 新しい usable canonical target が成立しなかった
```

11 段を S1〜S11 に番号付け:

| step | 条件 | 失敗時 |
| --- | --- | --- |
| S1 | input validation PASS | `false`（既存 artifact 無変更） |
| S2 | residue classification 完了 | — |
| S3 | recovery if and only if safe 完了 | `UnsafeResidue` なら `false`（既存 artifact 無変更） |
| S4 | stage creation PASS | `false`（既存 target 無傷） |
| S5 | stage complete write PASS | `false`（既存 target 無傷） |
| S6 | stage flush PASS（`FlushFileBuffers` 相当、戻り値検査） | `false`（既存 target 無傷） |
| S7 | existing target safely transitioned（`target → bak` 成功、または target 不在） | `false`。bak から target を復元してから `false` |
| S8 | stage → target PASS（`MoveFileExW` 系） | `false`。bak から target を復元してから `false` |
| S9 | target structural validation PASS | `false`。bak から target を復元してから `false` |
| S10 | target usable validation PASS | `false`。bak から target を復元してから `false` |
| S11 | `.bak` cleanup（**戻り値に無関係**） | `save = true`。`ResidueDisposition = PendingCleanup` |

**核心**: S9/S10 が `true` の時点で target は usable なので、**S11 の失敗で `false` を返してはならない**（N3）。`ResidueDisposition` は内部 state のみで保持（§6.3）。

**判定**: **DESIGN-VALIDATED**。現行は S1 なし・S6 戻り値無視（`:77373`）・S7/S8 が `ReplaceFileW`（S-20）・S9/S10 なし。

### 11.5 T0〜T5（transaction 途中状態、Owner 指示 §1）

| 状態 | target | stage | bak | 到達 | target 存否 |
| --- | --- | --- | --- | --- | --- |
| T0 | valid | valid | absent | S4 完了直後 | 有 |
| T1 | absent | valid | valid | S7 成功・S8 未実行 | **不在** |
| T2 | valid（新） | absent | valid | S8 成功・S11 未実行 | 有 |
| T3 | absent | valid | valid | S8 失敗 | **不在** |
| T4 | valid | valid | valid | S7 失敗 | 有 |
| T5 | valid | valid | valid | 既存 bak があるため S7 が失敗 | 有 |

**既存 `.bak` がある状態で S7 が失敗する理由**（Owner 指示 §2 のケース B）:

MS Learn `MoveFileExW` の `MOVEFILE_REPLACE_EXISTING`（1 / 0x1）:
> If a file named *lpNewFileName* exists, the function replaces its contents with the contents of the *lpExistingFileName* file, provided that security requirements regarding access control lists (ACLs) are met.

→ **フラグ無しでは既存 destination を置換できない**。よって:
- `既存 bak を削除 → target → bak`: 旧 bak を失う（**非推奨**）
- `target → bak`（`MOVEFILE_REPLACE_EXISTING` 付き）: 既存 bak を**上書き**（**非推奨**。step 失敗時に旧 bak へ戻れない）
- `target → bak`（フラグ無し、bak 存在）: **失敗**。target / stage とも無傷（**推奨。安全な no-op**）
- `target → bak`（フラグ無し、bak absent）: 成功。target 一時不在（T1）

→ **`target exists` かつ `bak exists` は save 開始前に必ず解消される**（§10 の residue classification + cleanup が bak を処理してから S4 に入る）。これが Owner §2 の「既存 `.bak` は startup recovery で処理してから save」方式の必然性。

**判定**: **DESIGN-VALIDATED**。MS Learn の引用は **SOURCE-PROVEN**（一次資料）。

---

## §12. load / recovery acceptance contract（Owner 指示 §10）

### 12.1 structural validation（save / load 双方、Owner 指定 11 条件）

| # | 条件 | Owner 指定 | SOURCE-PROVEN 裏付け |
| --- | --- | --- | --- |
| 1 | `1 <= channels <= 8` | yes | 上限 `:53874`。下限は producer `:49902` |
| 2 | `1 <= samples <= 2097152` | yes | `:56049` |
| 3 | `1 <= targetLength <= 2097152` | yes | `:56049` と同源 |
| 4 | `samples == targetLength` | yes | producer chain `:50417` + `:49900` + `:49945` |
| 5 | `0 <= sections <= 20` | yes | producer `:50222`（最大 20） |
| 6 | `rho.size == theta.size` | yes | producer `:50434-50439` |
| 7 | `rho.size <= 20` | yes | `:50222` |
| 8 | `theta.size <= 20` | yes | `:50222` |
| 9 | `sampleRate finite` かつ `44100 <= sampleRate <= 768000` | yes | 上限 `:45872`。下限は **Owner spec**（§7.2） |
| 10 | `freqStart finite` かつ `100 <= freqStart <= 400` | yes | `:56024-56025` |
| 11 | `freqEnd finite` かつ `700 <= freqEnd <= 1300` | yes | `:56027-56028` |
| 12 | `freqEnd >= freqStart + 10` | yes | producer `:50704-50706`, `:52906-52908` の `jlimit(max(MIXED_F2_MIN_HZ, f1 + 10.0f), ...)` → **SOURCE-PROVEN** |
| 13 | `phaseMode ∈ {0, 1, 2}` | （Owner 一覧に無い。§7.4 で追加） | `:55916-55921` |

**すべて save boundary と load boundary の双方で検証する**（producer proof と API boundary validation を混同しない）。

**判定**: 条件 1〜5, 7〜8, 10〜13 は **SOURCE-PROVEN**（producer が保証）／条件 6 は **SOURCE-PROVEN**（同一ループ push）／条件 9 の下限 44100 は **DESIGN-VALIDATED**（Owner spec）。**現行は 1〜13 のいずれも実装していない**（S-03, S-15）。

### 12.2 payload validation（Usable のみ）

| 条件 | 採用 | 判定 |
| --- | --- | --- |
| IR 全要素 finite | **必須** | **DESIGN-VALIDATED**。producer では `:50346-50357` で PROVEN（S-23）だが、`load` は検証ゼロ（S-04） |
| `rho` 全要素 finite | **必須** | **DESIGN-VALIDATED**（Owner 指示 §10「`rho finite` を必須 cache contract」）。producer の `std::clamp` は NaN を除去しない（`:3543`, `:3813`） |
| `theta` 全要素 finite | **必須** | **DESIGN-VALIDATED**（Owner 指示 §10）。theta の domain は無界（`:3544`, `:3814`） |
| `0 <= rho < 1` | **採用しない** | **NOT PROVEN**（Owner 指示 §10「producer implementation detail に依存した必須条件にはしない方向で検証してください」に一致）。producer の 4 経路（`:4004`, `:3543`, `:3769`, `:3813`）から `[0, 0.995]` を導出できるが、`std::clamp` の NaN 透過性により無条件化できない |

### 12.3 exact-size 3 条件（Owner 指示 §7）

```text
actualFileSize          == expectedFileSize
header.totalFileSize    == expectedFileSize
expectedSize            <= kMaxPersistentCacheFileBytes  (== 134,218,124)
```

加えて §7.4 の `expectedSize <= INT64_MAX` を `header.totalFileSize` への格納前提とする。

**判定**: **DESIGN-VALIDATED**。現行は検証ゼロ（S-05）。

### 12.4 `load()` lifecycle（N4、Owner 指定）

```text
load(key)
  ├─ Open H                                       ← read/write 両 access、同一 handle
  │    filename = encodeIdentityHex(key)
  │    if (!exists) return false
  ├─ read 76 bytes → header
  │    apply §10.1 の 9 ステップ → state
  │    if (state != CurrentV3Canonical) { Close H; return false; }
  ├─ caller identity ↔ header identity（bit-exact）  ← §12.5
  │    if mismatch { Close H; return false; }
  ├─ read IR (channels × samples doubles)
  ├─ read rho / theta (sections doubles each)
  ├─ payload validation（§12.2）                  ← Usable 判定
  │    if fail { Close H; return false; }
  ├─ same H へ lastUsedTime を in-place 書込
  │    （失敗しても diagnostic。return value には影響しない: Owner N4）
  ├─ FlushFileBuffers(H)
  ├─ Close H
  └─ return true
```

**解決される 5 問題**（§6.4.3 に同一）: read/write handle 分離 / load→touch race / touch による別 transaction / touch 単独の破壊 / mutex lifecycle 分断。

**§3.1 の訂正との関係**: `FileOutputStream`（`OPEN_ALWAYS` + `FILE_END`）を**使わなくなる**ため、追記による 2 倍化機構が構造的に消える。**in-place write は `SetFilePointer(h, lastUsedTimeOffset, ..., FILE_BEGIN)` + `WriteFile(h, 8 bytes)`** であり、`SetEndOfFile` を伴わないためサイズ不変。

**判定**: **DESIGN-VALIDATED**（`touch()` 廃止の caller 0 件は SOURCE-PROVEN）。

### 12.5 caller identity ↔ header identity は bit-exact

現行 `load` は tolerance 比較を使う（S-06、`:77255`, `:77257`, `:77258`）。v3 は filename identity が bit-exact であるため、tolerance 比較は構造的に無意味になる（filename が caller key から一意に決まるため、header が 1 ulp 違えば hit 自体が起きない）。

producer が save と load で同一値を渡すことは SOURCE-PROVEN（`:50413` `key.sampleRate = sampleRate;` は save/load 共通）。→ **bit-exact への統一は DESIGN-VALIDATED で妥当**。

---

## §13. 1177 と power-loss（Owner 指示 §13, §14）

### 13.1 責務分離（Owner N2, §13）

| 系統 | 責務 | 使用 API |
| --- | --- | --- |
| **SM-A: normal v3 save** | 通常の save transaction | **`MoveFileExW` 系**。`ReplaceFileW` の 1175/1176/1177 を基本機構に**しない** |
| **SM-B: legacy/recovery-specific** | 1177 の解析（文書化のみ） | `ReplaceFileW`。**production code path に入れない** |

### 13.2 1177 の仕様（MS Learn。一次資料 = SOURCE-PROVEN）

> **ERROR_UNABLE_TO_MOVE_REPLACEMENT_2** - 1177 (0x499) — The replacement file could not be moved. The replacement file still exists under its original name; however, **it has inherited the file streams and attributes from the file it is replacing**. The file to be replaced still exists with a different name. If *lpBackupFileName* is specified, it will be the name of the replaced file.

他:
- 1176: `lpBackupFileName` 未指定なら **the replaced file no longer exists**
- 1175: replaced と replacement は元名のまま
- その他 error code（例 `ERROR_INVALID_PARAMETER`）: 両者元名のまま、**backup file は存在しない**
- `lpBackupFileName == NULL` → **no backup file is created**

### 13.3 attributes を canonicality に含めないことの帰結（N2）

canonicality の判定入力は `filename identity + header + payload + expected size` のみ（§9.1）。attributes / ADS / ACL は判定対象外。したがって **SM-B（1177）の「attributes 継承」は canonicality の問題ではなく、SM-B 固有の artifact 管理の問題**として分離できる。

**判定**: **DESIGN-VALIDATED**（Owner 裁定の採用）。1177 を SM-A に入れないことで、この分離が実装上も成立する。

### 13.4 `MoveFileExW` の atomicity 表現（Owner 明示）

MS Learn `MoveFileExW` の `MOVEFILE_WRITE_THROUGH`（8 / 0x8）:
> The function does not return until the file is actually moved on the disk. Setting this value guarantees that **a move performed as a copy and delete** is flushed to disk before the function returns.

**表現してよいこと**: 「同一 volume 内で directory entry の置換を行い、失敗時は source / destination のいずれかが保持される」

**表現してはならないこと**: 「`MoveFileExW` は atomic transaction である」/「`MoveFileExW` は power-loss durability を提供する」

**判定**: **SOURCE-PROVEN**（MS Learn の明記）。

### 13.5 power-loss（Owner 指示 §14）

```text
strict power-loss durability  = out of contract
corrupt artifact acceptance  = forbidden
```

矛盾しない整理:

```text
power loss
    ↓
cache may disappear / residue may remain / partial artifact may remain
    ↓                                （いずれも contract 違反ではない）
startup recovery
    ↓
structural validation  +  usable validation     ← in contract
    ↓
invalid artifact は never accepted              ← in contract
```

形式化: `power-loss` を入力とする変換 `f` について
```text
∀ f ∈ PowerLoss :  ¬( ∃ a : f(a) = a ∧ state(a) = CurrentV3Canonical ∧ a ∉ Usable )
```
すなわち **power-loss が valid-but-unusable artifact を生成しうるなら、その責務は cache 側にない**。cache 側の責務は「accept しないこと」のみ。

**判定**: **DESIGN-VALIDATED**。`actualFileSize == expectedFileSize`（§12.3）が partial file を必ず `Malformed` に落とす。

---

## §14. Test matrix

### 14.1 既存テスト基盤（SOURCE-PROVEN）

| 項目 | 値 | 出典 |
| --- | --- | --- |
| CTest 有効化 | `option(CONVOPEQ_ENABLE_ISR_TESTS "Enable minimal ISR runtime regression tests" ON)` + `enable_testing()` | `CMakeLists.txt:128`, `:136` |
| test ソース数 | `src/tests/**/*.cpp` = **71 ファイル** | 実測 |
| test executable 数 | 40+ | `CMakeLists.txt:143-770` |
| 新規 test の標準パターン | `add_executable(T src/tests/T.cpp)` → `target_include_directories` → `target_compile_features(cxx_std_20)` → `target_link_libraries(T PRIVATE juce::juce_core juce::juce_gui_extra juce::juce_gui_basics r8brain)` → `add_dependencies(T ConvoPeq)` → `add_test(NAME T COMMAND T)` | `CMakeLists.txt:736-740`, `:744-752` |
| test-only seam | `CONVO_TESTING=1`（`:691`）、`CONVOPEQ_UNIT_TESTS=1`（`:280`, `:479`, `:535`, `:2095`） | `CMakeLists.txt` |
| ASan 一覧 | `CONVOPEQ_ASAN_TEST_TARGETS`（`:2237-2254`） | `CMakeLists.txt` |
| **現行 C1 専用 unit test** | **存在しない**。`MixedPhasePersistentCache` の参照は `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` の **1 箇所のみ** | 実測 |
| `maxCacheEntries` への test 参照 | **0 件** | 実測 |
| `IRCacheKey` / `evictOldestCacheEntry` への test 参照 | **0 件** | 実測 |

→ **C1（今回の改修対象）のテスト網羅率は実質 0%**。これが §15 の Phase 0 で test を先に書く理由。

### 14.2 v3 テストマトリクス（Owner 指定条件の反映）

| # | テスト ID | 対象 | 入力 | 期待結果 | 種別 |
| --- | --- | --- | --- | --- | --- |
| T-01 | `encodeIdentityHex` round-trip | §8 | 300,000 件の random identity | `decode(encode(x)) == x` | 単体 |
| T-02 | zero-padding | §8 | `fileHash = 0`, `targetLength = 0` | 16 桁 / 8 桁の `0` 埋めで出力 | 単体 |
| T-03 | all-FF | §8 | `fileHash = 0xFFFFFFFFFFFFFFFF` | 有効な 64 hex | 単体 |
| T-04 | non-hex | §8 | `g` を含む 64 文字 | `InvalidCandidate` | 単体 |
| T-05 | 桁数 63 / 65 | §8 | — | `InvalidCandidate` | 単体 |
| T-06 | uppercase emit | §8 | — | 常に lowercase | 単体 |
| T-07 | NaN/Inf identity | §8.5 | `sampleRate = NaN`, `F1 = +Inf` | `InvalidCandidate` | 単体 |
| T-08 | expectedSize | §7.5 | max shape | `== 134,218,124` かつ `<= INT64_MAX` | 単体 |
| T-09 | header size | §7.5 | `sizeof(DiskHeader)` | `== 76`（`static_assert`） | compile-time |
| T-10 | channels 境界 | §12.1 #1 | `0`, `1`, `8`, `9` | `0`/`9` を reject | 単体 |
| T-11 | samples 境界 | §12.1 #2 | `0`, `1`, `2097152`, `2097153` | `0`/`2097153` を reject | 単体 |
| T-12 | `samples == targetLength` | §12.1 #4 | 不一致 | reject | 単体 |
| T-13 | sections 境界 | §12.1 #5 | `-1`, `0`, `20`, `21` | `-1`/`21` を reject | 単体 |
| T-14 | SR domain | §12.1 #9 | `44099.9`, `44100`, `768000`, `768000.1`, `NaN`, `±Inf` | 端点のみ accept | 単体 |
| T-15 | F1/F2 domain | §12.1 #10-11 | `99.9`, `100`, `400`, `400.1`, `699.9`, `700`, `1300`, `1300.1` | 端点のみ accept | 単体 |
| T-16 | `f2 >= f1 + 10` | §12.1 #12 | `f1=100, f2=109.9` / `f1=100, f2=110` | 109.9 reject / 110 accept | 単体 |
| T-17 | rho/theta size 一致 | §12.1 #6 | `rho.size != theta.size` | reject | 単体 |
| T-18 | rho/theta size 上限 | §12.1 #7-8 | `21` | reject | 単体 |
| T-19 | IR finite | §12.2 | 1 要素に NaN / `+Inf` / `-Inf` | reject | 単体 |
| T-20 | rho finite | §12.2 | 1 要素に NaN / `±Inf` | reject | 単体 |
| T-21 | theta finite | §12.2 | 同上 | reject | 単体 |
| T-22 | **`rho` range 不採用** | §12.2 | `rho = 1.5`（有限） | **accept**（range は必須でないことの固定） | 単体 |
| T-23 | exact-size 3 条件 | §12.3 | `actualFileSize != expectedSize` | `Malformed` | 単体 |
| T-24 | `totalFileSize` 不一致 | §12.3 | header 値を書き換えた file | `Malformed` | 単体 |
| T-25 | version 2 artifact | §10.1 step 4 | v2 file | `Obsolete`。load 不可。**削除されない** | 統合 |
| T-26 | future version artifact | §10.1 step 3 | `version = 4` | `Future`。load 不可。**削除されない**。save は `false` | 統合 |
| T-27 | uppercase filename | §8.2 | `<64HEX>.mph` | `CurrentV3NonCanonical`。load 不可。**削除されない**。**rename されない** | 統合 |
| T-28 | digit 数不正 | §10.1 step 9 | `<63hex>.mph` | `NonCanonical`。**削除されない** | 統合 |
| T-29 | **`evictLRU(0)`** | §7 | `maxCount = 0` | **no-op**。`*.mph` が残る | 統合 |
| T-30 | `evictLRU` ordering | §6.5 | ts 有効/無効/同値混在 | `valid asc → invalid last → filename asc` の順で削除 | 統合 |
| T-31 | `evictLRU` の target 限定 | §10.8 | v2 / future / `.stage` / `.bak` を混在 | **それらは削除されない** | 統合 |
| T-32 | **`clear()` の target 限定** | §10.8 | 同上 | canonical のみ削除。**他は残る** | 統合 |
| T-33 | `remove()` の target 限定 | §10.8 | v2 / future | 削除されない | 統合 |
| T-34 | **`maxCacheEntries` normalization** | §7 | `"abc"`, `true`, `NaN`, `±Inf`, `1.7`, `-1`, `0`, `8`, `64`, `100` | `8`, `8`, `8`, `8`, `8`, `8`, `8`, `8`, `64`, `64` | 単体 |
| T-35 | **`save()` bool 契約** | §11.4 | S11 cleanup 失敗のみ | `save() == true` + `ResidueDisposition == PendingCleanup` | 統合 |
| T-36 | save の truncation 禁止 | §11.2 | 途中状態観測 | target が truncate されない | 統合 |
| T-37 | delete-then-create 禁止 | §11.2 | 同上 | target 不在窓が S7-S8 のみ | 統合 |
| T-38 | staging namespace 分離 | §11.3 | save 中の `clear()` 相当 | staging が削除されない | 統合 |
| T-39 | **T0〜T5 遷移** | §11.5 | 各ステップ直後に強制中断 | 期待 state と一致 | 統合 |
| T-40 | **residue decision table** | §10.5 / §10.6 | 49 セルを網羅生成 | 各セルの action が §10.6 と一致 | 統合 |
| T-41 | **identity 混在防止** | §10.3 | `A.mph`, `A.mph.stage`, `B.mph.bak` | `A.mph.stage` と `B.mph.bak` が同一 transaction として扱われない | 統合 |
| T-42 | **recovery 2 段構え** | §9.3 | bak は structural OK だが payload NaN | restore 後 `Usable` 失敗 → `NotRecovered` | 統合 |
| T-43 | **`touch()` 廃止** | §6.4 / §12.4 | load 3 回連続 | ファイルサイズが `expectedSize` のまま変わらない | 統合 |
| T-44 | timestamp 更新失敗 | §6.4.4 | write 失敗を注入 | `load() == true`（diagnostic のみ） | 統合 |
| T-45 | `GetSystemTimeAsFileTime` | §7 | — | `0` invalid / `1..INT64_MAX` valid / `>INT64_MAX` invalid | 単体 |
| T-46 | v2 artifact の二重保護 | §8.3 | v2 file | `Obsolete` **かつ** `NonCanonical` | 統合 |
| T-47 | **C1/C2/C3 隔離** | §5 | C1 `.mph` / C2 `.bin` / C3 memory | C1 の操作が C2/C3 に影響しない | 統合 |
| T-48 | ASan | §14.1 | 上記 T-01〜T-47 | `-DENABLE_ASAN=ON` で clean | 統合 |

**ASan 追加**: T-01〜T-47 の target を `CONVOPEQ_ASAN_TEST_TARGETS`（`CMakeLists.txt:2237-2254`）に追加する。

---

## §15. Implementation phase 分割

### 15.1 Phase 設計原則

1. **test-first**: Phase 0 で test 基盤と identity 定数のみを先に作る（production 変更 0）
2. **各 Phase は独立して build/CTest を通す**
3. **docs commit → implementation commit の順で単独 commit**（`git push origin HEAD:main`）
4. **read-only 監査期間（Phase -1）を挟まない**

### 15.2 Phase 分割

| Phase | 対象 | 内容 | 判定の前提 |
| --- | --- | --- | --- |
| **-1** | 本監査 | D-1R-2I 完了 | ✅ 完了 |
| **0** | test + 定数のみ（production 変更 0） | `src/tests/PersistentCacheV3Tests.cpp` 新規。CMake に `PersistentCacheV3Tests` target + `add_test`。T-01〜T-09, T-14〜T-22, T-34, T-45 を**失敗する状態で**追加。`kPersistentCache*` 定数を header に追加（未使用なら `[[maybe_unused]]`） | §7, §8, §12 |
| **1** | identity serialization | `encodeIdentityHex` / `decodeIdentityHex` を実装。`getCacheFile` を 64-hex に変更。`computeKeyHash` 削除 | §8 |
| **2** | artifact classification | `ArtifactState` 3 軸分解 + 9 ステップ判定を実装。`classifyCanonicalArtifact(actualFilename, header, actualFileSize, expectedIdentity)` / `classifyRecoveryCandidate(kind, actualFilename, header, actualFileSize)` | §6.1, §10.1 |
| **3** | load lifecycle | `load()` を §12.4 の 12 段に再構成。`touch()` を廃止（外部 caller 0 件、PROVEN）。same-handle `lastUsedTime` 更新 | §6.4, §12.4 |
| **4** | save transaction | `save()` を §11.1 の 11 段に再構成。`MoveFileExW` 系 commit。`.stage` / `.bak` namespace。`save()` bool 契約維持（`ResidueDisposition` は内部） | §10, §11 |
| **5** | residue / recovery | `classifyResidue` / `runRecovery`（§10.2, §10.6）。recovery 2 段構え（§9.3） | §9.3, §10.6 |
| **6** | destructive API | `evictLRU` / `remove` / `clear` / `getEntryCount` を `CurrentV3Canonical` のみに限定。`evictLRU(0) = no-op`。ordering（§6.5）。`remove` / `clear` / `getEntryCount` は外部 caller 0 件（PROVEN）なので**削除**も検討 | §10.8, §6.5 |
| **7** | caller 統合 | `:50440-50446` を `const bool saved = save(...); if (saved) evictLRU(normalizedMaxCacheEntries);` に変更。`maxCacheEntries` normalization を §7 の契約に差し替え（`jlimit(0,64,...)` → 正規化後に clamp） | §11.5, §7 |
| **8** | 統合 test + ASan | T-10〜T-44 を有効化。ASan target 追加。CTest 全 green | §14.2 |

### 15.3 Phase 間の依存

```text
0 → 1 → 2 → 3
            ↘ 4 → 5 → 6 → 7 → 8
```

- Phase 3（load）は Phase 2（classification）に依存
- Phase 4（save）は Phase 2 に依存（`classifyResidue` の判定に state が必要）
- Phase 5（recovery）は Phase 4 に依存（`.bak` / `.stage` 命名が必要）
- Phase 7 は Phase 4 と 6 の両方 に依存

---

## §16. remaining NOT PROVEN 一覧

### 16.1 NOT PROVEN 項目（Owner decision または production 実証が必要）

| # | 項目 | 現在の状態 | 閉じ方 |
| --- | --- | --- | --- |
| N-P-01 | `target → bak` transaction の production 実証 | **DESIGN-VALIDATED のみ**。`MoveFileExW` の実挙動は未実証 | Phase 4 + T-39 |
| N-P-02 | `stage → target` transaction の production 実証 | **DESIGN-VALIDATED のみ** | Phase 4 + T-39 |
| N-P-03 | existing `.bak` handling の production 実証 | **DESIGN-VALIDATED のみ**（MS Learn の引用は SOURCE-PROVEN だが自前実装の挙動は未実証） | Phase 5 + T-40 |
| N-P-04 | stale stage handling の production 実証 | **DESIGN-VALIDATED のみ** | Phase 5 + T-40 |
| N-P-05 | stale bak handling の production 実証 | **DESIGN-VALIDATED のみ** | Phase 5 + T-40 |
| N-P-06 | post-success `.bak` cleanup の production 実証 | **DESIGN-VALIDATED のみ**（`ResidueDisposition` 機構は設計済み） | Phase 4 + T-35 |
| N-P-07 | **`0 <= rho < 1` を cache contract に採らないことの妥当性** | **NOT PROVEN**（Owner が「採用しない方向で検証」と指示）。producer 4 経路からは `[0, 0.995]` を導出できるが `std::clamp` の NaN 透過性で無条件化不可。**最終判断は Owner** | Owner 確認要（T-22 で挙動を固定） |
| N-P-08 | `kPersistentCacheMinSampleRate = 44100.0` の source 裏付け | **NOT PROVEN**（Owner が明示的に「source 内の既存定数から PROVEN されたものとは扱わないでください」と指示）。Owner specification のみ | Owner spec として確定済み（§7.2）。追加の実証は**不要** |
| N-P-09 | `kDefaultPersistentCacheEntries = 8` の source 裏付け | **NOT PROVEN**（Owner spec。`MAX_CACHE_ENTRIES = 8` は C3 の別系統） | Owner spec として確定済み（§7.2）。追加の実証は**不要** |
| N-P-10 | in-memory C3 の `evictOldestCacheEntry()` の `lastUsedTime == 0` 扱い | **NOT PROVEN**（`:49715` の `minTime = uint32 max` により `0` は最古扱い。**改修対象外**） | 改修対象外として記録 |
| N-P-11 | diagnostics の外部 API 化 | **NOT PROVEN**（§6.3 で内部 state のみとする方針が成立することを論証したが、外部公開の必要性判断は保留） | Owner が別途提案する |
| N-P-12 | Phase 4 の `MoveFileExW` 呼び出しが `MOVEFILE_REPLACE_EXISTING` を必要とする場面の完全な網羅 | **NOT PROVEN**（MS Learn の仕様に基づく設計だが、自前実装での全 scenario 網羅は未実証） | Phase 4 + T-40 |

### 16.2 SOURCE-PROVEN だが現行が CONTRADICTED の項目（改修対象）

| # | 項目 | SOURCE-PROVEN 事実 | 現行 |
| --- | --- | --- | --- |
| C-01 | `evictLRU(0)` | `clear()` を呼ぶ（S-10） | CONTRACT 違反 |
| C-02 | `maxCacheEntries` normalization | narrowing が validation より先（S-13） | CONTRACT 違反 |
| C-03 | `maxCacheEntries` 既定値 | 0（S-11, S-12） | CONTRACT 違反 |
| C-04 | save → eviction | 戻り値無視（S-14） | CONTRACT 違反 |
| C-05 | `load` header upper-bound | 検査なし（S-03） | CONTRACT 違反 |
| C-06 | `load` payload finite | 検査なし（S-04） | CONTRACT 違反 |
| C-07 | `load` exact size | 検査なし（S-05） | CONTRACT 違反 |
| C-08 | `clear()` の destructive eligibility | 無条件削除（S-16） | CONTRACT 違反 |
| C-09 | `evictLRU` の destructive eligibility | magic のみ（S-17） | CONTRACT 違反 |
| C-10 | `remove()` の destructive eligibility | 検査なし（S-18） | CONTRACT 違反 |
| C-11 | staging namespace | `*.mph` に一致（S-21） | CONTRACT 違反 |
| C-12 | normal save の API | `ReplaceFileW`（S-20） | CONTRACT 違反 |
| C-13 | `lastUsedTime` 更新 | 別 handle + 追記（S-07, S-08, S-09） | CONTRACT 違反 |

---

## §17. implementation GO を出してよい条件

### 17.1 必須条件（すべて満たされた後に限り GO を出せる）

| # | 条件 | 現在の状態 |
| --- | --- | --- |
| G-01 | §7 の v3 定数が Owner により最終確定されている | ☐ Owner 待ち |
| G-02 | §8 の identity serialization（Option A）が Owner により最終確定されている | ☐ Owner 待ち |
| G-03 | §9 の Canonical / Usable / Recovery 境界が Owner により最終確定されている | ☐ Owner 待ち |
| G-04 | §10.6 の residue decision table（R1〜R4）が Owner により最終確定されている | ☐ Owner 待ち |
| G-05 | §11.4 の save success contract（S1〜S11）が Owner により最終確定されている | ☐ Owner 待ち |
| G-06 | §12 の load/recovery acceptance contract が Owner により最終確定されている | ☐ Owner 待ち |
| G-07 | N-P-07（`rho` range を採らないこと）の最終判断 | ☐ Owner 待ち |
| G-08 | N-P-11（diagnostics の外部 API 化）の判断 | ☐ Owner 待ち |
| G-09 | §15 の Phase 分割が Owner により承認されている | ☐ Owner 待ち |
| G-10 | Phase 0（test-first）の target 名 / ファイル名が確定している | ☐ Owner 待ち |

### 17.2 GO が出た後の実行順序（各 Phase 単独 commit）

```text
docs commit      : doc/work113/ の D-1R-2I + 以降の監査書
implementation commit : Phase 0 → 8 を 1 Phase ずつ
push             : git push origin HEAD:main
検証             : HEAD == origin/main / worktree clean / staged 0
```

### 17.3 現時点の結論

**本 D-1R-2I の時点で Owner の implementation GO は出ない。**（Owner が明示的に「現時点では implementation GO ではありません」と宣言している）

G-01〜G-10 のうち **10 項目すべてが Owner 待ち**。特に G-01（定数）と G-05（save success contract）は本監査が DESIGN-VALIDATED として論証したものであり、**現行 source で成立しているわけではない**ため、Owner の最終確定が先行する。

---

## 付録 A. authority 引用一覧（最新 snapshot の行番号）

`C:\VSC_Project\ConvoPeq\doc\ConvoPeqMD\20261002231921\ConvoPeq.md`
（5,941,824 bytes / 131,019 lines / SHA-256 `3a51ab3414dba90005b99540376a93d6ab76ae1a96a754ce9e7050e2e0593045`）

### A.1 cache 実装（`src\MixedPhasePersistentCache.cpp`、`:77150` から）

| 行 | 内容 |
| --- | --- |
| `:77167-77170` | `hashCombine` |
| `:77172-77192` | `computeKeyHash`（`memcpy` bit cast） |
| `:77198-77210` | `getCacheDirectory` → `%APPDATA%/ConvoPeq/MixedPhaseCache` |
| `:77212-77222` | `getCacheFile` → `toHexString(int64) + ".mph"`（可変長） |
| `:77228-77296` | `load` |
| `:77252-77253` | `magic != kMagic \|\| version != kVersion` |
| `:77254-77260` | identity 検証（**tolerance 比較**） |
| `:77263-77266` | `numChannels <= 0 \|\| numSamples <= 0`（**上限検査なし**） |
| `:77268` | `outIr.setSize(numChannels, numSamples)` |
| `:77279-77290` | rho/theta read（**上限検査なし**） |
| `:77293` | `touch(...)`（**load 内部の唯一の呼び出し**） |
| `:77302-77377` | `save` |
| `:77321` | `juce::TemporaryFile tempFile(file)` |
| `:77329` | `numSec = static_cast<int>(rho.size())` |
| `:77342` | `lastUsedTime = getMillisecondCounter()`（32-bit） |
| `:77368` | `header.totalFileSize = stream.getPosition()` |
| `:77373` | `stream.flush()`（**戻り値無視**） |
| `:77376` | `return tempFile.overwriteTargetFileWithTemporary();` |
| `:77383-77423` | `touch` |
| `:77394-77395` | 「切り詰めを回避する」コメント（**根拠が誤り**） |
| `:77402`, `:77406` | `fileSize` / `buffer.resize(fileSize)` |
| `:77416` | `now = getMillisecondCounter()` |
| `:77419` | `juce::FileOutputStream outStream(file)`（**追記**） |
| `:77422` | `outStream.write(...)`（**戻り値無視**） |
| `:77425-77466` | `evictLRU` |
| `:77427-77431` | `maxCount == 0 → clear()` |
| `:77435` | `findChildFiles(..., "*.mph")` |
| `:77454` | `magic != kMagic → continue`（**version 不問**） |
| `:77460-77461` | `std::sort`（**invalid 末尾規則なし**） |
| `:77468-77478` | `remove`（**分類検査なし**） |
| `:77480-77490` | `clear`（**分類検査なし**） |
| `:77492-77501` | `getEntryCount`（**分類なし**） |
| `:77526-77603` | class 宣言 |
| `:77574` | `kMagic = 0x4D69786564506800ULL` |
| `:77575` | `kVersion = 2` |
| `:77577-77593` | `struct DiskHeader`（`#pragma pack(1)`、14 field） |
| `:77596` | `static_assert(sizeof(DiskHeader) <= 128)` |

### A.2 producer chain

| 行 | 内容 |
| --- | --- |
| `:45872` | `static constexpr int kMaxSampleRate = 768000;` |
| `:50222` | `numSections`（`{2, 8, 20}`） |
| `:50311` | `lastAllpassSections = allpass_sections` |
| `:50341-50342` | RMS 規格化 `dst[i] *= gain` |
| `:50346-50357` | **全 channel × 全 sample finite sweep** |
| `:50413` | `key.sampleRate = sampleRate;` |
| `:50414` | `key.phaseMode = ConvolverProcessor::PhaseMode::Mixed;` |
| `:50417` | `key.targetLength = linearIR.getNumSamples();` |
| `:50434-50439` | `rho.push_back` / `theta.push_back`（同一ループ） |
| `:50440-50443` | `MixedPhasePersistentCache::save(...)` |
| `:50446` | `evictLRU(owner->getMaxCacheEntries())`（**戻り値無視**） |
| `:49900-49902` | `numSamples` / `numChannels` / `<= 0` return |
| `:49945` | `mixedIR(numChannels, numSamples)` |

### A.3 allpass / rho

| 行 | 内容 |
| --- | --- |
| `:4003-4008` | `struct SecondOrderAllpass` / `isStable()` → `rho ∈ [0, 1)` |
| `:3543` | `rho = std::clamp(std::abs(gain), 0.0, 0.995)` |
| `:3544` | `theta = 2π f0 / sampleRate`（**無界**） |
| `:3552-3556` | `unconstrainedToRho(x) = 0.98 * stableSigmoid01(x)` |
| `:3559` | `kThetaMax = 0.99π` |
| `:3561-3566` | `unconstrainedToTheta(x) = kThetaMax * stableSigmoid01(x)` |
| `:3769-3770` | `section.rho/theta = unconstrainedTo*` |
| `:3813-3814` | `section.rho = std::clamp(std::abs(best_gain), 0, 0.995)` / `theta = 2π best_f0 / SR` |

### A.4 定数 / enum

| 行 | 内容 |
| --- | --- |
| `:53874` | `inline constexpr unsigned kMaxIRLoadChannels = 8u;` |
| `:55916-55921` | `enum class PhaseMode : int { AsIs=0, Mixed=1, Minimum=2 }` |
| `:56024` | `MIXED_F1_MIN_HZ = 100.0f` |
| `:56025` | `MIXED_F1_MAX_HZ = 400.0f` |
| `:56026` | `MIXED_F1_DEFAULT_HZ = 200.0f` |
| `:56027` | `MIXED_F2_MIN_HZ = 700.0f` |
| `:56028` | `MIXED_F2_MAX_HZ = 1300.0f` |
| `:56029` | `MIXED_F2_DEFAULT_HZ = 1000.0f` |
| `:56049` | `static constexpr int MAX_IR_LATENCY = 2097152;` |
| `:50701-50706`, `:50792-50797` | `jlimit(max(MIXED_F2_MIN_HZ, clampedMixedF1 + 10.0f), MIXED_F2_MAX_HZ, ...)` |
| `:52903-52908` | `jlimit(max(MIXED_F2_MIN_HZ, mixedTransitionStartHz + 10.0f), MIXED_F2_MAX_HZ, ...)` |
| `:12833-12834` | `hasFiniteDouble("mixedF1Hz"/"mixedF2Hz", ...)` |

### A.5 maxCacheEntries と cache 系統

| 行 | 内容 |
| --- | --- |
| `:13603-13611` | `getConvolverMaxCacheEntries` / `setConvolverMaxCacheEntries(int → size_t)` |
| `:46119-46131` | `CacheManager::getCacheDirectory` → `%APPDATA%/ConvoPeq/IRCache` |
| `:46135-46138` | `CacheManager::getCacheFile` → `<key>_<fftSize>.bin` |
| `:46422` | `touch(key, fftSize)`（CacheManager 側） |
| `:46425-46452` | `CacheManager::touch`（**in-memory のみ。ファイル I/O なし**） |
| `:46460-46474` | `CacheManager::clear` = `dir.deleteRecursively()` |
| `:46484-46500+` | `CacheManager::evictLRU`（`isEntrySafeToDelete` 検査あり） |
| `:46553-46600` | `class CacheManager` |
| `:46570` | `void evictLRU(size_t maxEntries = 10);` |
| `:48986-48995` | `ConvolverProcessor::setMaxCacheEntries`（`jlimit<size_t>(1, 64, ...)`） |
| `:48997-49001` | `ConvolverProcessor::getMaxCacheEntries` |
| `:49003-49008` | `ConvolverProcessor::clearCache` → **C2 のみ** |
| `:49709-49727` | `ConvolverProcessor::evictOldestCacheEntry`（`MAX_CACHE_ENTRIES`） |
| `:52927` | `pendingOverride.maxCacheEntries = juce::jlimit(0, 64, ...)` |
| `:53108` | `setMaxCacheEntries(static_cast<size_t>(static_cast<int>(v.getProperty(...))))` |
| `:55888`, `:56882` | `int maxCacheEntries = 0;` |
| `:56102-56105` | cache API 宣言 |
| `:57141-57155` | `struct IRCacheKey` / `operator<` |
| `:57160-57164` | `struct CacheEntry`（`uint32_t lastUsedTime`） |
| `:57166-57168` | `irCache` / `cacheMutex` / `MAX_CACHE_ENTRIES = 8` |
| `:84826` | `cacheManager.evictLRU(processor.getMaxCacheEntries())` |

### A.6 JUCE tree（`C:\VSC_Project\ConvoPeq\JUCE\modules\juce_core\`）

| ファイル:行 | 内容 |
| --- | --- |
| `files/juce_FileOutputStream.cpp:39-45` | ctor → `openHandle()` |
| `files/juce_FileOutputStream.cpp:53-56` | `getPosition()` → `currentPosition` |
| `files/juce_FileOutputStream.cpp:88-125` | `write()` |
| `native/juce_Files_windows.cpp:429-450` | **`FileOutputStream::openHandle` = `OPEN_ALWAYS` + `SetFilePointer(FILE_END)`** |
| `native/juce_Files_windows.cpp:432` | `FILE_SHARE_READ`（sharing mode） |
| `native/juce_Files_windows.cpp:452-455` | `closeHandle()` |
| `native/juce_Files_windows.cpp:457-466` | `writeInternal` |
| `native/juce_Files_windows.cpp:468-473` | `flushInternal` = `FlushFileBuffers` |
| `native/juce_Files_windows.cpp:475-483` | **`truncate()` = `SetEndOfFile`（`touch()` から呼ばれない）** |
| `native/juce_Files_windows.cpp:372-378` | `replaceInternal` → `ReplaceFile(dest, this, nullptr, ...)` |
| `files/juce_TemporaryFile.cpp:61-70` | staging 名 = `<nameNoExt>_temp<randhex><ext>` |
| `files/juce_TemporaryFile.cpp:94-104` | `overwriteTargetFileWithTemporary`（5 回 retry） |
| `text/juce_String.cpp:2625-2633` | `toHexString` の可変長・非 zero-pad の実証 |
| `text/juce_String.cpp:1135-1169` | `WildCardMatcher`（full match） |

### A.7 CMakeLists.txt / tests（`C:\VSC_Project\ConvoPeq\`）

| ファイル:行 | 内容 |
| --- | --- |
| `CMakeLists.txt:128` | `option(CONVOPEQ_ENABLE_ISR_TESTS ... ON)` |
| `CMakeLists.txt:136` | `enable_testing()` |
| `CMakeLists.txt:143-770` | test executable 定義群 |
| `CMakeLists.txt:736-740`, `:744-752` | 新規 test の標準パターン |
| `CMakeLists.txt:691` | `CONVO_TESTING=1` |
| `CMakeLists.txt:280`, `:479`, `:535`, `:2095` | `CONVOPEQ_UNIT_TESTS=1` |
| `CMakeLists.txt:2040-2165` | `AudioEngineHarness` |
| `CMakeLists.txt:2237-2254` | `CONVOPEQ_ASAN_TEST_TARGETS` |
| `src/tests/**` | `.cpp` 71 ファイル |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `MixedPhasePersistentCache` の唯一の test 参照（1 箇所） |

### A.8 external evidence

| 出典 | 引用 |
| --- | --- |
| MS Learn `ReplaceFileW` | 1175 / 1176 / 1177 / `ERROR_INVALID_PARAMETER` の各説明、`lpBackupFileName == NULL` で backup なし |
| MS Learn `MoveFileExW` | `MOVEFILE_REPLACE_EXISTING` (1/0x1) の記載、`MOVEFILE_WRITE_THROUGH` (8/0x8) が copy-and-delete の flush のみを対象とする記載 |

---

## 付録 B. 判定索引（SOURCE-PROVEN / DESIGN-VALIDATED / NOT PROVEN）

| ID | 項目 | 判定 |
| --- | --- | --- |
| S-01 | 現在の persistent cache version = 2 | **SOURCE-PROVEN** |
| S-02 | 現在の filename 生成は variable-length hash | **SOURCE-PROVEN** |
| S-03 | `load()` は header upper-bound validation をしていない | **SOURCE-PROVEN** |
| S-04 | `load()` は payload finite 検証をしていない | **SOURCE-PROVEN** |
| S-05 | `load()` は `actualFileSize` / `totalFileSize` を検証していない | **SOURCE-PROVEN** |
| S-06 | `load()` は tolerance 比較で identity を検証する | **SOURCE-PROVEN** |
| S-07 | `touch()` が別 handle で書き戻している | **SOURCE-PROVEN** |
| S-08 | `touch()` は追記によりファイルサイズを 2 倍にする | **SOURCE-PROVEN** |
| S-09 | `touch()` の `write()` 戻り値を無視する | **SOURCE-PROVEN** |
| S-10 | `evictLRU(0)` が `clear()` を呼ぶ | **SOURCE-PROVEN** |
| S-11 | `maxCacheEntries` の既定値が 0 | **SOURCE-PROVEN** |
| S-12 | snapshot 復元の `jlimit` 下限が 0 | **SOURCE-PROVEN** |
| S-13 | narrowing が validation より先 | **SOURCE-PROVEN** |
| S-14 | save 後に return value を無視して evict | **SOURCE-PROVEN** |
| S-15 | `save()` は input validation を行わない | **SOURCE-PROVEN** |
| S-16 | `clear()` は `*.mph` を無条件削除 | **SOURCE-PROVEN** |
| S-17 | `evictLRU` は `magic` のみ検査 | **SOURCE-PROVEN** |
| S-18 | `remove()` は分類検査なし | **SOURCE-PROVEN** |
| S-19 | `getEntryCount()` は分類なし | **SOURCE-PROVEN** |
| S-20 | 現行 `.bak` は存在しない | **SOURCE-PROVEN** |
| S-21 | staging file は `*.mph` namespace 内に居る | **SOURCE-PROVEN** |
| S-22 | producer は rho/theta を同一ループで push | **SOURCE-PROVEN** |
| S-23 | producer は mixedIR 全要素に isfinite を検査 | **SOURCE-PROVEN** |
| S-24 | `remove/clear/getEntryCount/touch` の外部呼び出し元 0 件 | **SOURCE-PROVEN** |
| S-25 | C1 専用 unit test が存在しない | **SOURCE-PROVEN** |
| S-26 | producer は `f2 >= f1 + 10` を強制 | **SOURCE-PROVEN** |
| S-27 | `F1 ∈ [100,400]`, `F2 ∈ [700,1300]` の定数 | **SOURCE-PROVEN** |
| S-28 | `PhaseMode ∈ {0,1,2}` | **SOURCE-PROVEN** |
| S-29 | in-memory `CacheEntry::lastUsedTime` は `uint32_t` | **SOURCE-PROVEN** |
| S-30 | C1 / C2 は directory と拡張子が異なる | **SOURCE-PROVEN** |
| S-31 | `CacheManager::touch` は in-memory のみで無害 | **SOURCE-PROVEN** |
| S-32 | `irCache` の上限は `MAX_CACHE_ENTRIES = 8`（maxCacheEntries と無関係） | **SOURCE-PROVEN** |
| S-33 | `CacheManager::evictLRU` の default は 10 | **SOURCE-PROVEN** |
| D-01 | version = 3 | **DESIGN-VALIDATED** |
| D-02 | 64-hex identity filename（Option A） | **DESIGN-VALIDATED** |
| D-03 | Canonical / Usable separation | **DESIGN-VALIDATED** |
| D-04 | `evictLRU(0) = no-op` | **DESIGN-VALIDATED** |
| D-05 | `if (saved) evictLRU(...)` | **DESIGN-VALIDATED** |
| D-06 | power-loss durability out-of-contract | **DESIGN-VALIDATED** |
| D-07 | corrupt artifact never accept | **DESIGN-VALIDATED** |
| D-08 | `.mph.stage` staging namespace 分離 | **DESIGN-VALIDATED** |
| D-09 | `touch()` を load lifecycle に統合 | **DESIGN-VALIDATED** |
| D-10 | residue classification | **DESIGN-VALIDATED** |
| D-11 | normal save を `MoveFileExW` 系にする | **DESIGN-VALIDATED** |
| D-12 | `lastUsedTime` を same-handle 更新 | **DESIGN-VALIDATED** |
| D-13 | `GetSystemTimeAsFileTime` を wall-clock に | **DESIGN-VALIDATED** |
| D-14 | destructive API = `CurrentV3Canonical` only | **DESIGN-VALIDATED** |
| D-15 | header upper-bound validation | **DESIGN-VALIDATED** |
| D-16 | exact-size 3 条件 + checked arithmetic | **DESIGN-VALIDATED** |
| D-17 | 3 軸 ArtifactState 合成規則（total/disjoint） | **DESIGN-VALIDATED** |
| D-18 | residue decision table R1〜R4（total/disjoint） | **DESIGN-VALIDATED** |
| D-19 | save success contract S1〜S11 | **DESIGN-VALIDATED** |
| D-20 | load lifecycle 12 段 | **DESIGN-VALIDATED** |
| D-21 | recovery 2 段構え | **DESIGN-VALIDATED** |
| D-22 | identity membership = 64-hex prefix 完全一致 | **DESIGN-VALIDATED** |
| D-23 | evictLRU ordering（valid asc / invalid last / filename asc） | **DESIGN-VALIDATED** |
| D-24 | `ResidueDisposition` を内部 state のみで保持 | **DESIGN-VALIDATED** |
| D-25 | `expectedSize <= INT64_MAX` を格納前提とする | **DESIGN-VALIDATED** |
| D-26 | `NaN/Inf identity = InvalidCandidate` | **DESIGN-VALIDATED** |
| D-27 | LRU timestamp 更新失敗を load failure と同一視しない | **DESIGN-VALIDATED** |
| N-P-01〜12 | §16.1 の 12 項目 | **NOT PROVEN** |

---

## 付録 C. mojibake scan

| 対象 | U+FFFD | Cyrillic | Hangul | Thai | 簡体字 | Latin-ext |
| --- | --- | --- | --- | --- | --- | --- |
| 本報告書（最終） | 0 | 0 | 0 | 0 | 0 | `§` / `×` のみ |

走査対象コードポイント: `U+FFFD` / `U+0400-04FF` / `U+1100-11FF` / `U+AC00-D7AF` /
`U+0E00-0E7F` / `U+1EA0-1EF9`、および個別指定の `U+4E4B`（之）/ `U+4E3A`（为）/
`U+8D39`（费）。

初版作成時に以下 12 件の混入を検出し修正済み:

| 検出 | 修正 |
| --- | --- |
| Hangul `것` | `もの` |
| Hangul `전혀` | `全く` |
| 簡体字 `消费`（费） | `消費` |
| 簡体字 `问题的实测` | `実測上の問題` |
| 簡体字 `消费的` | `消費される` |
| 簡体字 `之外` | `以外`（2 箇所） |
| 簡体字 `隔离` | `隔離` |
| 簡体字 `正常化后再` | `正規化後に` |
| 簡体字 `场景` | `場面` |
| 混入 fragment `ambange` | `する`（`提案する`） |
| 混入 fragment `Because of` + `inis` | 文全体を日本語へ置換 |
| 混入 fragment `Descriptions` / `asim` | `Description` / `すれば` |

最終走査で `U+00C0-024F` に残る文字は `§`（187 回）と `×`（6 回、数学記号）のみ。
`determinism` の部分一致（`inis`）は誤検出。
走査対象コードポイント: `U+FFFD` / `U+0400-04FF` / `U+1100-11FF` / `U+AC00-D7AF` / `U+0E00-0E7F` / `U+1EA0-1EF9` / `U+00C0-024F`。

---

## 付録 D. read-only 遵守の記録

| 対象 | 変更 |
| --- | --- |
| `src/**` | 0 |
| `tests/**` | 0 |
| `CMakeLists.txt` | 0 |
| `build/**` | 0 |
| CTest 実行 | 0 |
| `ConvoPeq.md` 再生成 | 0 |
| `git commit` | 0 |
| `git push` | 0 |
| 作成したファイル | `doc/work113/P1-5-IR-P2_MIXEDPHASE-BG-CMAES_D1R2I_7864_20261006.md` のみ |

---

## 付録 E. 変更履歴

- 2026-10-06: D-1R-2I 初版作成（§0-§17 + 付録 A-E）。