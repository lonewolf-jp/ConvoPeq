# v3.78.64-D1R-2 — Implementation-Precondition Audit

- Date: 2026-10-05
- Authority: **Owner 指定 `C:\VSC_Project\ConvoPeq\ConvoPeq.md`**（`ConvoPeq(20261002-142258).md` は実在しない。standing instruction に従い同一ファイルとみなす）
- Type: **read-only implementation-precondition audit**
- production source / test / CMake / build / version / ConvoPeq 再生成 / commit / push: **すべて 0**
- D-1R-a / D-1R-a-b の裁定: **本版では行わない**（Owner が D-1R-2 結果確認後に裁定）

---

## 0. Owner 指摘の受諾（D-1R の不完全性）

### 0.1 指摘 1 — `std::uint64_t` に 256 bit は格納できない

D-1R §9.2 は次の API を提示した。

```cpp
std::uint64_t encodeIdentityHex (const Identity&) noexcept;
bool          decodeIdentityHex (std::uint64_t hex, Identity&) noexcept;
```

canonical filename は

```
fileHash 16 + sampleRateBits 16 + phaseMode 8 + freqStartBits 8 + freqEndBits 8 + targetLength 8
= 64 hex characters = 256 bit
```

**64 hex = 256 bit > 64 bit**。したがって `std::uint64_t` の戻り値型は**型として成立しない**（最大 16 hex = 64 bit しか表現できず、切り詰めまたはコンパイル不能になる）。Owner の指摘は正しい。

**判定: D-1R §9.2 の型設計は INVALID。§3 で再設計する。**

### 0.2 指摘 2 — T-09 のテストケースが誤り

D-1R §11.2 T-09 は `decodeIdentityHex` の不正入力として `0xFFFFFFFFFFFFFFFF` を挙げた。これは **`f` が 16 個並ぶ有効な 16 桁 hex 値**であり、非 hex 入力ではない。non-hex のテストには文字列としての非 hex 文字（`G` 等）が必要。

**判定: T-09 は無効。§6 で修正し、他 8 ケースも 256 bit 型に合わせて再設計する。**

### 0.3 本版で新たに検出した不備（D-1R 未記載）

| # | 不備 | 内容 |
| --- | --- | --- |
| N-1 | **`.stage` 残留時の `CREATE_NEW` 衝突** | D-1R §5.4/§5.5 は `CreateFileW (S, GENERIC_WRITE, CREATE_NEW)` を使う。`.stage` が残留していると `ERROR_FILE_EXISTS` で**常に失敗**する。D-1R §5.2 は同一 identity の `.stage` を処理すると記すが、順序が明示されておらず、処理が `CREATE_NEW` より**前**であることを保証していない |
| N-2 | **`/fp:fast` が icx Release で有効** | authority `CMakeLists.txt:1783-1784` `set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /fp:fast /Gy /Zi /utf-8 /EHsc")`（`:1778-1779` のコメント「/fp:fast は icx の性能上の意図的選択として icx Release のみに残置する」で確定）。D-1R は bit-pattern identity の前提として `/fp:precise` を暗黙に想定していた。**SR / F1 / F2 の domain 検証を浮動小数点比較で行うのは不適切** |
| N-3 | `F2 >= F1 + 10` の扱い | D-1R は cross-field FP 比較が必要と記したが、per-field range から**自動成立**する。FP 比較を不要にできるため N-2 の影響範囲を縮小できる |

---

## 1. authority 前提の再確認（Owner 指定 authority）

### 1.1 浮動小数点コンパイルフラグ

```
src: CMakeLists.txt（Owner 指定 authority のコードフェンス本文）

:1685-1687  # ★ work92 C-8: /fp:fast を除去（MSVC 既定の /fp:precise に戻す）。
            #   DSP 数値精度を優先し、MSVC ビルドの AMD 動作を標準 AVX2 + precise で保証する。
:1688       set(CMAKE_CXX_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /Gw /Gy /Zi /utf-8 /EHsc")   ← MSVC: /fp:fast 無し
:1693       set(CMAKE_CXX_FLAGS_DEBUG  "/D_DEBUG /bigobj /Zm400 /Ob0 /Od /Zi /RTC1 /utf-8 /EHsc")    ← Debug: /fp:fast 無し
:1756-1757  # /fp:fast: 高速浮動小数点（icx デフォルト）…
:1778-1779  #   /fp:fast は icx の性能上の意図的選択（LLVM OOM 回避で /O2 + fp:fast 維持）として
            #   icx Release のみに残置する。
:1783       set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /fp:fast /Gy /Zi /utf-8 /EHsc")                ← IntelLLVM: /fp:fast 有効
:1784       set(CMAKE_C_FLAGS_RELEASE   "/O2 /DNDEBUG /fp:fast /Gy /Zi /utf-8 /EHsc")
```

**結論**:

| toolchain / config | `/fp:fast` | canonical path への影響 |
| --- | --- | --- |
| MSVC Release | **無し**（`/fp:precise`） | 問題なし |
| MSVC Debug | **無し** | 問題なし |
| **IntelLLVM (icx) Release** | **あり** | **canonical validator は浮動小数点比較に依存してはならない** |

### 1.2 cache directory 構造（same-volume の根拠）

```
src/MixedPhasePersistentCache.cpp:48-50
    auto dir = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                   .getChildFile ("ConvoPeq")
                   .getChildFile ("MixedPhaseCache");
src/MixedPhasePersistentCache.cpp:69
    return getCacheDirectory ().getChildFile (filename);
```

したがって

```
cacheDir = %APPDATA%/ConvoPeq/MixedPhaseCache/
T = cacheDir/<64hex>.mph
S = cacheDir/staging/<64hex>.mph.stage
B = cacheDir/staging/<64hex>.mph.bak
```

**`T` と `S`/`B` は同一親ディレクトリ配下のサブディレクトリであり、同一 volume**（§19 で証明）。

### 1.3 DiskHeader v3 の layout（§9 の前提）

```
src/MixedPhasePersistentCache.h:68-85（現行 v2、#pragma pack(push,1)）
  offset  0  uint64_t magic              8
  offset  8  uint32_t version            4
  offset 12  uint32_t headerReserved     4
  offset 16  uint64_t fileHash           8
  offset 24  double    sampleRate        8
  offset 32  int32_t   phaseMode          4
  offset 36  float     freqStartHz        4
  offset 40  float     freqEndHz          4
  offset 44  int32_t   targetLength       4
  offset 48  uint64_t lastUsedTime        8
  offset 56  int32_t   numChannels        4
  offset 60  int32_t   numSamples         4
  offset 64  int32_t   numAllpassSections 4
  offset 68  int64_t   totalFileSize      8
  sizeof = 76
```

identity に属する field は `fileHash` / `sampleRate` / `phaseMode` / `freqStartHz` / `freqEndHz` / `targetLength` の **6 個**（合計 8+8+4+4+4+4 = 32 byte = 256 bit）。§3 の `IdentityBits` と 1:1 対応する。

---

## 2. 修正後の implementation contract（全体）

```text
[Disk identity, 256 bit]
  fileHash       64 bit  ─┐
  sampleRate     64 bit   │
  phaseMode      32 bit   ├─ encodeIdentityHex ──> IdentityHex (std::array<char,64>) ──> + ".mph" = 68 chars
  freqStartHz    32 bit   │
  freqEndHz      32 bit   │
  targetLength   32 bit  ─┘

[canonical namespace]  T = cacheDir/<64hex>.mph          （契約の管理対象）
[reserved namespace]   S = cacheDir/staging/<64hex>.mph.stage
                        B = cacheDir/staging/<64hex>.mph.bak

[判定]  classifyArtifact (path, requested) ──> CanonicalVerdict   （副作用ゼロ）
[操作]  decideAction (verdict, B の存在)   ──> Create | Replace | PreserveAndFail
[削除]  R1: B 削除可 iff classify(T) == current
        R2: （Owner 裁定待ち）current T の授権削除と同時なら B も削除可
        R3: classify(T) != current の B は何れの自動操作からも削除しない
        S1: in-flight handle が保持しない S は削除可
        S2: DeleteFileW 失敗は非致命的（preserve 扱い）
```

---

## 3. `IdentityHex` の型再設計（256 bit / Owner 中核指摘の修正）

### 3.1 型の定義

```cpp
namespace convo {

// 256-bit textual serialization（authoritative form）
using IdentityHex = std::array<char, 64>;

static_assert(sizeof (IdentityHex) == 64, "IdentityHex must be exactly 64 chars");
static_assert(std::is_trivially_copyable_v<IdentityHex>);

// 256-bit integer view（比較専用。永続化しない）
struct IdentityBits
{
    std::uint64_t fileHash;          //  64 bit
    std::uint64_t sampleRateBits;    //  64 bit
    std::uint32_t phaseModeBits;     //  32 bit
    std::uint32_t freqStartBits;     //  32 bit
    std::uint32_t freqEndBits;       //  32 bit
    std::uint32_t targetLengthBits;  //  32 bit
    // 合計 256 bit = 32 bytes
};

static_assert(sizeof (IdentityBits) == 32, "IdentityBits must be exactly 256 bit");
static_assert(std::is_trivially_copyable_v<IdentityBits>);

// Identity は DiskHeader の identity field の値 view（padding に依存しない field-by-field 比較を行う）
struct Identity
{
    std::uint64_t fileHash;          //  8
    double        sampleRate;        //  8
    std::int32_t  phaseMode;         //  4
    float         freqStartHz;       //  4
    float         freqEndHz;         //  4
    std::int32_t  targetLength;      //  4
};
```

**`IdentityBits` は 256 bit を整数 6 項目で保持する**ため、`IdentityHex`（64 char）↔ `IdentityBits`（32 byte）の変換は**過不足なく双方向**に対応する。

### 3.2 hex field layout（固定・不変）

| field | char offset | char width | bit range |
| --- | --- | --- | --- |
| `fileHash` | 0 | 16 | `[0, 64)` |
| `sampleRateBits` | 16 | 16 | `[64, 128)` |
| `phaseMode` | 32 | 8 | `[128, 160)` |
| `freqStartBits` | 40 | 8 | `[160, 192)` |
| `freqEndBits` | 48 | 8 | `[192, 224)` |
| `targetLength` | 56 | 8 | `[224, 256)` |
| 合計 | — | **64** | `[0, 256)` |

```cpp
inline constexpr std::array<std::size_t, 6> kIdentityHexFieldOffset = { 0, 16, 32, 40, 48, 56 };
inline constexpr std::array<std::size_t, 6> kIdentityHexFieldWidth  = { 16, 16,  8,  8,  8,  8 };
static_assert(kIdentityHexFieldOffset [0] ==  0);
static_assert(kIdentityHexFieldOffset [5] + kIdentityHexFieldWidth [5] == 64);
```

文字集合は **`[0-9a-f]` のみ**（`juce_String.cpp:1918 static const char hexDigits[] = "0123456789abcdef";` と同じ集合だが、**可変幅ではない** Dedicated serializer を使う）。

### 3.3 なぜ `uint64_t` では不可か（明示）

```
64 hex characters = 64 * 4 = 256 bit
std::uint64_t    =  64 bit
```

容量比 4:1。`uint64_t` 1 個では **最大 16 hex（= `fileHash` のみ）**しか表現できず、`sampleRateBits` 以降が切り詰められる。したがって:

| 型 | 容量 | 判定 |
| --- | --- | --- |
| `std::uint64_t` | 64 bit | **不可**（Owner 指摘どおり） |
| `std::array<char, 64>` | 64 char = 256 bit | **採用** |
| `std::array<std::uint64_t, 4>` | 256 bit | 整数 view として利用可能（`IdentityBits` がこれに相当） |
| `juce::String` | 可変 | **不可**（`hexToString` の可変幅性により長さの保証がない） |

---

## 4. encode / decode API の型設計

### 4.1 公開 API

```cpp
// --- 符号化（pure / 副作用ゼロ） ---
IdentityHex encodeIdentityHex (const Identity& id) noexcept;

// --- 復号（pure / 副作用ゼロ） ---
bool decodeIdentityHex (const IdentityHex& hex, Identity& out) noexcept;

// --- filename からの取り出し（pure / 副作用ゼロ） ---
bool parseIdentityHexFromFileName (const juce::String& fileName, IdentityHex& out) noexcept;
bool identityFromCanonicalFileName (const juce::String& fileName, Identity& out) noexcept;

// --- 整数 view（pure / 副作用ゼロ） ---
IdentityBits toIdentityBits (const Identity& id) noexcept;

// --- bit-pattern equality（pure / 整数比較のみ） ---
bool sameIdentity (const Identity& a, const Identity& b) noexcept;
bool sameIdentity (const IdentityBits& a, const IdentityBits& b) noexcept;

// --- filename 合成（pure） ---
juce::String canonicalFileNameFor (const Identity& id);   // 常に 68 文字
```

### 4.2 戻り値型の一覧

| 関数 | 入力型 | 出力型 | 失敗表現 |
| --- | --- | --- | --- |
| `encodeIdentityHex` | `const Identity&` | `IdentityHex`（64 char） | 失敗しない（domain 不正も hex 化される。domain 検証は別関数） |
| `decodeIdentityHex` | `const IdentityHex&` | `Identity&` | `false`（非 hex 文字・長さ不正のみ。domain 不正も `true` で復元され得る） |
| `parseIdentityHexFromFileName` | `const juce::String&` | `IdentityHex&` | `false`（68 文字形式違反・非 hex 文字） |
| `identityFromCanonicalFileName` | `const juce::String&` | `Identity&` | `false`（上記 + 合成のみ） |
| `toIdentityBits` | `const Identity&` | `IdentityBits`（32 byte） | 失敗しない |
| `sameIdentity` | `Identity` / `IdentityBits` | `bool` | 失敗しない |
| `canonicalFileNameFor` | `const Identity&` | `juce::String`（68 char） | 失敗しない |

### 4.3 「domain 検証」と「hex 変換」の分離

`encodeIdentityHex` は **domain 検証を行わない**。domain 検証は:

```
identityDomainValid (const Identity&) -> bool          // §7.3
classifyArtifact / inspectArtifact                      // §7.4 の検査 3 / 15
```

，位于する。`getCacheFile (invalid identity) -> empty File`（Owner 契約 I）は

```
canonicalFileNameForInternal (id):
    if (! identityDomainValid (id)) return juce::String();   // 空文字
    return canonicalFileNameFor (id);
```

で実現し、**invalid identity では filesystem access も directory creation も発生しない**。

### 4.4 hex 変換の実装要件（固定）

| 要件 | 内容 |
| --- | --- |
| 桁数 | 各 field を**ゼロ埋め**で固定幅出力する。`hexToString` 的な可変幅输出は禁止 |
| 文字集合 | `[0-9a-f]` のみ |
| `fileHash` | `std::uint64_t` を `16` 桁ゼロ埋め小文字 hex |
| `sampleRateBits` | `std::bit_cast<std::uint64_t> (double)` を `16` 桁ゼロ埋め小文字 hex。**`static_cast` 禁止** |
| `phaseMode` / `targetLength` | `std::int32_t` を `static_cast<std::uint32_t>` した値を `8` 桁ゼロ埋め小文字 hex（2 の補数表現。`phaseMode = -1` は `ffffffff`） |
| `freqStartBits` / `freqEndBits` | `std::bit_cast<std::uint32_t> (float)` を `8` 桁ゼロ埋め小文字 hex |
| NUL 終端 | `IdentityHex` は **NUL 終端しない**（64 文字ちょうど）。`juce::String` 合成時に別途 68 文字バッファを作る |

---

## 5. 全 identity field の bit-pattern equality

### 5.1 整数 view への変換（浮動小数点演算ゼロ）

```cpp
IdentityBits toIdentityBits (const Identity& id) noexcept
{
    return
    {
        id.fileHash,
        std::bit_cast<std::uint64_t> (id.sampleRate),
        static_cast<std::uint32_t> (id.phaseMode),
        std::bit_cast<std::uint32_t> (id.freqStartHz),
        std::bit_cast<std::uint32_t> (id.freqEndHz),
        static_cast<std::uint32_t> (id.targetLength)
    };
}

bool sameIdentity (const Identity& a, const Identity& b) noexcept
{
    return toIdentityBits (a) == toIdentityBits (b);       // IdentityBits の整数比較のみ
}
```

**`toIdentityBits` は `std::bit_cast` と整数変換のみ。浮動小数点の比較・演算を一切含まない。**

### 5.2 `std::bit_cast` の要件

| 要件 | 状態 |
| --- | --- |
| C++20 | authority `CMakeLists.txt:1143 set(CMAKE_CXX_STANDARD 20)` / `:1144 REQUIRED ON` |
| `<bit>` include | 必須 |
| trivially copyable な型であること | `double` / `float` は満たす |
| 同一サイズ | `double`(8) ↔ `uint64_t`(8) / `float`(4) ↔ `uint32_t`(4) |

### 5.3 値比較（`==`）を禁止する理由

| 入力 | bit-pattern | 値比較 `==` | 本設計 |
| --- | --- | --- | --- |
| `sampleRate = NaN` | `0x7FF8000000000000` | **false** | **true**（同一 bits なら） |
| `sampleRate = +0.0` vs `-0.0` | `0x0000…` vs `0x8000…` | **true**（IEEE: `+0.0 == -0.0`） | **false** |
| `sampleRate = 48000.0` vs `48000.0 + 1ulp` | 相異なる | **false** | **false**（正しい） |
| `freqStartHz` 同様 | — | — | 同上 |

bit-pattern identity は「**値として同じに見えるものも異なるなら別 identity**」という契約であるため、`==` は誤り。`sameIdentity` は整数 view の比較のみで構成する。

**現行 source の epsilon 比較（authority `:103`/`:105`/`:106`）在这一設計で完全に排除される。**

### 5.4 `/fp:fast` との関係（§1.1 の証拠に基づく）

icx Release で `/fp:fast` が有効なため、**canonical path 上の浮動小数点比較・演算は信用できない**。具体例: `/fp:fast` は「NaN 不会被生成される」という仮定を許すため、`v == v` や `std::isfinite(v)` のような式が定数畳み込みでCASE 既是 `true` に潰される可能性がある。

したがって:

| 箇所 | 浮動小数点演算を使うか |
| --- | --- |
| `encodeIdentityHex` / `decodeIdentityHex` / `toIdentityBits` / `sameIdentity` | **否**（整数のみ） |
| `canonicalFileNameFor` | **否** |
| SR / F1 / F2 の **domain 検証** | **否**（§7.3 の整数ビット範囲比較） |
| `F2 >= F1 + 10` の検証 | **不要**（§7.5 で per-field range から自動成立） |
| DiskHeader の構築（`header.sampleRate = id.sampleRate;`） | 値代入のみ（演算なし） |
| `touchOnHandleUnlocked` の timestamp 計算 | `juce::Time::currentTimeMillis()` の返却値のみ（§13） |

**canonical path 全体で浮動小数点演算ゼロ**が、本 contract の不変条件となる。

---

## 6. T-01〜T-15 の再設計 + 新規ケース

production test は read-only 制約のため作成しない。**仕様のみ確定。**

### 6.1 参照値（実測済み bit pattern）

```
44100.0   -> 0x40E5888000000000      768000.0  -> 0x4127700000000000
prev(44100.0) -> 0x40E5887FFFFFFFFF  （下界未満 -> reject）
next(768000.0)-> 0x4127700000000001  （上界超過 -> reject）
NaN(+)    -> 0x7FF8000000000000      NaN(-)    -> 0xFFF8000000000000  （両方とも reject）
+Inf      -> 0x7FF0000000000000      -Inf      -> 0xFFF0000000000000  （両方とも reject）
-0.0      -> 0x8000000000000000      （下界未満 -> reject）
+0.0      -> 0x0000000000000000      （下界未満 -> reject）

100.0f    -> 0x42C80000   1 ULP 下  -> 0x42C7FFFF (99.99999237060547)   （下界未満 -> reject）
400.0f    -> 0x43C80000
700.0f    -> 0x442F0000
1300.0f   -> 0x44A28000   1 ULP 上  -> 0x44A28001 (1300.0001220703125)（上界超過 -> reject）
120.0f    -> 0x42F00000
900.0f    -> 0x44610000
-0.0f     -> 0x80000000             （下界未満 -> reject）
NaNf      -> 0x7FC00000             （上界超過 -> reject）

48000.0   -> 0x40E7700000000000
```

### 6.2 参照 encode 出力（instance）

```
Identity { fileHash = 0x0123456789ABCDEF, sampleRate = 48000.0, phaseMode = 1,
           freqStartHz = 120.0f, freqEndHz = 900.0f, targetLength = 262144 }

IdentityHex (64 chars):
0123456789abcdef  40e7700000000000  00000001  42f00000  44610000  00040000
|<-- fileHash -->|<- sampleRateBits ->|<- pm ->|<-f1->|<--f2-->|<-- tl -->|

filename (68 chars):
0123456789abcdef40e77000000000000000000142f000004461000000040000.mph
```

### 6.3 再設計したテスト仕様

| # | ケース | 入力 | 期待 |
| --- | --- | --- | --- |
| **T-01** | identity round-trip（代表値） | §6.2 の `Identity` | `decode (encode (x))` が **bit-pattern 同一**（`sameIdentity == true`） |
| **T-02** | 固定幅（上位ゼロ埋め） | `fileHash = 0x0000000000000001` | hex 先頭 16 桁が `0000000000000001` |
| **T-03** | 固定幅（`phaseMode` / `targetLength`） | `phaseMode = 0x01`, `targetLength = 1` | `00000001` / `00000001` |
| **T-04** | 小文字のみ | 代表値 | 出力が `[0-9a-f]` に限る（大文字 `A`–`F` を含まない） |
| **T-05** | 文字数 | 代表値 | `IdentityHex::size() == 64`、`canonicalFileNameFor` の長さが **68** |
| **T-06** | `sampleRate` bit-pattern 感度 | `48000.0` と `48000.0 + 1ulp` | hex の `[16,32)` が**異なる**（他 field は不変） |
| **T-07** | `freqStartHz` bit-pattern 感度 | `120.0f` と `nextafterf(120.0f)` | hex の `[40,48)` が**異なる** |
| **T-08** | domain 外 `phaseMode` | `phaseMode = -1` | encode は `ffffffff`（8 桁ゼロ詰め）。`decodeIdentityHex` は `true` を返すが `identityDomainValid` は `false` |
| **T-09（新）** | **`decodeIdentityHex` の非 hex 文字** | 64 文字文字列 `"000000000000000000000000000000000000000000000000000000000000000g"`（**末尾 1 文字が `g`**） | **`false`** を返す |
| **T-10（新）** | **`decodeIdentityHex` の先頭非 hex 文字** | `"G000000000000000000000000000000000000000000000000000000000000000"` | **`false`** |
| **T-11** | `parseIdentityHexFromFileName` — 正常 | `0123456789abcdef…00040000.mph`（68 文字） | `true` + `IdentityHex` が `encodeIdentityHex` の出力と一致 |
| **T-12（新）** | 長さ不一致（63 文字） | `"0"*63 + ".mph"` | `false` |
| **T-13（新）** | 長さ不一致（65 文字 hex） | `"0"*65 + ".mph"` | `false` |
| **T-14（新）** | 拡張子不一致 | `<64hex>.mpc` | `false` |
| **T-15（新）** | 大文字 hex | `"AB" + "0"*62 + ".mph"` | `false`（canonical は lowercase のみ） |
| **T-16（新）** | 旧 hash 名（可変幅） | `"1234abcd.mph"` | `false`（canonical でない） |
| **T-17（新）** | `IdentityHex` の型特性 | — | `sizeof == 64`、`std::is_trivially_copyable_v == true` |
| **T-18（新）** | `IdentityBits` の型特性 | — | `sizeof == 32`、`std::is_trivially_copyable_v == true` |
| **T-19（新）** | field layout の整合 | — | `offset[i] + width[i] == offset[i+1]`（`i = 0..4`）、`offset[5] + width[5] == 64` |
| **T-20（新）** | field 単一変更の局所性 | 各 field のうち 1 個のみを 1 bit 変化させた 6 通り | hex の該当範囲のみが変化し、他 5 field の範囲が不変 |
| **T-21（新）** | NaN の bit-pattern equality | `sampleRate = NaN` の 2 インスタンス（同一 bits） | `sameIdentity == true`（**値比較なら false**） |
| **T-22（新）** | `+0.0` と `-0.0` の非同一性 | `freqStartHz = +0.0f` / `-0.0f` | `sameIdentity == false`（bit-pattern identity の帰結） |
| **T-23（新）** | SR domain の整数ビット境界 | `0x40E5887FFFFFFFFF`（下界 1 ULP 下）/ `0x40E5888000000000` / `0x4127700000000000` / `0x4127700000000001`（上界 1 ULP 上） | `false` / `true` / `true` / `false` |
| **T-24（新）** | SR domain の非有限・符号 | `0x7FF8000000000000`(NaN+) / `0xFFF8000000000000`(NaN-) / `0x7FF0000000000000`(+Inf) / `0xFFF0000000000000`(-Inf) / `0x8000000000000000`(-0.0) / `0x0000000000000000`(+0.0) | **全 6 通り `false`** |
| **T-25（新）** | F1 domain の整数ビット境界 | `0x42C7FFFF` / `0x42C80000` / `0x43C80000` / `0x43C80001` | `false` / `true` / `true` / `false` |
| **T-26（新）** | F2 domain の整数ビット境界 | `0x442EFFFF` / `0x442F0000` / `0x44A28000` / `0x44A28001` | `false` / `true` / `true` / `false` |
| **T-27（新）** | `F2 >= F1 + 10` が自動成立 | 全 domain 妥当な `(F1, F2)` 組 | `F2 >= F1 + 10` が**常に成立**（§7.5） |
| **T-28（新）** | 副作用ゼロの静的確認 | — | `classifyArtifact` / `inspectArtifact` / `encodeIdentityHex` / `decodeIdentityHex` / `parseIdentityHexFromFileName` / `canonicalFileNameFor` の各本体に §8.1 の禁止 API が現れない（source-level test） |
| **T-29（新）** | `computeKeyHash` 不在 | — | `MixedPhasePersistentCache.cpp` に `computeKeyHash` が残っていない（source-level test） |
| **T-30（新）** | `toHexString` 不在 | — | canonical filename 経路に `juce::String::toHexString` が現れない（source-level test） |

### 6.4 T-09 の修正理由（Owner 指摘）

```
旧 T-09: decodeIdentityHex (0xFFFFFFFFFFFFFFFF) -> false     ← 誤り
```

`0xFFFFFFFFFFFFFFFF` は `f` が **16 個**並ぶ **有効な 16 桁 hex 値**であり、64 桁 hex 文字列としても `f` のみからなる有効入力である。したがって `false` を返す根拠にならない。

新 T-09 は **文字列**として非 hex 文字 `g` を含む 64 文字入力を用いる。`decodeIdentityHex` の引数は `uint64_t` ではなく `IdentityHex`（`std::array<char,64>`）であるため、非 hex 文字の表現が可能になる。

---

## 7. canonical validator（副作用ゼロ）

### 7.1 公開 API

```cpp
enum class CanonicalVerdict
{
    absent,          // path が存在しない
    current,         // current v3 canonical
    future,          // version >= 4
    obsolete,        // version < 3
    malformed,       // ours だが構造的に不正
    noncanonical,    // filename shape 不正
    indeterminate    // I/O failure / 判定不能
};

struct CanonicalReport
{
    CanonicalVerdict verdict { CanonicalVerdict::indeterminate };
    std::uint32_t    observedVersion { 0 };
    DiskHeader       header {};
    IdentityBits     headerIdentity {};
    IdentityHex      headerHex {};
    std::int64_t     expectedSize { 0 };
    std::int64_t     actualSize   { 0 };
    juce::String     observedFilename {};
    const char*      reason {};          // 診断ラベルのみ。行動には影響しない
};

// 判定のみ。delete / overwrite / rename / directory 作成を一切行わない（§8）。
CanonicalVerdict classifyArtifact (const juce::File& path, const Identity* requested) noexcept;
CanonicalReport   inspectArtifact  (const juce::File& path, const Identity* requested) noexcept;
```

### 7.2 verdict → action

```cpp
enum class SaveAction { create, replace, preserveAndFail };

SaveAction decideSaveAction (CanonicalVerdict v) noexcept
{
    switch (v)
    {
        case CanonicalVerdict::absent:  return SaveAction::create;
        case CanonicalVerdict::current: return SaveAction::replace;
        default:                        return SaveAction::preserveAndFail;
    }
}
```

```
current       -> operation 対象
future        -> preserve
obsolete      -> preserve
malformed     -> preserve
noncanonical  -> preserve
indeterminate -> preserve
```

### 7.3 domain 検証（整数ビット比較のみ）

```cpp
// --- 境界の bit pattern（すべて実測値 / §6.1） ---
inline constexpr std::uint64_t kSrMinBits = 0x40E5888000000000ULL;  // 44100.0
inline constexpr std::uint64_t kSrMaxBits = 0x4127700000000000ULL;  // 768000.0
inline constexpr std::uint32_t kF1MinBits = 0x42C80000U;            // 100.0f
inline constexpr std::uint32_t kF1MaxBits = 0x43C80000U;            // 400.0f
inline constexpr std::uint32_t kF2MinBits = 0x442F0000U;            // 700.0f
inline constexpr std::uint32_t kF2MaxBits = 0x44A28000U;            // 1300.0f

bool sampleRateInDomain (double sr) noexcept
{
    const auto bits = std::bit_cast<std::uint64_t> (sr);          // 整数比較のみ
    return kSrMinBits <= bits && bits <= kSrMaxBits;
}

bool freqInDomainF1 (float hz) noexcept
{
    const auto bits = std::bit_cast<std::uint32_t> (hz);
    return kF1MinBits <= bits && bits <= kF1MaxBits;
}

bool freqInDomainF2 (float hz) noexcept
{
    const auto bits = std::bit_cast<std::uint32_t> (hz);
    return kF2MinBits <= bits && bits <= kF2MaxBits;
}
```

**整数範囲比較が `isfinite(v) && 44100.0 <= v <= 768000.0` と等価である証明**

IEEE 754 binary では、符号ビットが 0（非負）である値について **unsigned integer 順序が数値順序と一致する**。したがって:

| bits の位置 | 値の分類 | 判定 |
| --- | --- | --- |
| `bits < kSrMinBits` | `+0.0` / 正の小さな値 / **`-0.0`**（`0x8000…` は大きいので別）/ 負の有限 / **`-Inf`** / **負の NaN**（`0xFFF8…`） | **reject**（下界未満） |
| `kSrMinBits <= bits <= kSrMaxBits` | `44100.0` 以上 `768000.0` 以下の有限正値 | **accept** |
| `bits > kSrMaxBits` | `768000.0` 超 / **`+Inf`**（`0x7FF0…`）/ **正の NaN**（`0x7FF8…`） | **reject**（上界超過） |

`isfinite` も `>=` / `<=` も必要ない。**浮動小数点比較が完全に排除される。**

`float`（F1 / F2）も同一の議論が成り立つ（`0x7FC00000` NaNf は `0x44A28000` より大きい → reject、`0x80000000` `-0.0f` は `0x42C80000` より小さい → reject）。

### 7.4 `identityDomainValid`

```cpp
bool identityDomainValid (const Identity& id) noexcept
{
    return id.fileHash != 0                                    // fileHash != 0
        && sampleRateInDomain (id.sampleRate)                   // SR domain（整数比較）
        && id.phaseMode >= static_cast<std::int32_t> (ConvolverProcessor::PhaseMode::AsIs)
        && id.phaseMode <= static_cast<std::int32_t> (ConvolverProcessor::PhaseMode::Minimum)
        && freqInDomainF1 (id.freqStartHz)
        && freqInDomainF2 (id.freqEndHz)
        && id.targetLength >= 1
        && id.targetLength <= ConvolverProcessor::MAX_IR_LATENCY;
}
```

PhaseMode の authority は `src/ConvolverProcessor.h:117-122 enum class PhaseMode : int { AsIs = 0, Mixed = 1, Minimum = 2 };`。**PersistentCache 側で enum / range を再定義しない**（整数範囲比較のみ）。`phaseMode` は `int32_t` なので整数比較のみで表現可能。

### 7.5 `F2 >= F1 + 10` は per-field range から自動成立

```
max(F1) + 10.0f = 400.0f + 10.0f = 410.0f
min(F2)         = 700.0f
700.0f > 410.0f  -> 成立
```

したがって **domain 妥当な `(F1, F2)` 組すべてで `F2 >= F1 + 10` が必然的に成立**し、cross-field の浮動小数点比較は**不要**。整数 view では `kF2MinBits (0x442F0000) > kF1MaxBits (0x43C80000)` であることも確認できる（§6.1 で実測）。

**Owner 契約「PersistentCache 側で `F2 <= Nyquist` を追加しない」を維持する**（v3.78.62 §11 で確定）。SR の domain が十分小さい（768 kHz に対して F2 上限 1300 Hz）ため、Nyquist 制約は構造的に充足される。

### 7.6 判定順（22 段）

```
 1 filename が 68 文字 & [0-9a-f]{64} + ".mph"                    -> noncanonical
 2 parseIdentityHexFromFileName 成功                             -> noncanonical
 3 復元 identity が identityDomainValid                           -> noncanonical
 4 GetFileAttributesEx 成功                                      -> indeterminate
 5 FILE_ATTRIBUTE_DIRECTORY が立つ                               -> malformed
 6 reparse tag != 0                                              -> indeterminate
 7 GetFileSizeEx 成功                                            -> indeterminate
 8 GENERIC_READ で Open 成功                                     -> indeterminate
 9 ReadFile で 76 byte ちょうど                                  -> malformed
10 magic == kMagic                                              -> malformed
11 version >= 4                                                  -> future      (以降検査しない)
12 version < 3                                                   -> obsolete    (以降検査しない)
13 headerReserved == 0                                           -> malformed
14 fileHash != 0                                                 -> malformed
15 header identity の domain妥当（§7.3）                          -> malformed
16 header identity の shape domain妥当（§10.2）                   -> malformed
17 filename identity == header identity（IdentityBits 整数比較）  -> malformed
18 requested supplied なら requested == header identity            -> malformed
19 expectedSize = checkedArithmetic が overflow しない             -> malformed
20 header.totalFileSize == expectedSize                           -> malformed
21 actual file size == expectedSize                               -> malformed
22 expectedSize <= kMaxPersistentCacheFileBytes                   -> malformed
-- 全通過 -> current
```

**検査 11-12 で打ち切る理由**: v4 以降の header layout は我々が知らないため、構造を検査せず `future` として返す。**検査 13 を version より先に置かない**と future artifact を誤って `malformed` に倒す。

**検査 17 の実装**: `IdentityBits` の整数比較のみ。`header` から `toIdentityBits (identityFromDiskHeader (header))` を作り、`parseIdentityHexFromFileName` の結果（hex）を `decodeIdentityHex` したものと比較する。**浮動小数点比較を一切含めない。**

---

## 8. canonical validator の副作用ゼロ（Owner 指示 11）

### 8.1 使用してよい API（すべて非変更）

| API | 性質 |
| --- | --- |
| `GetFileAttributesEx` | attribute の読み取りのみ |
| `GetFileInformationByHandleEx (FileAttributeTagInfo)` | reparse tag の読み取りのみ |
| `CreateFileW (…, GENERIC_READ, FILE_SHARE_READ\|FILE_SHARE_WRITE\|FILE_SHARE_DELETE, …, OPEN_EXISTING, …)` | 書き込み access を要求しない。`FILE_FLAG_DELETE_ON_CLOSE` / `FILE_ATTRIBUTE_TEMPORARY` は**不使用** |
| `GetFileSizeEx` | サイズの読み取りのみ |
| `ReadFile` | 読み取りのみ |
| `CloseHandle` | handle の解放 |

### 8.2 禁止 API（validator 内で使用してはならない）

```
DeleteFileW / DeleteFileA
MoveFileExW / MoveFile / ReplaceFileW / ReplaceFile
CreateDirectoryW / createDirectory
FileOutputStream / WriteFile / SetEndOfFile / FlushFileBuffers
SetFileTime / SetFilePointerEx（書き込み位置変更用途）
SetFileInformationByHandle（FileDispositionInfo / FileRenameInfo / FileEndOfFileInfo）
FILE_FLAG_DELETE_ON_CLOSE / FILE_ATTRIBUTE_TEMPORARY
```

### 8.3 検証方法

`classifyArtifact` / `inspectArtifact` の**本体に対して** §8.2 の識別子が現れないことを source-level test で確認する（T-28）。テストは `src/tests/` に追加するが、本版では read-only 制約のため作成しない。

---

## 9. DiskHeader v3 の layout assertions

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
static_assert(offsetof (DiskHeader, magic)             ==  0);
static_assert(offsetof (DiskHeader, version)           ==  8);
static_assert(offsetof (DiskHeader, headerReserved)    == 12);
static_assert(offsetof (DiskHeader, fileHash)          == 16);
static_assert(offsetof (DiskHeader, sampleRate)        == 24);
static_assert(offsetof (DiskHeader, phaseMode)         == 32);
static_assert(offsetof (DiskHeader, freqStartHz)       == 36);
static_assert(offsetof (DiskHeader, freqEndHz)         == 40);
static_assert(offsetof (DiskHeader, targetLength)      == 44);
static_assert(offsetof (DiskHeader, lastUsedTime)      == 48);
static_assert(offsetof (DiskHeader, numChannels)       == 56);
static_assert(offsetof (DiskHeader, numSamples)        == 60);
static_assert(offsetof (DiskHeader, numAllpassSections)== 64);
static_assert(offsetof (DiskHeader, totalFileSize)     == 68);
static_assert(std::is_trivially_copyable_v<DiskHeader>);
```

### 9.1 identity と DiskHeader field の対応（1:1）

```cpp
// pure / 副作用ゼロ。std::memcpy や構造体比較を行わない。
Identity identityFromDiskHeader (const DiskHeader& h) noexcept
{
    return { h.fileHash, h.sampleRate, h.phaseMode, h.freqStartHz, h.freqEndHz, h.targetLength };
}

void storeIdentityToDiskHeader (const Identity& id, DiskHeader& h) noexcept
{
    h.fileHash     = id.fileHash;
    h.sampleRate   = id.sampleRate;
    h.phaseMode    = id.phaseMode;
    h.freqStartHz  = id.freqStartHz;
    h.freqEndHz    = id.freqEndHz;
    h.targetLength = id.targetLength;
}
```

**epsilon 比較（authority `:103`/`:105`/`:106` の `std::abs(...) > 1.0e-6`）を完全に排除する。** 比較は全て `IdentityBits` の整数比較（§5.1）。

### 9.2 version lifecycle

```
version == 3   -> current
version  >= 4  -> future     / preserve / reject / no auto-conversion / no auto-delete
version  <  3  -> obsolete   / preserve / reject / no auto-conversion / no auto-delete
runtime headerReserved != 0  -> malformed / preserve
```

**自動 migration / 自動 delete は行わない。**

### 9.3 訂正（維持）

現行 v2 layout（§1.3）でも `lastUsedTime == 48` / `totalFileSize == 68` / `sizeof == 76` は**既に成立**している。不足は 4 点:

1. `static_assert` の表明（現行 authority `MixedPhasePersistentCache.h:87` は `<= 128` のみ）
2. `std::is_trivially_copyable_v<DiskHeader>` の static_assert
3. runtime `headerReserved == 0` 検証（`load()` に無い）
4. `kVersion = 2 -> 3`

---

## 10. max file size の checked arithmetic

### 10.1 導出式

```
sizeof(DiskHeader)
+ kMaxIRLoadChannels * MAX_IR_LATENCY * sizeof(double)
+ kMaxPersistentCacheSections * sizeof(double)
+ kMaxPersistentCacheSections * sizeof(double)
```

### 10.2 shape domain（validator 検査 16 の入力）

```
numChannels       1 .. kMaxIRLoadChannels   (1 .. 8)
numSamples        1 .. MAX_IR_LATENCY        (1 .. 2,097,152)
numAllpassSections 0 .. kMaxPersistentCacheSections (0 .. 20)
rho.size() == theta.size()   ← save() の入力 validation 側
numAllpassSections == rho.size()  ← save() が header に書くため自動一致
```

`rho.size() == theta.size()` は**ファイル側から検証できない**（ファイルは `numAllpassSections` と `2 * numAllpassSections` 個の double を持ち `rho`/`theta` の区別を持たない）ため、`save()` の入力 validation に置く。

### 10.3 実装

```cpp
constexpr std::size_t checkedAdd (std::size_t a, std::size_t b) noexcept
{
    return (b > (std::numeric_limits<std::size_t>::max) () - a) ? 0 : a + b;
}
constexpr std::size_t checkedMul (std::size_t a, std::size_t b) noexcept
{
    return ((a != 0) && (b > (std::numeric_limits<std::size_t>::max) () / a)) ? 0 : a * b;
}

constexpr std::size_t deriveMaxPersistentCacheFileBytes() noexcept
{
    constexpr std::size_t headerBytes  = 76;                                  // = sizeof(DiskHeader)
    constexpr std::size_t channelBound = convo::irload::kMaxIRLoadChannels;    // 8
    constexpr std::size_t sampleBound  = static_cast<std::size_t> (ConvolverProcessor::MAX_IR_LATENCY);
    constexpr std::size_t sectionBound = 20;                                  // kMaxPersistentCacheSections

    std::size_t total = headerBytes;
    total = checkedAdd (total, checkedMul (checkedMul (channelBound, sampleBound), sizeof (double)));
    total = checkedAdd (total, checkedMul (sectionBound, sizeof (double)));   // rho
    total = checkedAdd (total, checkedMul (sectionBound, sizeof (double)));   // theta
    return total;
}

constexpr std::size_t kMaxPersistentCacheFileBytes = deriveMaxPersistentCacheFileBytes();
static_assert(kMaxPersistentCacheFileBytes == 134218124);   // 導出値の凍結（magic number の直接使用はしない）
```

### 10.4 導出値

```
76
+ 8 * 2,097,152 * 8 = 134,217,728
+ 20 * 8            = 160
+ 20 * 8            = 160
= 134,218,124 bytes
```

**Owner 指示「既定の checked arithmetic により 134,218,124 bytes とします」に一致。**

### 10.5 representability

| 型 | 判定 |
| --- | --- |
| `std::size_t`（Windows x64 = 64 bit） | `134,218,124 < 2^63` -> representable。`checkedAdd`/`checkedMul` で全段階を検査 |
| `std::uint64_t` | representable |
| `std::int64_t` | representable（`header.totalFileSize` は `int64_t` のまま） |
| `DWORD`（`WriteFile` の `nNumberOfBytesToWrite` の型） | `134,218,124 < 2^32` -> representable。ただし API は `size_t` を取る |
| `std::int32_t` / `std::uint32_t` | **非 representable**。`totalFileSize` は `int64_t` のままであり narrowing しない |

---

## 11. maxCacheEntries の semantic normalization

### 11.1 実装（`src/convolver/ConvolverProcessor.StateAndUI.cpp:382` の修正対象）

```cpp
static int normalizeMaxCacheEntriesFromDouble (double d) noexcept
{
    if (! std::isfinite (d)) return kDefaultPersistentCacheEntries;   // NaN / +-Inf -> 8
    if (std::trunc (d) != d) return kDefaultPersistentCacheEntries;   // fractional   -> 8
    if (d < static_cast<double> (kMinPersistentCacheEntries)) return kDefaultPersistentCacheEntries;
    if (d > static_cast<double> (kMaxPersistentCacheEntries)) return kMaxPersistentCacheEntries;
    return static_cast<int> (d);                                      // 1..64 が保証済み
}

static int normalizeMaxCacheEntries (const juce::var& v) noexcept
{
    if (v.isDouble()) return normalizeMaxCacheEntriesFromDouble (static_cast<double> (v));
    if (v.isInt())
    {
        const auto r = static_cast<int> (v);
        if (r < kMinPersistentCacheEntries) return kDefaultPersistentCacheEntries;
        if (r > kMaxPersistentCacheEntries) return kMaxPersistentCacheEntries;
        return r;
    }
    if (v.isInt64())
    {
        const auto r = static_cast<std::int64_t> (v);
        if (r < static_cast<std::int64_t> (kMinPersistentCacheEntries)) return kDefaultPersistentCacheEntries;
        if (r > static_cast<std::int64_t> (kMaxPersistentCacheEntries)) return kMaxPersistentCacheEntries;
        return static_cast<int> (r);
    }
    return kDefaultPersistentCacheEntries;   // bool / String / void / undefined / その他
}
```

### 11.2 写像表

| 入力 | 結果 |
| --- | --- |
| non-numeric（void / undefined / bool / String / Array / Binary / Object / Method） | `8` |
| NaN / +Inf / -Inf | `8` |
| fractional | `8` |
| integer < 1 | `8` |
| `1 .. 64` | unchanged |
| `> 64` | `64` |

### 11.3 `juce::var` の unchecked 経路の回避

`JUCE/modules/juce_core/containers/juce_Variant.cpp` の全変換:

| 経路 | 実装 | 本設計での扱い |
| --- | --- | --- |
| `:96 defaultToInt` | `return 0;` | `isDouble/isInt/isInt64` のいずれにも一致しないので到達しない |
| `:146 intToInt` | `return data.intValue;` | `isInt()` 分岐で型が保証される |
| `:179 int64ToInt` | `return (int) data.int64Value;` | **`isInt64()` 分岐で独自処理するため到達しない**（unchecked narrowing を回避） |
| `:212 doubleToInt` | `return (int) data.doubleValue;` | **`isDouble()` 分岐で独自処理するため到達しない**（unchecked narrowing を回避） |
| `:242 boolToInt` | `return data.boolValue ? 1 : 0;` | bool は non-numeric として `8` に写像（分岐に到達しない） |
| `:274 stringToInt` | `return getString (data)->getIntValue ();` | String は non-numeric として `8` に写像（分岐に到達しない） |

### 11.4 修正後の呼び出し側

```cpp
// src/convolver/ConvolverProcessor.StateAndUI.cpp:382
if (v.hasProperty ("maxCacheEntries"))
    setMaxCacheEntries (static_cast<std::size_t> (normalizeMaxCacheEntries (v.getProperty ("maxCacheEntries"))));
```

**`int -> size_t` の無検査変換（現行の `static_cast<size_t>(static_cast<int>(...))`）を排除する。** narrowing は `normalizeMaxCacheEntries` が `1..64` を保証した後にのみ起きる。

### 11.5 `setMaxCacheEntries` 側

`src/convolver/ConvolverProcessor.LoadPipeline.cpp:182-191` の `juce::jlimit<size_t>(1, 64, maxEntries)` は正しい。`1` と `64` を `kMinPersistentCacheEntries` / `kMaxPersistentCacheEntries` に置き換える（値の重複定義は避ける）。

---

## 12. save -> eviction（D-2 / Owner 指示 18）

### 12.1 修正対象（authority で確認済み）

```
src/convolver/ConvolverProcessor.MixedPhase.cpp:695  convo::MixedPhasePersistentCache::save(
src/convolver/ConvolverProcessor.MixedPhase.cpp:701  convo::MixedPhasePersistentCache::evictLRU(owner->getMaxCacheEntries());
```

`:695` の戻り値は捨てられ、`:701` は `:687 if (optimizedByCmaes)` のガードのみで**無条件実行**される。

### 12.2 修正後

```cpp
if (convo::MixedPhasePersistentCache::save (
        key.fileHash, key.sampleRate, static_cast<int> (key.phaseMode),
        key.f1, key.f2, key.targetLength, mixedIR, rho, theta))
    convo::MixedPhasePersistentCache::evictLRU (owner->getMaxCacheEntries ());
```

```
save == false         -> eviction しない
save == true          -> eviction を attempt
eviction failure      -> save 成功を rollback しない
```

**Owner が D-2 を承認済み。** 本版では実装しない。

---

## 13. timestamp（Owner 指示 13）

### 13.1 取得元

```cpp
const auto nowMs = juce::Time::currentTimeMillis ();   // int64 epoch millis
```

**`juce::Time::getMillisecondCounter ()` は使用しない。** JUCE 8.0.12 exact source:

```
JUCE/modules/juce_core/time/juce_Time.cpp:239   uint32 juce_millisecondsSinceStartup () noexcept;
JUCE/modules/juce_core/time/juce_Time.cpp:241   uint32 Time::getMillisecondCounter () noexcept
JUCE/modules/juce_core/time/juce_Time.cpp:220   int64  Time::currentTimeMillis () noexcept
   #if JUCE_WINDOWS
       struct _timeb t;
       _ftime_s (&t);
       return ((int64) t.time) * 1000 + t.millitm;
```

`getMillisecondCounter ()` は **`uint32` 起動経過ミリ秒**であり `2^32 ms` ≒ **49.7 日で wrap**する。`static_cast<uint64_t>` による昇格は wrap を防げない（authority `:190` / `:264` が現状この経路を使用）。

### 13.2 domain と符号化

```
0                  -> invalid / LAST
1 .. INT64_MAX     -> valid
> INT64_MAX        -> invalid / LAST
```

```cpp
struct LruKey
{
    bool          invalid;
    std::uint64_t lastUsedTime;
    juce::String  canonicalFilename;
};

LruKey makeLruKey (const DiskHeader& h, const juce::String& canonicalFileName) noexcept
{
    const auto raw = h.lastUsedTime;
    const bool invalid = (raw == 0) || (raw > static_cast<std::uint64_t> (INT64_MAX));
    return { invalid, raw, canonicalFileName };
}
```

**`lastUsedTime` の書込値**: `currentTimeMillis ()` が `1..INT64_MAX` ならその `uint64_t` 値を書き、`0` または範囲外なら `0` を書く（invalid を明示する）。

**future timestamp / clock rollback は単独では invalid の理由にしない。** `invalid` は `0` と `> INT64_MAX` の **2 条件のみ**。

### 13.3 LRU ordering（Owner 指示 15）

```cpp
bool operator< (const LruKey& a, const LruKey& b)
{
    if (a.invalid != b.invalid) return ! a.invalid;                  // valid が先
    if (! a.invalid && a.lastUsedTime != b.lastUsedTime) return a.lastUsedTime < b.lastUsedTime;
    return a.canonicalFilename < b.canonicalFilename;                // tie-break 昇順
}
```

---

## 14. `kMaxPersistentCacheSections` の位置付け（Owner 指示 14）

producer 側の現行設定（authority `src/convolver/ConvolverProcessor.MixedPhase.cpp:477`）:

```cpp
designer_config.numSections = liveReconfigure ? (highRateLive ? 2 : 8) : 20;
```

**PersistentCache 側は producer の現在の設定を拘泥せず、安全上限 20 を採用する。** producer 側が将来 32 に変更しても、validator の検査 16（`numAllpassSections ∈ 0..20`）が `malformed` として拒否する。**これが安全上限としての意味**であり、producer の値への依存はない。

---

## 15. `evictLRU(0)` = no-op（Owner 指示 19）

```cpp
private void evictLRUUnlocked (std::size_t maxCount)
{
    if (maxCount == 0)
        return;                                    // ★ clear() を呼ばない。deleteAllCurrentCanonicalUnlocked も呼ばない
    if (maxCount > kMaxPersistentCacheEntries) maxCount = kMaxPersistentCacheEntries;
    ...
}
```

**candidate は `classifyArtifact (...) == current` のみ。** `future` / `obsolete` / `malformed` / `noncanonical` / `indeterminate` は候補外 = preserve。

**production 到達可能性**（v3.78.63 §6.4 より）: `src/ConvolverProcessor.h:89 / :1083 int maxCacheEntries = 0;` と `src/convolver/ConvolverProcessor.StateAndUI.cpp:201 pendingOverride.maxCacheEntries = juce::jlimit (0, 64, snapshot.maxCacheEntries);` により `getMaxCacheEntries ()` が 0 を返し得るため、`evictLRU (0)` は production から到達し得る。**現在の `clear ();` alias は明示的な violation**（authority `:277`）。

---

## 16. `.stage` が canonical artifact の唯一の保存コピーにならない invariant（Owner 指示 8）

### 16.1 補題

**補題 L-1（昇格と退避の経路）**
`T`（canonical path）へ artifact のバイト列を「新規配置」する操作は `MoveFileExW (S -> T)` のみ。`T` の**既存**バイト列を退避させる操作は `ReplaceFileW (T, S, B, …)` のみで、退避先は必ず `B`。**`S` は昇格先であり退避先ではない。**

根拠: `ReplaceFileW` の `lpReplacedFileName = T` / `lpReplacementFileName = S` / `lpBackupFileName = B` であり、退避先は第 3 引数に明示される（`juce_Files_windows.cpp:372-378` はこの第 3 引数を `nullptr` にしているため退避先が存在せず、1176 で target が消滅する。D-1R 設計は `B` を必ず渡す）。

**補題 L-2（`B` の消費）**
`B` から退避・移設させる操作は rollback の `MoveFileExW (B -> T)` のみで、これは `B` を**消費**する（`MoveFileExW` は rename であり、`MOVEFILE_COPY_ALLOWED` を使わないため §19.1 の copy+delete 劣化は起きない）。

**補題 L-3（`T` の明示削除）**
`T` を `DeleteFileW` で削除するのは `clear ()` / `evictLRUUnlocked` / `removeUnlocked` の 3 つで、いずれも `classifyArtifact (T) == current` の場合に限る（Owner 契約 13/14）。**退避ではなく契約上の明示削除**である。

### 16.2 定理

任意の時点において、ある identity `X` の canonical artifact のバイト列は、`T (X)` または `B (X)` の少なくとも一方に存在する。ただし §16.1 補題 L-3 の契約上の明示削除を実行した直後を除く。

**証明**

`T (X)` のバイト列が現れる経路を列挙する。

1. **path A の commit**: `MoveFileExW (S -> T)`。`T` は空（`classifyArtifact (T) == absent` が path A の前提）なので、退避は不要。生成後 `T` にバイト列が存在する。✓
2. **path B の commit**: `ReplaceFileW (T, S, B, …)` 成功。旧バイト列は `B` へ、新バイト列は `T` へ。✓
3. **rollback 成功**: `MoveFileExW (B -> T)`。`B` を消費して `T` に旧バイト列を復元。✓
4. **rollback 失敗**: `T` は不定、`B` に旧バイト列（1177 の場合）。**`S` は退避先でない**（補題 L-1）。✓
5. **明示削除**: `clear`/`evictLRU`/`remove` による契約上の削除。例外として除外。

`T` から退避させうる操作は 2 と 4 のみ（補題 L-1）で、どちらも退避先は `B`。したがって `T` から退避したバイト列は `B` に存在する。∎

### 16.3 帰結（`.stage` 削除の安全性）

`S` のバイト列は常に次のいずれかである。

1. **未昇格の新候補**（`save ()` が書いたばかり、または `ReplaceFileW` 1177 で promotion に失敗した新 artifact）
2. **破棄された候補**（`ReplaceFileW` 1176 / 1175 / その他の失敗で `S` に残ったもの）

**`S` は canonical artifact でも superseded canonical artifact でもない**（補題 L-1）。したがって `S` の削除は如何なる artifact の喪失も招かない。∎

**これが S1（`in-flight handle が保持していなければ `.stage` を削除可）の証明である。**

---

## 17. R1 / R2 / R3 の全状態組合せ（Owner 指示 5・6）

### 17.1 状態空間

```
T ∈ { absent, current, future, obsolete, malformed, noncanonical, indeterminate }   (7)
B ∈ { absent, present }                                                              (2)
S ∈ { absent, present }                                                              (2)
                                          合計 28 セル（網羅的）
```

### 17.2 28 セルの完全表

凡例: `C` = `CreateFileW(S, CREATE_NEW)` / `R` = `ReplaceFileW(T,S,B)` / `P` = `PRESERVE_AND_FAIL` / `K` = `KEEP（R3 により保持）` / `Del` = `DeleteFileW`

| # | T | B | S | `save()` action | `save()` 終了時 B | `save()` 終了時 S | `.bak` 規則 | `.stage` 規則 | `evictLRU` | `clear` | `remove` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | absent | absent | absent | `C` → path A | absent | 昇格で消費 | R1 該当なし | — | T なし | T なし | no-op |
| 2 | absent | absent | present | **S1 で S を削除してから `C`** | absent | 昇格で消費 | R1 該当なし | S1 で削除 | T なし | T なし | no-op |
| 3 | absent | present | absent | `C` → path A。**B に触れない** | **K** | 昇格で消費 | **R3** | — | T なし。**B は K** | T なし。**B は K** | no-op。**B は K** |
| 4 | absent | present | present | **S1 で S を削除してから `C`**。**B に触れない** | **K** | 昇格で消費 | **R3** | S1 で削除 | T なし。**B は K** | T なし。**B は K** | no-op。**B は K** |
| 5 | current | absent | absent | `C` → path B → `R` | 成功: `Del`（D1）／失敗: absent | 成功: 消費／失敗: `Del` | R1 | — | T は候補 | T `Del` + **R2** | T `Del` + **R2** |
| 6 | current | absent | present | **S1 で S を削除してから `C` → path B → `R`** | 同 5 | 同 5 | R1 | S1 で削除 | T は候補 | T `Del` + **R2** | T `Del` + **R2** |
| 7 | current | present | absent | **D0 で B を `Del`** → path B → `R` | 成功: `Del`（新 B）／失敗: absent | 同 5 | **R1**（canonical == current） | — | T は候補 | T `Del` + **R2** | T `Del` + **R2** |
| 8 | current | present | present | **D0 で B を `Del`**、**S1 で S を削除**、`C` → path B → `R` | 同 7 | 同 5 | **R1** | S1 で削除 | T は候補 | T `Del` + **R2** | T `Del` + **R2** |
| 9 | future | absent | absent | `P`（S を作らない） | absent | absent | R1 該当なし | — | preserve | preserve | preserve |
| 10 | future | absent | present | `P`（S は残留。**S1 で削除可**） | absent | **S1 で削除** | R1 該当なし | S1 で削除 | preserve | preserve | preserve |
| 11 | future | present | absent | `P`。**B に触れない** | **K** | absent | **R3** | — | preserve。**B は K** | preserve。**B は K** | preserve。**B は K** |
| 12 | future | present | present | `P`。**B に触れない** | **K** | **S1 で削除** | **R3** | S1 で削除 | preserve。**B は K** | preserve。**B は K** | preserve。**B は K** |
| 13 | obsolete | absent | absent | `P` | absent | absent | R1 該当なし | — | preserve | preserve | preserve |
| 14 | obsolete | absent | present | `P` | absent | **S1 で削除** | R1 該当なし | S1 で削除 | preserve | preserve | preserve |
| 15 | obsolete | present | absent | `P`。**B に触れない** | **K** | absent | **R3** | — | preserve。**B は K** | preserve。**B は K** | preserve。**B は K** |
| 16 | obsolete | present | present | `P`。**B に触れない** | **K** | **S1 で削除** | **R3** | S1 で削除 | preserve。**B は K** | preserve。**B は K** | preserve。**B は K** |
| 17 | malformed | absent | absent | `P` | absent | absent | R1 該当なし | — | preserve | preserve | preserve |
| 18 | malformed | absent | present | `P` | absent | **S1 で削除** | R1 該当なし | S1 で削除 | preserve | preserve | preserve |
| 19 | malformed | present | absent | `P`。**B に触れない** | **K** | absent | **R3** | — | preserve。**B は K** | preserve。**B は K** | preserve。**B は K** |
| 20 | malformed | present | present | `P`。**B に触れない** | **K** | **S1 で削除** | **R3** | S1 で削除 | preserve。**B は K** | preserve。**B は K** | preserve。**B は K** |
| 21 | noncanonical | absent | absent | `P` | absent | absent | R1 該当なし | — | preserve | preserve | preserve |
| 22 | noncanonical | absent | present | `P` | absent | **S1 で削除** | R1 該当なし | S1 で削除 | preserve | preserve | preserve |
| 23 | noncanonical | present | absent | `P`。**B に触れない** | **K** | absent | **R3** | — | preserve。**B は K** | preserve。**B は K** | preserve。**B は K** |
| 24 | noncanonical | present | present | `P`。**B に触れない** | **K** | **S1 で削除** | **R3** | S1 で削除 | preserve。**B は K** | preserve。**B は K** | preserve。**B は K** |
| 25 | indeterminate | absent | absent | `P` | absent | absent | R1 該当なし | — | preserve | preserve | preserve |
| 26 | indeterminate | absent | present | `P` | absent | **S1 で削除** | R1 該当なし | S1 で削除 | preserve | preserve | preserve |
| 27 | indeterminate | present | absent | `P`。**B に触れない** | **K** | absent | **R3** | — | preserve。**B は K** | preserve。**B は K** | preserve。**B は K** |
| 28 | indeterminate | present | present | `P`。**B に触れない** | **K** | **S1 で削除** | **R3** | S1 で削除 | preserve。**B は K** | preserve。**B は K** | preserve。**B は K** |

### 17.3 表から読み取れる不変条件

| # | 不変条件 | 該当セル |
| --- | --- | --- |
| N-1 | **`.bak` が削除されるのは `classify(T) == current`（R1）の場合か、`T` の授権削除と同時（R2）の場合のみ** | 5–8（R1）、R2 セル |
| N-2 | **`classify(T) == absent` のとき `.bak` は削除されない** | **3, 4, 27 は該当せず、3・4 の B は `K`** |
| N-3 | **`classify(T) != current`（future/obsolete/malformed/noncanonical/indeterminate）のとき `.bak` は `evictLRU` / `clear` / `remove` からも削除されない** | 9–28 の B=present セルすべて `K` |
| N-4 | **`.stage` は in-flight handle が保持していなければ常に削除可** | 全 `S=present` セル |
| N-5 | **`T` が `current` でない場合、`save()` は `.stage` を作らない**（セル 9–28 の `P`） | 9–28 |
| N-6 | **`T` が `absent` の場合、`save()` は `.bak` に触れない** | 3, 4 |

**Owner が D-1R で指定した 3 ケース_maps**:

| Owner のケース | 該当セル | 結果 |
| --- | --- | --- |
| **T/B 同時存在** | 7, 8 | `T == current` なので **R1 成立 → D0 で B を削除してよい** |
| **T absent + B current** | **3, 4** | **B は `K`（R3）。削除しない。** T が `current` に再生成された後に限り D1/R1 で削除可 |
| **rollback failure** | §18 の transition 3 | §16 の定理により `B` に旧 artifact。**§17.7 で `rB == current` なら `K`** |

### 17.4 rollback failure 後の再分類（transition 専用）

```cpp
CanonicalReport rT = inspectArtifact (T, nullptr);
CanonicalReport rB = inspectArtifact (B, nullptr);

if (rT.verdict == CanonicalVerdict::current)
{
    // 復旧済み。残存 B は R1 で削除可
    if (rB.verdict != CanonicalVerdict::absent) DeleteFileW (B);
    DeleteFileW (S);
    return false;
}

if (rB.verdict == CanonicalVerdict::current)
{
    // ★ B が旧 artifact の唯一の保護コピー。R3 により削除しない
    DeleteFileW (S);
    juce::Logger::writeToLog ("MixedPhasePersistentCache: rollback failed; backup retained as sole protective copy");
    return false;
}

if (rT.verdict == CanonicalVerdict::absent && rB.verdict == CanonicalVerdict::absent)
{
    // 旧 artifact の所在が不明。indeterminate として何も削除しない
    juce::Logger::writeToLog ("MixedPhasePersistentCache: rollback failed; old artifact location unknown");
    return false;
}

// rT が future/obsolete/malformed/noncanonical -> R3 により B は自動保護
DeleteFileW (S);
return false;
```

### 17.5 T=current, B=current の同時存在セル（7, 8）の正当性

セル 7/8 は「`T` が current、`B` が存在」。ここで `B` が**旧 artifact の唯一の保護コピー**である可能性は**ない**。理由:

`B` が作られるのは `ReplaceFileW (T, S, B, …)` の完了時のみ。その瞬間 `T` は新 artifact を保持する。すなわち **`B` が存在時点で `T` は `current` である**。したがって R1 が成立し、`B` は superseded バージョンである。

逆に「`T` が current で `B` が旧 artifact の唯一の保護コピー」という状態は**到達不能**である（§16 の定理が `T` または `B` の少なくとも一方に旧 artifact の存在を保証しており、`T` が `current` なら `T` がそれ fulfilled する）。

### 17.6 `.stage` 残留と `CREATE_NEW` の衝突（N-1 の修正）

D-1R §5.4/§5.5 は `CreateFileW (S, GENERIC_WRITE, CREATE_NEW)` を使う。`.stage` が残留していると `ERROR_FILE_EXISTS`（`CreateFileW` dwCreationDisposition 表「`CREATE_NEW` (1) — Creates a new file, only if it does not already exist. If the specified file exists, the function fails and the last-error code is set to **ERROR_FILE_EXISTS** (80)」）で失敗する。

**修正**: `saveUnlocked` は `CreateFileW (S, …, CREATE_NEW)` の**直前**に同一 identity の `.stage` を `DeleteFileW` する（S1 により可能）。§17.2 の `S=present` セル（2, 4, 6, 8）がこれに対応する。`DeleteFileW` が失敗した場合は `save` を `false` で終了する（`ERROR_FILE_EXISTS` をそのまま `CREATE_NEW` にぶつけると原因が曖昧になるため）。

### 17.7 R2 採否に依存しない範囲

R1 / R3 / S1 / S2 / D0 / D1 の成立条件は **R2 の採否に依存しない**。R2 が採用されても棄却されても、N-1〜N-6 は成立する。

---

## 18. process crash before / after `ReplaceFileW`（Owner 指示 7）

### 18.1 crash 遷移表

| # | crash 時点 | `T` | `S` | `B` | 旧 artifact | 新 artifact | 次回 `save ()` の状態セル | 回復 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CR-1 | path B の `CreateFileW (S, …)` **前** | current（旧） | absent | absent | **T** | — | 7 or 5 | D0 で B 削除を判断（不在 → no-op） |
| CR-2 | `S` の書き込み**中** | current（旧） | partial | absent | **T** | 破棄 | 7 or 5 | 次回 §17.2 の `S=present` セルで **S1 により S 削除**。`T` は無傷 |
| CR-3 | `S` の `FlushFileBuffers` **後・`CloseHandle` 前** | current（旧） | complete | absent | **T** | `S` | 7 or 5 | 同上。`T` は無傷 |
| CR-4 | `ReplaceFileW` **呼出直前** | current（旧） | complete | absent | **T** | `S` | 7 or 5 | 同上 |
| CR-5 | `ReplaceFileW` **実行中** | 不定 | complete | 不定 | 1176/1175/他: `T` / 1177: `B` / 不明: 判定不能 | `S` | `classify(T)` で 5-8 / 3-4 / 9-28 へ分岐 | `classify` が実態を判定する |
| CR-6 | `ReplaceFileW` **成功後・`DeleteFileW (B)` 前** | **新（current）** | absent | 旧（superseded） | **`B`** | **`T`** | **7** | **R1 成立 → D0 で B 削除。→ 永久増殖しない（INV-6）** |
| CR-7 | rollback `MoveFileExW (B->T)` **実行中** | 不定 | complete | 不定 | §17.4 の再分類で確定 | `S` | `classify` で判定 | §17.4 |
| CR-8 | rollback **成功後・`DeleteFileW (S)` 前** | **旧（current）** | 新（complete） | absent | **T** | `S` | **5** | 次回 §17.2 の `S=present` セルで **S1 により S 削除** |

### 18.2 「`T` が部分的に書かれた state は存在しない」

`T` のバイト列は §16.1 補題 L-1 に従って **`MoveFileExW (S -> T)` と `ReplaceFileW (T, S, B, …)` のみ**で生成される。両者とも metadata 级别的 rename/replace であり、**write-in-place ではない**（whole-file RMW は §4.4 で廃止済み）。

したがって `T` は常に「完全な artifact」または「不在」であり、**部分的な `T` は存在しない**。これが crash recovery の核心である。

対比: 現行 authority の `touch (`:255` 全体読み → `:267 FileOutputStream` → `:270 write`) は `T` 自体を **in-place で 2 倍に伸ばす**ため、`T` が破損する。本設計はその経路を撤去する。

### 18.3 CR-6 が唯一の「削除漏れ」であり、回復可能

CR-6（`ReplaceFileW` 成功後に backup 削除前に crash）が「`.bak` が取り残され canonical は新しい」という状態。このとき:

```
classify(T) == current   ->  R1 成立  ->  D0 で B を削除  ->  セル 7 へ
```

したがって **CR-6 は次回 `save ()` で必ず解消され、`.bak` は永続増殖しない**。

---

## 19. Win32 failure semantics（Owner 指示 9）

### 19.1 `MoveFileExW`（target-absent commit と rollback）

`dwFlags`（Microsoft Learn 原文）:

| flags | 原文 | 本設計での帰結 |
| --- | --- | --- |
| `MOVEFILE_WRITE_THROUGH` 8 | 「**The function does not return until the file is actually moved on the disk.** Setting this value guarantees that a move performed as a copy and delete operation is flushed to disk before the function returns. The flush occurs at the end of the copy operation. This value has no effect if `MOVEFILE_DELAY_UNTIL_REBOOT` is set.」 | path A に適用。`S` と `T` は同一 volume（§19.4） |
| `MOVEFILE_COPY_ALLOWED` 2 | 「If the file is to be moved to a different volume, the function simulates the move by using the `CopyFile` and `DeleteFile` functions. **If the file is successfully copied to a different volume and the original file is unable to be deleted, the function succeeds leaving the source file intact.**」 | **使用しない**。copy+delete 劣化を構造的に除外 |
| `MOVEFILE_REPLACE_EXISTING` 1 | 「If a file named *lpNewFileName* exists, the function replaces its contents with the contents of the *lpExistingFileName* file, **provided that security requirements regarding access control lists (ACLs) are met**. If *lpNewFileName* names an existing directory, an error is reported.」 | rollback のみ。**ACL 条件で失敗し得る。文書は失敗後の destination 状態を規定しない** → §17.4 で **T と B を再分類**（仮定しない） |
| `MOVEFILE_DELAY_UNTIL_REBOOT` 4 | reboot 実行まで移動しない | **使用しない** |
| `MOVEFILE_CREATE_HARDLINK` 16 | 「Reserved for future use.」 | **使用しない** |
| `MOVEFILE_FAIL_IF_NOT_TRACKABLE` 32 | link source の追跡不能時に失敗 | **使用しない** |

### 19.2 `ReplaceFileW`

| `GetLastError ()` | 原文 | `lpBackupFileName` 指定時 |
| --- | --- | --- |
| `1176` | 「If `lpBackupFileName` was specified, the replaced and replacement files retain their original file names. **Otherwise, the replaced file no longer exists** and the replacement file exists under its original name.」 | `T` = 旧（intact）。`B` 不在 |
| `1177` | 「The replacement file still exists under its original name… The file to be replaced still exists with a different name. **If `lpBackupFileName` is specified, it will be the name of the replaced file.**」 | `T` 不在。**`B` = 旧** |
| `1175` | 「The replaced and replacement files retain their original file names.」 | `T` = 旧（intact） |
| その他（`ERROR_INVALID_PARAMETER` ほか） | 「the replaced and replacement files will retain their original file names. In this scenario, a backup file does not exist」 | `T` = 旧（intact） |

**L1 不変条件**: `lpBackupFileName` 指定時、`ReplaceFileW` が 0 を返した時点で**旧 artifact のバイト列は `T` か `B` のいずれかに存在する**。

### 19.3 `DeleteFileW`

| 状況 | 文書 | 扱い |
| --- | --- | --- |
| 対象が存在しない | 「the `DeleteFile` function fails with **ERROR_FILE_NOT_FOUND**」 | **S2**: 非致命的。already-deleted として proceed 可 |
| read-only file | 「the function fails with **ERROR_ACCESS_DENIED**」 | **S2**: 非致命的。preserve |
| 他 handle が open | 「The operating system delays file deletion until all handles to the file are closed.」（`CreateFileW` Remarks）→ 後続の `CreateFile` は `ERROR_ACCESS_DENIED` | **S2**: 非致命的。preserve。cross-process 副作用があるため §19.5 の sweep は identity 限定 |
| ACL | 「To delete or rename a file, you must have either delete permission on the file, or delete child permission in the parent directory.」 | **S2**: 非致命的。preserve |
| `lpNewFileName` が既存 directory（`MoveFileExW` の REPLACE_EXISTING） | 「If *lpNewFileName* names an existing directory, an error is reported.」 | `T` が directory の場合は validator 検査 5 で `malformed` となり `save ()` は `P`。rollback で発生しても §17.4 の再分類が `T` を `malformed` と判定 |

**`DeleteFileW` の失敗はいずれも `save ()` の logical result を変更しない**（S2）。`.bak` の削除失敗は INV-1 の「削除可能性の唯一の出所」を変えない（削除されなかった = preserved）。

### 19.4 same-volume

```
cacheDir = %APPDATA%/ConvoPeq/MixedPhaseCache/      (authority :48-50)
T        = cacheDir/<64hex>.mph
S        = cacheDir/staging/<64hex>.mph.stage
B        = cacheDir/staging/<64hex>.mph.bak
```

`S` / `B` は `cacheDir` の**サブディレクトリ**であり、`T` と同一 volume を共有する。**`ReplaceFileW` / `MoveFileExW` の same-volume 要件は構造的に充足される。** `MOVEFILE_COPY_ALLOWED` を必要としない。

**注意**: `staging/` の directory 作成は `ensureCacheDirectoryUnlocked ()` の責務であり、`getCacheDirectory ()`（pure path construction）からは行わない（Owner 契約 I）。

### 19.5 sharing semantics

| 用途 | `dwDesiredAccess` | `dwShareMode` | 根拠 |
| --- | --- | --- | --- |
| classification（validator） | `GENERIC_READ` | `FILE_SHARE_READ \| FILE_SHARE_WRITE \| FILE_SHARE_DELETE` | 他 process / AV scanner を妨げない。JUCE `FileInputStream::openHandle`（`juce_Files_windows.cpp:398-400`）と同じ方針 |
| `load` / `touch` | `GENERIC_READ \| GENERIC_WRITE` | `FILE_SHARE_READ \| FILE_SHARE_WRITE`（**`FILE_SHARE_DELETE` 禁止**） | `FILE_SHARE_DELETE` を外すことで、同一パスへの `DeleteFile` / `MoveFile` / `ReplaceFile` を OS が拒否する。`CreateFileW` dwShareMode「**FILE_SHARE_DELETE** (0x00000004) — Enables subsequent open operations on a file or device to request delete access.」 |
| staging 書込 | `GENERIC_WRITE` | `FILE_SHARE_READ` | 自 process のみ。`CREATE_NEW`（§17.6） |

**相互作用**: `ReplaceFileW` は `T` を「opened with the **GENERIC_READ, DELETE, and SYNCHRONIZE** access rights. The sharing mode is **FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE**」で開く。`load` / `touch` の handle は `FILE_SHARE_DELETE` を持たないため **`ReplaceFileW` の `DELETE` 要求は OS に拒否される**。

| 局面 | 結果 | 正当性 |
| --- | --- | --- |
| 同一 process 内の `load` と `save` の並行 | §20 の mutex が直列化するため発生しない | 正常 |
| cross-process の `load` と `save` の並行 | `save` が `ERROR_SHARING_VIOLATION` で失敗 | **正しい方向**（preserve）。cross-process 同期は契約外 |
| classification handle と `ReplaceFileW` の並行 | classification は `FILE_SHARE_DELETE` を含むため許容 | 正常（かつ分類は commit 前に閉じる） |

### 19.6 process-local mutex boundary

| 対象 | lock |
| --- | --- |
| `load` / `save` / `touch` / `evictLRU` / `remove` / `clear` / `getEntryCount` | **取得** |
| `getCacheDirectory` / `getCacheFile` / `canonicalFileNameFor` / `encodeIdentityHex` / `decodeIdentityHex` | **取得しない**（pure） |
| `classifyArtifact` / `inspectArtifact` / `toIdentityBits` / `sameIdentity` | **取得しない**（pure。ただし public API から呼ばれるときのみ呼び出し側の lock 下で実行される） |
| `ensureCacheDirectoryUnlocked` | **lock 済み前提**（private unlocked helper） |

- **1 個の process-local mutex**。non-recursive。
- **public → public の nested lock は 0 件**: `loadUnlocked → touchOnHandleUnlocked`（private→private）、`evictLRUUnlocked` は `clear ()` も `deleteAllCurrentCanonicalUnlocked ()` も呼ばない（`maxCount == 0` で `return`）、`clear → deleteAllCurrentCanonicalUnlocked`（public→private）。
- **cross-process 同期は契約外**（v3.78.62 §13 で確定）。
- **Audio Thread 到達なし**（T1 = `LoaderThread`, T2 = `AudioEngine::rebuildThread`）。

---

## 20. `R2` 採用 / 不採用の比較（Owner 指示 20）

| 軸 | **R2 を採用**（D-1R-a） | **R2 を採用しない**（D-1R-a-b） |
| --- | --- | --- |
| **ownership** | `.bak` が canonical 削除を跨いで蓄積しない。`clear ()` / `evictLRU ()` / `remove ()` が identity 全体を回収する | `.bak` は常に reserved namespace の私物。canonical 検索に現れない |
| **safety（唯一保護コピーの削除）** | **発生しない**。R2 は `T` が `current` の場合のみ適用され、その時点で `T` が同一 identity の有効な artifact を保持している（INV-2 と同じ論理） | **発生しない**。R3 のみが削除を許可する |
| **safety（1177 rollback failure）** | **発生しない**。`T` が不在なので R2 の前提が成立しない。R3 により保持 | **発生しない**。同上 |
| **capacity** | `clear ()` 後に staging namespace は空に近くなる。`.bak` の残留は identity 個数以下 | `clear ()` 後にも identity 個数分の `.bak` が残る。canonical namespace が空でも staging に容量が発生 |
| **Owner 契約 13/14 への準拠** | **準拠**（`T` が `current` の場合のみ削除）。ただし **reserved namespace を destructive operation の対象に拡張する**ため**契約の拡張** | **完全に準拠**（staging は `.mph` canonical namespace 外のため-destructive operation の対象外） |
| **N-1〜N-6 の成立** | **成立** | **成立** |

### 20.1 結論

**R2 の採否は安全性の差ではなく、容量と契約解釈の選択である。**

- どちらを選んでも **Owner の核心不変条件「旧 artifact の唯一のコピーを自動削除しない」は充足される**（N-2 / N-3 が R3 のみで成立するため）。
- **R2 を採用しない場合の方が Owner 契約 13/14 への準拠は厳密**である（staging namespace を契約の管理対象外に保つ）。
- **R2 を採用する場合は容量が改善されるが、契約に 1 文の追記**（`clear ()` / `evictLRU ()` / `remove ()` が current canonical を削除する場合に限り、同 identity の staging backup も削除する）が必要。

**本版では裁定しない。Owner の判断待ち。**

---

## 21. 定数（Owner 指示: 暫定案として維持）

```cpp
namespace convo {

constexpr double kPersistentCacheMinSampleRate = 44100.0;
constexpr double kPersistentCacheMaxSampleRate = 768000.0;

constexpr int kMinPersistentCacheEntries     = 1;
constexpr int kDefaultPersistentCacheEntries = 8;
constexpr int kMaxPersistentCacheEntries    = 64;

constexpr int kMaxPersistentCacheSections    = 20;

} // namespace convo
```

| 定数 | 状態 | 根拠 |
| --- | --- | --- |
| `kPersistentCacheMinSampleRate` / `MaxSampleRate` | **暫定** | internal OS の SR が 44.1 kHz〜768 kHz で変動するため。inclusive + finite（§7.3 の整数ビット比較で判定） |
| `kMinPersistentCacheEntries` / `kDefaultPersistentCacheEntries` / `kMaxPersistentCacheEntries` | **暫定** | `non-numeric → 8` / `integer < 1 → 8` / `1..64 unchanged` / `>64 → 64`（§11.2） |
| `kMaxPersistentCacheSections` | **暫定** | producer の現行設定（authority `MixedPhase.cpp:477` の 2 / 8 / 20）に依存しない **PersistentCache の安全上限**（§14） |
| `kMaxPersistentCacheFileBytes` | **導出** | `134,218,124` bytes（§10.4）。magic number として直接記述しない |

**再利用（重複定義しない）**:

| 値 | authority |
| --- | --- |
| PhaseMode | `src/ConvolverProcessor.h:117-122` `enum class PhaseMode : int { AsIs = 0, Mixed = 1, Minimum = 2 }` |
| F1 | `:225-226` `MIXED_F1_MIN_HZ = 100.0f` / `MIXED_F1_MAX_HZ = 400.0f` |
| F2 | `:228-229` `MIXED_F2_MIN_HZ = 700.0f` / `MIXED_F2_MAX_HZ = 1300.0f` |
| `MAX_IR_LATENCY` | `:250` `= 2097152` |
| `kMaxIRLoadChannels` | `src/convolver/IRLoadAdmission.h:43` `= 8u` |

**`F2 <= Nyquist` は PersistentCache 側で追加しない**（v3.78.62 §11 で Owner が確定、v3.78.64 §9 で維持）。

---

## 22. 判定

### 22.1 Owner 指定 20 項目の閉鎖状況

| # | 指示 | 節 | 状態 |
| --- | --- | --- | --- |
| 1 | Identity → 64 hex filename → Identity の完全 round-trip | §3, §4, §6 | **CLOSED** |
| 2 | 全 identity field の bit-pattern equality | §5 | **CLOSED** |
| 3 | encode/decode API の型設計 | §3.1, §4 | **CLOSED**（`uint64_t` 不成立を修正） |
| 4 | T-01〜T-15 の再設計 | §6.3（T-09 修正 + T-16〜T-30 追加） | **CLOSED** |
| 5 | R1/R2/R3 の全状態組合せ | §17.2（**28 セル網羅**） | **CLOSED** |
| 6 | T/B 同時存在、T absent + B current、rollback failure | §17.3, §17.4, §17.5 | **CLOSED** |
| 7 | process crash before/after ReplaceFileW | §18.1（CR-1〜CR-8） | **CLOSED** |
| 8 | `.stage` が canonical artifact の唯一の保存コピーにならない invariant | §16（補題 L-1〜L-3 + 定理 + 証明） | **CLOSED** |
| 9 | ReplaceFileW / MoveFileExW / DeleteFileW の failure semantics | §19.1, §19.2, §19.3 | **CLOSED** |
| 10 | same-volume / sharing / process-local mutex boundary | §19.4, §19.5, §19.6 | **CLOSED** |
| 11 | canonical validator の副作用ゼロ | §8 | **CLOSED** |
| 12 | SR domain inclusive + finite | §7.3（**整数ビット比較**） | **CLOSED** |
| 13 | timestamp は `juce::Time::currentTimeMillis ()` | §13.1 | **CLOSED** |
| 14 | `kMaxPersistentCacheSections = 20` | §14, §21 | **CLOSED** |
| 15 | max file size の checked arithmetic | §10（`134,218,124`） | **CLOSED** |
| 16 | DiskHeader v3 の layout assertions | §9（全 field offset を assert） | **CLOSED** |
| 17 | maxCacheEntries の semantic normalization | §11 | **CLOSED** |
| 18 | save success → eviction の D-2 | §12 | **CLOSED**（実装は次段階） |
| 19 | `evictLRU (0)` = no-op | §15 | **CLOSED** |
| 20 | R2 採用 / 不採用の比較 | §20 | **CLOSED**（裁定は Owner） |

### 22.2 本版で新たに固定した 3 点

| # | 内容 |
| --- | --- |
| F-1 | `IdentityHex = std::array<char,64>`（256 bit textual）と `IdentityBits`（256 bit integer view）を導入。`uint64_t` は**不成立** |
| F-2 | canonical path 全体で**浮動小数点演算ゼロ**。SR/F1/F2 の domain 検証は整数ビット範囲比較。根拠は authority `CMakeLists.txt:1783-1784` の `/fp:fast`（icx Release 有効） |
| F-3 | `.stage` 残留を `CreateFileW (S, …, CREATE_NEW)` の**直前**に削除する（D-1R の N-1 修正） |

### 22.3 Gate

```
Authority            = Owner 指定 ConvoPeq.md
Contract Gate        = PASS
Evidence Gate        = PASS
D-1R-2 residual ambiguity = 0
Owner ratification required = D-1R-a / D-1R-a-b（1 件の裁定のみ）
Implementation status = NO-GO（production source 未変更）
```

### 22.4 停止条件の遵守

```
production source 変更 0 / test 変更 0 / CMake 変更 0 / build 0 / version 変更 0
ConvoPeq 再生成 0 / commit 0 / push 0
git reset / clean / rebase / amend / squash / force push 0
staged 0 / HEAD = b0b46941（不変）
```

本成果物は **修正後の implementation contract のみ**。production implementation には進まない。