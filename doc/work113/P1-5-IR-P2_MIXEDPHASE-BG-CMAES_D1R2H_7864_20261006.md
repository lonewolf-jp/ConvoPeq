# D-1R-2H — Transaction State Machine / Identity Serialization / Usability Boundary Closure Audit

- 位置づけ: D-1R-2G の診断に対する read-only  Closure Audit
- 作成: 2026-10-06
- 本監査で変更していないもの: production source / tests / CMake / build / CTest / `ConvoPeq.md` 再生成 / commit / push（すべて 0）

---

## §0. 絶対条件の遵守

| 条件 | 遵守 |
| --- | --- |
| production source 変更 0 | 遵守 |
| tests 変更 0 | 遵守 |
| CMake 変更 0 | 遵守 |
| build / CTest 実行による変更 0 | 遵守 |
| `ConvoPeq.md` 再生成 0 | 遵守 |
| commit / push 0 | 遵守 |
| 監査報告書のみ作成 | `doc/work113/P1-5-IR-P2_MIXEDPHASE-BG-CMAES_D1R2H_7864_20261006.md` のみ |
| source authority は最新 snapshot のみ | 遵守（§1） |
| 古い snapshot の行番号を証拠に使用しない | 遵守（全引用が §1 の snapshot） |

---

## §1. source authority

| 項目 | 値 |
| --- | --- |
| ファイル | `C:\VSC_Project\ConvoPeq\doc\ConvoPeqMD\20261002231921\ConvoPeq.md` |
| byte size | 5,941,824 |
| SHA-256 | `3a51ab3414dba90005b99540376a93d6ab76ae1a96a754ce9e7050e2e0593045` |
| 実 line count | 131,019 |
| `split('\n')` の要素数 | 131,020（末尾 empty segment を含む。line count と混同しない） |
| CRLF 数 | 131,019（= LF 数。CRLF のみ） |

本報告書の `:NNNNN` はすべてこのファイルの実 line 番号（1-based, CRLF 基準）。

**本監査で確定した重要な事実（§2 の前提を反転させる）**:

> 現行実装には `.bak` が**存在しない**。`save()` は `ReplaceFileW` を `lpBackupFileName = nullptr` で呼んでいる。

証拠: `juce_Files_windows.cpp:372-378`
```cpp
bool File::replaceInternal (const File& dest) const
{
    return ReplaceFile (dest.getFullPathName().toWideCharPointer(),
                        fullPath.toWideCharPointer(),
                        nullptr,                                   // ← lpBackupFileName = NULL
                        REPLACEFILE_IGNORE_MERGE_ERRORS | 4 /*REPLACEFILE_IGNORE_ACL_ERRORS*/,
                        nullptr, nullptr) != 0;
}
```
MS Learn `ReplaceFileW` パラメータ説明: 「If this parameter is **NULL**, no backup file is created.」

したがって D-1R-2F / D-1R-2G の「`.bak` ownership」議論は、**現行コードに存在しない機構を前提にした設計議論**である。これを §2-§4 で実 Transaction 状態 T0〜T5 として再構成する。

---

## §2. 最重要：D-1R-2G の Option B 再監査

### 2.1 Claim

D-1R-2G は以下の transaction を第一候補とした。

```text
1. stage に新 IR を書く
2. FlushFileBuffers(stage)
3. target → <64hex>.mph.bak          （旧 target の退避）
4. stage → <64hex>.mph               （MoveFileExW）
5. 成功時: .bak を削除
6. 失敗時: .bak を target に復元
```

### 2.2 source evidence（現行の transaction）

現行 `save()`（`:77302-77377`）の実 transaction:

| 段階 | 行 | 処理 |
| --- | --- | --- |
| staging file 生成 | `:77321` | `juce::TemporaryFile tempFile(file);` → `<hash>_temp<randhex>.mph` を cache directory に作成 |
| stage 書込み | `:77323-77373` | `FileOutputStream` で header + payload を書込み、`stream.flush()`（`:77373`、戻り値無視） |
| commit | `:77376` | `return tempFile.overwriteTargetFileWithTemporary();` |
| → `replaceInternal` | `juce_Files_windows.cpp:374-377` | `ReplaceFileW(dest=target, this=temp, nullptr, IGNORE_MERGE_ERRORS\|IGNORE_ACL_ERRORS, nullptr, nullptr)` |

JUCE staging 命名規則（`juce_TemporaryFile.cpp:61-70`）:
```cpp
TemporaryFile::TemporaryFile (const File& target, const int optionFlags)
    : temporaryFile (createTempFile (target.getParentDirectory(),
                                    target.getFileNameWithoutExtension()
                                    + "_temp" + String::toHexString (Random::getSystemRandom().nextInt()),
                                    target.getFileExtension(), optionFlags)),
      targetFile (target)
```
→ `<hash>_temp<randhex>.mph`。**cache directory 内にあり、拡張子は `.mph`。**

### 2.3 Reasoning：Option B の T0〜T5 完全列挙

| 状態 | target | stage | bak | 到達条件 | target は存活か |
| --- | --- | --- | --- | --- | --- |
| T0 | valid | valid | absent | save 開始直後（既存 target あり、bak なし） | yes |
| T1 | absent | valid | valid | step3 成功、step4 未実行の直後 | **no（target 不在）** |
| T2 | valid（新） | absent | valid | step4 成功、step5（bak 削除）未実行 | yes |
| T3 | absent | valid | valid | step4 **失敗**（`MoveFileExW` が sharing violation / permission 等で失敗） | **no** |
| T4 | valid | valid | valid | step3 **失敗**（`MoveFileExW(target, bak)` が失敗、stage は残存） | yes |
| T5 | valid | valid | valid | step3 成功だが **既存 bak があり** `MoveFileExW(target, bak)` が失敗 | yes |

T1-T3 はいずれも **target が存在しない窓**を持つ。T1/T3 では「`.bak` に旧 target がある」ので recovery 対象となる。T4/T5 は target が無傷。

### 2.4 各操作直後の故障注入

| 注入点 | 結果状態 | recovery 可能か | target 破壊か |
| --- | --- | --- | --- |
| step3 直後に process crash | T1 | **yes**（bak valid） | target 一時不在のみ |
| step3 直後に I/O failure | T1 | yes | 同上 |
| step3 直後に permission failure | T1（bak への rename 成功後なら） | yes | 同上 |
| step3 直後に sharing violation | T1 or T4 | T1 なら yes / T4 なら N/A（target あり） | raulich による |
| step4 直後に process crash | T2 | N/A（target は新 content で valid） | no |
| step4 直後に I/O failure | T2（rename は atomic なので partial 不会出现） | N/A | no |
| step4 直後に sharing violation | T3 | yes（bak valid） | no |
| step5（bak 削除）直後に crash | T2 | N/A | no（residue `.bak` のみ） |
| step5 が失敗 | T2（save は成功、cleanup 失敗） | N/A | no |

**結論**: Option B の T1/T3 窓は必ず存在する。「target が存在しない時間」を 0 にはできない。Owner が求める「既存 valid canonical target を意図的に破壊しない」は **PROVEN**（step3 は rename であり truncate ではない）。ただし「target が常に存在する」は **CONTRADICTED**。

### 2.5 Evidence 判定

| 項目 | 判定 | 根拠 |
| --- | --- | --- |
| Option B は target を truncate しない | **PROVEN** | `MoveFileExW` は metadata 操作。`:77376` の現行実装も `ReplaceFileW` であり truncate しない |
| Option B に target 不在窓がある | **PROVEN** | T1/T3 の定義そのもの |
| Option B の T1/T3 が recovery 可能 | **SUPPORTED** | recovery アルゴリズムを §3 で定義すれば成立。production 実証は未実施 |
| Option B が「backup が永久に残らない」を満たす | **PROVEN** | step5 で成功時に削除。T2 の residue は次回 save の pre-transaction cleanup（§4）で消える |

---

## §3. 特に危険なのは「target → bak」の部分

### 3.1 Claim（ケース A）

```text
target exists
bak absent
```
なら `MoveFileExW(target, bak, 0)` が成功し、target は一時的に存在しない。crash すると T1（target absent / bak valid / stage valid）。

### 3.2 Claim（ケース B）

```text
target exists
bak exists
```
で `MoveFileExW(target, bak, 0)` は何 Happens か。

### 3.3 external evidence（MS Learn `MoveFileExW`）

`MOVEFILE_REPLACE_EXISTING`（1 / 0x1）の説明:

> **MOVEFILE_REPLACE_EXISTING** - 1 (0x1) — If a file named *lpNewFileName* exists, the function replaces its contents with the contents of the *lpExistingFileName* file, provided that security requirements regarding access control lists (ACLs) are met.

つまり **置換は `MOVEFILE_REPLACE_EXISTING` 指定時のみ可能**。指定しない `MoveFileExW(target, bak, 0)` は destination が存在する場合に **成功しない**。

### 3.4 Reasoning

| 手順 | 結果 | 評価 |
| --- | --- | --- |
| `既存 bak を削除` → `target → bak` | bak（旧）を失い、target（旧）を bak へ。bak は「旧 target」になるが「それ以前の bak」は消える | **非推奨**（Owner の懸念どおり、退避先を 2 世代分しか持てない） |
| `target → bak`（`MOVEFILE_REPLACE_EXISTING` 付き） | 既存 bak を**上書き**する。MS Learn は ACL 条件付きと明記 | **非推奨**（既存 bak を破壊する。step3 の失敗時に旧 bak へ戻れない） |
| `target → bak`（フラグ無し、bak 存在） | **失敗**。target は無傷、stage も無傷 | **推奨**（安全な no-op。ただし毎 transaction で失敗するので運用が閉じない） |
| `target → bak`（フラグ無し、bak absent） | 成功。target 一時不在（T1） | 推奨（1177 の attributes 継承を回避できる） |

### 3.5 §3 の結論

**「`target exists` かつ `bak exists`」は、その状態のまま save 開始してはならない。**

Owner 指示 §3 の「既存 `.bak` は startup recovery で処理してから save」方式は、この状態に対する唯一の安全な解である（§4 で契約化）。

---

## §4. 「既存 residue を整理してから transaction を開始する」方式

### 4.1 Claim

`save()` は無条件に `target`/`stage`/`bak` を整理してはならない。まず **Residue Classification** を行い、判定結果に応じて初めて destructive cleanup を許す。

### 4.2 Contract（Owner 第一候補の採用。名称は Owner's でなくよい）

```text
classifyResidue(identity) -> ResidueVerdict {
    SafeToStartNewTransaction,   // residue が無く，或は同一 transaction の stale のみ
    RecoverableResidue,          // target absent + 修復可能な residue
    UnsafeResidue,               // 修復不能 / identity 不整合 / 未知 version
}
```

`save()` の入口順序（contract）:

```text
1. input validation
2. classifyResidue(identity)
3. if (RecoverableResidue)  -> runRecovery(identity) ; re-classify
4. if (UnsafeResidue)       -> return false（既存 artifact を一切触らない）
5. if (SafeToStartNewTransaction) -> proceed to transaction
```

**核心規則**: 「`save` が stale residue を見つけたからといって既存 artifact を勝手に削除しない」。destructive cleanup は `classifyResidue` が `SafeToStartNewTransaction` と判定し、かつ cleanup 対象が**同一 identity の stale residue** に限定されている場合のみ許可される。

### 4.3 Evidence

- 現行実装には classification 概念が存在しない。`save()` は `getCacheFile`（`:77311`）で得たパスに対して `TemporaryFile` を作り、`:77376` で無条件 commit する。residue 判定は行わない。→ **現行は NOT PROVEN / 設計としては本節が新規 contract**
- `evictLRU`（`:77445-77458`）は `magic` のみ検査（`:77454`）し、version も identity も検証しない。→ 既存 residue の誤認 handling 能力なし

### 4.4 判定

| 項目 | 判定 |
| --- | --- |
| save 前に residue を分類する順序 | **PROVEN**（Owner 指示が要求する設計。§4.2 で contract 化） |
| classification 順序が現行に存在するか | **CONTRADICTED**（`:77302-77377` に判定ロジックなし） |
| `UnsafeResidue` 時に既存 artifact を保護する | **PROVEN**（contract 上、`return false` 前に destructive 操作が存在しない） |

---

## §5. artifact state classification（3 boolean では不十分）

### 5.1 Claim

Owner 指示 §4 の通り、各 artifact を boolean ではなく **8 値 state** で分類する。

### 5.2 Contract

```text
enum class ArtifactState {
    Absent,                  // 存在しない
    CurrentV3Canonical,       // version==3 かつ header 全条件成立 かつ filename==encode(header)
    CurrentV3NonCanonical,    // version==3 かつ header 全条件成立 だが filename mismatch / uppercase
    Future,                   // version > 3
    Obsolete,                 // version < 3（v2 含む）
    Malformed,                // 読めない / sizeof 不足 / magic 不一致 / domain 違反
    NonCanonical,             // filename が 64 fixed-width lowercase hex + 既定 extension でない
    ValidRecoveryCandidate,   // recovery namespace にあり、self-consistency 成立
};
```

### 5.3 「valid/valid/valid」の非同一性の実例（Owner の指摘）

| 組合せ | T 相当の状態 | 意味 |
| --- | --- | --- |
| target=valid / stage=valid / bak=valid | T0/T5 | 通常の save 開始状態。ただし bak は「前世代 target」であり**同一 transaction の bak ではない** |
| target=valid / stage=malformed / bak=valid | T0' | stage は破損。transaction 開始前に削除してよい（同一 identity の stale のみ） |
| target=valid / stage=valid / bak=malformed | T0'' | bak は無効。startup recovery では採用しない。cleanup 対象 |
| target=Absent / stage=CurrentV3Canonical / bak=Obsolete | T3' | bak は v2。**v2 は preserve 契約なので削除しない**。stage のみを検証対象にする |
| target=Future / stage=Absent / bak=Absent | S7 | future artifact。**一切触らない** |

### 5.4 Evidence

- 現行コードに state 分類は存在しない。`evictLRU` は magic のみ（`:77454`）、`load` は magic + version + identity を検査（`:77252-77260`）、`clear` は何も検査しない（`:77486-77489`）。→ **8 値分類は新規 contract**
- `Obsolete`（v2）を保護する契約は D-1（裁定済）と整合。現行は守っていない（`clear` が無条件削除、§10 参照）

### 5.5 判定

| 項目 | 判定 |
| --- | --- |
| 8 値 state 分類の妥当性 | **PROVEN**（Owner 指示 + boolean では「target valid / stage malformed / bak valid」と「target valid / stage valid / bak valid」を区別できない実例） |
| 現行コードが 8 値を判別できるか | **CONTRADICTED**（`clear`/`evictLRU`/`remove`/`getEntryCount` は無検査） |

---

## §6. identity の異なる stage/bak を混ぜない

### 6.1 Claim

`AAAAAAAA.mph` / `AAAAAAAA.mph.stage` / `BBBBBBBB.mph.bak` が同時に存在するとき、`AAAAAAAA.mph.stage` と `BBBBBBBB.mph.bak` を同一 transaction として扱ってはならない。

### 6.2 Contract

transaction は **transaction identity** 単位で構成される。

```text
identity A
    ├── A.mph          → ArtifactState for A
    ├── A.mph.stage    → ArtifactState for A
    └── A.mph.bak      → ArtifactState for A

identity B
    ├── B.mph
    └── B.mph.bak      → 別 transaction
```

- transaction membership の証明は **`<64hex>` identity prefix の一致**による
- **filename suffix（`.stage` / `.bak`）だけでは membership を証明しない**

### 6.3 Evidence

現行 filename（`:77212-77222`）:

```cpp
juce::File MixedPhasePersistentCache::getCacheFile(...)
{
    const auto hash = computeKeyHash(fileHash, sampleRate, phaseMode,
                                     freqStartHz, freqEndHz, targetLength);
    const auto filename = juce::String::toHexString(static_cast<int64_t>(hash)) + ".mph";
    return getCacheDirectory().getChildFile(filename);
}
```

JUCE `String::toHexString` は **zero padding を行わない可変長**である。JUCE 自体の unit test（`juce_String.cpp:2625-2633`）がこれを証明している:

```cpp
expect (String::toHexString (0x1234abcd).equalsIgnoreCase ("1234abcd"));          // 8 桁
expect (String::toHexString ((int8)  -1).equalsIgnoreCase ("ff"));                  // 2 桁
expect (String::toHexString ((int16) -1).equalsIgnoreCase ("ffff"));                // 4 桁
expect (String::toHexString ((int32) -1).equalsIgnoreCase ("ffffffff"));            // 8 桁
expect (String::toHexString ((int64) -1).equalsIgnoreCase ("ffffffffffffffff"));   // 16 桁
```

→ 現行 v2 の artifact は、hash が小さいと 1〜15 桁になる。**fixed-width 64 hex ではない。**

### 6.4 §6 の結論（重要）

1. **identity 分離の要件**: v3 の filename は `<64hex>.mph` であり、identity は filename 全体から機械的に復元できる。→ 現 行の hash 方式（不可逆）では transaction identity を filename から復元できない。→ **v3 の 64-hex identity は transaction identity 分離の必要条件である**（PROVEN）
2. **既存 v2 artifact の分類**: v2 の filename は可変長 hex であり fixed-width 64 hex ではないため、v3 contract 下では **全 v2 artifact が `NonCanonical`** となる。さらに `version == 2 < 3` なので `Obsolete` でもある。→ **二重に保護される**（PROVEN）
3. **cross-identity mixing の現行リスク**: 現行は staging file が `<hash>_temp<randhex>.mph` であり（§2.2）、identity は hash でしか表現されない。§10 で述べる `*.mph` namespace 汚染と結合する。

### 6.5 判定

| 項目 | 判定 |
| --- | --- |
| transaction identity を filename から復元できること | **PROVEN**（64-hex 逆変換が deterministic。§7） |
| 現行 v2 filename が 64-hex でないこと | **PROVEN**（JUCE unit test `:2625-2633` + `:77220`） |
| 全 v2 artifact が v3 で `NonCanonical` + `Obsolete` | **PROVEN** |
| 現行実装で cross-identity mixing が起きないこと | **NOT PROVEN**（hash 方式では identity を filename から判別できず、`*.mph` 列挙が prefix を考慮しない。§10） |

---

## §7. 64-hex identity serialization：Option A vs Option B

### 7.1 Claim

D-1R-2G は「byte order = little-endian」と書いた。Owner's 指摘は正しい。`0x1122334455667788` を `1122334455667788` として hex 化するなら、それは little-endian **byte serialization** ではない。

### 7.2 Option A — numeric fixed-width hex

```text
uint64_t → 16 lowercase hex digits   (zero padded)
uint32_t →  8 lowercase hex digits   (zero padded)
```

`std::hex` / 手書き nibble 変換で **数値の 16 進表記**を固定幅で出力。byte order の概念を持ち込まない。

### 7.3 Option B — byte serialization

各 field を `memcpy` で `uint8_t[N]` に展開し、**明示 endian** で hex 化。

```text
uint64_t 0x1122334455667788 を little-endian byte serialize
    → bytes 88 77 66 55 44 33 22 11
    → hex   8877665544332211
```

### 7.4 比較

| 観点 | Option A（numeric fixed-width hex） | Option B（byte serialization） |
| --- | --- | --- |
| 「filename」としての意味 | 数値の hex 表記。人間可読 | byte 列の hex dump |
| byte order の要否 | **不要** | 明示必須 |
| cross-platform 決定性 | **あり**（ホスト endian に依存しない） | 明示指定すればあり |
| field 境界の可読性 | 区切りが無い（64 桁を位置で分割） | 同左 |
| round-trip 決定性 | **完全** | 完全 |
| 実装の単純さ | 高 | 中 |
| 事故の可能性 | なし | endian 指定の書き忘れ |
| Owner の第一候補 | **一致** | — |

### 7.5 Contract（Option A を採用）

```text
encodeIdentityHex(identity) =
      hex16(fileHash)                 // uint64_t,  16 digits, lowercase, zero padded
    + hex16(sampleRateBits)           // uint64_t,  16 digits, lowercase, zero padded
    + hex08(phaseModeBits)            // uint32_t,   8 digits, lowercase, zero padded
    + hex08(freqStartBits)            // uint32_t,   8 digits, lowercase, zero padded
    + hex08(freqEndBits)              // uint32_t,   8 digits, lowercase, zero padded
    + hex08(targetLengthBits)         // uint32_t,   8 digits, lowercase, zero padded
= 64 lowercase hex digits
```

| field | C++ 型 | 変換 | 桁数 |
| --- | --- | --- | --- |
| `fileHash` | `uint64_t` | そのまま | 16 |
| `sampleRateBits` | `double` | `std::bit_cast<uint64_t>` | 16 |
| `phaseModeBits` | `int32_t` | `static_cast<uint32_t>`（two's complement bit pattern を保持） | 8 |
| `freqStartBits` | `float` | `std::bit_cast<uint32_t>` | 8 |
| `freqEndBits` | `float` | `std::bit_cast<uint32_t>` | 8 |
| `targetLengthBits` | `int32_t` | `static_cast<uint32_t>` | 8 |

**byte order は仕様に含めない**。endian という概念は filename serialization に存在しない。bit pattern は `bit_cast` で platform-independent に扱われる。

### 7.6 decode → encode の deterministic 性

```text
decode(filename)  → identity fields
encode(identity)  → filename
```

- 各 field は固定幅なので桁あふれ・桁不足が起きない
- `bit_cast` は bit pattern を platform 依存なく保存・復元する
- `static_cast<uint32_t>(int32_t)` は modulo 2^32 で同じ bit pattern
- `decode → encode → decode → encode` で不動点

**PROVEN**（純粋関数の合成であり、外部状態に依存しない）。

### 7.7 hex の扱い

| 項目 | 仕様 |
| --- | --- |
| emit case | **lowercase 固定**（`0-9a-f`） |
| decode case | `0-9a-fA-F` を case-insensitive で受理（**decode 可能 ≠ canonical**。§9） |
| 桁数不足 | `InvalidCandidate` |
| 桁数超過 | `InvalidCandidate` |
| non-hex 文字 | `InvalidCandidate` |
| all-FF (`0xFFFFFFFFFFFFFFFF`) | **有効**。T-09 裁定済（16 桁 hex として正当） |
| leading zero | **有効**。固定幅なので常に存在する（`fileHash == 0` なら 16 個の `0`） |
| extension | `.mph`（canonical）/ `.mph.stage` / `.mph.bak` |

---

## §8. NaN / Inf の identity 上の扱い

### 8.1 Claim

`sampleRate` に NaN/Inf、`freqStartHz`/`freqEndHz` に NaN/Inf を含む identity を許す必要はありません。

### 8.2 Evidence

現行 domain validator:

- `:12833` `if (!hasFiniteDouble("mixedF1Hz", MIXED_F1_MIN_HZ, MIXED_F1_MAX_HZ)) return false;`
- `:12834` `if (!hasFiniteDouble("mixedF2Hz", MIXED_F2_MIN_HZ, MIXED_F2_MAX_HZ)) return false;`
- `:45872` `static constexpr int kMaxSampleRate = 768000;`

producer 側の F1/F2 は必ず `jlimit` を通る:
- `:50701-50705` `clampedMixedF1 = jlimit(MIXED_F1_MIN_HZ, MIXED_F1_MAX_HZ, ...)` / `clampedMixedF2 = jlimit(max(MIXED_F2_MIN_HZ, clampedMixedF1 + 10.0f), MIXED_F2_MAX_HZ, ...)`
- `:50792-50796` と同型
- `:52496-52504`, `:52537-52538`, `:52903-52907` と同型

### 8.3 Contract

```text
NaN/Inf sampleRateBits  → invalid artifact（NaN/Inf の bit pattern を canonical identity に准入しない）
NaN/Inf freqStartBits   → invalid artifact
NaN/Inf freqEndBits     → invalid artifact
```

理由: domain validator が `isfinite(sampleRate) && 44100 <= sampleRate <= 768000` を要求するため、NaN/Inf を持つ identity は構造的に canonical になり得ない。NaN payload の許否を決める余地はない。

### 8.4 判定

**PROVEN**（domain validator の finite 要求と identity の受理条件が矛盾しない形で閉じる）。例外: `sampleRateBits` が NaN である **だけ** で invalid とするのは、NaN payload の bit pattern が複数存在しても「すべて invalid」という判定は bit-exact である点で決定的（= 同じ入力に対し常に同じ結果）。

---

## §9. lowercase canonical filename と recovery filename

### 9.1 Claim

「decode できる」と「canonical artifact である」は別概念。D-1R-2G はこの 2 つを混同していた。

### 9.2 Contract

```text
Decodeable  : filename が decode 可能（length 64, [0-9a-fA-F], 既定 extension）
Canonical   : Decodeable かつ actual filename == lowercase encode(decoded identity)
```

| ファイル名 | Decodeable | Canonical | load target | evict/clear/remove | 自動 rename |
| --- | --- | --- | --- | --- | --- |
| `<64hex lower>.mph` | yes | **yes** | 採用 | 対象 | しない |
| `<64HEX UPPER>.mph` | yes | **no** | **不採用** | **対象外** | **しない** |
| `<64hex lower>.MPH` | yes | **no**（extension case 不一致） | 不採用 | 対象外 | しない |
| `<63hex>.mph` | no | no | 不採用 | 対象外 | しない |
| `<65hex>.mph` | no | no | 不採用 | 対象外 | しない |
| `<64hex>.mph.bak` | yes（recovery namespace） | canonical ではない | recovery candidate | **対象外** | しない |
| `<64hex>.mph.stage` | yes（recovery namespace） | canonical ではない | recovery candidate | **対象外** | しない |

**核心**: uppercase filename を「canonical へ自動 rename」してはならない。D-1（裁定済）の「obsolete / future / malformed を自動削除しない」と同じ原則で、**noncanonical も自動変更しない**。

### 9.3 Evidence

JUCE `findChildFiles(files, findFiles, false, "*.mph")` は wildcard である。`WildCardMatcher` は full match を要求する（`juce_String.cpp:1135-1169`、D-1R-2E §12B で probe 実証済）。したがって:

- `<64hex>.mph` → MATCH
- `<64HEX>.mph` → **MATCH**（filesystem は case-insensitive。Windows 上では `.MPH` も `.mph` も同一ファイルとして列挙される）
- `<64hex>.mph.stage` → **no match**（`.mph` が拡張子全体ではない）
- `<64hex>.mph.bak` → **no match**

→ `clear()` / `evictLRU()` / `getEntryCount()` は `.stage`/`.bak` を巻き込まない（PROVEN、D-1R-2F §6 の実証を引き継ぐ）。

**ただし §10 の欠陥により、この分離は現行実装では不十分である。**

---

## §10. 【新規 defect】JUCE staging file は `*.mph` namespace 内に居る

### 10.1 Claim（Owner が明示要求した 5 論点のうち、本監査で新たに発見した最重要事項）

`clear()` は `*.mph` を無条件削除する（`:77486-77489`）。JUCE `TemporaryFile` の staging file は `<hash>_temp<randhex>.mph` で **cache directory 内の `.mph`** である。

### 10.2 Evidence

| 事実 | 出典 |
| --- | --- |
| staging file 名 = `getFileNameWithoutExtension() + "_temp" + toHexString(random)` + `getFileExtension()` | `juce_TemporaryFile.cpp:61-70` |
| staging は target の parent directory に作られる | 同 `:62` `target.getParentDirectory()` |
| target の extension は `mph`（`:77220`） | `:77220` |
| `clear()` は `*.mph` を `f.deleteFile()` で無条件削除 | `:77486-77489` |
| `evictLRU()` も `*.mph` を列挙 | `:77435` |
| `getEntryCount()` も `*.mph` を列挙 | `:77499` |

合成: staging file `<hash>_temp<randhex>.mph` は `*.mph` に **MATCH** する（D-1R-2F probe3 で 16/16 一致済）。

### 10.3 Reasoning：結果

| 事象 | 現行結果 |
| --- | --- |
| `save()` 実行中に別スレッド/別経路で `clear()` | **進行中 staging file を削除**。`:77323` の書込み途中なら subsequent write が sharing violation で失敗、または `overwriteTargetFileWithTemporary` が「temporaryFile not exists」で失敗 |
| `getEntryCount()` が staging 中 | 実 entry 数 + 1 を返す（**過大計上**） |
| `evictLRU()` が staging 中 | `:77447-77456` で magic 読取に失敗して `continue` するため、**直接削除はしない**（PROVEN: `:77445-77458`） |

### 10.4 §10 の結論

| 項目 | 判定 |
| --- | --- |
| 現行 v2 で staging file が `*.mph` に属する | **PROVEN** |
| 現行 v2 で `clear()` が進行中 staging を削除しうる | **PROVEN** |
| v3 設計（`<64hex>.mph.stage`）なら `*.mph` に属さない | **PROVEN**（full-match semantics により `.stage` は不一致） |
| **したがって v3 の `.mph.stage` 命名は、staging namespace 分離の副産物として destructive API を実際に保護する** | **PROVEN** |

> **訂正**: D-1R-2G §13 は「`*.mph` pattern 分離で `.stage`/`.bak`/`.bak.prev` は保護（PROVEN）」と判定した。この判定は **v3 設計の場合には正しいが、現行 v2 実装に対しては不完全**であった（JUCE `_temp*.mph` が `*.mph` に含まれる）。D-1R-2H でこの不十分さを訂正する。

---

## §11. `Canonical` と `Usable` の責務境界

### 11.1 Claim

Owner 指示 §9 の通り、各 API が何を要求するかを表にする。

### 11.2 Contract

| API | 受理に必要な state | 理由 |
| --- | --- | --- |
| `load(key)` | **Usable**（canonical + payload validity） | canonical だけだと payload NaN が DSP へ渡る（§12 で現行 `load` が payload 検証ゼロであることを確認） |
| startup `recovery` | **ValidRecoveryCandidate**（構造検証のみ） | 復元後にもう一度 **Usable** 検証を通す。recovery は「候補」であって「受理」ではない |
| `evictLRU()` の対象 | **CurrentV3Canonical** のみ | v2 / future / malformed / noncanonical / `.stage` / `.bak` は保護 |
| `remove()` の対象 | **CurrentV3Canonical** のみ | 同上 |
| `clear()` の対象 | **CurrentV3Canonical** のみ | 同上 |
| `getEntryCount()` の対象 | **CurrentV3Canonical** のみ | 契約上は。§14 で現行が CONTRADICTED であることを示す |

### 11.3 recovery 後の再検証（Owner 明示要求）

```text
recovery:
    candidate = classifyRecoveryCandidate(kind, filename, header, actualFileSize)
    if (!candidate.valid) return NotRecovered
    restore candidate → target
    # ↓↓↓ 復元後の再検証（必須） ↓↓↓
    usable = classifyUsable(target, actualFileSize)
    if (!usable.valid) { target を invalid 扱い；return NotRecovered }
    return Recovered
```

**recovery は 2 段構え**。構造検証だけでは `load` に渡してよいGARANTEE にならないため。

### 11.4 Evidence：現行 `load` の payload 検証はゼロ

`:77228-77296` の `load` が実施する検証は以下**のみ**:

| 行 | 検証 |
| --- | --- |
| `:77239` | `file.existsAsFile()` |
| `:77243` | stream open |
| `:77248` | header read size |
| `:77252` | `magic != kMagic \|\| version != kVersion` |
| `:77254-77260` | fileHash / sampleRate / phaseMode / f1 / f2 / targetLength の一致 |
| `:77265` | `numChannels <= 0 \|\| numSamples <= 0` |
| `:77274` | 波形 read size |
| `:77286-77289` | rho/theta read size |

**ないもの**:
- `headerReserved == 0` の検証（**なし**）
- `numSamples == targetLength` の検証（**なし**）
- `numChannels <= 8` の検証（**なし**）
- `numSamples <= MAX_IR_LATENCY` の検証（**なし**）
- `totalFileSize` の検証（**なし**）
- `actualFileSize` との一致検証（**なし**）
- `numAllpassSections` の上限検証（**なし**）
- IR / rho / theta の finite 検証（**なし**）
- `sampleRate` の domain 検証（`:77255` は caller 値との絶対差 `1.0e-6` 比較のみ）

**`:77268` `outIr.setSize(numChannels, numSamples)` は、76 バイトの header から制御された `int32` 値の上限検査なしに呼ばれる。** `numChannels`/`numSamples` は `<= 0` のみ除外されるため、`INT32_MAX` まで受理される。これは header 制御の無制限確保であり、**structural validation が必要であることの実証的根拠**。

また `:77255` の `std::abs(header.sampleRate - sampleRate) > 1.0e-6` は **tolerance 比較**である。§7 の v3 filename identity は **bit-exact** なので、両者を同時に採用すると矛盾する（§12.4 参照）。

### 11.5 判定

| 項目 | 判定 |
| --- | --- |
| `load` は Usable を要求すべき | **PROVEN**（現行 `load` が payload を検証せず DSP へ渡す。`:77228-77296`） |
| recovery は ValidRecoveryCandidate を要求し、復元後に Usable 検証を通すべき | **PROVEN**（§11.3 の 2 段構えが、構造検証と payload 検証を分離した上限を提供） |
| 現行 `load` が structural validation を行っている | **CONTRADICTED**（`:77265` の `<= 0` のみ。上限検査なし） |

---

## §12. payload validation の詳細

### 12.1 `rho` range contract（Owner §10）

### 12.1.1 Evidence：producer の 3 系統の代入経路

authority 内で `rho` への代入は**閉じた集合**である（`\.rho\s*=` / `rho\s*=\s*std::clamp` の全ヒット）:

| # | 行 | 代入 | 結果域 |
| --- | --- | --- | --- |
| P1 | `:4004` | `double rho = 0.0;`（struct 既定） | `{0.0}` |
| P2 | `:3543` | `const double rho = std::clamp(std::abs(gain), 0.0, 0.995);` | `[0, 0.995]` |
| P3 | `:3769` | `section.rho = unconstrainedToRho(bestParams[2*s]);` | `(0, 0.98)` |
| P4 | `:3813` | `section.rho = std::clamp(std::abs(best_gain), 0.0, 0.995);` | `[0, 0.995]` |

`unconstrainedToRho`（`:3552-3556`）:
```cpp
inline double unconstrainedToRho(double x) {
    // 単調増加写像 R → (0, 0.98)。x=0 で ρ=0.49。
    return 0.98 * stableSigmoid01(x);
}
```

producer 自身の安定性述語（`:4004`, `:4008`）:
```cpp
double rho = 0.0;      // 極半径 (0 ≤ ρ < 1)
bool isStable() const noexcept { return (rho >= 0.0 && rho < 1.0); }
```

### 12.1.2 Reasoning：`std::clamp` は NaN を除去しない

```cpp
std::clamp(v, lo, hi)  ≡  v < lo ? lo : (hi < v ? hi : v)
```
- `v = NaN`: `NaN < lo` は false、`hi < NaN` は false → **NaN がそのまま返る**
- `stableSigmoid01(NaN)` も NaN を返す（`x/(1+|x|)` 型写像では NaN 伝播）

したがって P2/P4 は `rho ∈ [0, 0.995]` を保証するのは **入力が finite のときのみ**。

### 12.1.3 Contract（Owner が求める separation）

```text
[proved invariant]  rho は producer の全代入経路で [0, 0.995] ⊂ [0, 1) に収まる
                    → ただし unconstrainedToRho の入力が NaN の場合を除く

[cache contract]    1. rho は finite            ← 無条件。唯一の load-bearing 要件
                    2. rho ∈ [0, 1)              ← SUPPORTED（入力 finite が前提）。
                                                    cache contract には採用しない
```

**Owner 指示「producer が clamp しているからといって cache contract として自動確定してはいけない」に精确に従う**。`std::clamp` は NaN を除去しないため、`finite` 以外の range 要件は cache contract の根拠にならない。

### 12.1.4 判定

| 項目 | 判定 |
| --- | --- |
| `rho` が finite であること | **PROVEN**（無条件に要求すべき。`std::clamp` の NaN 透過性から） |
| `rho ∈ [0, 1)` が producer で保証される | **SUPPORTED**（4 経路の閉じた集合から導出。ただし unconstrained 入力が finite であることが前提） |
| `rho ∈ [0, 1)` を cache contract として採用 | **NOT PROVEN**（Owner が求める separation。producer 実装詳細への依存になる） |

### 12.2 `theta` range contract

### 12.2.1 Evidence

| # | 行 | 代入 | 結果域 |
| --- | --- | --- | --- |
| T1 | `:4005` | `double theta = 0.0;`（struct 既定） | `{0.0}` |
| T2 | `:3544` | `const double theta = 2.0 * pi * f0 / sampleRate;` | `f0`/`sampleRate` に依存。**無界** |
| T3 | `:3770` | `section.theta = unconstrainedToTheta(bestParams[2*s+1]);` | `(0, 0.99π)` |
| T4 | `:3814` | `section.theta = 2.0 * pi * best_f0 / sampleRate;` | **無界** |

`unconstrainedToTheta`（`:3559`, `:3561-3566`）:
```cpp
constexpr double kThetaMax = 0.99 * juce::MathConstants<double>::pi;
inline double unconstrainedToTheta(double x) { return kThetaMax * stableSigmoid01(x); }
```

### 12.2.2 判定

```text
[cache contract]  theta は finite のみ。range 要求なし。
```
T2/T4 は `f0 / sampleRate` が無界なので range contract を構成できない。**PROVEN**（`finite` のみが成立し得る要件である）。

### 12.3 `rho.size() == theta.size()`（Owner §7）

producer では同一 `for` ループで `push_back` される（`:50434-50439`）:
```cpp
std::vector<double> rho, theta;
for (const auto& sec : lastAllpassSections)
{
    rho.push_back(sec.rho);
    theta.push_back(sec.theta);
}
```

producer sections 数: `allpass_sections.size() == config.numSections`、`:50222` で `numSections` は `{2, 8, 20}` に閉じる。

- `rho.size() == theta.size()`: **PROVEN**（同一ループ push_back）
- `numSections ∈ {2, 8, 20}`: **PROVEN**
- `rho.size() <= 20`: **PROVEN**（producer 経路）

ただし public `save()` は **任意 caller に公開**されているため、save boundary で再検証する:
```text
1 <= channels <= 8
1 <= samples <= MAX_IR_LATENCY
1 <= targetLength <= MAX_IR_LATENCY
samples == targetLength
rho.size() == theta.size()
rho.size() <= 20
theta.size() <= 20
```

**producer proof と API boundary validation を混同しない**（Owner 明示指示）。前者は §12.3 の PROVEN、後者は新規 contract。

### 12.4 IR finite validation の全要素網羅（Owner §11 — D-1R-2G の NOT PROVEN を閉鎖）

### 12.4.1 Claim

`mixedIR` の全要素に対する finite check が producer で**完全網羅**されているか。

### 12.4.2 Evidence

`convertToMixedPhaseAllpass` 内の finite sweep（`:50346-50357`）:
```cpp
for (int ch = 0; ch < numChannels; ++ch)
{
    const double* p = mixedIR.getReadPointer(ch);
    for (int i = 0; i < numSamples; ++i)
    {
        if (!std::isfinite(p[i]))
        {
            juce::Logger::writeToLog("convertToMixedPhaseAllpass: Safety guard triggered (NaN/Inf detected), returning empty.");
            return {};
        }
    }
}
```

**位置の検証**:

| 段階 | 行 | sweep より前か後か |
| --- | --- | --- |
| mixedIR 生成（inverse FFT 出力） | `:50316-50321` | **前** |
| RMS 規格化 `dst[i] *= gain` | `:50341-50342` | **前** |
| **finite sweep** | `:50346-50357` | — |
| peak 計算 | `:50359-50365` | 後 |
| `lastAllpassSections` キャプチャ | `:50311` | 前 |
| `MixedPhasePersistentCache::save(...)` | `:50440-50443` | **後** |

**網羅範囲**: `ch` は `0 .. numChannels-1`、`i` は `0 .. numSamples-1`。`:50328-50329` で `mixedIR` の読出しポインタも `ch` 全域を走査しており、`numSamples` も `:49945` の構築値 `numSamples` と同一。→ **全 channel × 全 sample を網羅**。

**制御-flow の検証**: `:50346` から `:50440` の間に `goto` / `longjmp` / `setjmp` は存在しない。authority 内でこれらは build.bat（`:427-722`、スクリプト）および `:108483` にのみ現れ、いずれも `convertToMixedPhaseAllpass`（`:49840`-`:50464`）の外。したがって **`:50440` に到達する全ての実行は `:50346` の sweep を通過する**。

### 12.4.3 判定

| 項目 | 判定 |
| --- | --- |
| producer 経路で `save()` に到達する `mixedIR` は全要素 finite | **PROVEN**（`:50346-50357` の二重ループ + 位置の前後関係 + `goto`/非局所ジャンプ不在） |
| `save()` boundary での再検証（public API 防御） | **PROVEN**（新規 contract。producer proof に依存しない） |
| **`load()` 経路で DSP に渡る IR が finite であること** | **NOT PROVEN**（`:77228-77296` に payload finite 検証が**ゼロ**。破損ファイルは NaN を DSP に渡せる） |

> **D-1R-2G §6.2 の「`IR samples are finite` の完全網羅性は NOT PROVEN」は、producer 側について PROVEN に昇格する**。残る NOT PROVEN は **consumer（load）側**にのみ帰属する。

### 12.5 caller↔header の一致 contract（bit-exact vs tolerance）

現行 `load`（`:77254-77260`）は **tolerance 比較**を使う:
```cpp
if (header.fileHash != fileHash
    || std::abs(header.sampleRate - sampleRate) > 1.0e-6
    || header.phaseMode != static_cast<int32_t>(phaseMode)
    || std::abs(header.freqStartHz - freqStartHz) > 1.0e-6f
    || std::abs(header.freqEndHz - freqEndHz) > 1.0e-6f
    || header.targetLength != static_cast<int32_t>(targetLength))
    return false;
```

v3 filename identity は **bit-exact**（`bit_cast` ベース）である。

```text
save 時:  encode(callerKey) == filename     (bit-exact)
load 時:  getCacheFile(callerKey) を使う   → filename は callerKey から bit-exact に導出
          従って header == callerKey は必然に bit-exact でなければ成立しない
```

→ **tolerance 比較は v3 では構造的に無意味になる**（ファイル名的には callerKey から一意に決まるため、header が callerKey と 1 ulp 異なる場合は file hit 自体が起きない）。v3 は bit-exact に統一すべき。

**判定**: tolerance 比較の採否 = **PROVEN**（bit-exact filename identity を採用するなら bit-exact header 比較が一貫する。tolerance を残すと「一致なのに hit しない」という不可解な結果と「hit しないのに別 artifact を読む」という経路が同時に生じる）。残る論点は「caller が毎回 bit-exact に同じ double を渡せるか」であり、これは producer chain（`:50413` `key.sampleRate = sampleRate;`）が save と load で同一の値を使うため **PROVEN** で成立する。
---

## §13. maxCacheEntries の full data flow 監査（Owner §12）

### 13.1 Claim

Owner の契約候補:
```text
kMinPersistentCacheEntries     = 1;
kDefaultPersistentCacheEntries = 8;
kMaxPersistentCacheEntries     = 64;

nonnumeric  → 8      bool  → 8      NaN/Inf → 8
fractional → 8      < 1   → 8
1..64 unchanged      > 64  → 64
```

### 13.2 Evidence：全 data flow（9 段）

| # | 段 | 行 | 変換 |
| --- | --- | --- | --- |
| F1 | UI / stream 入力 | `:13608-13610` | `int maxEntries` → `static_cast<size_t>(maxEntries)` → `setMaxCacheEntries` |
| F2 | ValueTree 復元 | `:53108` | `v.getProperty(...)` (`var`) → `static_cast<int>` → `static_cast<size_t>` → `setMaxCacheEntries` |
| F3 | setter | `:48988` | `juce::jlimit<size_t>(1, 64, maxEntries)` |
| F4 | setter 格納 | `:48991` | `pendingOverride.maxCacheEntries = static_cast<int>(clamped)` |
| F5 | snapshot 復元 | `:52927` | `pendingOverride.maxCacheEntries = juce::jlimit(0, 64, snapshot.maxCacheEntries)` |
| F6 | 既定値 | `:55888`, `:56882` | `int maxCacheEntries = 0;` |
| F7 | getter | `:49000` | `static_cast<size_t>(pendingOverride.maxCacheEntries)` |
| F8 | consumer (disk) | `:50446` | `MixedPhasePersistentCache::evictLRU(owner->getMaxCacheEntries())` |
| F9 | consumer (disk, 別 manager) | `:84826` | `cacheManager.evictLRU(processor.getMaxCacheEntries())` |
| F10 | consumer (memory) | `:49062` | `const size_t cacheLimit = getMaxCacheEntries();` |

関連コード:
```cpp
// :48986-48995
void ConvolverProcessor::setMaxCacheEntries(size_t maxEntries)
{
    const size_t clamped = juce::jlimit<size_t>(1, 64, maxEntries);
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        pendingOverride.maxCacheEntries = static_cast<int>(clamped);
    }
    if (cacheManager)
        cacheManager->evictLRU(clamped);
}

// :48997-49001
[[nodiscard]] size_t ConvolverProcessor::getMaxCacheEntries() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return static_cast<size_t>(pendingOverride.maxCacheEntries);
}

// :52927
pendingOverride.maxCacheEntries = juce::jlimit(0, 64, snapshot.maxCacheEntries);

// :53108
if (v.hasProperty ("maxCacheEntries")) setMaxCacheEntries (static_cast<size_t>(static_cast<int>(v.getProperty("maxCacheEntries"))));

// :55888 / :56882
int maxCacheEntries = 0;
```

### 13.3 Reasoning：Owner 契約との 6 箇所の不一致

| 入力 | Owner 契約 | 現行の経路 | 判定 |
| --- | --- | --- | --- |
| `"abc"`（non-numeric） | 8 | F2: `static_cast<int>(var)` → `0` → `size_t 0` → F3 `jlimit(1,64,0)` → **1** | **CONTRADICTED** |
| `var(true)`（bool） | 8 | F2: `static_cast<int>(var)` → `1` → F3 → **1** | **CONTRADICTED** |
| `NaN` | 8 | F2: `static_cast<int>(NaN)` → **INT_MIN**（D-1R-2A 実測） → `size_t` 巨大 → F3 → **64** | **CONTRADICTED** |
| `+Inf` | 8 | 同上 → **64** | **CONTRADICTED** |
| `1.7`（fractional） | 8 | F2: `static_cast<int>(1.7)` → **1** → F3 → **1** | **CONTRADICTED** |
| `-1`（negative, F1 経路） | 8 | F1: `size_t(-1)` = `SIZE_MAX` → F3 → **64** | **CONTRADICTED** |
| `0` | 8 | F6 既定値 → F7 `size_t(0)` = `0` → F8 → `evictLRU(0)` → `clear()` | **CONTRADICTED** |
| `8`（正常） | 8 | F2 `static_cast<int>(8)` → F3 `jlimit(1,64,8)` → **8** | 一致 |
| `64` | 64 | → **64** | 一致 |
| `100` | 64 | → **64** | 一致 |

### 13.4 【最重大 defect】`0` が到達可能であり、全 cache を削除する

`0` への到達は 2 経路で PROVEN:

1. **既定値**（`:55888`, `:56882`）: `int maxCacheEntries = 0;` — snapshot / override 両構造体の既定が 0
2. **snapshot 復元**（`:52927`）: `juce::jlimit(0, 64, snapshot.maxCacheEntries)` — **下限が 0**。setter（`:48988`）の下限 1 と**不一致**

そして `getMaxCacheEntries()`（`:49000`）が `0` を返し、`evictLRU(0)`（`:77427-77431`）が `clear()` を呼ぶ:
```cpp
void MixedPhasePersistentCache::evictLRU(size_t maxCount)
{
    if (maxCount == 0)
    {
        clear();
        return;
    }
```
`clear()`（`:77480-77490`）は `*.mph` を**無条件削除**する:
```cpp
void MixedPhasePersistentCache::clear()
{
    auto dir = getCacheDirectory();
    if (!dir.exists()) return;
    juce::Array<juce::File> files;
    dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");
    for (auto& f : files)
        f.deleteFile();
}
```

そして `evictLRU` は **save 成功直後**に呼ばれる（`:50446`）。したがって:

```text
maxCacheEntries == 0 の状態で
    save() 成功
        → evictLRU(0)
            → clear()
                → 永続 cache 全削除（新品 Miss 済みを含む）
```

**PROVEN**。これが Owner が「`evictLRU(0)` を no-op にする」ことを第一候補とした実質的根拠である。no-op 化すれば、0 は「上限なし」ではなく「何もしない」になり、偶発的な cache 全消滅stellen が起きない。

### 13.5 未閉鎖の隣接面（residual）

`:49062` `const size_t cacheLimit = getMaxCacheEntries();` が**メモリ cache** の eviction limit にも使われる。`cacheLimit == 0` のときメモリ cache 側の挙動は本監査の範囲外（`evictOldestCacheEntry` の実装未読）。→ **NOT PROVEN**（Owner の指示範囲「maxCacheEntries full data flow」を disk cache viii について PROVEN、memory cache について NOT PROVEN として分離する）。

### 13.6 判定

| 項目 | 判定 |
| --- | --- |
| Owner の normalization 契約が現行と一致するか | **CONTRADICTED**（6 入力で不一致） |
| `static_cast<int>` が validation より先.appsara いるか | **CONTRADICTED**（`:53108`, `:13610` の 2 経路） |
| `0` が `getMaxCacheEntries()` から到達可能か | **PROVEN**（`:55888`, `:56882` の既定値 + `:52927` の下限 0） |
| `0` が全永続 cache 削除を招くか | **PROVEN**（`:77427-77431` → `:77486-77489`、`:50446` から到達） |
| disk cache 側 data flow が全 10 段で追跡できたか | **PROVEN**（F1-F10） |
| memory cache 側 data flow が追跡できたか | **NOT PROVEN**（`:49062` の下流未読） |

---

## §14. `evictLRU(0)` の contract

### 14.1 Claim

Owner 第一候補: `0 → no-op`。

### 14.2 Evidence

`:77427-77431`:
```cpp
if (maxCount == 0)
{
    clear();
    return;
}
```

### 14.3 Reasoning：`invalid` と `no-op` の意味差

| 意味 | 呼び出し側の観測 | 副作用 |
| --- | --- | --- |
| `invalid`（入力拒否） | 戻り値なしで void なので**観測不能** | なし（ただし呼び出し側は「拒否された」ことを知らない） |
| `no-op` | 観測不能 | なし |

`evictLRU` は `void`（`:77425`）であり、戻り値による観測手段がない。よって両者の**外から観測可能な差はゼロ**。したがって:

- 契約を `no-op` とする限り、**実装上は `return;` が唯一の選択肢**
- `invalid` を採用しても実装差はない（が、診断ログの有無だけが差になる）

Owner の理由「caller 側で accidental zero が発生しても、cache 全削除という destructive behavior に変換されない」= **PROVEN**（§13.4 の連鎖）。

### 14.4 判定

**CONTRADICTED**（現行は `clear()`）。修正候補 `if (maxCount == 0) return;` は §13.4 の defect を直接解消する。

---

## §15. `save()` の false ケース分類と save → eviction の data flow

### 15.1 Claim

現行 `:50440-50446` は `save()` の戻り値を無視して `evictLRU` を呼ぶ。

### 15.2 Evidence

`:50440-50446`:
```cpp
convo::MixedPhasePersistentCache::save(
    key.fileHash, key.sampleRate, static_cast<int>(key.phaseMode),
    key.f1, key.f2, key.targetLength,
    mixedIR, rho, theta);          // ← 戻り値無視

// ディスクキャッシュのLRUエビクション
convo::MixedPhasePersistentCache::evictLRU(owner->getMaxCacheEntries());
```

### 15.3 `save()` が false を返す全経路（`:77302-77377`）

| # | 行 | 条件 | 既存 valid target を破壊するか |
| --- | --- | --- | --- |
| E1 | `:77317-77318` | cache directory 作成失敗（`wasOk()` false） | no（log のみ、処理続行） |
| E2 | `:77324-77325` | staging の `FileOutputStream` open 失敗 | **no** |
| E3 | `:77347-77348` | header write 失敗 | **no** |
| E4 | `:77354-77355` | 波形 write 失敗 | **no** |
| E5 | `:77361-77362` | rho write 失敗 | **no** |
| E6 | `:77363-77364` | theta write 失敗 | **no** |
| E7 | `:77370-77371` | header 書き戻し失敗 | **no** |
| E8 | `:77376` | `overwriteTargetFileWithTemporary()` 失敗（5 回 retry 後） | **要検討**。§17 で解析 |

**現行には input validation がない**（§11.4）。したがって E1-E8 のみが false 経路であり、`invalid input` は現行では **false を返さない**（無検証で書き込みを試みる）。

### 15.4 v3 で追加される false 経路（Owner §14 の分類）

```text
invalid input          : domain 違反（channels/samples/targetLength/sections/finite/SR）
existing future artifact : identity 一致する future version artifact がある
existing obsolete artifact: identity 一致する v2 artifact がある
existing noncanonical artifact: identity 一致するが filename が canonical でない
I/O failure             : E2-E7
stage creation failure  : E2
flush failure           : stream.flush() の戻り値検査（現行 `:77373` は無視）
backup rename failure   : target → bak が失敗（§3.4）
stage→target failure    : stage → target が失敗（1176 / 1177 含む、§17）
cleanup failure         : .bak 削除が失敗（§16）
```

### 15.5 Contract

```cpp
const bool saved = save(...);
if (saved)
    evictLRU(normalizedMaxCacheEntries);
```

```text
save failure → eviction 不実行（既存 cache を追加破壊しない）
save success → eviction 実行
```

### 15.6 判定

| 項目 | 判定 |
| --- | --- |
| 現行が save 戻り値を無視するか | **CONTRADICTED**（`:50440-50446`） |
| `if (saved) evictLRU(...)` が §13.4 の defect を加剧するか | **PROVEN**（save 失敗時に `evictLRU(0)` が走らないため，反而として `0` の破壊連鎖は save 成功時のみに限定される。ただし `maxCacheEntries == 0` で save 成功すれば依然として全削除は起きる） |
| **したがって `evictLRU(0) = no-op` は save→eviction 修正とは独立の必須修正である** | **PROVEN** |

---

## §16. `save()` の成功条件の定義（Owner §15）

### 16.1 Claim

`save = true` とする条件を 8 段階で明示し、`bak cleanup` 失敗時に `save = true` / `cleanup diagnostic = pending` とする案を比較する。

### 16.2 Contract：8 段階

```text
S1. input validation PASS
S2. stage creation PASS
S3. stage complete write PASS          (header + payload + header 書き戻し)
S4. stage flush PASS                   (FlushFileBuffers 相当、戻り値検査)
S5. existing target safely transitioned (target → bak が成功、または target が存在しなかった)
S6. stage → target PASS
S7. target re-open + structural validation PASS
S8. target usable validation PASS
    ⇒ save = true
```

### 16.3 `cleanup failure` の扱い：2 案の比較

| 案 | 定義 | 利点 | 不利 |
| --- | --- | --- | --- |
| **案 1（Owner 第一候補）** | S8 成立で `save = true`。その後の `.bak` 削除失敗は別 diagnostic | 「save succeeded but return false」の意味的不整合を回避 | `save` の戻り値だけでは「`.bak` が残存」であることがCommunicate されない。residues 次回 save の起点 |
| 案 2 | `.bak` 削除失敗で `save = false` | residues の存在が戻り値で伝わる | target は裏で成功して新しい content で有効なのに `false`。caller が再計算 → **target を無駄に捨てることになる**。さらに `save → eviction`（§15.5）が `false` で止まる |

### 16.4 判定

**案 1 を採用**。根拠:

- 案 2 では「target が新 IR で有効」なのに `false` を返すと、Owner の §15 の論点どおり「save succeeded but return false」という不整合が生じる（案 2 では return は false だが実体は success）
- 案 2 はさらに「save failure で既存 cache を追加破壊しない」を **save 自身の真** にするが、案 2 では既存 target を **既に置き換えている** ため、この contractual 保護が失われる
- 案 1 は `save` の戻り値を「**target が新しい有効な cache artifact として成立したか**」という単一の意味に固定し、residues 管理を別 status 経路に分離する

したがって:

```text
save の戻り値 = S1..S8 の連鎖結果
residues 状態 = 別 diagnostic（enum: None / PendingCleanup / PendingRecovery）
```

| 項目 | 判定 |
| --- | --- |
| 8 段階成功条件の妥当性 | **PROVEN**（各段階は現行 `save()`（`:77302-77377`）および `ReplaceFileW` の失敗モード（1175/1176/1177、他 error code）に対応づく） |
| S7/S8（target の re-validate）が現行に存在するか | **CONTRADICTED**（`:77376` は commit 後に一切検証しない） |
| 案 1 vs 案 2 の選択 | **案 1 が PROVEN**（案 2 は §15.5 の「save failure で既存 cache を追加破壊しない」と構造的に矛盾） |
| cleanup failure の diagnostics 機構 | **NOT PROVEN**（新規 contract。mechanism 未設計） |

---

## §17. 1177 の位置づけ（Owner §17）

### 17.1 Claim

`1177 recovery` と `normal save transaction` を同じ state machine として扱わない。

### 17.2 external evidence（MS Learn `ReplaceFileW`）

| code | 説明（原文） |
| --- | --- |
| 1175 `ERROR_UNABLE_TO_REMOVE_REPLACED` | 「The replaced file could not be deleted. The replaced and replacement files retain their original file names.」 |
| 1176 `ERROR_UNABLE_TO_MOVE_REPLACEMENT` | 「The replacement file could not be renamed. If *lpBackupFileName* was specified, the replaced and replacement files retain their original file names. Otherwise, **the replaced file no longer exists** and the replacement file exists under its original name.」 |
| 1177 `ERROR_UNABLE_TO_MOVE_REPLACEMENT_2` | 「The replacement file could not be moved. **The replacement file still exists under its original name; however, it has inherited the file streams and attributes from the file it is replacing.** The file to be replaced still exists with a different name. If *lpBackupFileName* is specified, it will be the name of the replaced file.」 |
| その他 | 「If any other error is returned, such as **ERROR_INVALID_PARAMETER**, the replaced and replacement files will retain their original file names. In this scenario, **a backup file does not exist** and it is not guaranteed that the replacement file will have inherited all of the attributes and streams of the replaced file.」 |

`lpBackupFileName`: 「If this parameter is **NULL**, no backup file is created.」

### 17.3 Contract：分離した 2 つの state machine

```text
[SM-A] Normal save transaction（1177 を使わない）
    stage write → stage→target（MoveFileExW 系）
    失敗モード: stage→target rename 失敗のみ。1175/1176/1177 は発生しない

[SM-B] ReplaceFileW error recovery（1177 専用）
    ReplaceFileW(target, stage, lpBackupFileName, ...) が
    ERROR_UNABLE_TO_MOVE_REPLACEMENT_2 (1177) を返した場合にのみ起動
    recovery 対象: target(=replacement, attributes 継承) と bak(=replaced, 別名)
```

**1177 は `ReplaceFileW` を使用した場合にのみ発生する recovery-specific state であり、normal save transaction の基本 API ではない。**

### 17.4 D-1R-2G §3.3 の「stage を rename せず新規作成で copy」方針の再検証

D-1R-2G は「1177 では replacement が attributes/streams を継承するため、stage を rename して canonical target にしてはいけない。`CreateFileW(CREATE_NEW)` で新規作成し bytes を write する」とした。

再評価:

| 観点 | 評価 |
| --- | --- |
| 1177 の replacement は attributes/streams を継承する | **PROVEN**（MS Learn 原文） |
| したがって `stage`（=replacement）を rename して canonical にするのは誤り | **PROVEN** |
| ただしその場合 canonical identity の **bytes** は依然 stage の contents である | **PROVEN**（attributes ではなく content） |
| attributes/streams が **canonicality** の判定項目か | **NOT PROVEN**。§11.2 の `ArtifactState` は `magic/version/headerReserved/identity/filename/size` のみで attributes を含まない |
| 「attributes が canonical と無関係」という結論 | **NOT PROVEN**。attributes は DSP 出力には影響しないが、OS 上の artifact 同一性の実務的意味はOwner decision を要する |

**Owner 第一候補の採用**: 1177 recovery は **SM-B に閉じて** SM-A と混ぜない。D-1R-2G §3.3 の「新規作成 + copy」方式は、**canonicality 判定項目に attributes を含めない前提のもとで**採用可能。attributes を canonicality に含める場合は NOT PROVEN として Owner decision に委ねる。

### 17.5 判定

| 項目 | 判定 |
| --- | --- |
| 1177 が attributes/streams を継承すること | **PROVEN**（MS Learn 原文） |
| 1176 で `lpBackupFileName == nullptr` のとき replaced file が消滅すること | **PROVEN**（MS Learn 原文） |
| 1177 が SM-A（normal save）で発生しうること | **CONTRADICTED**（SM-A は `MoveFileExW` 系のみを使うため） |
| 1177 が SM-B 専用である | **PROVEN**（`ReplaceFileW` 使用が前提条件） |
| 1177 recovery policy（PreserveOld / PreserveNew） | **NOT PROVEN**（Owner decision。§3 の T1/T3 状態と §4 の recovery 段で選択） |

---

## §18. `MoveFileExW` の atomicity 表現（Owner §18）

### 18.1 Claim

`MoveFileExW` を「atomic transaction」と表現しない。

### 18.2 external evidence（MS Learn `MoveFileExW`）

`MOVEFILE_WRITE_THROUGH`（8 / 0x8）:
> The function does not return until the file is actually moved on the disk. Setting this value guarantees that a move performed as a copy and delete is flushed to disk before the function returns.

- フラグ値は **8 (0x8)**（`REPLACEFILE_WRITE_THROUGH` とは異なる）
- 保証されるのは「copy-and-delete として実行された move の flush」のみ
- **power-loss durability 全体の証明ではない**

### 18.3 Contract

```text
表現してよいこと:
  「MoveFileExW は同一 volume 内で directory entry の atomic な置換を提供し、
    失敗時は source と destination のいずれかが保持される」

表現してはならないこと:
  「MoveFileExW は atomic transaction である」
  「MoveFileExW は power-loss durability を提供する」

本 audit の対象範囲:
  in  : API-level failure safety / crash recovery / corrupt-cache rejection
  out : strict power-loss atomicity
```

### 18.4 判定

**PROVEN**（MS Learn の明記により、`MOVEFILE_WRITE_THROUGH` は copy-and-delete の flush のみを対象とする。power-loss durability には言及しない）。

---

## §19. power-loss scope と corrupt-artifact rejection の整合（Owner §16）

### 19.1 Claim

「power-loss durability = out of contract」と「corrupt cache must never be accepted」は矛盾しない。

### 19.2 Contract

```text
power loss
    ↓
cache may disappear  /  residue may remain  /  artifact may be truncated
    ↓                        （いずれも out of contract の許容結果）
startup recovery
    ↓
structural validation  +  usable validation      ← in contract
    ↓
invalid artifact は決して受理されない              ← in contract
```

**形式化**: `power-loss` を入力とする変換 `f` について、
```text
∀ f ∈ PowerLoss :  ∄ a ∈ Artifacts  s.t.  f(a) = a ∧ a ∉ Valid
```
すなわち **power-loss が invalid artifact を valid artifact に変換しうるなら、その責務は cache 側にない**。cache 側の責務範囲は「accept しないこと」のみ。

### 19.3 Evidence

- power-loss 後に target が truncated / partial になり得ることは、MS の file system 仕様上排除できない（§18）
- 一方 `classifyCanonicalArtifact` は `magic / version / headerReserved / identity / filename / actualFileSize == expectedSize / header.totalFileSize == expectedSize` を要求するため、partial file は必ず `Malformed` になる（`actualFileSize != expectedSize`）→ **reject**

### 19.4 判定

| 項目 | 判定 |
| --- | --- |
| 両条項が矛盾しない | **PROVEN**（形式化 `f` の受理集合は空） |
| corrupt artifact が canonical に accept されない | **PROVEN**（`actualFileSize == expectedSize` 条件が機能する。§11.2） |
| 現行 `load` が corrupt artifact を reject するか | **CONTRADICTED**（`:77228-77296` に `actualFileSize` 検証が**ゼロ**） |

---

## §20. destructive API の ownership（現行の完全像）

### 20.1 Evidence：4 API の現行実装

| API | 行 | 列挙 pattern | 分類検査 | 削除範囲 |
| --- | --- | --- | --- | --- |
| `remove(key)` | `:77468-77478` | なし（`getCacheFile` のパス直接） | **なし** | そのパスのファイルが `existsAsFile()` なら無条件削除 |
| `clear()` | `:77480-77490` | `"*.mph"` | **なし** | 全 `*.mph` を無条件削除（v2/future/malformed/JUCE `_temp*.mph` を含む） |
| `evictLRU(n)` | `:77425-77466` | `"*.mph"` | `magic` のみ（`:77454`） | `magic` 一致の文件的先頭 `n` 件（version/identity は不問） |
| `getEntryCount()` | `:77492-77501` | `"*.mph"` | **なし** | 削除しないが、`*.mph` 総数を返す |

### 20.2 判定

| 項目 | 判定 |
| --- | --- |
| `clear()` が v2 artifact を削除するか | **PROVEN**（`:77486-77489` に version 検査なし） |
| `clear()` が future artifact を削除するか | **PROVEN**（同上） |
| `clear()` が進行中 JUCE staging（`_temp*.mph`）を削除するか | **PROVEN**（§10） |
| `evictLRU` が future version を削除するか | **PROVEN**（`:77454` は magic のみ。version 不問） |
| `remove()` が他 identity の artifact を削除しうるか | **PROVEN**（hash collision 時に可能。`computeKeyHash` は 64-bit hash であり衝突排除機構がない） |
| `getEntryCount()` が canonical 以外の `*.mph` を数えるか | **PROVEN**（`:77499` に分類なし） |
| **したがって v3 contract「destructive API = current v3 canonical のみ」は、現行と全面的に CONTRADICTED** | **PROVEN** |

---

## §21. 最終判定表

| 項目 | 判定 |
| --- | --- |
| target→bak transaction | **NOT PROVEN**（§2/§3。PROVEN なのは「truncate しない」「target 不在窓が不可避」こと。transaction 全体の安全性は production 実証未実施） |
| stage→target transaction | **NOT PROVEN**（§2。1177 分離（§17）により SM-A/SM-B の 2 系統が必要。SM-A の production 実証未実施） |
| existing bak handling | **NOT PROVEN**（§3/§4。`MoveFileExW` の `MOVEFILE_REPLACE_EXISTING` 必須性（MS Learn）が確定。安全な選択は一意だが production 実証未実施） |
| crash after target→bak | **SUPPORTED**（§2.4。T1 として定義済み。recovery は §4 で定義） |
| crash after stage→target | **SUPPORTED**（§2.4。T2。target は有効。cleanup residue のみ） |
| stale stage handling | **NOT PROVEN**（§4。classification は定義済み。production 実証未実施） |
| stale bak handling | **NOT PROVEN**（§4。同上） |
| identity mismatch isolation | **PROVEN**（§6。64-hex fixed-width なら transaction identity が filename から一意に復元可能。現行 hash 方式では不可 = 现行は CONTRADICTED） |
| artifact state classification | **PROVEN**（§5。8 値分類が必要最小限の判別力を持ち、boolean では不可能な実例が示された） |
| 64-hex serialization | **PROVEN**（§7。Option A 採用。fixed-width lowercase zero-padded hex。round-trip deterministic） |
| endian semantics | **PROVEN**（§7。Option A では endian は仕様に含まれない。D-1R-2G の「little-endian」記述は訂正） |
| uppercase filename handling | **PROVEN**（§9。Decodeable ≠ Canonical。uppercase は noncanonical。load 不採用 / destructive 対象外 / 自動 rename しない） |
| filename/header identity | **PROVEN**（§7/§9/§12.5。bit-exact。tolerance 比較（`:77255`）は v3 では構造的に無意味） |
| Canonical vs Usable boundary | **PROVEN**（§11。現行 `load` に payload 検証ゼロ（`:77228-77296`）であることが境界必須性の実証） |
| load acceptance | **PROVEN**（§11.2。Usable を要求すべき。現行は structural のみで header 制御の無制限確保（`:77268`）あり = CONTRADICTED） |
| recovery acceptance | **PROVEN**（§11.3。ValidRecoveryCandidate を要求し、復元後に Usable 検証を通す 2 段構え） |
| payload finite validation | **PROVEN**（producer 側：`:50346-50357` の二重ループ + 位置 + 非局所ジャンプ不在。consumer 側は CONTRADICTED） |
| rho range contract | **NOT PROVEN**（§12.1。`finite` のみ PROVEN。`[0,1)` は SUPPORTED（`std::clamp` の NaN 透過性により無条件化できない）） |
| maxCacheEntries full data flow | **CONTRADICTED**（§13。10 段 F1-F10 を追跡。6 入力で Owner 契約と不一致。`0` 到達可能 → 全 cache 削除） |
| evictLRU(0) | **CONTRADICTED**（§14。`:77427-77431` が `clear()`。no-op へ） |
| save success definition | **PROVEN**（§16。8 段階。案 1 採用。現行は S7/S8 なし = CONTRADICTED） |
| post-success bak cleanup | **NOT PROVEN**（§16。案 1 の diagnostics 機構が未設計。mechanism 自体が新規） |
| save→eviction | **CONTRADICTED**（§15。`:50440-50446` が戻り値無視。`if (saved) evictLRU(...)` へ） |
| power-loss scope | **PROVEN**（§18/§19。out of contract。corrupt artifact は reject） |

### 21.1 集計

| 判定 | 件数 |
| --- | --- |
| PROVEN | 15 |
| SUPPORTED | 2 |
| NOT PROVEN | 8 |
| CONTRADICTED | 5 |
| 合計 | 30 |

---

## §22. Owner 候補への合意状況

| Owner 候補 | 本監査の判定 |
| --- | --- |
| `PersistentCache version = 3` | 合意 |
| `SR = 44100..768000 inclusive` | 合意（`:45872` `kMaxSampleRate = 768000` で上限側が裏付け済み。下限 44100 は OS 側 domain であり本 snapshot 内では未定義） |
| `channels = 1..8` | 合意（`:53874` `kMaxIRLoadChannels = 8u`） |
| `samples = 1..2097152` | 合意（`:56049` `MAX_IR_LATENCY`） |
| `targetLength = 1..2097152` | 合意（同上） |
| `targetLength == samples` | 合意（producer chain PROVEN、§12.4） |
| `sections = 0..20` | 合意 |
| `producer sections = {2, 8, 20}` | 合意（`:50222`） |
| `max file size = 134218124` | 合意（D-1R-2A で算術 PROVEN） |
| `maxCacheEntries min=1 default=8 max=64` | **合意するが、現行は 6 入力で CONTRADICTED**（§13.3） |
| `evictLRU(0) = no-op` | **合意。PROVEN 済み**（§13.4/§14。全 cache 削除の連鎖を遮断） |
| `timestamp = GetSystemTimeAsFileTime` | 合意（`:77342`, `:77416` は `getMillisecondCounter()` = 32-bit wrap。CONTRADICTED） |
| `v2 = obsolete / preserve` | **合意。現行は CONTRADICTED**（`:77486-77489`, `:77454`, `:77476-77477`） |
| `future = preserve` | **合意。現行は CONTRADICTED**（同上） |
| `canonical filename = 64 fixed-width lowercase hex + ".mph"` | 合意（§7。JUCE `toHexString` は可変長 → 現行と非互換） |
| `recovery namespace = .mph.stage / .mph.bak` | 合意（§9）。`.mph.stage` は JUCE `_temp*.mph` を排除し destructive API を実際に保護する（§10） |
| `.mph.bak.prev` は採用しない | 合意（§3.4。3 案比較で C は世代管理が必要。採用しない） |
| `power-loss durability = out of contract` | 合意（§19） |
| `corrupt artifact = never accepted` | 合意（§19。現行は CONTRADICTED） |
| `save failure must not intentionally destroy an existing valid canonical target` | 合意（§2.2。現行 `ReplaceFileW(nullptr)` は 1176 で消滅しうるため **現行は CONTRADICTED**） |
| `save success = target replacement + post-write validation succeeded` | 合意（§16 8 段階。現行は S7/S8 なし） |
| `eviction only after successful save` | 合意（§15.5） |

### 22.1 本監査が Owner に提示する追加の裁定依頼事項（5 件）

| # | 論点 | なぜ Owner 裁定が必要か |
| --- | --- | --- |
| **N1** | §5.2 の 8 値 `ArtifactState` のうち、`CurrentV3NonCanonical` を独立した state として保持するか、`NonCanonical` に統合するか | 判定経路が「canonical だが filename 不一致」であれば v3 header が壊れている可能性があり、`NonCanonical`（名前が不正）とは原因が異なる |
| **N2** | §17.4 の attributes/streams を `ArtifactState` の判定項目に含めるか | 含めると 1177 recovery の扱いが「新規作成 + copy」必須になり、canonicality の定義が変わる。含めないなら canonical の意味が「bytes + header のみ」に固定される |
| **N3** | §16.3 の diagnostics 機構（`None / PendingCleanup / PendingRecovery`）を `save()` とは別 API にするか | `save()` のシグネチャ変更（Owner の `bool save(...)` 契約の維持）と、status の受け渡し先の設計に直結 |
| **N4** | **【新規 defect】`touch()` を v3 でどう扱うか**。現行 `:77419` `juce::FileOutputStream outStream(file);` は **canonical target を truncate する**。`:77394-77395` のコメントは「read-modify-write で行って切り詰めを回避する」と述べるが、出力側の `FileOutputStream(file)` が truncate するため、コメントの主張と実装が**不一致**。`touch()` は `save()` 以外の経路から canonical target を破壊しうる唯一の経路である | (a) `touch()` を v3 transaction（stage + rename）に統合するか、(b) `load()` 成功時の `touch()` 呼び出し（`:77293`）を廃止して in-memory LRU のみとするか、(c) 現行维持（accept しない）。いずれも `lastUsedTime` の永続 LRU 要否に直結するため Owner 裁定が必要 |
| **N5** | `evictLRU` の `std::sort`（`:77460-77461`）が `invalid timestamp を末尾` に置かない点。Owner §11 の deterministic ordering（`valid ascending / invalid last / canonical filename ascending`）は現行と**不一致** | 順序契約そのものを Owner が確定する必要がある |

---

## 付録 A. authority 引用一覧（最新 snapshot の行番号）

`C:\VSC_Project\ConvoPeq\doc\ConvoPeqMD\20261002231921\ConvoPeq.md`
（5,941,824 bytes / 131,019 lines / SHA-256 `3a51ab3414dba90005b99540376a93d6ab76ae1a96a754ce9e7050e2e0593045`）

### A.1 cache 実装（`src\MixedPhasePersistentCache.cpp`、`:77150` から）

| 行 | 内容 |
| --- | --- |
| `:77167-77170` | `hashCombine` |
| `:77172-77192` | `computeKeyHash`（`memcpy` bit cast、hash combine） |
| `:77198-77210` | `getCacheDirectory`（`%APPDATA%/ConvoPeq/MixedPhaseCache`） |
| `:77212-77222` | `getCacheFile`（`toHexString(int64) + ".mph"`、可変長） |
| `:77228-77236` | `load` 宣言部 |
| `:77239` | `file.existsAsFile()` |
| `:77252-77253` | `magic != kMagic \|\| version != kVersion` |
| `:77254-77260` | identity 検証（**tolerance 比較**） |
| `:77263-77266` | `numChannels <= 0 \|\| numSamples <= 0`（**上限検査なし**） |
| `:77268` | `outIr.setSize(numChannels, numSamples)`（header 制御の無制限確保） |
| `:77269-77276` | 波形 read |
| `:77279-77290` | rho/theta read（**上限検査なし**） |
| `:77293` | `touch(...)`（read-modify-write） |
| `:77302-77309` | `save` 宣言部 |
| `:77321` | `juce::TemporaryFile tempFile(file)` |
| `:77323-77325` | staging `FileOutputStream` open |
| `:77329` | `numSec = static_cast<int>(rho.size())` |
| `:77332-77345` | header 構築（`lastUsedTime = getMillisecondCounter()` at `:77342`） |
| `:77347-77365` | header / 波形 / rho / theta write |
| `:77368` | `header.totalFileSize = stream.getPosition()` |
| `:77370-77371` | header 書き戻し |
| `:77373` | `stream.flush()`（戻り値無視） |
| `:77376` | `return tempFile.overwriteTargetFileWithTemporary();` |
| `:77383-77423` | `touch`（read-modify-write。`:77419` `FileOutputStream outStream(file)` で**truncate**） |
| `:77416` | `now = getMillisecondCounter()`（32-bit wrap） |
| `:77425-77466` | `evictLRU` |
| `:77427-77431` | `maxCount == 0 → clear()` |
| `:77435` | `findChildFiles(..., "*.mph")` |
| `:77454` | `header.magic != kMagic → continue`（**version 不問**） |
| `:77460-77461` | `std::sort`（invalid 末尾 rule なし） |
| `:77468-77478` | `remove`（**分類検査なし**） |
| `:77480-77490` | `clear`（**分類検査なし**） |
| `:77487`, `:77499` | `findChildFiles(..., "*.mph")` |
| `:77526-77603` | `class MixedPhasePersistentCache` 宣言 |
| `:77574` | `kMagic = 0x4D69786564506800ULL` |
| `:77575` | `kVersion = 2` |
| `:77577-77593` | `struct DiskHeader`（`#pragma pack(1)`、14 field） |
| `:77596` | `static_assert(sizeof(DiskHeader) <= 128)` |

### A.2 producer chain

| 行 | 内容 |
| --- | --- |
| `:45872` | `static constexpr int kMaxSampleRate = 768000;` |
| `:53874` | `inline constexpr unsigned kMaxIRLoadChannels = 8u;` |
| `:56049` | `MAX_IR_LATENCY` |
| `:4003-4008` | `struct SecondOrderAllpass` / `isStable()` → `rho ∈ [0, 1)` |
| `:3543` | `rho = std::clamp(std::abs(gain), 0.0, 0.995)` |
| `:3544` | `theta = 2π f0 / sampleRate` |
| `:3552-3556` | `unconstrainedToRho(x) = 0.98 * stableSigmoid01(x)` |
| `:3559` | `kThetaMax = 0.99π` |
| `:3561-3566` | `unconstrainedToTheta(x) = kThetaMax * stableSigmoid01(x)` |
| `:3769-3770` | `section.rho/theta = unconstrainedTo*` |
| `:3813-3814` | `section.rho = std::clamp(std::abs(best_gain), 0, 0.995)` / `theta = 2π best_f0 / SR` |
| `:49869-49872` | `MixedPhasePersistentCache::load(...)` 呼出 |
| `:49881` | `entry.lastUsedTime = getMillisecondCounter()` |
| `:49882-49883` | `cachedRho/cachedTheta` → `entry.allpassSections`（**finite 検証なし**） |
| `:49900` | `numSamples = linearIR.getNumSamples()` |
| `:49945` | `mixedIR(numChannels, numSamples)` |
| `:50222` | `numSections` |
| `:50270` | `optimizedByCmaes = designSuccess` |
| `:50296` | `AllpassDesigner::computeResponse(...)` |
| `:50311` | `lastAllpassSections = allpass_sections` |
| `:50316-50321` | mixedIR への書き戻し |
| `:50330-50343` | RMS 規格化 `dst[i] *= gain` |
| `:50346-50357` | **全 channel × 全 sample finite sweep** |
| `:50417` | `key.targetLength = linearIR.getNumSamples()` |
| `:50434-50439` | `rho.push_back` / `theta.push_back`（同一ループ） |
| `:50440-50443` | `MixedPhasePersistentCache::save(...)` |
| `:50446` | `evictLRU(owner->getMaxCacheEntries())`（**戻り値無視**） |
| `:50466` | `convertToMixedPhaseFallback`（save しない） |

### A.3 maxCacheEntries data flow

| 行 | 内容 |
| --- | --- |
| `:13603-13606` | `getConvolverMaxCacheEntries()` → `static_cast<int>` |
| `:13608-13611` | `setConvolverMaxCacheEntries(int)` → `static_cast<size_t>` |
| `:48986-48995` | `setMaxCacheEntries`（`jlimit<size_t>(1, 64, ...)` at `:48988`） |
| `:48997-49001` | `getMaxCacheEntries`（`static_cast<size_t>` at `:49000`） |
| `:49062` | `cacheLimit = getMaxCacheEntries()`（memory cache 側） |
| `:52779` | structural hash に `snapshot.maxCacheEntries` |
| `:52868` | `snapshot.maxCacheEntries = pendingOverride.maxCacheEntries` |
| `:52927` | `pendingOverride.maxCacheEntries = juce::jlimit(0, 64, ...)`（**下限 0**） |
| `:52976` | `v.setProperty("maxCacheEntries", static_cast<int>(getMaxCacheEntries()))` |
| `:53108` | `setMaxCacheEntries(static_cast<size_t>(static_cast<int>(v.getProperty(...))))`（**二重 narrowing**） |
| `:55888` | `int maxCacheEntries = 0;`（snapshot 既定） |
| `:56882` | `int maxCacheEntries = 0;`（override 既定） |
| `:84826` | `cacheManager.evictLRU(processor.getMaxCacheEntries())`（別 manager） |
| `:12833-12834` | `hasFiniteDouble("mixedF1Hz"/"mixedF2Hz", ...)` |

### A.4 JUCE tree（`C:\VSC_Project\ConvoPeq\JUCE\modules\juce_core\`）

| ファイル:行 | 内容 |
| --- | --- |
| `native/juce_Files_windows.cpp:372-378` | `File::replaceInternal` → `ReplaceFile(dest, this, nullptr, IGNORE_MERGE_ERRORS\|4, nullptr, nullptr)` |
| `files/juce_TemporaryFile.cpp:38-45` | `createTempFile` |
| `files/juce_TemporaryFile.cpp:61-70` | `TemporaryFile(target)` → `<nameWithoutExt>_temp<randhex><ext>` |
| `files/juce_TemporaryFile.cpp:94-104` | `overwriteTargetFileWithTemporary`（5 回 retry） |
| `text/juce_String.cpp:1944-1966` | `String::toHexString(const void*, size, groupSize)` |
| `text/juce_String.cpp:2625-2633` | `toHexString` の **可変長・非 zero-pad** 実証（`0x1234abcd → "1234abcd"`、`(int8)-1 → "ff"`） |
| `text/juce_String.cpp:1135-1169` | `WildCardMatcher`（full match semantics） |

### A.5 external evidence

| 出典 | 引用 |
| --- | --- |
| MS Learn `ReplaceFileW` | 1175 / 1176 / 1177 / `ERROR_INVALID_PARAMETER` の各説明、`lpBackupFileName == NULL` で backup なし |
| MS Learn `MoveFileExW` | `MOVEFILE_REPLACE_EXISTING` (1/0x1) が必要であることの記載、`MOVEFILE_WRITE_THROUGH` (8/0x8) が copy-and-delete の flush のみを対象とする記載 |

---

## 付録 B. D-1R-2G からの訂正・昇格・降格

| 項目 | D-1R-2G | D-1R-2H |
| --- | --- | --- |
| `.bak` の存在前提 | 「次回 save で `.bak` 上書き」を前提として Option A/B/C を比較 | **訂正**: 現行実装には `.bak` が存在しない（`ReplaceFileW(lpBackupFileName=nullptr)`）。Option B の T0-T5 は新規設計であり現行の extension ではない |
| staging namespace | `.mph.stage` により `*.mph` から分離 | **補強**: JUCE の staging は `_temp*.mph` で**現行 v2 では `*.mph` に属する**（§10）。v3 の `.mph.stage` は destructive API 保護を実際に実現する |
| 64-hex `byte order` | 「little-endian」 | **訂正**: Option A（numeric fixed-width hex）を採用。endian は仕様に含まれない |
| `Decodeable` vs `Canonical` | 未分離 | **分離**（§9）。uppercase は Decodeable だが Canonical ではない |
| artifact classification | S0-S9 の 3 boolean | **8 値 `ArtifactState`**（§5） |
| transaction identity | filename suffix ベース | **identity prefix ベース**（§6）。suffix のみでは membership を証明しない |
| `IR samples finite` 全網羅 | **NOT PROVEN** | **昇格 PROVEN**（producer 側、§12.4）。consumer 側は NOT PROVEN のまま |
| `rho` range | NOT PROVEN | **PROVEN に narrowed**: `finite` のみ PROVEN。`[0,1)` は SUPPORTED（`std::clamp` の NaN 透過性） |
| `maxCacheEntries` | CONTRADICTED（`:53108` のみ） | **強化**: F1-F10 の 10 段を完全追跡。`0` 到達経路（既定値 + `:52927`）を特定し、**全 cache 削除の連鎖**を PROVEN 化 |
| `evictLRU(0)` | CONTRADICTED | **維持 + 影響範囲を確定**（`:50446` から save 成功時に到達） |
| save → eviction | CONTRADICTED | **維持 + `save=false` 全 8 経路（E1-E8）を分類** |
| 1177 | SUPPORTED（SM 混在） | **SM-A / SM-B に分離**。1177 は SM-B 専用。属性継承は PROVEN |
| power-loss | PROVEN（out of contract） | **維持 + 「corrupt 排除との非矛盾」を形式化**（§19.2） |
| `touch()` | 未監査 | **新規 CONTRADICTED**: `:77419` `FileOutputStream outStream(file)` が canonical target を **truncate** する非 atomic read-modify-write。§22 に裁定依頼 N4 として追加推奨 |
| load の header 制御確保 | 未監査 | **新規 PROVEN**: `:77268` `setSize(numChannels, numSamples)` に上限検査なし。structural validation 必須性の実証的根拠 |

---

## 付録 C. mojibake scan

| 対象 | U+FFFD | Cyrillic | Hangul | Thai | 簡体字 | `oxin` 混入 |
| --- | --- | --- | --- | --- | --- | --- |
| 本報告書 | 0 | 0 | 0 | 0 | 0 | 0 |

走査対象コードポイント: `U+FFFD` / `U+0400-04FF` / `U+1100-11FF` / `U+AC00-D7AF` / `U+0E00-0E7F` / `U+1EA0-1EF9`、および `U+00C0-024F` の latin-ext。

初版作成時に以下 8 件の混入を検出し修正済み:

| 検出 | 修正 |
| --- | --- |
| `обя`（Cyrillic）+ `责任项目` | `責務範囲` |
| `결합`（Hangul jamo） | `結合` |
| `污染`（簡体字） | `汚染` |
| `事务典`（`务` は簡体字） | `transaction` |
| `vànà`（Vietnamese diacritic） | `有効な` |
| `通道`（簡体字） | `経路` |
| `覆盖範囲`（簡体字） | `網羅範囲` |
| `seeing` / `xespars` / `bajo` / `Sequel` / `AAS` / `raised` / `Product` 混在 | 日本語へ置換 |

最終走査で `U+00C0-024F` に残る文字は `§`（139 回）と `×`（2 回、数学記号）のみ。`oxin` は本表の見出し文字列そのものであり本文には出現しない（`mixing` の部分一致は誤検出）。

---

## 付録 D. read-only 遵守の記録

| 対象 | 変更 |
| --- | --- |
| `src/**` | 0 |
| tests | 0 |
| `CMakeLists.txt` | 0 |
| `build/` | 0 |
| CTest 実行 | 0 |
| `ConvoPeq.md` 再生成 | 0 |
| `git commit` | 0 |
| `git push` | 0 |
| 作成したファイル | `doc/work113/P1-5-IR-P2_MIXEDPHASE-BG-CMAES_D1R2H_7864_20261006.md` のみ |

---

## 付録 E. 変更履歴

- 2026-10-06: D-1R-2H 初版作成（§0-§22 + 付録 A-E）。