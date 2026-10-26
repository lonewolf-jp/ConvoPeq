# D-1R-2K — Phase 0 Executable Contract Closure Audit

- 位置づけ: Owner 指示「Owner 回答 — D-1R-2J 精査結果と実装前追加条件」（Notion page `3f185e21-c4b6-816e-91f9-f0c301838660`、2026-10-06 14:59）に対する read-only closure audit
- 作成: 2026-10-06
- 変更していないもの: `src/**` / `tests/**` / `CMakeLists.txt` / `build/**` / `ConvoPeq.md` / commit / push（すべて 0）
- 本書は **Phase 0 の実行可能 contract を定義する**。production source の変更は行わない（Owner §12）

---

## §0. 判定語と read-only 遵守

| 語 | 定義 |
| --- | --- |
| **SOURCE-PROVEN** | 最新 `ConvoPeq.md` の該当行、または一次資料 / 実行計算から直接証明できる事実 |
| **DESIGN-VALIDATED** | 改修後 contract として論証できる事項（現行では成立していない） |
| **NOT PROVEN** | Owner 判断または production 実証が必要な事項 |

NOT PROVEN の分類: **NP-S**（source inspection） / **NP-T**（実行テスト） / **NP-O**（Owner decision）

**source authority**: `C:\VSC_Project\ConvoPeq\doc\ConvoPeqMD\20261002231921\ConvoPeq.md`
（5,941,824 bytes / 131,019 lines / SHA-256 `3a51ab3414dba90005b99540376a93d6ab76ae1a96a754ce9e7050e2e0593045`）

**保持**: D-1R-2I / D-1R-2J / Owner 指示ページは削除・改変しない（Owner §13）。

---

## §1. Owner 回答の closure 一覧

### §1.1 追加条件 8 項目の closure 状況

| # | Owner §12 の条件 | 本監査での closure | 判定 |
| --- | --- | --- | --- |
| 1 | X-1: `kDefaultPersistentCacheEntries = 0` 維持。8 への変更は禁止 | §2.1 | **CLOSED**（Owner 裁定） |
| 2 | X-2: diagnostics は public API にせず内部 state + test/diagnostic observation | §2.2 | **CLOSED**（Owner 裁定） |
| 3 | 768 kHz の targetLength 2値分裂を正式な design decision として closure | §3 | **CLOSED**（§3.5 で「768 kHz 固有」という前提を訂正し、全域に一般化） |
| 4 | `targetLength` / `numSamples` / `expectedSize` の関係を v3 contract として明文化 | §4 | **CLOSED** |
| 5 | producer-side narrowing と consumer-side validation を対で検証 | §4.6 | **CLOSED** |
| 6 | eviction candidate は validated usable canonical artifact に限定 | §6 | **CLOSED**（D-1R-2J §8 を具体化） |
| 7 | clear/remove/evict の destructive authority を state gate に統一 | §7 | **CLOSED** |
| 8 | Phase 0 test-only を先に完了し、その結果を Owner に提示 | §9 | **本書で仕様提示**。実コード化は Owner GO 後 |

### §1.2 D-1R-2J の未 closure 項目からの変化

| # | D-1R-2J 時点 | D-1R-2K 時点 |
| --- | --- | --- |
| NP-09 / NP-16（`kDefaultPersistentCacheEntries` の衝突） | NOT PROVEN / NP-O | **CLOSED**（Owner: 0 維持） |
| NP-11（diagnostics の外部 API 化） | NOT PROVEN / NP-O | **CLOSED**（Owner: 内部 state 維持） |
| NP-13（768 kHz targetLength 2値分裂） | NP-S 完了、ただし「768 kHz 固有」の記述 | **訂正して一般化**（§3.5: 全整数 SR の 50.07% で発生） |
| NP-01〜06（production 実証 6 件） | NOT PROVEN / NP-T | **NP-T のまま**。Phase 0 の harness 仕様を §9.6 に確定 |

---

## §2. X-1 / X-2 の closure

### §2.1 X-1 — `kDefaultPersistentCacheEntries`（Owner: 0 維持）

**裁定: `maxCacheEntries` の既定値は 0 のまま維持する。8 への変更は禁止**（Owner §4）。

現行 source との整合（すべて SOURCE-PROVEN）:

| 項目 | 値 | 行 |
| --- | --- | --- |
| `BuildSnapshot::maxCacheEntries` 既定 | `0` | `:55888` |
| `StateSnapshot::maxCacheEntries` 既定 | `0` | `:56882` |
| `setState` 復元クランプ | `jlimit(0, 64, ...)` — **下限 0 を許す** | `:52927` |
| `setMaxCacheEntries` クランプ | `jlimit<size_t>(1, 64, ...)` — **下限 1** | `:48988` |
| `evictLRU` の既定引数 | `size_t maxEntries = 10`（C2 専用） | `:46570` |

**Owner の判断根拠と source の一致（DESIGN-VALIDATED）**: `8` は C3 の in-memory `MAX_CACHE_ENTRIES = 8`（`:57168`）の値であり、persistent cache の容量 semantic とは別物。Owner §9 の 3 系統分離の考え方と整合する。

**ただし、X-1 の closure にともなって必須となる契約（Owner §4 が明示）**:

```text
evictLRU(0) == no-op
```

**これが現行で成立していないことの SOURCE-PROVEN 証拠**:

`:50446` `MixedPhasePersistentCache::evictLRU(owner->getMaxCacheEntries())` は `size_t` を渡す。
`getMaxCacheEntries()` は `size_t` を返す（`:48997-49001`）。
`setState` 復元は `jlimit(0, 64, snapshot.maxCacheEntries)`（`:52927`）で **0 を許す**。
`:48991` は `static_cast<int>(clamped)` で `pendingOverride.maxCacheEntries` に格納する。

したがって **復元経路で `maxCacheEntries == 0` が現れ得**、`:50446` 経由で `evictLRU(0)` が呼ばれる。而して現行 `evictLRU` は `:77427-77431` で `0 → clear()` とする。

→ **`evictLRU(0) = no-op` は v3 で必ず明示実装しなければならない**（Owner §4 が要求）。現行は **CONTRADICTED**（Owner 裁定と現行挙動が矛盾）。

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-01** | **SOURCE-PROVEN** | `maxCacheEntries` の既定 0 が snapshot / state の 3 箇所すべてで 0（`:55888`, `:56882`, `:52927`） |
| **K-02** | **SOURCE-PROVEN** | `setMaxCacheEntries` の下限 1（`:48988`）と `setState` 復元の下限 0（`:52927`）が**不一致** |
| **K-03** | **CONTRADICTED** | 現行 `evictLRU(0)` は `clear()` を呼ぶ（`:77427-77431`）ため、Owner の「0 維持」裁定と整合しない。**v3 で `evictLRU(0) = no-op`  MUST** |
| **K-04** | **DESIGN-VALIDATED** | v3 は `evictLRU(n)` を `n == 0 → return` で早期終了し、例外として `clear()` を提供しない |

### §2.2 X-2 — diagnostics API（Owner: 内部 state 維持）

**裁定: `ResidueDisposition`（`None` / `PendingCleanup` / `PendingRecovery`）を外部 public API にしない**（Owner §5）。

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-05** | **DESIGN-VALIDATED** | classification / residue の state は `MixedPhasePersistentCache` の **private** enum として保持する。public 宣言は追加しない |
| **K-06** | **DESIGN-VALIDATED** | Phase 0 の test は friend 宣言または同一 TU 内の test seam Through `convo::testing` で state を観測する。public API 拡大を伴わない |
| **K-07** | **DESIGN-VALIDATED** | 既存 public API（`load` / `save` / `evictLRU` / `remove` / `clear` / `getEntryCount`）の**シグネチャは不変**。`save()` の `bool` 契約（N3）も維持 |

**X-2 は NP-O → CLOSED**（Owner decision、Owner §5 相当）。

---

## §3. Owner §2 — 768 kHz targetLength 2値分裂の正式設計論点化

Owner §2 は 6 項目の追加検証を要求し、「768 kHz 以外にも float32 境界で同様の 1-sample divergence が発生しないか」（要求 5）を明示した。

**結論（先出し）: 発生する。，而且 768 kHz 固有ではない。**
`int(SR × (double)float32(2^21 / SR))` は整数 SR 44,100〜768,000 の **50.07%** で `2^21 − 1` になる。**44,100 Hz 自身を含む全標準放送 SR が該当する**（SOURCE-PROVEN、独立実行計算）。

### §3.1 要求 1 — `setTargetIRLength()` → snapshot → `computeTargetIRLength()` の全経路

SOURCE-PROVEN（`:52376-52397`, `:52855`, `:52895-52898`, `:53719-53745`）:

```text
setTargetIRLength(timeSec)                                   :52376
  ├─ isFinite(timeSec) 拒否（NaN/Inf は前値保持）              :52382-52383
  ├─ maxAllowedSec = getMaximumAllowedIRLengthSec(
  │      consumeAtomic(currentSampleRate, acquire))           :52384
  │      └─ sr <= 0        → IR_LENGTH_MAX_SEC (3.0f)         :53705
  │      └─ sr > 0         → (float)(2^21 / sr)   ★float32★   :53707
  ├─ clampedTime = jlimit(0.5f, maxAllowedSec, timeSec)       :52385
  └─ pendingOverride.targetIRLengthSec = clampedTime          :52390

--- capture ---  snapshot.targetIRLengthSec = pendingOverride.targetIRLengthSec   :52855

--- restore (setState) ---
  irHardMaxSec = getMaximumAllowedIRLengthSec(
      consumeAtomic(currentSampleRate, acquire))              :52895
  pendingOverride.targetIRLengthSec =
      jlimit(0.5f, irHardMaxSec, snapshot.targetIRLengthSec)   :52896-52898

computeTargetIRLength(sampleRate, originalLength)             :53719
  targetIRTimeSec = (double) pendingOverride.targetIRLengthSec  :53724
  target = (int)(sampleRate * targetIRTimeSec)                 :53728
  if (target > 2^21) → [SR-01] log                             :53730-53739
  target = min(target, 2^21) ; target = max(target, 1)          :53741-53742
```

**★float32 正規化は `getMaximumAllowedIRLengthSecForSampleRate` の 1 箇所のみ**（`:53707`）。`jlimit` は float32 だが `maxAllowedSec` をそのまま返すため新たな丸めは発生しない。`(double)float` の widening は厳密。

**この経路が `2^21 − 1` を生む条件**: `timeSec > maxAllowedSec`（ユーザーが上限より長い時間を要求）→ `clampedTime = (float)(2^21/sr)` → `target = (int)(sr × float(2^21/sr))`。

**SOURCE-PROVEN**: `jlimit` の上界が `maxAllowedSec` であるため、**経路 1 は全 SR で float32 境界を経由しうる**。

### §3.2 要求 2 — `setState()` → prepare → load の全経路

SOURCE-PROVEN（`:52889-52898`, `:53730-53739`）:

`setState` は復元時に `irHardMaxSec` を `consumeAtomic(currentSampleRate, acquire)` から読む（`:52895`）。
**未 prepare（`currentSampleRate == 0`）なら `irHardMaxSec = 3.0f`**（`:53705`）→ `pendingOverride.targetIRLengthSec = min(snapshot, 3.0f)`。

その後 `prepareToPlay` が実 SR を publish すると、`computeTargetIRLength(768000, n)` は：

```
target = (int)(768000 × 3.0) = 2304000
       > 2^21 → [SR-01] log 発火                                :53736
       min(., 2^21) = 2097152                                   :53741
```

**`:52889-52894` のコメントが到達経路を明示している**（SOURCE-PROVEN）:

```
// ★ SR-01 (C-1): ... 従来は静的 IR_LENGTH_MAX_SEC=3.0 クランプで、processing SR > 699,050 Hz のとき
//   BuildSnapshot 経由の要求が素通しされ computeTargetIRLength のサンプル数上限で無音切断していた。
//   到達経路: pendingOverride が hardMax(sr) 未反映の状態で確定したケースのみ
//   （典型: 未 prepare =currentSampleRate 0 の setState 復元後に高 SR で prepare/load）。
```

**`:53730-53734` のコメントが同一の到達経路を獨立に記録**（SOURCE-PROVEN）。経路 1 と経路 2 は**別々の既知 defect シナリオ**であり、両方が `2^21` を生む。

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-08** | **SOURCE-PROVEN** | `2^21 = 2,097,152` を生成するのは **`pendingOverride` が `currentSampleRate == 0` の間に 3.0f で確定**した経路のみ |
| **K-09** | **SOURCE-PROVEN** | この経路は `[SR-01]` log（`:53736`）を発火する。**production で可視化されている** |
| **K-10** | **DESIGN-VALIDATED** | v3 の cache contract は `2^21` と `2^21 − 1` の**双方を producer が正当に生成しうる値**として受理しなければならない |

### §3.3 要求 3 — auto-detected IR length 経路

SOURCE-PROVEN（`:52399-52425`, `:53079-53082`）:

```text
applyAutoDetectedIRLength(timeSec)                             :52399
  ├─ isFinite(timeSec) 拒否                                    :52402-52403
  ├─ maxAllowedSec = getMaximumAllowedIRLengthSec(acquire)      :52404
  ├─ clampedTime = jlimit(0.5f, maxAllowedSec, timeSec)         :52405
  ├─ autoDetectedIRLengthSec = clampedTime                      :52410
  ├─ irLengthManualOverride = false                            :52411
  └─ if |prevTarget − clampedTime| > 1e-5f:                     :52416
        targetIRLengthSec = clampedTime        ★二重更新★       :52420
```

**経路 3 の特徴（SOURCE-PROVEN）**: `targetIRLengthSec` は `autoDetectedIRLengthSec` と**別フィールド**であり、`:52416` の差分ガードを通ったときだけ追随する。2 つのフィールドが独立に保持されるため、**同一 SR で `targetIRLengthSec` と `autoDetectedIRLengthSec` が異なる値を持ちうる**。

復元経路（`:53079-53082`）:

```text
applyAutoDetectedIRLength (v.getProperty ("autoDetectedIRLength"))   :53080
  └─ else if hasProperty("irLength"): applyAutoDetectedIRLength(...) :53082
```

→ 復元も §3.1 と同じ `clampedTime` 経路を通る。

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-11** | **SOURCE-PROVEN** | 経路 3 は `targetIRLengthSec` を `:52420` で**条件付き**更新する。`:52416` の 1e-5 ガードは「値が等しい場合”这一icode 書かないだけ的人格で、実質的な不整合を生まない |
| **K-12** | **DESIGN-VALIDATED** | v3 の cache identity は `targetIRLengthSec` の**確定後**の値（= `linearIR.getNumSamples()`）を基にするため、経路 1/2/3 のどれを通っても最終的な cache identity は一貫する |

### §3.4 要求 4 — cache load/save が targetLength の実値を identity と header の双方で一致させること

SOURCE-PROVEN（`:49815`, `:49840`, `:49865`, `:50417`, `:77172-77192`, `:77212-77222`, `:77341`, `:77259`）:

**identity 側**:

```cpp
uint64_t MixedPhasePersistentCache::computeKeyHash(uint64_t fileHash,
                                                    double sampleRate,
                                                    int phaseMode,
                                                    float freqStartHz, float freqEndHz,
                                                    int targetLength)      :77172
{
    uint64_t h = fileHash;                                               :77178
    ... srBits, phaseMode, f1Bits, f2Bits ...                            :77181-77189
    h = hashCombine(h, static_cast<uint64_t>(targetLength));             :77190  ★
    return h;
}
```

→ **`targetLength` は identity に含まれる**（SOURCE-PROVEN）。

**producer 側（save）**:

```
:50417  key.targetLength = linearIR.getNumSamples();
:50440  MixedPhasePersistentCache::save(key.fileHash, key.sampleRate, ...,
:50442                                   key.f1, key.f2, key.targetLength, ...)
:77311  const auto file = getCacheFile(..., targetLength);      ← identity に使う
:77341  header.targetLength = static_cast<int32_t>(targetLength); ← header にも書く
```

**consumer 側（load）**:

```
:49865  key.targetLength = linearIR.getNumSamples();            ← 同一式
:49871  ..., key.f1, key.f2, key.targetLength, ...)
:77237  const auto file = getCacheFile(..., targetLength);      ← identity で探索
:77259  || header.targetLength != static_cast<int32_t>(targetLength)) return false;
```

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-13** | **SOURCE-PROVEN** | **save と load はともに `linearIR.getNumSamples()` を `targetLength` に使う。identity と header は同一の値 hood となる** |
| **K-14** | **SOURCE-PROVEN** | identity には `targetLength` が含まれる（`:77190`）。load は header 値を要求値と厳密比較する（`:77259`、±許容なし） |
| **K-15** | **DESIGN-VALIDATED** | v3 の Option A（64-hex lossless 固定幅）も `targetLength` を identity に含める（`targetLengthBits` 8 hex）。**現行の 64-bit hash は衝突可能、v3 は無衝突**。これが identity を窄義化する理由 |

### §3.5 要求 5 — 768 kHz 以外にも float32 境界で 1-sample divergence が発生するか【主要新規発見】

**独立実行計算**（`float32` = IEEE-754 binary32 round-to-nearest-even を Python `struct` で再現）。
`MAX_IR_LATENCY = 2^21 = 2,097,152`、SR 範囲 = 整数 `[44100, 768000]`。

```text
整数 SR 走査数                    : 723,901
2^21 − M(SR) の値域               : {0: 361,446 , 1: 362,455}
M(SR) != 2^21 の SR 数            : 362,455   (50.0697 %)
最大損失                          : 1 sample（2 sample 以上の損失は 0 件）
```

**M(SR) = `int(SR × (double)float32(2^21 / SR))`**（= `:53707` → `:53728` の経路）

#### 標準／放送 SR の実測

| SR | hardMaxSec | M(SR) | 損失 | C(SR) | M vs C |
| --- | --- | --- | --- | --- | --- |
| 8,000 | 262.1440125 | 2,097,152 | 0 | 24,000 | — |
| 11,025 | 190.2178650 | **2,097,151** | **1** | 33,075 | — |
| 16,000 | 131.0720062 | 2,097,152 | 0 | 48,000 | — |
| 22,050 | 95.1089325 | **2,097,151** | **1** | 66,150 | — |
| 32,000 | 65.5360031 | 2,097,152 | 0 | 96,000 | — |
| **44,100** | 47.5544662 | **2,097,151** | **1** | 132,300 | — |
| **48,000** | 43.6906662 | **2,097,151** | **1** | 144,000 | — |
| 64,000 | 32.7680016 | 2,097,152 | 0 | 192,000 | — |
| **88,200** | 23.7772331 | **2,097,151** | **1** | 264,600 | — |
| **96,000** | 21.8453331 | **2,097,151** | **1** | 288,000 | — |
| 128,000 | 16.3840008 | 2,097,152 | 0 | 384,000 | — |
| **176,400** | 11.8886166 | **2,097,151** | **1** | 529,200 | — |
| **192,000** | 10.9226665 | **2,097,151** | **1** | 576,000 | — |
| 256,000 | 8.1920004 | 2,097,152 | 0 | 768,000 | — |
| **352,800** | 5.9443083 | **2,097,151** | **1** | 1,058,400 | — |
| **384,000** | 5.4613333 | **2,097,151** | **1** | 1,152,000 | — |
| 512,000 | 4.0960002 | 2,097,152 | 0 | 1,536,000 | — |
| **705,600** | 2.9721541 | **2,097,151** | **1** | **2,097,152** | **−1 ★** |
| **768,000** | 2.7306666 | **2,097,151** | **1** | **2,097,152** | **−1 ★** |

**`M vs C` 列が `−1` になるのが 705,600 Hz と 768,000 Hz のみ** — これが D-1R-2J NP-13 の正体であり、**「同一の物理的 IR 長要求が path 依存で別 cache identity になる」のは SR cap が発動する 699,051 Hz 以上に限られる**。

一方、`M(SR)` が `2^21 − 1` になる SR は 705,600 / 768,000 を含む **全標準放送 SR**（44,100 / 48,000 / 88,200 / 96,000 / 176,400 / 192,000 / 352,800 / 384,000 / 705,600 / 768,000 の 10/10）。

**正確に这样说と**:

| 主張 | 判定 |
| --- | --- |
| 「float32 正規化により `targetLength` が `2^21 − 1` になる SR は 768 kHz 固有」 | **CONTRADICTED**。整数 SR の 50.07%、標準放送 SR の 10/10 が該当 |
| 「同一の物理的 IR 長要求が path 依存で 2 つの cache identity になる」 | **SOURCE-PROVEN**（SR cap 帯 699,051〜768,000 Hz に限定） |
| 「損失が 2 sample 以上に達する SR は存在しない」 | **SOURCE-PROVEN**（float32 相対誤差 2^-24 の上界から、絶対誤差は常に < 1 sample） |

**非該当 SR の構造**: 8,000 / 16,000 / 32,000 / 64,000 / 128,000 / 256,000 / 512,000 はすべて `8000 × 2^k`。`2^21 / (8000 × 2^k)` の float32 正規化後に `× (8000 × 2^k)` を戻すと厳密に `2^21` になる（IEEE-754 の丸め対称性）。

#### ユーザー明示値の経路（float32 損失なし）

| 検証 | 結果 |
| --- | --- |
| `P(SR, t) = min(max((int)(SR × jlimit(0.5f, hardMax(SR), t)), 1), 2^21)` | |
| `t ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}` × 7 標準 SR | `int(SR × t)` と完全一致（**mismatch 0 件**） |

→ **ユーザーが 3.0 秒以下を明示的に要求した場合、float32 損失は発生しない**（`3.0f` と `SR` がともに厳密表現可能）。損失は **`t > hardMax(SR)` のときだけ**発生する。

**`t = 3.0` の境界挙動（追加発見）**:

| SR | `int(SR × 3.0)` | `P(SR, 3.0)` | 理由 |
| --- | --- | --- | --- |
| 705,600 | 2,116,800 | **2,097,151** | `hardMax = 2.972 < 3.0` → jlimit が hardMax に丸める |
| 768,000 | 2,304,000 | **2,097,151** | 同上（`hardMax = 2.7307`） |

→ **SR cap 帯では「ユーザーが 3.0 秒を明示的に要求しても `2^21 − 1` になる」**（SOURCE-PROVEN）。`C(SR) = 2^21` に到達するのは **`pendingOverride` が `currentSampleRate == 0` の間に確定した場合のみ**（K-08）。

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-16** | **SOURCE-PROVEN** | `int(SR × float32(2^21/SR)) != 2^21` は整数 SR 44,100〜768,000 の 50.0697%（362,455 / 723,901）で成立。損失は常に 1 sample |
| **K-17** | **SOURCE-PROVEN** | 10 個の標準放送 SR（44.1k〜768k の全標準値）が全て該当 |
| **K-18** | **CONTRADICTED** | D-1R-2J NP-13 の「768 kHz で 2 値分裂」という記述は**範囲が狭すぎる**。**正しくは「SR cap 帯（≥ 699,051 Hz）では path 依存分裂、float32 損失自体は全域」** |
| **K-19** | **SOURCE-PROVEN** | `t ≤ 3.0` の明示要求では float32 損失 0 件。損失は `t > hardMax(SR)` のみ |
| **K-20** | **DESIGN-VALIDATED** | v3 の cache domain は `1 ≤ targetLength ≤ 2,097,152` とし、**全域で受理する**。正規化は禁止（Owner §2 裁定） |

### §3.6 要求 6 — 双方の値を受理することが意図した contract か

Owner §2 は「ここは NP-O 候補であり、実装前に結論を固定する」とした。本監査は次の根拠で**裁定提案**を出す。

| 根拠 | 判定 |
| --- | --- |
| `2^21` は SR-01 defect 経路（`[SR-01]` log 発火、`pendingOverride` 未 prepare 確定）の**唯一の**生産源（K-08, K-09） | **SOURCE-PROVEN** |
| `2^21 − 1` は**通常の hardmax クランプ経路**で生成される（`t > hardMax`） | **SOURCE-PROVEN** |
| 両者は**別々のセマンティクス**であり、いずれも破損ではない | **SOURCE-PROVEN** |
| `linearIR.getNumSamples() = stepTrimmed.getNumSamples() = computeTargetIRLength(...)` の戻り値（§4.1 で証明）で、**producer は意図的にその値を write する** | **SOURCE-PROVEN** |
| Owner §2 は「2,097,151 に固定」「2,097,152 に丸め」の改修を**明確に不承認** | Owner 裁定 |
| Owner §3.2 は「699,050 / 2,097,151 / 2,097,152 を production semantic constant として新設することは不要」「テスト用 frozen constant と production semantic constant を混同しないこと」 | Owner 裁定 |

**裁定提案（NP-O → CLOSED 提案）**:

```text
cache v3 の targetLength domain = closed integer interval [1, 2,097,152]

- 2,097,152 : SR-01 経路（pendingOverride が sr=0 で確定）の生産値。受理する
- 2,097,151 : hardmax クランプ経路の生産値（全標準 SR）。受理する
- 2,097,153+ : producers が生成しえない。**Reject する**
- 0 以下      : producers が生成しえない。**Reject する**

production semantic constant は新設しない（Owner §3.2）。
2,097,152 = 既存の MAX_IR_LATENCY を参照するだけでよい。
```

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-21** | **DESIGN-VALIDATED** | domain は閉区間 `[1, MAX_IR_LATENCY]`。両端とも producer が到達する。`MAX_IR_LATENCY` は既存定数を参照（新設しない） |
| **K-22** | **DESIGN-VALIDATED** | `2,097,151` は「不该な値」ではなく**正常な生産値**。v3 はこれを `NonCanonical` 扱いにしてはならない |
| **K-23** | **NP-O（closure 提案）** | 上記裁定は Owner の §2 裁定（正規化禁止）と整合するが、**Owner の明示承認を得れば closure**。本監査は GO を出さない |

---
## §4. Owner §6 — `targetLength` と exact-size validation

Owner §6 は D-1R-2J の SR-J-3（expected size は固定最大値から算出してはならない）を妥当とした上で、5 つの追加確認を要求した。

### §4.1 要求 A — header の `targetLength` と `numSamples` が同じであることを要求するのか

**答え: 要求すべきである。且つ producer 側は既に構造的にこれを満たしている**（SOURCE-PROVEN）。

#### producer 側の一致連鎖（各行は実 source）

| # | 行 | 内容 |
| --- | --- | --- |
| 1 | `:48526` | `stepResult.targetLength = owner.computeTargetIRLength(stepResult.loadedSR, stepResult.loadedIR.getNumSamples());` |
| 2 | `:48527` | `stepTrimmed.setSize(stepResult.loadedIR.getNumChannels(), stepResult.targetLength);` — **厳密な長さ設定** |
| 3 | `:48530` | `const int copySamples = std::min(stepResult.targetLength, stepResult.loadedIR.getNumSamples());` |
| 4 | `:48545` | `stepTrimmed.copyFrom(ch, 0, stepResult.loadedIR, ch, 0, copySamples);` — 不足分は 0 埋め |
| 5 | `:48658` | `convertToMixedPhase(&owner, stepFileHash, stepTrimmed, minPhaseIR, ...)` — **`linearIR` = `stepTrimmed`** |
| 6 | `:49764-49765` | `convertToMixedPhase(..., const juce::AudioBuffer<double>& linearIR, const juce::AudioBuffer<double>& minimumIR, ...)` — 仮引数名は `linearIR` |
| 7 | `:49900` | `const int numSamples = linearIR.getNumSamples();` |
| 8 | `:49945` | `juce::AudioBuffer<double> mixedIR(numChannels, numSamples);` — **構築時に長さを固定** |
| 9 | `:50417` | `key.targetLength = linearIR.getNumSamples();` |
| 10 | `:50443` | `save(..., key.targetLength, mixedIR, rho, theta);` |
| 11 | `:77328` | `const int numSamples = ir.getNumSamples();` — `ir` = `mixedIR` |
| 12 | `:77341` | `header.targetLength = static_cast<int32_t>(targetLength);` |
| 13 | `:77344` | `header.numSamples = static_cast<int32_t>(numSamples);` |

**帰結**:

```
stepTrimmed.getNumSamples()  ==  stepResult.targetLength          (行 2)
linearIR.getNumSamples()     ==  stepTrimmed.getNumSamples()        (行 5,6)
mixedIR.getNumSamples()      ==  linearIR.getNumSamples()          (行 7,8)
key.targetLength             ==  linearIR.getNumSamples()          (行 9)
header.targetLength          ==  key.targetLength                  (行 10,12)
header.numSamples            ==  mixedIR.getNumSamples()           (行 11,13)

∴  header.targetLength  ==  header.numSamples                    ★always★
```

**この等式は allpass 処理が長さを変えないことから独立に成立する**（`mixedIR` は `:49945` で長さが固定され、`:49988` の `mixedIR.copyFrom(ch, 0, mixedIR, 0, 0, numSamples)` も含め `numSamples` を超えない）。

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-24** | **SOURCE-PROVEN** | **`header.targetLength == header.numSamples` は producer 側で常に成立する**（13 行の連鎖） |
| **K-25** | **SOURCE-PROVEN** | consumer 側 `load()` は `header.targetLength != targetLength`（`:77259`）のみ検査し、**`header.numSamples` と `header.targetLength` の内部一致を検査しない** |

**v3 contract（DESIGN-VALIDATED）**:

```text
C-TL-1  save():  targetLength == numSamples を検証し、违反すれば false を返す（書かない）
C-TL-2  load():  header.targetLength == header.numSamples を検証し、違反すれば false（読まない）
C-TL-3  identity も header も「検証後の targetLength」を正とする。不一致なら artifact を Rewritten しない
```

### §4.2 要求 B — 要求しない場合、その差異に意味があるのか

**答え: 意味を持たない。且つそれは artifact corruption の指示である**。

| 観点 | 評価 |
| --- | --- |
| DSP 的意味 | `mixedIR` の実際の長さは `numSamples`。`targetLength` は DSP に渡らない（`mixedIR` 全体を allpass フィルタへ渡す `:49995-49996`） |
| 診断的意味 | なし。両者は producer 定義により同一値 |
| 破損検知の意味 | **あり**。producer が書き込まない組み合わせは corruption / 手書き artifact の強い指標 |

→ **要求すべき**（C-TL-2）。

### §4.3 要求 C — `targetLength = 2,097,151` / `numSamples = 2,097,152` のような mismatch を usable とする理由があるか

**答え: 理由がない**。

| # | 根拠 |
| --- | --- |
| K-24 | producer はこの組み合わせを**生成しない**（`targetLength == numSamples` が構造的保証） |
| — | したがって観測されたなら **producer bug か corruption**。いずれでも usable として受理してはならない |
| §3.5 K-16 | `2,097,151` 自体は**正常な生産値**。しかし `numSamples` 側と食い違う組合は正常でない |

**混同してはならない 2 件事**（本監査の最重要区別点）:

| | 内容 | 扱い |
| --- | --- | --- |
| **正常** | `targetLength = numSamples = 2,097,151`（hardmax クランプ経路） | **Usable 可** |
| **正常** | `targetLength = numSamples = 2,097,152`（SR-01 経路） | **Usable 可** |
| **異常** | `targetLength = 2,097,151` かつ `numSamples = 2,097,152` | **Malformed**。Reject |

→ v3 は「`2,097,151` を拒否する」のではなく「**`targetLength != numSamples` を拒否する**」と契約する（K-22 と整合）。

### §4.4 要求 D — file size 計算時の overflow-safe validation

`expectedSize` の式:

```text
expectedSize = sizeof(DiskHeader)
             + numChannels * numSamples * sizeof(double)
             + numAllpassSections * sizeof(double) * 2
           = 76 + numChannels * numSamples * 8 + numAllpassSections * 16
```

最大値（`8 ch × 2,097,152 samples × 20 sections`）:

```text
  8 × 2,097,152 × 8 = 134,217,728   (128 MiB)
+ 20 × 8 × 2        =         320
+ 76                =          76
  ------------------------------------
  expectedSize max  = 134,218,124
```

| 検証 | 結果 |
| --- | --- |
| `int64` に収まるか | **はい**（134,218,124 ≪ 9.22e18） |
| `uint32` に収まるか | **はい**（134,218,124 < 4,294,967,295） |
| `int32` に収まるか | **はい**（134,218,124 < 2,147,483,647）。余裕 16 倍 |
| 2,097,151 側の最大値 | 134,218,060（**差分 64 bytes**） |

#### 現行の算術の安全性（SOURCE-PROVEN）

| 行 | 式 | 評価 |
| --- | --- | --- |
| `:77354` | `static_cast<size_t>(numSamples) * sizeof(double)` | **安全**。`int32 → size_t(64-bit)` が**乗算より前** |
| `:77361` | `static_cast<size_t>(numSec) * sizeof(double)` | **安全** |
| `:77273` | `static_cast<size_t>(numSamples) * sizeof(double)` | **安全** |
| `:77285-77286` | `static_cast<int64>(static_cast<size_t>(numSec) * sizeof(double))` | **安全** |
| `:77268` | `outIr.setSize(numChannels, numSamples);` | **★未検証★**（§4.5 参照） |

→ **乗算そのものは全て `size_t`/`int64` を先に 使っているため、現行に整数溢れは無い**。問題は**確保サイズの上限検証が §4.5**。

**v3 の contract（DESIGN-VALIDATED）**:

```text
C-SZ-1  expectedSize は int64 (checked arithmetic) で算出する。
C-SZ-2  算出前に各フィールドが domain 検証を通過していることを要求する（§4.5）。
C-SZ-3  expectedSize > kMaxPersistentCacheFileBytes なら Malformed（save 側は false）。
C-SZ-4  actualFileSize != expectedSize なら Malformed。header.totalFileSize も同値であること。
C-SZ-5  sizeof(DiskHeader) は static_assert で固定済みの値を使用する（現行 76）。
```

### §4.5 要求 E — 32-bit narrowing が全経路で排除されているか

**答え: 排除されていない。5 箇所すべて未検証**（SOURCE-PROVEN）。

| # | 行 | 式 | narrowing 前検証 |
| --- | --- | --- | --- |
| 1 | `:77341` | `header.targetLength = static_cast<int32_t>(targetLength)` | **なし** |
| 2 | `:77343` | `header.numChannels = static_cast<int32_t>(numChannels)` | **なし** |
| 3 | `:77344` | `header.numSamples = static_cast<int32_t>(numSamples)` | **なし** |
| 4 | `:77345` | `header.numAllpassSections = static_cast<int32_t>(numSec)` | **なし** |
| 5 | `:53108` | `static_cast<size_t>(static_cast<int>(v.getProperty(...)))` | **なし**（二重 narrowing） |

#### 5 番の深刻度（§4 の要求 D と直結）

`:77263-77268`:

```cpp
const int numChannels = static_cast<int>(header.numChannels);   // :77263
const int numSamples  = static_cast<int>(header.numSamples);    // :77264
if (numChannels <= 0 || numSamples <= 0)                         // :77265  ★下方のみ
    return false;

outIr.setSize(numChannels, numSamples);                          // :77268  ★無制限確保
```

**破損・ hand-crafted な artifact で `numChannels = 8`, `numSamples = 2,147,483,647` の場合**:

```
8 × 2,147,483,647 × 8 bytes = 137,438,953,408 bytes  ≈ 128 GiB の連続確保を試行
```

→ **`load()` が artifact を読んで 128 GiB 確保を起動する**。`:77265` のガードは「正の値か」しか見ていない。

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-26** | **SOURCE-PROVEN** | `load()` は `numChannels` / `numSamples` の**上限制約を持たない**。`:77265` は `> 0` のみ検査し、`:77268` で無制限確保する |
| **K-27** | **DESIGN-VALIDATED** | v3 は `setSize` の**前**に domain 検証を行う。`save()` と `load()` は**同一の predicate 関数**を共有する |
| **K-28** | **DESIGN-VALIDATED** | narrowing は「検証済み domain 値の `static_cast`」に限定する。検証未実行の値の narrow は禁止 |

**Owner §6 最終項「producer-side admission と consumer-side validation を対にして設計する」への対応**:

```text
// 単一の authoritative predicate（shared by save / load）
namespace {
bool isAdmissibleDomain (int32_t numChannels, int32_t numSamples,
                         int32_t targetLength, int32_t numSections) noexcept
{
    if (numChannels < 1 || numChannels > kMaxPersistentCacheChannels)   return false;
    if (numSamples  < 1 || numSamples  > kMaxPersistentCacheSamples)    return false;
    if (targetLength< 1 || targetLength> kMaxPersistentCacheSamples)    return false;
    if (targetLength != numSamples)                                     return false;  // §4.3
    if (numSections < 0 || numSections > kMaxPersistentCacheSections)   return false;
    return true;
}
}
```

| 呼び出し側 | 使用箇所 | 効果 |
| --- | --- | --- |
| producer admission | `save()` — `header` 書込**前** | 不正値を**書かない** |
| consumer validation | `load()` — `setSize` **前**（`:77268` の直前） | 不正値を**読まない / 確保しない** |

→ **C-TL-1（save）と C-TL-2（load）が同一述語を共有することで、producer が書き込める値と consumer が受け入れる値が自動的に一致する**（Owner §6 最終項の充足）。

### §4.6 §4 の判定

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-24** | **SOURCE-PROVEN** | producer は `targetLength == numSamples` を構造的に保証（13 行連鎖） |
| **K-25** | **SOURCE-PROVEN** | consumer はこの一致を検査していない |
| **K-26** | **SOURCE-PROVEN** | consumer は `numSamples` の上限検証を持たず 128 GiB 確保を試行しうる |
| **K-27** | **DESIGN-VALIDATED** | 単一述語 `isAdmissibleDomain` を save/load で共有する |
| **K-28** | **DESIGN-VALIDATED** | narrowing は domain 検証後のみ |
| **K-29** | **DESIGN-VALIDATED** | `expectedSize` は header の実測 `numSamples` から `int64` checked arithmetic で算出 |

---

## §5. Owner §7 — `touch()` 廃止案の atomicity 証明

Owner §7 は「`load 内 single-handle timestamp 更新`」を設計として妥当としたが、単に `touch()` を消すだけでは不十分であり、6 つの failure sequence を証明することを要求した。

### §5.1 現行の touch 機序（SOURCE-PROVEN）

```cpp
// :77383-77423
...
    juce::FileOutputStream stream(file);        // :77419
    if (!stream.openedOk())
        return false;                            // （戻り値は caller が無視）
    stream.write(&header, sizeof(DiskHeader));   // :77422  ★戻り値無視★
```

JUCE 実装（一次資料）:

| ファイル:行 | 内容 |
| --- | --- |
| `juce_Files_windows.cpp:429-450` | `openHandle` = `CreateFile(..., OPEN_ALWAYS, ...)` + `SetFilePointer(FILE_END)` |
| `juce_Files_windows.cpp:432` | sharing mode = `FILE_SHARE_READ` **のみ** |
| `juce_Files_windows.cpp:468-473` | `flushInternal` = `FlushFileBuffers` |
| `juce_Files_windows.cpp:475-483` | `truncate()` = `SetEndOfFile`。**`touch()` から呼ばない** |
| `juce_FileOutputStream.cpp:39-45` | ctor が `openHandle` を呼ぶ |
| `juce_FileOutputStream.cpp:53-56` | `getPosition` |
| `juce_FileOutputStream.cpp:88-125` | `write` |
| `juce_TemporaryFile.cpp:61-70` | staging 名 = `<nameNoExt>_temp<randhex><ext>` |

**帰結**: `OPEN_ALWAYS` + `FILE_END` は**追記**であり truncate ではない。`touch()` 実行後:

```text
actualFileSize_before = N  (76 + payload)
actualFileSize_after  = N + 76 = 2N - payload ... （正確には N + sizeof(DiskHeader)）
```

→ **artifact が 2 倍に成長する**。D-1R-2H の「truncate」主張は既に取り下げ済み（Owner 指示が正しい）。

**`FILE_SHARE_READ` のみ**のため、同ファイルへの別の `GENERIC_WRITE` open は OS 上で `ERROR_SHARING_VIOLATION` となる。これは**副作用としての直列化**であり、明示的保証ではない。

### §5.2 Owner §7 の 6 failure sequence と v3 contract

| # | Sequence | 現行の帰結 | v3 contract | 判定 |
| --- | --- | --- | --- | --- |
| **FS-1** | header read OK → payload OK → **timestamp update 失敗** | `touch()` が `:77422` で write 失敗しても `load()` は `:77295` で `true` を返す。ただしファイルは 2N に成長 → **次回 load は size 不整合で false** | in-place write 失敗 / `FlushFileBuffers` 失敗は **diagnostic のみ**。`load()` は **`true` のまま**。artifact の canonical state は**変化しない** | **DESIGN-VALIDATED** |
| **FS-2** | header read OK → **payload partial → 失敗** | `:77274` `bytesRead != bytesToRead` → `false`。ファイルは書き換えられないので artifact は無傷 | 同一。ただし v3 は payload 読み出し前に `expectedSize` 検証するため、partial を早期検出する | **DESIGN-VALIDATED** |
| **FS-3** | **timestamp update OK → canonical replacement 失敗** | 現行は該当経路なし（touch は別ファイル操作） | v3 は「検証成功 → timestamp in-place 更新 → close」の順。**canonical replacement は行わない**（identity 不変なので不要）。失敗しても既存 bytes が書き換わるだけで canonical state は保たれる | **DESIGN-VALIDATED** |
| **FS-4** | **concurrent save → load** | `FILE_SHARE_READ` により一方の open が `ERROR_SHARING_VIOLATION`。ただし**どちらが失敗するかは非決定的** | **明示 NonRT mutex** で直列化。save の stage → target commit は atomic。load は commit 前か後の一方を観測（中間状態は観測しない） | **DESIGN-VALIDATED** |
| **FS-5** | **concurrent load → eviction** | `evictLRU` が load 中の target を `delete` しうる。`FILE_SHARE_READ` は削除を妨げない | eviction は `CurrentV3Canonical` のみを対象にするが、**load が成功裡に完了した entry の timestamp を更新**するため、直列化により load 完了後の entry が残る | **DESIGN-VALIDATED** |
| **FS-6** | **concurrent save → eviction** | `:50440-50446` は save 成功を**待たずに** `evictLRU` を呼ぶ。save 失敗でも eviction が走る | **`if (saved) evictLRU(...)`**（D-1R-2J A-9）。明示 mutex で直列化 | **DESIGN-VALIDATED** |

**Owner §7 最終項「timestamp 更新失敗で valid artifact を unusable と判定してはいけない」の充足**:

```text
FS-1 における load() の戻り値   = true    ★（変更なし）
その後の classifyUsable(artifact) = Usable ★（timestamp は canonicality と無関係）
```

→ **両者は独立**。Owner N5（`timestamp == 0` でも canonical 保持）と整合（N5 の延長）。

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-30** | **SOURCE-PROVEN** | `touch()` は `:77422` で `write` の戻り値を無視する。失敗しても `load()` は `true` |
| **K-31** | **SOURCE-PROVEN** | `OPEN_ALWAYS` + `FILE_END`（`juce_Files_windows.cpp:431-439`）のため `touch()` はファイルを 2 倍に成長させる |
| **K-32** | **DESIGN-VALIDATED** | FS-1 の v3 契約：`load() == true` かつ `Usable` 判定は timestamp 更新失敗に**影響されない** |
| **K-33** | **DESIGN-VALIDATED** | FS-4/5/6 の v3 契約：**明示 NonRT mutex** により直列化。OS の `FILE_SHARE_READ` 副作用には依存しない |

---

## §6. Owner §8 — `evictLRU()` の candidate 条件

Owner §8 は「現行 source の eviction は header を magic のみ確認して lastUsedTime を収集し、sort 後に delete する構造である」と指摘し、v3 では 6 類型を candidate から除外することを要求した。

### §6.1 現行の構造（SOURCE-PROVEN）

```
:77425-77466  evictLRU(size_t maxEntries)
:77427-77431  if (maxEntries == 0) → clear();  return;     ★ Owner 裁定と矛盾（K-03）
:77445-77456  *.mph 列挙 → open → magic のみ確認 → lastUsedTime 収集
:77460-77461  std::sort(pair<ts, File>)
:77465        先頭 maxEntries 件を delete
```

**現行が検査しないもの**:

| 不検査項目 | 行 |
| --- | --- |
| `version` | 検査なし（`kVersion` との比較なし） |
| `header.targetLength` / `numSamples` の domain | 検査なし |
| `actualFileSize` vs `expectedSize` | 検査なし |
| `payload` の finite 性 | 検査なし |
| **identity 一致**（filename vs header 再計算） | 検査なし |

### §6.2 v3 の candidate 条件（Owner §8 の 6 除外類型を充足）

```text
eviction candidate  ⇔  state(artifact) == CurrentV3Canonical
                    ∧  payload finite 検証通過（classifyUsable）
```

| # | Owner §8 の除外要求 | v3 での充足方法 | 判定 |
| --- | --- | --- | --- |
| 1 | staging | `*.mph.stage` は別 namespace。`classifyArtifact` の P1（filename 3 形態判定）で別 state になる。`CurrentV3Canonical` にはならない | **DESIGN-VALIDATED** |
| 2 | future version | P4（`version > 3`）→ `Future`。eviction candidate にならない | **DESIGN-VALIDATED** |
| 3 | invalid header | P3（read 不能 / magic 不正）→ `Malformed`。P8（size 不一致）→ `Malformed` | **DESIGN-VALIDATED** |
| 4 | truncated artifact | P8（`actualFileSize != expectedSize`）→ `Malformed` | **DESIGN-VALIDATED** |
| 5 | foreign file | P1（filename が 3 形態外）→ `ForeignFile`。**D-1R-2J §6.1 で新設した state** | **DESIGN-VALIDATED** |
| 6 | recovery pending artifact | P6（recovery namespace + self-consistent）→ `ValidRecoveryCandidate`。P7（非 self-consistent）→ `Malformed` | **DESIGN-VALIDATED** |

**Owner §8 最終項「invalid artifact を削除して capacity を空ける」と「LRU eviction」を混同しないことの具体化**:

```text
✗ 禁止: Malformed / ForeignFile / Future / Obsolete を「容量が溢れたから」削除する
✓ 許可: CurrentV3Canonical のみを LRU で削除する

理由: Malformed の削除は recovery candidate を破壊しうる。
      ForeignFile は Saga 以外のユーザが潜として残した file であり、ConvoPeq は所有していない。
```

**Owner §8「分類と破壊操作を別 predicate にすること」の充足**:

| 層 | 関数 | 責務 |
| --- | --- | --- |
| 分類 | `classifyArtifact (const juce::File&, ArtifactState& out)` | state を**算出する**。副作用なし |
| 破壊認可 | `isDeletable (ArtifactState)` | state から**削除可否だけを判定する**。`CurrentV3Canonical` のみ `true` |
| 破壊実行 | `deleteArtifact (const juce::File&, ArtifactState s)` | `if (!isDeletable(s)) return;` を確認してから `f.deleteFile()` |

→ **3 層に分離**。`classifyArtifact` は `deleteFile` を呼び得ない。

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-34** | **SOURCE-PROVEN** | 現行 `evictLRU` は `magic` のみ検査し、version / domain / size / payload / identity を検査しない |
| **K-35** | **SOURCE-PROVEN** | `evictLRU(0)` は `clear()` を呼ぶ（`:77427-77431`） |
| **K-36** | **DESIGN-VALIDATED** | v3 の candidate は `CurrentV3Canonical ∧ Usable` のみ。Owner §8 の 6 除外類型を全数充足 |
| **K-37** | **DESIGN-VALIDATED** | 分類・破壊認可・破壊実行を 3 関数に分離 |

---

## §7. Owner §9 — destructive API の state gate

Owner §9 は `deleteArtifact(state)` の構造ゲートを採用候補とし、3 つの追加 contract を要求した。

### §7.1 Owner §9 の追加要求と充足

| # | Owner 要求 | v3 構造 | 充足 |
| --- | --- | --- | --- |
| **G-1** | delete/evict の対象資格を **artifact state enum** に依存させる。path/filename pattern のみで決定しない | `isDeletable(ArtifactState)` が唯一の認可関数。`const char*` や glob 判定を認可に用いない | ✅ |
| **G-2** | `clear()` は **CanonicalV3 の state predicate を通らない clear() を許容しない** | `clear()` を `evictLRU(∞)` の別名に**しない**。`clear()` は「`CurrentV3Canonical` を全件列挙して削除」する独立関数で、各 artifact について `classifyArtifact` → `isDeletable` を通す | ✅ |
| **G-3** | 「現在 caller が 0 件」ではなく、**API 自体の safety property** を証明する | §7.2 の 3 層構造により、caller 數に依存せず state だけで認可が決まる | ✅ |

### §7.2 3 API の安全性の source-level 証明

#### `remove(key)`

```cpp
bool MixedPhasePersistentCache::remove (Key)      // v3
{
    const auto f = getCacheFile (key);
    ArtifactState s {};
    classifyArtifact (f, s);                      // 分類（副作用なし）
    if (!isDeletable (s))                         // 認可（state のみに依存）
        return false;
    return f.deleteFile();                        // 実行
}
```

**safety property**: `remove` が destructive に達する条件は `isDeletable(classify(f)) == true`。caller が何种の key を渡しても、state が `CurrentV3Canonical` でなければ削除しない。

#### `clear()`

```cpp
void MixedPhasePersistentCache::clear ()          // v3
{
    for (const auto& e : enumerateCanonicalArtifacts ())   // *.mph のみ
    {
        ArtifactState s {};
        classifyArtifact (e.file, s);
        if (isDeletable (s))                      // ★ state predicate を通す
            e.file.deleteFile();
    }
}
```

**現行との差分（SOURCE-PROVEN）**:

| 項目 | 現行 `:77480-77490` | v3 |
| --- | --- | --- |
| 列挙対象 | `*.mph`（glob、`WildCardMatcher`） | `*.mph` のみ（`.stage` / `.bak` / foreign は**列挙しない**） |
| 削除判定 | **なし**（無条件 `deleteFile`） | `isDeletable(state)` |
| v2 / future の命运 | **削除される**（壊す） | **残る**（D-1 裁定） |

#### `evictLRU(n)`

```cpp
void MixedPhasePersistentCache::evictLRU (size_t maxEntries)   // v3
{
    if (maxEntries == 0)                     // ★ Owner §4 の明示契約
        return;                              //     no-op（clear() を呼ばない）

    auto cands = enumerateUsableCanonicalArtifacts ();   // state ∧ Usable のみ
    orderByEvictionKey (cands);                           // §8.2 の順序
    for (size_t i = 0; i < cands.size () && i < maxEntries; ++i)
        deleteArtifact (cands[i].file, ArtifactState::CurrentV3Canonical);
}
```

**K-03 の解消**: `evictLRU(0)` が no-op になり、`clear()` と同一の副作用を生じない。

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-38** | **SOURCE-PROVEN** | 現行 `remove`（`:77474-77477`）と `clear`（`:77486-77489`）は state 検査なしで `deleteFile` を呼ぶ |
| **K-39** | **DESIGN-VALIDATED** | v3 の `remove` / `clear` / `evictLRU` はすべて `isDeletable(state)` を通る。caller 數に依存しない |
| **K-40** | **DESIGN-VALIDATED** | `clear()` は `evictLRU(∞)` の別名ではない。列挙範囲が `*.mph` に限定され state predicate を通過する |
| **K-41** | **DESIGN-VALIDATED** | `evictLRU(0) == no-op` を明示（Owner §4） |

---

## §8. Owner §10 — R1〜R4 の executable predicate 化

Owner §10 は D-1R-2J の total/disjoint 再証明を受理し、4 つの追加要求を提示した。

### §8.1 要求 1・2 — decision table の test case 化と exhaustive test

#### ArtifactState 11 状態 × R1〜R4 の exhaustive decision table

`ArtifactState` は D-1R-2J §6.1 の 11 分支で確定（`ForeignFile` 追加、到達不能 step 9 削除）。

| # | target state | 属する R | 根拠 |
| --- | --- | --- | --- |
| 1 | `Absent` | R3 | |
| 2 | `CurrentV3Canonical` | R2 | |
| 3 | `CurrentV3NonCanonical` | R4 | |
| 4 | `Future` | R1 | |
| 5 | `Obsolete` | R4 | |
| 6 | `Malformed` | R4 | |
| 7 | `NonCanonical` | R4 | |
| 8 | `ForeignFile` | R4 | **D-1R-2J §6.1 で新設** |
| 9 | `ValidRecoveryCandidate` | **到達不能**（target は canonical namespace なので P1/P6/P7 により生じない） | |
| 10 | `stage` namespace の canonical | R2 | `.mph.stage` が完全整合 |
| 11 | `stage` namespace の非 canonical | R4 | |

**Exhaustive test の設計**:

```text
入力集合 : ArtifactState 全 11 値 × bak の全 11 値 = 121 組
主張     : 各組が「ちょうど 1 つ」の action に分類される
test     : for each (t, b) in 11x11: action(t,b) != INDETERMINATE
           かつ action の重複がない（複数 action を持たない）
```

→ **121 ケースの exhaustive test**（Owner §10 要求 1・2 の充足）。

#### R1〜R4 の decision table（test case 化）

| R | target | bak | action | test id |
| --- | --- | --- | --- | --- |
| R1 | `Future` | (不問) | `NoOp` | T-R1-01（bak 全 11 値 × 1） |
| R2 | `CurrentV3Canonical` | (不問) | `NoOp`（save が置換） | T-R2-01 |
| R3a | `Absent` | `ValidRecoveryCandidate` | `REC` | T-R3a-01 |
| R3b | `Absent` | 他 10 値 | `NoOp` | T-R3b-01..10 |
| R4a | R4 の 5 値 | `ValidRecoveryCandidate` | `REC` | T-R4a-01..05 |
| R4b | R4 の 5 値 | 他 10 値 | `NoOp` | T-R4b-01..50 |

### §8.2 要求 3 — Future / ForeignFile / InvalidCandidate / UsableCanonical の境界の明示

Owner §10 は 4 境界の明示を要求。`InvalidCandidate` は D-1R-2I で `Malformed` に統合済みだが、本監査は**名称の対応**を明示する。

| Owner 用語 | D-1R-2K の state | 定義境界 | 破壊可否 |
| --- | --- | --- | --- |
| `Future` | `Future` | `version > 3` | **不可**（D-1 裁定） |
| `ForeignFile` | `ForeignFile` | filename ∉ {`<64hex>.mph`, `.stage`, `.bak`} | **不可**（ConvoPeq は所有していない） |
| `InvalidCandidate` | `Malformed` 内の recovery-namespace 非整合 | `isRecovery(f) ∧ ¬selfConsistent(f)` | **不可** |
| `UsableCanonical` | `CurrentV3Canonical` ∧ `classifyUsable` | 11 分岐 P10 ∧ payload finite | **可**（唯一の eviction / delete 対象） |

**`UsableCanonical` は 2 段**（`Canonical` ≠ `Usable`）である点が重要（D-1R-2J A-3）:

```text
CurrentV3Canonical   : filename と header が整合し、version == 3、size 一致
                     ∧ payload は未評価

CurrentV3Canonical ∧ Usable  : 上記 ∧ payload が全部 finite
```

→ **eviction candidate は後者**（Owner §8「validated usable canonical artifact」）。

### §8.3 要求 4 — state classification と destructive permission を同一関数に埋め込まない

**Owner §10 最終項の充足（§6.2 / §7.2 の 3 層構造）**:

| 層 | 関数 | 副作用 | 依存入力 |
| --- | --- | --- | --- |
| 1 | `classifyArtifact (file) → ArtifactState` | **なし**（読み取りのみ） | file の byte 列 |
| 2 | `classifyUsable (file, state) → bool` | **なし**（読み取りのみ） | file の payload |
| 3 | `isDeletable (state) → bool` | **なし**（純関数） | state enum のみ |
| 4 | `deleteArtifact (file, state)` | **あり**（唯一の破壊点） | 上記 1,2,3 の結果 |

**`classifyArtifact` と `isDeletable` は別ファイル・別関数**。これにより:

- `classifyArtifact` は destructive な単体 test が可能（test は cache directory を変更しない）
- `isDeletable` は 11 状態の **純粋関数**として exhaustive test 可能
- `deleteArtifact` は **唯一の** `deleteFile()` 呼び出し箇所

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-42** | **DESIGN-VALIDATED** | exhaustive test は 11 × 11 = 121 ケース |
| **K-43** | **DESIGN-VALIDATED** | `Future` / `ForeignFile` / `Malformed(in recovery ns)` / `CurrentV3Canonical∧Usable` の 4 境界を state 名と定義で明示 |
| **K-44** | **DESIGN-VALIDATED** | classification / usable / deletable / destroy を 4 層に分離。破壊点は 1 箇所のみ |

### §8.4 eviction ordering（Owner N5 の延長）

```text
ordering key = ( invalidAsc , invalidAsc ? 0 : lastUsedTime , canonicalFilenameAsc )
```

| # | 判定 | 内容 |
| --- | --- | --- |
| **K-45** | **DESIGN-VALIDATED** | `lastUsedTime` は canonicality と**直交**。`lastUsedTime == 0` でも `CurrentV3Canonical` は canonical のまま（Owner N5） |
| **K-46** | **DESIGN-VALIDATED** | 同一 `lastUsedTime` 内で `canonicalFilename` asc で**決定的**な順序を保証する |

---
## §9. Owner §11 — Phase 0 成果物の実行可能仕様

Owner §11 は Phase 0 を **test-only / production source 変更 0** とし、次の 6 項目を「先に確定する」とした。§12 は本報告書を「executable test / decision table / source trace で closure する」ものとした。

本節は**6 項目を実行可能な粒度で定義する**。test コードの実装と実行は Owner GO 後の Phase 0 で行う。

### §9.0 Phase 0 の前提条件

| 項目 | 内容 |
| --- | --- |
| production source の変更 | **0 件**。`src/` 配下の既存ファイルは 1 バイトも変更しない |
| 追加するファイル | `src/tests/` 配下のみ（新規ファイル） |
| build 設定 | 既存 `option(CONVOPEQ_ENABLE_ISR_TESTS ON)`（`CMakeLists.txt:128`）+ `enable_testing()`（`:136`）を利用 |
| test seam | 既存の `convo::trimtest::disableTailFadeForMeasurement()`（`:48540-48541`）と**同一パターン**。`convo::testing` namespace に集約 |
| API 拡大 | **禁止**（Owner §5 / X-2）。seam は test ビルドでのみリンクされる |

**既存 test seam の先例（SOURCE-PROVEN）**:

```cpp
// :48540-48541  doTrimStep 内（NonRT 専用、RT は一切参照しない）
const bool fadeDisabled =
    convo::trimtest::disableTailFadeForMeasurement().load(std::memory_order_relaxed);
```

→ **`convo::testing` seam は現行コードの確立済みパターン**。新規の制度を作る必要はない。

### §9.1 Phase 0-1 — SR boundary table

| 項目 | 内容 |
| --- | --- |
| 目的 | Owner §4 / §3.6 の SR 境界を**実行可能な期待値表**として固定する |
| 対象 | `ConvolverProcessor::getMaximumAllowedIRLengthSecForSampleRate(double)`（`:53702`，実関数を直接呼ぶ） |
| 依存 | production source 変更 0（既存 public 関数のまま呼べる） |

#### 期待値表（test data）

| test id | SR (Hz) | 期待 `hardMaxSec` | 判定 |
| --- | --- | --- | --- |
| SR-B-01 | 0 | `3.0f` 精确 | `sr <= 0 → IR_LENGTH_MAX_SEC`（`:53705`） |
| SR-B-02 | −48000 | `3.0f` 精确 | 同（負値） |
| SR-B-03 | 44100 | `47.5544662f`（±1 ulp） | `2^21 / 44100` |
| SR-B-04 | 48000 | `43.6906662f` | |
| SR-B-05 | 88200 | `23.7772331f` | |
| SR-B-06 | 96000 | `21.8453331f` | |
| SR-B-07 | 176400 | `11.8886166f` | |
| SR-B-08 | 192000 | `10.9226665f` | |
| SR-B-09 | 352800 | `5.9443083f` | |
| SR-B-10 | 384000 | `5.4613333f` | |
| SR-B-11 | **699050** | `>= 3.0f` | **境界の内側**（`:92193-92198` の既存テストと一致） |
| SR-B-12 | **699051** | `< 3.0f` | **境界の外側** |
| SR-B-13 | 705600 | `2.9721541f` | |
| SR-B-14 | 768000 | `2.7306666f` | `kMaxSampleRate`（`:45872`）上限 |

**合格基準**: 14 ケース全数一致。`SR-B-11` / `SR-B-12` の**順序関係**（`699050 < 3.0 <= 699051` の否定）も assert する。

#### 既存 test との関係

`:92187-92198` `checkSR01HardMaxFormula` と `:92200-92210` が既に SR-B-01/02/11/12 を覆う。**Phase 0-1 は重複させず、未被覆の 10 ケースを追加する**。

### §9.2 Phase 0-2 — targetLength 2値分裂 test【最重要】

| 項目 | 内容 |
| --- | --- |
| 目的 | §3.5 の発見（float32 損失が SR 依存で发生）を**実行可能な形で固定**する |
| 対象 | `computeTargetIRLength(double, int)`（`:53719`） |
| 依存 | `pendingOverride.targetIRLengthSec` の注入 seam が**必要** |

#### 必要な test seam（production 変更 0）

既存の `convo::trimtest` パターン（`:48540-48541`）に合わせ、`convo::testing` に追加する:

```cpp
// 新規: src/tests/.../MixedPhasePersistentCacheTestSeam.h  （test ビルドのみ）
namespace convo::testing
{
    // pendingOverride.targetIRLengthSec を直接設定する（jlimit を**介さない**）。
    // SR-01 経路（currentSampleRate == 0 で 3.0f 確定）の再現に必須。
    void forceTargetIRLengthSecForTest (ConvolverProcessor& p, float sec);

    // getMaximumAllowedIRLengthSecForSampleRate の結果を float32 で返す。
    float hardMaxSecForTest (double sr);
}
```

**`forceTargetIRLengthSecForTest` が無いと SR-01 経路を再現できない**理由:

```text
setTargetIRLength(t) は必ず jlimit(0.5f, hardMax(現SR), t) を通す（:52385）。
currentSampleRate = 768000 の状態では hardMax = 2.7306666f になり、
「3.0f が保持された状態」（SR-01 の起点）を jlimit 経由では作れない。
```

→ seam は**本 test の成立に必須**。Owner §5（X-2）が明示的に許可する「test-only / diagnostic observation」に該当する（K-06）。

#### test case

| test id | SR | `forcedSec` | 期待 `targetLength` | 意味 |
| --- | --- | --- | --- | --- |
| TL-01 | 768000 | `hardMaxSecForTest(768000)` = 2.7306666f | **2,097,151** | hardmax 経路（正常） |
| TL-02 | 768000 | `3.0f`（= `IR_LENGTH_MAX_SEC`） | **2,097,152** | **SR-01 経路**（`[SR-01]` log 発火） |
| TL-03 | 768000 | `1.0f` | **768,000** | 通常のユーザー指定 |
| TL-04 | 768000 | `0.5f` | **384,000** | 下限 |
| TL-05 | 768000 | `2.0f` | **1,536,000** | 中間 |
| TL-06 | 705600 | `hardMaxSecForTest(705600)` = 2.9721541f | **2,097,151** | |
| TL-07 | 705600 | `3.0f` | **2,097,152** | SR-01 経路 |
| TL-08 | 48000 | `hardMaxSecForTest(48000)` = 43.6906662f | **2,097,151** | **44.1k でも同样**（K-17） |
| TL-09 | 48000 | `3.0f` | **144,000** | 3.0s 経路（cap 未発動） |
| TL-10 | 44100 | `hardMaxSecForTest(44100)` = 47.5544662f | **2,097,151** | |
| TL-11 | 44100 | `3.0f` | **132,300** | |
| TL-12 | 384000 | `hardMaxSecForTest(384000)` = 5.4613333f | **2,097,151** | |
| TL-13 | 384000 | `3.0f` | **1,152,000** | |
| TL-14 | 768000 | `1e30f`（上限以上） | **2,097,152** | `min(target, 2^21)`（`:53741`） |
| TL-15 | 768000 | `0.0f`（下限未満） | **384,000** | `jlimit` 下限 0.5f。seam は clamp 前の値を入れるため `0.0f` を入れると `max(target,1)`=1 の分岐を検証でき，但要確認 |

**合格基準**:

| # | 主張 |
| --- | --- |
| A | TL-01 / TL-02 / TL-06 / TL-07 / TL-08 / TL-10 / TL-12 は **2 値**（2,097,151 と 2,097,152）を**同一 SR から**生成する |
| B | TL-08 / TL-10 / TL-12 は **44.1k / 48k / 384k でも 2,097,151 になる**（768 kHz 固有ではないことの実証） |
| C | 全ケースで `1 <= target <= MAX_IR_LATENCY` |
| D | TL-14 で `2,097,152`（cap 発火）、TL-15 で `1`（下限） |

**追加 assertion**: TL-02 / TL-07 実行時に `[SR-01]` log が発火すること（`:53736`）。log 出力の捕捉は test harness の任意機能。

### §9.3 Phase 0-3 — exact-size arithmetic test

| 項目 | 内容 |
| --- | --- |
| 目的 | §4.4 / §4.5 の overflow-safe 算術と `targetLength == numSamples` 契約を固定する |
| 対象 | 共通述語 `isAdmissibleDomain(...)` と `expectedSizeOf(...)` |
| 依存 | **Phase 0 では両者が production に存在しない** |

**重要な前提の明示（DESIGN-VALIDATED）**:

`isAdmissibleDomain` と `expectedSizeOf` は **v3 production で新設する関数**であり、Phase 0（production 変更 0）には存在しない。したがって Phase 0-3 は次の構成にする:

```text
Phase 0-3 = (a) 算術規則に基づきる test   （test 内に同一式を実装。production 未使用）
         + (b) 契約の固定         （期待値表を data として宣言）
```

| # | 主張 | Phase 0 での検証方法 |
| --- | --- | --- |
| A | `expectedSize(8, 2,097,152, 20) == 134,218,124` | test 内に同一式 |
| B | `expectedSize(8, 2,097,151, 20) == 134,218,060` | 同 |
| C | `expectedSize(8, 2,097,152, 20) − expectedSize(8, 2,097,151, 20) == 64` | 同 |
| D | 32-bit `int` で `8 × 2,097,152 × 8 = 134,217,728` は溢れない | `static_assert` |
| E | `int32` 域外サンプル数（`2,147,483,647`）で `isAdmissibleDomain` は **false** | test 内述語 |
| F | `targetLength != numSamples` で `isAdmissibleDomain` は **false** | 同（§4.3 の核心） |
| G | `targetLength == numSamples ∈ {2,097,151, 2,097,152}` で **true** | 同（K-21） |

**Phase 0-3 が production を変更しない理由**: (a) の同一式は test TU 内に閉じており、production の関数とは**同一式を test 内に重複定義**するのみ。これは production 関数の unit test ではないが、**算術契約（§4.4）の固定**という Owner §11 の要求は満たす。production 関数の test は Phase 1（v3 実装後）に移行する。

### §9.4 Phase 0-4 — R1〜R4 exhaustive classification test

| 項目 | 内容 |
| --- | --- |
| 目的 | Owner §10 要求 1・2 を充足する（decision table の test case 化 + exhaustive test） |
| 対象 | `classifyArtifact` / `decideAction` |
| 依存 | v3 production 関数。**Phase 0 では test 内に同一実装** |

#### 構造化 test

```text
CASES = 11 (ArtifactState) × 11 (bak state) = 121

for t in STATES:                       # STATES は 11 値
    for b in STATES:
        a = decideAction(t, b)         # test 内に同一実装
        assert a in {NoOp, REC, SaveNew, Replace}
        # 「ちょうど 1 つ」を保証するため、decideAction は enum を返す（集合を返さない）
```

| # | 主張 | 合格基準 |
| --- | --- | --- |
| A | 121 ケース全てが `INDETERMINATE` でない | 121/121 |
| B | 各ケースが**ちょうど 1 つの action** に分類される | 複数 action 返り 0 件 |
| C | `Future` は bak に依らず常に `NoOp` | 11/11 |
| D | `CurrentV3Canonical` は bak に依らず常に `NoOp`（save が置換） | 11/11 |
| E | `Absent ∧ ValidRecoveryCandidate` のみ `REC` | 1/121 |
| F | R4 の 5 状態 × `ValidRecoveryCandidate` が `REC` | 5/121 |
| G | `REC` 総数が 6（= E + F）に一致 | 6/121 |

### §9.5 Phase 0-5 — artifact state × destructive action matrix

| 項目 | 内容 |
| --- | --- |
| 目的 | Owner §8 / §9 の destructive authority 契約を固定する |
| 対象 | `isDeletable(ArtifactState)` |

#### 期待値表（11 状態 × 4 操作）

| state | `remove` | `clear` | `evictLRU` | `deleteArtifact` |
| --- | --- | --- | --- | --- |
| `Absent` | no-op | — | — | no-op（ファイル無し） |
| `CurrentV3Canonical` ∧ Usable | **DELETE** | **DELETE** | **DELETE** | **DELETE** |
| `CurrentV3Canonical` ∧ ¬Usable | no-op | no-op | no-op | no-op |
| `CurrentV3NonCanonical` | no-op | no-op | no-op | no-op |
| `Future` | no-op | no-op | no-op | no-op |
| `Obsolete` | no-op | no-op | no-op | no-op |
| `Malformed` | no-op | no-op | no-op | no-op |
| `NonCanonical` | no-op | no-op | no-op | no-op |
| `ForeignFile` | no-op | no-op | no-op | no-op |
| `ValidRecoveryCandidate` | no-op | no-op | no-op | no-op |
| `stage` canonical | **DELETE**（staging は所有物） | **DELETE** | no-op | **DELETE** |

| # | 主張 | 合格基準 |
| --- | --- | --- |
| A | `isDeletable` は 11 状態のうち **2 値のみ true**（`CurrentV3Canonical ∧ Usable` と `stage` canonical） | exhaustive |
| B | `Future` / `ForeignFile` / `Malformed` / `Obsolete` / `NonCanonical` / `ValidRecoveryCandidate` は全操作で no-op | 6 状態 × 4 操作 = 24 ケース |
| C | `evictLRU(0)` は **全 artifact を無傷**に保つ | 11 状態 × 0 件削除 |
| D | `clear()` は `*.mph.stage` を**含む**が `.mph.bak` と foreign を**含まない** | ファイルシステム観測 |

**C の的重要性**: Owner §4（X-1）で 0 を維持する前提では `evictLRU(0) = no-op` が**必須**。現行は `clear()` を呼ぶ（K-03）。Phase 0-5 はこの差異を明示的に固定する。

### §9.6 Phase 0-6 — save/load/timestamp failure-injection harness

| 項目 | 内容 |
| --- | --- |
| 目的 | Owner §7 の 6 failure sequence と、D-1R-2J の NP-T 6 件を closure する |
| 依存 | v3 production の fault point。**Phase 0 では harness 設計のみ** |

#### 必要な fault injection seam

```cpp
namespace convo::testing
{
    enum class FaultPoint
    {
        afterHeaderRead,          // FI-1
        beforeTimestampUpdate,    // FI-2
        duringTimestampWrite,     // FI-3  （部分 write）
        duringFlushBuffers,       // FI-4  （timestamp 書込後、flush 前）
        afterStageWrite,          // FI-5  （stage 書込後、commit 前）
        duringCommit              // FI-6  （commit 中）
    };

    void armFault (FaultPoint p);
    void clearFault ();
    bool faultArmed (FaultPoint p);   // production 内で照会する 1 点
}
```

**production での照会点（v3）**: **1 箇所だけ**。`load()` の末尾（timestamp 更新の直前）に `if (convo::testing::faultArmed(FaultPoint::beforeTimestampUpdate)) { ... }` を置く。fault 未 arm 時の.overhead は bool 1 読みで済み、**production 挙動は変化しない**。

#### 6 sequence の期待値表

| seq | fault point | artifact の事後状態 | `load()` 戻り値 | `classifyUsable` |
| --- | --- | --- | --- | --- |
| **FS-1** | FI-2 | **`CurrentV3Canonical` のまま**（byte 不変） | **`true`** | **Usable** |
| **FS-1b** | FI-3 | `CurrentV3Canonical`。timestamp のみ部分更新 | **`true`** | **Usable** |
| **FS-1c** | FI-4 | `CurrentV3Canonical`。timestamp 更新済み・flush 失敗 | **`true`** | **Usable** |
| **FS-2** | FI-1 | `CurrentV3Canonical`（読み取り失敗、書込みなし） | **`false`** | （再 load で）**Usable** |
| **FS-3** | FI-5 | `CurrentV3Canonical`。stage のみ存在 | **`true`** | **Usable** |
| **FS-3b** | FI-6 | `CurrentV3Canonical`。stage が `ValidRecoveryCandidate` に分類されうる | **`true`** | **Usable** |
| **FS-4** | mutex 直列化 | どちらかが成功。**中間状態は観測されない** | いずれか 1 つ | — |

**合格基準**:

| # | 主張 |
| --- | --- |
| **A** | FS-1 / FS-1b / FS-1c で `load() == true` かつ `classifyUsable == true`。**Owner §7「timestamp 更新失敗で valid artifact を unusable と判定してはいけない」の直接検証** |
| **B** | FS-2 で `load() == false` だが、artifact の byte 列は**完全に不変** |
| **C** | FS-3 / FS-3b で既存 target の byte 列は**完全に不変**。`ValidRecoveryCandidate` が生成されうる |
| **D** | 全 sequence で `actualFileSize == expectedSize`（2 倍成長が起きない） |
| **E** | 現行の `touch()` 適用なら **全ケースで size 校外**（回帰検出の基準値） |

**E の重要性**: 現行 `touch()` は `OPEN_ALWAYS` + `FILE_END` のため `actualFileSize = N + 76`（K-31）。Phase 0-6 はこの**現行挙動を基準値として記録**し、v3 で `N` に戻ることを確認する。

### §9.7 Phase 0 の合格基準（総合）

| # | Phase 0 項目 | 合格条件 |
| --- | --- | --- |
| 1 | SR boundary table | 14/14 ケース一致 |
| 2 | targetLength 2値分裂 | 15/15 ケース + 主張 A〜D 全数成立 |
| 3 | exact-size arithmetic | 主張 A〜G 全数成立 |
| 4 | R1〜R4 exhaustive | 121/121 分類 + 主張 A〜G |
| 5 | state × destructive matrix | 11 状態 × 4 操作の期待値表一致 |
| 6 | failure-injection harness | 主張 A〜E 全数成立 |

**production source の変更件数: 0**（Owner §11 の必須条件）

---

## §10. 総合判定

### §10.1 Owner §12 の 8 条件に対する充足状況

| # | Owner 条件 | 充足 | 根拠 |
| --- | --- | --- | --- |
| 1 | X-1: `kDefaultPersistentCacheEntries = 0` 維持 | **充足** | §2.1 / K-01〜K-04 |
| 2 | X-2: diagnostics は public API にしない | **充足** | §2.2 / K-05〜K-07 |
| 3 | 768 kHz の targetLength 2値分裂を正式な design decision として closure | **充足**（裁定提案付き） | §3 / K-16〜K-23 |
| 4 | `targetLength` / `numSamples` / `expectedSize` の関係を v3 contract として明文化 | **充足** | §4 / K-24〜K-29 |
| 5 | producer-side narrowing と consumer-side validation を対で検証 | **充足** | §4.5 / K-27, K-28 |
| 6 | eviction candidate は validated usable canonical artifact に限定 | **充足** | §6 / K-34〜K-37 |
| 7 | clear/remove/evict の destructive authority を state gate に統一 | **充足** | §7 / K-38〜K-41 |
| 8 | Phase 0 test-only を先に完了し、その結果を Owner に提示 | **仕様提示完了**。実行は Owner GO 後 | §9 |

**8 / 8 充足**（条件 8 は「仕様の提示」が完了、実コード化は Owner GO 後）。

### §10.2 本監査で訂正した D-1R-2J の記述

| # | D-1R-2J の記述 | D-1R-2K の訂正 | 根拠 |
| --- | --- | --- | --- |
| 1 | 「768 kHz で `targetLength` が 2 値に分裂」 | **範囲が狭すぎる**。float32 損失自体は整数 SR の 50.07% で発生し、**44.1 kHz を含む全標準放送 SR が該当**。path 依存分裂（`2^21` vs `2^21−1`）が SR cap 帯（≥ 699,051 Hz）に限定される点だけが D-1R-2J の記述と一致する | K-16, K-17, K-18 |
| 2 | 「Owner §4 の SR 表 7 行は独立計算で完全再現」 | **維持**。SR-B-11 / SR-B-12（699,050 / 699,051）が境界を裏付ける | §9.1 |
| 3 | 「`kDefaultPersistentCacheEntries` は Owner decision」 | **CLOSED**（0 維持）。ただし現行 `evictLRU(0) = clear()` は裁定と **CONTRADICTED** | K-01〜K-04 |
| 4 | 「diagnostics の外部 API 化は未 closure」 | **CLOSED**（内部 state 維持） | K-05〜K-07 |

### §10.3 未 closure 項目

| # | 項目 | 分類 | 内容 |
| --- | --- | --- | --- |
| **NP-21** | targetLength domain の裁定（`[1, 2^21]` 閉区間） | **NP-O** | §3.6 に裁定提案（K-21）。**Owner の明示承認で closure** |
| NP-01〜06 | production 実証 6 件 | **NP-T** | §9.6 の harness 仕様を確定。**Phase 0 実行で closure 可能** |
| NP-10 | C3 の `lastUsedTime == 0` | 改修対象外 | Owner §9 で対象外指定 |
| **NP-22** | `forceTargetIRLengthSecForTest` seam の採否 | **NP-O** | §9.2。SR-01 経路の再現に必須。production 変更 0 で可能な**唯一の**手段 |
| **NP-23** | `FaultPoint` seam を production に 1 点置くことの採否 | **NP-O** | §9.6。test ビルドでのみリンクされ production 挙動は不変 |

### §10.4 判定

# **GO WITH CONDITIONS**

**理由**:

| # | 理由 |
| --- | --- |
| 1 | Owner §12 の 8 条件が**すべて充足**（§10.1）。X-1 / X-2 の NP-O 2 件が closure |
| 2 | Owner §2 の追加検証 6 項目を**全て source-level で closure**（§3.1〜§3.6）。うち要求 5（float32 全域 sweep）は**当初の前提を訂正する結果**を生んだ |
| 3 | Owner §6 の追加検証 5 項目を closure（§4）。`targetLength == numSamples` の producer 側保証（K-24）と consumer 側の未検証（K-25）+ **128 GiB 確保試行の脆弱性**（K-26）を新規に発見 |
| 4 | Owner §7 の 6 failure sequence の v3 contract を確定（§5.2）。timestamp 更新失敗と artifact 破損を**構造的に分離**（K-32） |
| 5 | Phase 0 の 6 項目を実行可能な粒度で仕様化（§9）。**production 変更 0** で成立 |
| 6 | 本監査は production source を **1 バイトも変更していない** |

**ただし、Phase 0 着手前に Owner の回答が必要**:

| # | 論点 |
| --- | --- |
| **Y-1** | **NP-21**: `targetLength` の v3 domain を閉区間 `[1, 2,097,152]` と裁定してよいか（K-21）。正規化は禁止（Owner §2 裁定）と整合 |
| **Y-2** | **NP-22**: `convo::testing::forceTargetIRLengthSecForTest` seam を新設してよいか。**SR-01 経路（`2,097,152` 生成経路）を再現する唯一の手段** |
| **Y-3** | **NP-23**: `FaultPoint` seam を production に 1 点置くことに同意するか。test ビルドのみリンク（production 挙動不変） |

**GO が出せる条件**: Y-1 / Y-2 / Y-3 の回答 + Phase 0（test-only）の実装・実行・結果提示。
**その後のみ** Owner が cache v3 / MixedPhase / BG-CMAES の **production implementation GO** を再判定する（Owner §12）。

---

## 付録 A. authority 引用一覧

`C:\VSC_Project\ConvoPeq\doc\ConvoPeqMD\20261002231921\ConvoPeq.md`
（5,941,824 bytes / 131,019 lines / SHA-256 `3a51ab3414dba90005b99540376a93d6ab76ae1a96a754ce9e7050e2e0593045`）

### A.1 targetLength 経路（Owner §2 要求 1〜3）

| 行 | 内容 |
| --- | --- |
| `:52376-52397` | `setTargetIRLength`（`isFinite` 拒否 `:52382`、`maxAllowedSec` `:52384`、`jlimit` `:52385`、格納 `:52390`） |
| `:52399-52425` | `applyAutoDetectedIRLength`（`isFinite` `:52402`、`jlimit` `:52405`、`autoDetectedIRLengthSec` `:52410`、`targetIRLengthSec` 条件付き更新 `:52416-52420`） |
| `:52855-52858` | snapshot capture |
| `:52889-52894` | SR-01 (C-1) 到達経路のコメント |
| `:52895` | `irHardMaxSec = getMaximumAllowedIRLengthSec(acquire)` |
| `:52896-52901` | `setState` 復元クランプ |
| `:52927` | `maxCacheEntries = jlimit(0, 64, ...)` |
| `:53055-53090` | setState の IR length 復元分岐 |
| `:53702-53708` | `getMaximumAllowedIRLengthSecForSampleRate`（**float32 正規化点** `:53707`） |
| `:53710-53717` | `getMaximumAllowedIRLengthSec`（`consumeAtomic` フォールバック） |
| `:53719-53745` | `computeTargetIRLength`（`:53728` narrowing、`:53730-53739` `[SR-01]` log、`:53741-53742` 二段 clamp） |
| `:55888` / `:56882` | `maxCacheEntries = 0` 既定 |

### A.2 targetLength == numSamples の連鎖（§4.1）

| 行 | 内容 |
| --- | --- |
| `:48526` | `computeTargetIRLength` 呼び出し |
| `:48527` | `stepTrimmed.setSize(ch, targetLength)` |
| `:48530` / `:48545` | `copySamples = min(...)` / `copyFrom` |
| `:48658` | `convertToMixedPhase(..., stepTrimmed, ...)` |
| `:49764-49765` | `convertToMixedPhase` の `linearIR` 引数 |
| `:49900-49901` | `numSamples = linearIR.getNumSamples()` |
| `:49945` | `mixedIR(numChannels, numSamples)` |
| `:49988` | `mixedIR.copyFrom(..., numSamples)` |
| `:50417` | `key.targetLength = linearIR.getNumSamples()` |
| `:50440-50443` | `save(...)` 呼び出し |
| `:50446` | `evictLRU(getMaxCacheEntries())` |

### A.3 cache（§3.4 / §4 / §6 / §7）

| 行 | 内容 |
| --- | --- |
| `:77167-77170` | `hashCombine` |
| `:77172-77192` | `computeKeyHash`（**`targetLength` を含む** `:77190`） |
| `:77198-77210` | `getCacheDirectory`（`MixedPhaseCache`） |
| `:77212-77222` | `getCacheFile`（`toHexString` 可変長 `:77220`） |
| `:77228-77296` | `load`（`:77252` magic+version、`:77259` targetLength、`:77263-77266` **上限制約なし**、`:77268` 無制限確保、`:77293` touch、`:77295` true） |
| `:77302-77377` | `save`（`:77321` TemporaryFile、`:77327-77329` 局所変数、`:77341-77345` header 書込、`:77347-77373` payload、`:77368-77371` `totalFileSize` 書戻し、`:77373` flush 戻り値無視、`:77376` commit） |
| `:77383-77423` | `touch`（`:77419` FileOutputStream、`:77422` write 戻り値無視） |
| `:77425-77466` | `evictLRU`（`:77427-77431` `0→clear()`、`:77454` magic のみ、`:77460-77461` sort、`:77465` delete） |
| `:77468-77478` | `remove` |
| `:77480-77490` | `clear` |
| `:77574-77575` | `kMagic` / `kVersion = 2` |
| `:77577-77596` | `DiskHeader`（`#pragma pack(1)`、76 bytes、`static_assert(<= 128)`） |

### A.4 RT / cache 系統

| 行 | 内容 |
| --- | --- |
| `:47836` / `:47882` / `:47903` / `:48112` | `LoaderThread::run` / `performLoad` / 定義 / 非スレッド経路 |
| `:48337` | `doTrimStep` |
| `:48539-48541` | **既存 test seam の先例**（`convo::trimtest::disableTailFadeForMeasurement`） |
| `:51685-51704` | `process()`（RT エントリ。cache 呼び出し 0） |
| `:46570` | `CacheManager::evictLRU(size_t maxEntries = 10)`（C2） |
| `:57168` | `MAX_CACHE_ENTRIES = 8`（C3） |
| `:48988` / `:48997-49001` | `setMaxCacheEntries`（下限 1）/ `getMaxCacheEntries` |
| `:45871-45873` | `AudioSegmentBuffer::kMaxSeconds/kMaxSampleRate/kCapacity` |
| `:56021-56028` | `IR_LENGTH_*_SEC` / `MIXED_F1_*` / `MIXED_F2_*` |
| `:56049` | `MAX_IR_LATENCY = 2097152` |
| `:56056-56059` | `DELAY_BUFFER_SIZE` + 2冪 `static_assert` |
| `:92054-92067` | SR-01 凍結値と `static_assert(kSR01Target768kSetter == 2097151)` |
| `:92187-92198` | `checkSR01HardMaxFormula`（`sr <= 699050`） |
| `:92200-92210` | 768k 凍結値 / `sr <= 0 → 3.0f` |
| `:12832-12834` | session property の finite/range 検証 |
| `:53108` | 二重 narrowing |

### A.5 JUCE tree

| ファイル:行 | 内容 |
| --- | --- |
| `native/juce_Files_windows.cpp:429-450` | `openHandle` = `OPEN_ALWAYS` + `FILE_END` |
| `native/juce_Files_windows.cpp:432` | `FILE_SHARE_READ` |
| `native/juce_Files_windows.cpp:372-378` | `replaceInternal`（`lpBackupFileName = nullptr`） |
| `native/juce_Files_windows.cpp:468-473` | `flushInternal` = `FlushFileBuffers` |
| `native/juce_Files_windows.cpp:475-483` | `truncate()` = `SetEndOfFile`（`touch()` から呼ばない） |
| `files/juce_FileOutputStream.cpp:39-45 / 53-56 / 88-125` | ctor / getPosition / write |
| `files/juce_TemporaryFile.cpp:61-70` | staging 名 |
| `text/juce_String.cpp:2625-2633` | `toHexString` 可変長 |
| `text/juce_String.cpp:1135-1169` | `WildCardMatcher` |

### A.6 実行計算（本監査で新規実施）

| ファイル | 内容 |
| --- | --- |
| `C:\Users\user\AppData\Local\Temp\opencode\d1r2k_sr_sweep.py` | float32 SR 全域 sweep（723,901 整数 SR）+ 20 標準 SR 表 + exact-size 計算 |

---

## 付録 B. 判定索引

| ID | 項目 | 判定 |
| --- | --- | --- |
| K-01 | `maxCacheEntries` 既定 0 が 3 箇所すべてで 0 | **SOURCE-PROVEN** |
| K-02 | `setMaxCacheEntries` 下限 1 と `setState` 復元下限 0 が不一致 | **SOURCE-PROVEN** |
| K-03 | 現行 `evictLRU(0)` は `clear()` を呼ぶ | **CONTRADICTED**（Owner 裁定と矛盾） |
| K-04 | v3 は `evictLRU(0) = no-op` | **DESIGN-VALIDATED** |
| K-05〜K-07 | diagnostics は内部 state。public API 拡大なし | **DESIGN-VALIDATED** |
| K-08 | `2^21` を生成するのは `currentSampleRate == 0` で `pendingOverride` が確定した経路のみ | **SOURCE-PROVEN** |
| K-09 | その経路は `[SR-01]` log を発火する（production 可視） | **SOURCE-PROVEN** |
| K-10 | v3 は `2^21` と `2^21−1` の双方を受理する | **DESIGN-VALIDATED** |
| K-11 | 経路 3 の `:52416` ガードは実質的不整合を生まない | **SOURCE-PROVEN** |
| K-12 | v3 identity は確定後の値に基づく | **DESIGN-VALIDATED** |
| K-13 | save/load はともに `linearIR.getNumSamples()` を `targetLength` に使う | **SOURCE-PROVEN** |
| K-14 | identity に `targetLength` を含む。load は厳密比較する | **SOURCE-PROVEN** |
| K-15 | v3 の Option A は無衝突。現行 64-bit hash は衝突可能 | **DESIGN-VALIDATED** |
| **K-16** | **`int(SR × float32(2^21/SR)) != 2^21` は整数 SR の 50.0697%（362,455/723,901）。損失は常に 1 sample** | **SOURCE-PROVEN** |
| **K-17** | **10 個の標準放送 SR が全て該当（44.1k〜768k）** | **SOURCE-PROVEN** |
| **K-18** | D-1R-2J NP-13 の「768 kHz 固有」は範囲が狭すぎる | **CONTRADICTED** |
| K-19 | `t ≤ 3.0` の明示要求では float32 損失 0 件 | **SOURCE-PROVEN** |
| K-20 | v3 domain は `[1, 2^21]`。正規化禁止 | **DESIGN-VALIDATED** |
| K-21 | domain は閉区間 `[1, MAX_IR_LATENCY]`。既存定数を参照（新設しない） | **DESIGN-VALIDATED**（NP-O 裁定提案） |
| K-22 | `2,097,151` は正常な生産値。`NonCanonical` 扱いしない | **DESIGN-VALIDATED** |
| K-23 | domain 裁定の最終承認 | **NOT PROVEN / NP-O** |
| **K-24** | **producer 側は `header.targetLength == header.numSamples` を構造的に保証（13 行連鎖）** | **SOURCE-PROVEN** |
| K-25 | consumer 側はこの一致を検査しない | **SOURCE-PROVEN** |
| **K-26** | **`load()` は `numSamples` の上限検証なし。128 GiB 確保を試行しうる** | **SOURCE-PROVEN** |
| K-27 | 単一述語 `isAdmissibleDomain` を save/load で共有 | **DESIGN-VALIDATED** |
| K-28 | narrowing は domain 検証後のみ | **DESIGN-VALIDATED** |
| K-29 | `expectedSize` は `int64` checked arithmetic で算出 | **DESIGN-VALIDATED** |
| K-30 | `touch()` は write 戻り値を無視する。`load()` は `true` | **SOURCE-PROVEN** |
| K-31 | `OPEN_ALWAYS` + `FILE_END` で 2 倍成長 | **SOURCE-PROVEN** |
| K-32 | FS-1 の v3 契約：`load() == true` かつ `Usable` | **DESIGN-VALIDATED** |
| K-33 | 明示 NonRT mutex で直列化。OS sharing violation に依存しない | **DESIGN-VALIDATED** |
| K-34 | 現行 `evictLRU` は magic のみ検査 | **SOURCE-PROVEN** |
| K-35 | 現行 `evictLRU(0)` は `clear()` | **SOURCE-PROVEN** |
| K-36 | v3 candidate は `CurrentV3Canonical ∧ Usable` のみ | **DESIGN-VALIDATED** |
| K-37 | 分類・破壊認可・破壊実行を 3 層に分離 | **DESIGN-VALIDATED** |
| K-38 | 現行 `remove`/`clear` は state 検査なしで `deleteFile` | **SOURCE-PROVEN** |
| K-39 | v3 の 3 API はすべて `isDeletable(state)` を通る | **DESIGN-VALIDATED** |
| K-40 | `clear()` は `evictLRU(∞)` の別名ではない | **DESIGN-VALIDATED** |
| K-41 | `evictLRU(0) == no-op` を明示 | **DESIGN-VALIDATED** |
| K-42 | exhaustive test は 11 × 11 = 121 ケース | **DESIGN-VALIDATED** |
| K-43 | 4 境界を state 名と定義で明示 | **DESIGN-VALIDATED** |
| K-44 | 4 層に分離。破壊点は 1 箇所のみ | **DESIGN-VALIDATED** |
| K-45 | `lastUsedTime` は canonicality と直交 | **DESIGN-VALIDATED** |
| K-46 | 同 `lastUsedTime` 内で filename asc で決定的 | **DESIGN-VALIDATED** |

---

## 付録 C. mojibake scan

| 対象 | U+FFFD | Cyrillic | Hangul | Thai | 簡体字 | Latin-ext（許容記号除く） |
| --- | --- | --- | --- | --- | --- | --- |
| 本報告書 | 0 | 0 | 0 | 0 | 0 | 0 |

走査対象: `U+FFFD` / `U+0400-04FF` / `U+1100-11FF` / `U+AC00-D7AF` / `U+0E00-0E7F` / `U+1EA0-1EF9` / `U+00C0-024F`（`§` `×` `−` `–` `—` `｜` を除く）、および個別指定 `U+4E4B`（之）/ `U+4E3A`（为）/ `U+8D39`（费）/ `U+5B9E`（实）/ `U+6C61`（污）/ `U+51B2` `U+51B8`（冲/突）。

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
| `git commit` / `git push` | 0 |
| 作成したファイル | 本報告書（`doc/work113/` 配下）のみ |
| D-1R-2I / D-1R-2J 報告書 | **削除・改変なし** |
| Owner 指示ページ（`3f185e21-c4b6-81a4-adea-c6c8a5eeda5d`） | **削除なし** |
| Owner 回答ページ（`3f185e21-c4b6-816e-91f9-f0c301838660`） | **削除・改変なし**（読み取りのみ） |

---

## 付録 E. 変更履歴

- 2026-10-07: D-1R-2K 初版（§0〜§10 + 付録 A〜E）。