# STG-11-D12 Fresh Discovery — Session-State Boundary & Integer-Domain Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D12_FRESH-DISCOVERY_20261001.md`
- Work item: **STG-11-D12** — session/state 境界・整数ドメインの残存 invalid-state 監査
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20261001-054723).md` と同一。
  D11 マーカー 8 件・FRESH を作業開始時に確認。過去資料・記憶・旧行番号は未使用）
- Commit / push: **禁止**

---

## 0. 判定

```
NO-GO — no concrete defect
```

session / ValueTree / UI から runtime に侵入する不正値の全域を field 単位で
追跡した結果、D12 の GO 条件（7 項目）をすべて満たす新規 defect は存在しない。
jlimit の存在だけを安全根拠にせず、各 field について
input → setter → storage → snapshot → consumer → harm まで実コードで確認した。
D7/D8/D9/D10/D11 の再報告ではない。

D1〜D11 の未 commit 状態はそのまま保持する。Implementation / Gate は作成しない。

---

## 1. Field-by-field trace（全件）

凡例：`S` = session 入力可、`G` = setter ガード、`N` = 中和点、`H` = harm なし確定。

### 1.1 Enum / mode（整数 OOB）

| field | input→setter→storage | consumer | invalid handling | verdict |
| --- | --- | --- | --- | --- |
| processingOrder | S→D7 guard→atomic | 等価比較のみ | OOB 到達不能 | SAFE（D7） |
| analyzerSource | S→D7 guard→atomic | 等価比較のみ | OOB 到達不能 | SAFE（D7） |
| convHC/LC/eqLPF | S→D7 guard→atomic | RT 配列 index | OOB 到達不能 | SAFE（D7） |
| noiseShaperType | S→B-3 guard→atomic | 等価 chain＋validator{0..3} | OOB 到達不能 | SAFE |
| oversamplingType | S→B-3 guard→atomic | 同上 | OOB 到達不能 | SAFE |
| EQ band type/channel | S→D10 guard→state | switch fallthrough／等価 chain | OOB 到達不能 | SAFE（D10） |
| phaseMode/tailMode/nucHC/nucLC/filterStructure | S→int jlimit/clamp→store | 等価比較／switch | int-domain 有界 | SAFE |
| tailL1L2Multiplier | S→[2,16] clamp→store | `l1Part=l0Part*mult`（int 演算）＋MKL `jlimit(2,16)` | 有界 | SAFE |

### 1.2 Float / double（NaN・Inf・極端値）

| field | input→setter→storage | consumer | invalid handling | verdict |
| --- | --- | --- | --- | --- |
| targetIRLengthSec/auto | S→D8 finite guard→store | `computeTargetIRLength`→trim | 非有限到達不能 | SAFE（D8） |
| tailStartSec/tailStrength | S→D9 finite guard→store | MKL gain／形状 | 非有限到達不能 | SAFE（D9） |
| totalGain | S→D11 finite guard→store | ramp／prepare | 非有限到達不能 | SAFE（D11） |
| mix/smoothingTime/dB gains/saturation | S→jlimit＋abs-gate→store | DSP | NaN は格納されず旧値維持（比較偽）。Inf は clamp | SAFE |
| nonlinearSaturation | S→jlimit→store（NaN 格納可） | `saturation > 0.0` が偽→適用 skip | NaN は適用されず無害 | SAFE（accidental だが確定） |
| EQ freq/gain/Q | S→無検証 store→state | `calcSVFCoeffs` 系 `isfinite`→bypass 係数 | NaN は bypass 化 | SAFE |
| adaptive coeffs | S→無検証 store→bank | `clampCoeff`（非有限→0、±0.85） | NaN/Inf/極値は中和 | SAFE |
| mixedF1/F2 | S→無検証 store→snapshot | `convertToMixedPhase`→`validateBuffer` が NaN 出力拒否 | NaN 出力は破棄（前段維持） | SAFE |
| cmaesRestarts | S→int store | learner loop 上限（cancellable worker） | 負値は skip、巨大値は cancel 可能 | ACCEPTABLE（harm なし） |
| coeffSafetyMargin | S→double store | `clampCoeff(v, margin)`＋stability check（既定 ON） | 有限入力＋既定 ON で安定 | ACCEPTABLE |
| ditherBitDepth | S→無検証 store→atomic | publish は validator{0,16,24,32} が拒否。placeholder prepare は bypass 付き | DSP 到達は validator 通過分のみ | SAFE |

### 1.3 Size / count / index（0・負・巨大・符号変換）

| field | handling | verdict |
| --- | --- | --- |
| maxCacheEntries | `[1,64]` clamp（NaN→INT_MIN→size_t 巨大→64 に clamp） | SAFE |
| targetUpgradeFFTSize | allowlist 写像（512/1024/2048/4096。NaN→512） | SAFE |
| rebuildDebounceMs | `[10,3000]` clamp＋consumer `max(1,...)` | SAFE |
| band index | `0<=i<NUM_BANDS` 検査（NaN→INT_MIN で除外） | SAFE |
| block size/channels/sample rate/FFT | host 契約＋`PrepareBlockSizingPolicy`＋validator | session 経由なし |

### 1.4 Bool（意味反転）

`agcEnabled` / bypass 群 / `irLengthManualOverride` / `useMinPhase` /
`experimentalDirectHeadEnabled` / `enableStabilityCheck` — var→bool 変換に
OOB 概念なし。`enableStabilityCheck=false` はユーザー明示操作であり defect ではない。

### 1.5 Restore–setter parity

D8/D9/D10/D11 の guard はいずれも **setter 内**にあるため、UI / session /
内部 path のすべてに自動適用される。parity は構造的に成立する。
setter 迂回の直接書き込みは `pendingOverride` 全件調査で不存在を確認
（全書き込みが setter-guard 済み／jlimit 済み／int-domain）。

---

## 2. jlimit 監査の結論（Owner §5）

`jlimit` は NaN に対して有限性を保証しない（比較偽→入力返却）ことを前提に
全 field を再評価した。NaN が格納され得る setter
（`setTotalGain` 旧版、`setNonlinearSaturation`、convolver tail 系旧版、
EQ band float）は、いずれも下流で中和されるか D8/D9/D11 で修正済みである。
残存する無条件 store（tailStart 除く tail 系は D9 済み、mixedF、saturation、
EQ band）は、下流の有限性ゲート（`> 0.0` 比較、`isfinite`、`validateBuffer`、
RT per-sample clamp）により無害であることを個別に確定した。

---

## 3. なぜ NO-GO か

D12 の GO 条件（§6 の 7 項目）に照らし、「既存 guard で完全に中和されていない」
「具体的なユーザー影響がある」を満たす候補が存在しない。
最も近い `nonlinearSaturation` の NaN 格納も、`saturation > 0.0` の偽により
適用 skip となることが実コードで確定している。

---

## 4. 推奨（将来の harm 証明時に再評価）

- `nonlinearSaturation` 等の「accidental safe」（比較偽による不格納）は
  意図的 guard ではないため、将来の consumer 変更時に再評価すること。
- `cmaesRestarts` の巨大値（cancel 可能な長時間学習）は現時点で harm なし。
- D6 NO-GO・R8 flake は引き続き未解決として残る。

---

## 5. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**
- D1〜D11 の未 commit 変更はすべて保持
- Implementation / Gate は作成しない（NO-GO のため）
