# P1-5-IR-P2 — Step 5-N / P3-5-R6: Execution Vehicle Audit（read-only設計）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R6）
- **種別**: read-only 設計監査。実装・build・実行なし。変更ゼロ。
- **目的**: 既存 public API だけで active world を安全かつ一義的に記録できる
  test-only vehicle を設計として固定する（R6-A/B/C 判定）。
- **結論**: **R6-C**（§10）。理由は §5 に固定する。

---

## 1. State Freeze

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
F vehicle＋R1 diagnostic     保持（f4723815…／a00d140d…・revertなし）
production/CMake/JUCE        0 diff（P1 TU以外）
```

## 2. Existing vehicle baseline

対象は P3-5 F vehicle の target pair のみ（新規 vehicle を設計しない）：

```text
process = fresh／IR = g0／OS = 1／conv = on／EQ = off／sc = 0/1／amp = -20 → -6
```

topology（runPair・ensureTestIr×2・sc0/sc1・diffStats）は保全前提。
R5-A の API 十分性を受けて R6 で観測 schema を固定する。

## 3. `runCase()` observation points（T0–T9・source固定）

現 F-vehicle `runCase()` への対応（行番号は drift するため symbol 基準）：

```text
T0  case開始                  runCase entry
T1  seqBefore取得             getLastCommittedPublicationSequence（:411相当）
T2  configureChain()          :412相当（dispatch diagnostics 差分の前点）
T3  rebuild dispatch diagnostics取得（T2前後差分・§8）
T4  publish wait終了          waitWorldPublished return（strict分岐・変更しない§11）
T5  seqAfter取得              wait直後
T6  active RuntimeWorld read  scoped short read（§4・要production accessor§5）
T7  capture開始               configureCapture＋setTap
T8  capture終了               clearTap
T9  active RuntimeWorld read  scoped short read（§4・同上）
```

lifecycle／backpressure 差分は T1–T5 および T7–T8 区間で取得する（§8）。

## 4. RCU read-handle lifetime（規約固定）

```text
capture直前 → 短時間read（fieldをlocalへcopy） → handle破棄
    → capture → 短時間read → handle破棄
```

- capture 全区間の handle 保持は**禁止**。read handle は RCU epoch を pin し、
  retire／reclaim を停滞させ pressure を誘発しうる（§R5-§5 申送り）。
- `RuntimeReadHandle` は move-only（copy delete・:2335）。scope 内完結を原則とする。
- sleep／wait／capture を跨ぐ保持は設計上禁止する。

## 5. Active snapshot field set（最小固定・R6-C の根拠）

Identity：`generation／worldId／publication.sequenceId`。
Routing：`convBypassed／eqBypassed／processingOrder`。
Automation：`softClipEnabled／saturationAmount／headroom／makeup／trim`。
DSP projection：`oversamplingFactor／irLoaded／irFinalized／structuralHash／sampleRate`。
Timing／overlap：`fadeTimeSec`。
追加は target identity への必要性が source で示せる場合のみ。

### 到達可能性の裁定（R6-C の核心）

- `makeRuntimeReadHandle／getRuntimeWorldFromReadHandle／
  getRuntimeSnapshotFromReadHandle` は public 領域にあるが、
  context 構築に要る `messageThreadRcuReader` は **private**（:4940）。
  test-only から代替 reader を調達する既存 public 経路は存在しない。
- 既存 public で読める committed-state は bypass mirror 2 bits
 （`isEQBypassed／isConvolverBypassed`＝committed compatibility mirror）のみ。
  `isSoftClipEnabled／getSaturationAmount／getOversamplingFactor／getOversamplingType`
  は intent 側 UI atomic であり committed ではない。
- 見せかけの別経路 `getActiveRuntimeDSP()` は placeholder-bootstrap専用 legacy mirror
  であり通常時は null（:2310-2314明記）。使用しない。
- よって committed os／softClip／sat／EQ／IR-identity の snapshot 同定には、
  最小でも production 側の追加（例：read-only snapshot accessor／reader accessor
  のいずれか1点）が必要である。これは production source 変更であり、
  R6 では実施せず停止する。P3-1-D（limiter envelope系）とは別物であり、
  P3-1-D は開始しない。

## 6. requested/active separation（log設計規約）

```text
requested:  os=1／convBypassed=0／eqBypassed=1／softClip=0／sat=1.0／amp=-20
active_before: {...}
active_after:  {...}
```

- vehicle 内に `requested == active` と解釈する code を入れない。比較は監査側で行う。
- order=F/R の意味判定を log に入れない（case sequence の相違のみ。§9）。

## 7. sequence/generation/worldId correlation（規約）

- 4者は別 numbering（R5 §6）：rebuildRequestGeneration／RuntimeWorld::generation／
  publicationSequence（reserve≠commit に注意）／worldId。
- log field は分離し、`generation=123` 単独表記を禁止する。
- `requestGeneration == worldGeneration` 前提の validation を入れない。
- 相関 key は `(sequenceId, generation, worldId)` 三つ組＋snapshot field とする。

## 8. dispatch/lifecycle/backpressure auxiliary evidence（補助・正本分離）

- Rebuild dispatch 8 counters／lifecycle 6 項目／backpressure 6 項目は
  いずれも public getter 群であり、区間差分として test-only で取得可能である。
- ただし publish の正本は **sequence＋active-world snapshot** のみとする。
  queued／drained／backlog 系を publish proof に使わない（R5維持）。

## 9. F/R symmetry（将来比較の保全）

- F（-20→-6）／R（-6→-20）で同一 observation schema を用いる。
- am-6 の merge 問題：seqBefore／dispatch delta／active_before を pass 毎に
  独立取得する構造とし、「-20でpublishされたから-6も測定できる」前提を入れない。
- capture 前後両読（T6＋T9）を必須とし、後読のみの target 判定実装を禁止する。
- wait 結果と snapshot は独立 evidence として保存し、
  `wait==true` からの target 断定を禁止する（§11相当）。
- `waitWorldPublished(...,30000)`・sleep・settle は変更しない。

## 10. R6 gate decision

```text
R6-A（Minimal vehicle feasible）: REJECTED
  sequence／delta 系は既存のみで可だが、active snapshot 全体は不可（§5）。
R6-B（schema可・runCase構造不足）: REJECTED
  不足は runCase 構造ではなく production accessibility にある。
R6-C（production API追加が必要）: ADOPTED
  production変更はせず停止し、最小 accessor の別途設計判断へ。
  P3-1-D（limiter系）とは分離して扱う。
```

停止点：R vehicle・F/R比較・P3-1-D・limiter帰属・stale帰属・production修正に進まない。
R4境界（測定値Known／publish未証明／active未同定／stale Possible／Δ未帰属）を維持する。
