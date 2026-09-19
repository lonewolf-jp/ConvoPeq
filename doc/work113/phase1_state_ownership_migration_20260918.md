# WORK113-13 — Phase 1: State Ownership Migration

- production changes: 2 files
- test-only changes: 0
- implementation: Phase 1 / 4（topology 未変更）
- commit: 0（未コミット。ユーザー指示「未コミットのまま継続」）

- **作成日**: 2026-09-18
- **種別**: Production（State ownership の移行のみ）
- **前工程**: `doc/work113/filter_application_contract_freeze_20260918.md`（113-12 = READY_WITH_OBSERVATION）
- **基準ソース**: `ConvoPeq.md` 5,276,742 B / 2026-09-18 22:33:38（HEAD `0654e7b5`）
- **保留**: 113-7 block sweep / WORK114

---

## 0. 変更境界（不変条件単位で切断）

113-13 は **「誰が HC/LC の変更意図を所有するか」だけを移す**段階。以下は**一切変更しない**：

```text
applySpectrumFilter / FilterSpec / DSPCoreDouble.cpp / DSPCoreFloat.cpp
convIsLast / OutputFilter / FFT・OLS / NUC の実フィルタ処理
StateIO・property・serialization
```

---

## 1. 変更内容（2 箇所）

### (a) `src/convolver/ConvolverProcessor.StateAndUI.cpp` — `getStructuralHash()`

`nucHCMode`/`nucLCMode` の `hashCombine` 2 行を除去（旧 :913-914）。

**安全根拠**：`structuralHash` は**永続化されない**。用途は
- `CrossfadeAuthority.cpp:28-29` — old/new world の同一 IR 判定（同一なら crossfade 不要）
- `RebuildDispatch.cpp:207` — rebuild admission の重複判定
- `RuntimeBuilder.cpp:249` — world projection への記録

HC/LC が IR から外れる以上、mode 変更は IR の structural change ではなく、同一性判定に含めるべきでない。

### (b) `src/audioengine/AudioEngine.Parameters.cpp` — `setConvHCFilterMode()` / `setConvLCFilterMode()`

`uiConvolverProcessor.setNUCFilterModes(...)` 呼出（fan-out (2)）を削除。
`publishAtomic(convHCFilterMode / convLCFilterMode)`（= intent 正本）は維持。

```text
Before: UI → publishAtomic(convHC/LC) ─┬→ snapshot → OutputFilter ①
                                       └→ setNUCFilterModes → nucHC/LC → FilterSpec → applySpectrumFilter
After:  UI → publishAtomic(convHC/LC) ──→ snapshot → OutputFilter ①（唯一の伝達経路）
                  （setNUCFilterModes への fan-out は停止）
```

---

## 2. 停止条件の検証（4 点）

| # | 条件 | 判定 | 根拠 |
|---|---|---|---|
| **[1]** | `setConvHC/LCFilterMode → setNUCFilterModes` の fan-out が消えている | **○** | `AudioEngine.Parameters.cpp` に呼出なし（残るのは説明コメント :668/:682 のみ）。production caller は `StateAndUI.cpp:390`（`setState` 内部・preset 復元）のみ |
| **[2]** | `getStructuralHash` に `nucHCMode/nucLCMode` が入っていない | **○** | `hashCombine(snapshot.nucHCMode/nucLCMode)` を除去 |
| **[3]** | StateIO / property / state は残っている | **○** | `getState`: `v.setProperty("nucHCMode", …)` :261 / `setState` 復元 :389-395 / clamp :202-204 / `BuildSnapshot::nucHCMode` は `ConvolverProcessor.h:98,1080` に残存 / test `AC-A1-1` が `saved.hasProperty("nucHCMode")` を検証し **PASS** |
| **[4]** | `FilterSpec` / `applySpectrumFilter` / routing は未変更 | **○** | `FilterSpec` は依然 populate される（`Lifecycle.cpp:303`, `LoaderThread.cpp:213`, `LoadPipeline.cpp:658` の `spec.hcMode = buildSnapshot.nucHCMode` は無変更） |

> **`[2]` の補足（重要な発見）**：`getStructuralHash` とは**別の** fingerprint
> `computeBuildSnapshotFingerprint()`（`StateAndUI.cpp:29-76`、`:32-33` に「用途が異なる」と明記）は
> `nucHCMode/nucLCMode` を**含んだまま**である（:56-57）。これは `snapshot.fingerprint`（:287）→
> `RebuildDispatch.cpp:171` → admission dedup（:206-208）に到達する。
> **意図的に保持する**。理由：`setState`（preset 復元）は依然 `nucHC/LC` を更新して IR を再焼き込みする必要があり、
> fingerprint から除外すると **preset 復元時の IR 再生成が抑止される**（回帰）。
> 除外が適切になるのは 113-15 で IR 側 HC/LC が完全に消えた後。

---

## 3. Phase 1 の特性（過渡状態の明示）

- **静的再生は bit 不変**。`convHC/LCFilterMode` は preset から load 前に復元され、IR に焼き込まれるため、
  ロード済み IR の HC/LC は従来どおり 1 回適用される。
- 変化するのは **実行時モード切替のみ**：既定 order `Conv→EQ` では `convIsLast=false` のため OutputFilter ① は呼ばれず、
  HC/LC は `applySpectrumFilter`（IR）由来 → **Phase 1 後はモード切替が音に反映されない**。
  これは 113-14/113-15 までの**意図的な中間状態**。
- したがって Phase 1 の判定は `STATE / TOPOLOGY INTERMEDIATE PASS` であり、`ACOUSTIC FINAL PASS` ではない。

---

## 4. 検証結果

| 項目 | 結果 |
|---|---|
| `cmake --build build --config Release --target AudioEngineHarness` | `[3/4] Linking CXX executable Release\AudioEngineHarness.exe` **成功** |
| `cmake --build build --config Release --target ConvoPeq` | `[15/16] Linking CXX executable ConvoPeq_artefacts\Release\ConvoPeq.exe` **成功** |
| `build\Release\AudioEngineHarness.exe`（引数なし = 全デフォルトシナリオ） | **EXIT=0 / 全 PASS** |
| `ConvolverStateRoundTripTests` | **PASS (AC-A1-1/2/3)** ← `nucHC/LC` round-trip 維持 |
| `IRLoadAdmissionTests` / `IRLoadPreviewAdmissionTests` | PASS |
| `DeferredFlowIntegrationTests` / `DeferredPublishViewStateMachineTests` | PASS |
| `AudioEngineHarness: all publish pipeline tests` | **PASS** |

- LSP（clangd）診断は JUCE グローバルヘッダ未設定による**既存偽陽性**（未変更ファイルにも同種が出る）→ MSVC ビルドで判定。
- 注：`[M03-2]`「direct ON = IR taps 0..31 raw (bypasses HC/LC); OFF = FFT path (filtered)」は
  `applySpectrumFilter` 依存の**計測専用・assert なし**テスト。113-14 で意味を失うため Phase 7（regression matrix）で扱う。

### 未計測（113-12 §6 から継承）
`mode 変更で REBUILD_TELEMETRY 発行 0 回` の**実測**。ただし現在は構造的保証が二重：
- (a) fan-out 停止により `setNUCFilterModes` → `postCoalescedChangeNotification()` が呼ばれない（rebuild intent 自体が submit されない）
- (b) `getStructuralHash` が mode 非依存 → 仮に他理由で rebuild が来ても `RebuildDispatch.cpp:207` の structuralHash 比較で no-op

---

## 5. 112-13 の停止条件判定

```text
STATE OWNERSHIP MIGRATION: COMPLETE
  [1] fan-out 消滅          ○
  [2] getStructuralHash 除外 ○（別 fingerprint は意図的に保持）
  [3] StateIO/state 残存     ○
  [4] topology 未変更        ○
```

→ **完了**。次は 113-14（NUC Filter Application Removal）。

---

## 6. 最終的な変更系列（ユーザー承認）

```text
113-12  Contract Freeze                    （完了）
113-13  State Ownership Migration          （完了・本文書）
113-14  NUC HC/LC Application Removal      （next、routing は変更しない）
113-15  OutputFilter Routing Migration     （exactly-once・音響修正本体）
113-16  Acoustic / Regression Validation
```

- 113-14 と 113-15 は**同一コミットにしない**（`state ownership` / `NUC operator` / `routing operator` の 3 因果を分離し、
  sideband 消失の帰属を判定可能にするため）。
- 113-14 の受入は `STATE / TOPOLOGY INTERMEDIATE PASS`（NUC 内から HC/LC が消えたこと）。
  NUC=0 回・OutputFilter も routing 次第で 0 回となり得るが、これは意図した中間状態。
- 最終的な exactly-once は 113-15 の受入条件。
