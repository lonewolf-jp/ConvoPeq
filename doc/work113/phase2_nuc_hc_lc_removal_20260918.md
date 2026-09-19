# WORK113-14 — Phase 2: NUC HC/LC Application Removal

- production changes: 1 file
- test-only changes: 0
- implementation: Phase 2 / 4（routing 未変更）
- commit: 0（未コミット。ユーザー指示「未コミットのまま継続」）

- **作成日**: 2026-09-18
- **種別**: Production（旧 HC/LC operator の除去のみ）
- **前工程**: `doc/work113/phase1_state_ownership_migration_20260918.md`（113-13 完了）
- **保留**: 113-7 block sweep / WORK114

---

## 0. 変更境界

113-14 は **旧 HC/LC operator（NUC 側）を runtime から外す**段階。以下は**変更しない**：

```text
FilterSpec（メンバ・populate 経路）/ applySpectrumFilter() 関数自体
DSPCoreDouble.cpp / DSPCoreFloat.cpp / convIsLast / OutputFilter / routing
FFT・OLS / tail processing（tailMode==0 Air tilt）/ direct head / L0/L1/L2 / StateIO
```

---

## 1. 変更内容（1 箇所）

### `src/MKLNonUniformConvolver.cpp:1164-1173` — IR build 内の `applySpectrumFilter` 呼出を停止

```text
Before:  if (filterSpec != nullptr) applySpectrumFilter(*filterSpec);
After:   (void)filterSpec;      ← 呼出を停止（関数・FilterSpec は残置）
```

- **call site は全 src で 1 箇所のみ**（`grep applySpectrumFilter` → 定義 `:361`／宣言 `h:484`／呼出 `:1165` のみ）
  → 1 行の切断で `FilterSpec → applySpectrumFilter → HC/LC` のデータ経路が完全に切れる。
- `applySpectrumFilter()` の実装（HC ゲイン `:399-433` / LC ゲイン `:435-456` / `vdMul` 適用 `:458-466`）は**無変更で残置**
  → 将来の再有効化は 1 行の復元で可能。
- 直後の `tailEnabled && tailMode == 0` Air tilt（`:1167+`）は**独立機能のため保持**（HC/LC とは別系統）。
- MSVC の未使用警告（C4100）回避のため `(void)filterSpec;` を明示。

---

## 2. 受入検証（113-12 §5 の機械的基準）

既存 seam `--buzz --nuc7c` の `irFreqEnergy` を使用。期待値は 113-12 §5 で事前定義済み
（raw = **2.049e+03**、フィルタ適用時 = 9.14375e+02）。

| 条件 | 113-14 前（WORK113-7C） | **113-14 後** | 判定 |
|---|---|---|---|
| `A_none_np3`（useSpec=0 = raw 参照） | 2.049e+03 | **2.049e+03** | 不変 ✓ |
| `B_spec_np3`（useSpec=1） | 9.14375e+02 | **2.049e+03** | raw に一致 ✓ |
| `D_spec_np32`（useSpec=1） | 9.14375e+02 | **2.049e+03** | raw に一致 ✓ |

- `irFreqNZ=1` / `irFreqPeak=1.000000e+00` は全条件で不変（構造は同一）。
- `numPartsIR`=3 と 32 が同一値 → **`numPartsIR` 非依存**も維持（WORK113-7C の結論を再確認）。

```text
ACCEPTANCE:  FilterSpec の ON/OFF が irFreq を変化させない
          →  nucHCMode/nucLCMode は runtime HC/LC authority として機能しない
RESULT:      STATE / TOPOLOGY INTERMEDIATE PASS   ✓ CONFIRMED
```

---

## 3. ビルド / 回帰

| 項目 | 結果 |
|---|---|
| `cmake --build build --config Release --target AudioEngineHarness` | `[2/3] Linking Release\AudioEngineHarness.exe` **成功** |
| `cmake --build build --config Release --target ConvoPeq` | `[2/3] Linking ConvoPeq_artefacts\Release\ConvoPeq.exe` **成功** |
| `build\Release\AudioEngineHarness.exe`（全デフォルトシナリオ） | **EXIT=0 / 全 PASS** |
| `ConvolverStateRoundTripTests` | **PASS (AC-A1-1/2/3)** |
| `IRLoadAdmissionTests` / `IRLoadPreviewAdmissionTests` | PASS |
| `DeferredFlowIntegrationTests` / `DeferredPublishViewStateMachineTests` | PASS |
| `AudioEngineHarness: all publish pipeline tests` | **PASS** |

- LSP（clangd）診断は JUCE グローバルヘッダ未設定による既存偽陽性（編集は行番号のみ +2 シフト）。MSVC ビルドで判定。

---

## 4. 意図した中間状態（音響）

- `FilterSpec` は**依然 populate される**（`spec.hcMode = buildSnapshot.nucHCMode`：`Lifecycle.cpp:303` / `LoaderThread.cpp:213` / `LoadPipeline.cpp:658`）が、
  `applySpectrumFilter` が呼ばれないため **IR は raw**。
- routing 未変更のため、既定 order `Conv→EQ` では `convIsLast=false` → OutputFilter ① は呼ばれず、
  **HC/LC は一時的に 0 回**（`EQThenConvolver` では ① が 1 回）。
- これは**意図した中間状態**。`ACOUSTIC FINAL PASS` は 113-15 の受入条件。
- 副次効果：`irFreqEnergy` が 2.049e+03 に戻るため、**この時点で block-rate 側帯波は IR 由来でなくなる**
  （113-15 後に sideband 再測定で帰属を判定する）。

---

## 5. 停止条件判定

```text
NUC HC/LC APPLICATION REMOVAL: COMPLETE
  [1] applySpectrumFilter 呼出 0（call site 1 箇所を切断）   ○
  [2] 関数・FilterSpec は残置（削除なし）                    ○
  [3] routing / convIsLast / OutputFilter 未変更              ○
  [4] tail / direct head / L0-L2 未変更                      ○
  [5] 機械的受入（irFreqEnergy == raw 2.049e+03）            ○ CONFIRMED
```

→ **完了**。次は 113-15（OutputFilter Routing Migration＝音響修正本体）。
