# D102-C2-3 — CLOSE 証明（O_denom 実測確定 / G1-G15 PASS / production 0-diff）

- **実施日**: 2026-08-25 23:45 (JST)
- **作業種別**: read-only CLOSE（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-25 22:11:33** + 実 `src/`
- **判定**: **CLOSE PASS**

---

## 1. 最終値固定（TBD 完全除去）

```text
O_denom       = 1  (campaign-wide maximum, 10 eligible Closed windows, 2026-08-25 23:40)
K_min         = ceil(4120 / 1) = 4120
R_required    = 1 + 4120 = 4121
R_cap,bounded = 5120 (=4096(D)+512(Q)+512(E))
Terminal dependency = max(0, 4121-5120) = 0
compatibility = PASS bounded (4121 < 5120, headroom 999)
```

- `D102-C2-3 Numerical Result` の TBD は完全除去済み（`grep -rn TBD evidence/D102-C2-3-Numerical-Result.md` → `CLOSE PASS ... TBD なし` の記述のみ、数値 TBD は 0）
- `O_denom=1` は 10 個の eligible window 全件から取得した campaign-wide maximum として確定（仮置きではない）

## 2. G1〜G15 最終 PASS 固定

| Gate | 条件 | 結果 |
|---|---|---|
| G1 | 22:11:33 ソースと実 `src/` が一致 | **PASS** (`head -n3 ConvoPeq.md` 22:11:33 / `git diff HEAD -- src/audioengine` 0 lines) |
| G2 | production source modification = 0 | **PASS** |
| G3 | campaign start/end recorded | **PASS** 104614058528 → 104623794884 |
| G4 | all Closed windows recorded | **PASS** 11 windows |
| G5 | eligibility mechanically applied | **PASS** |
| G6 | all exclusions + reasons recorded | **PASS** 1 WarmupExclusion |
| G7 | eligibleWindowCount >= 2 | **PASS** 10 |
| G8 | observed rate / contract rate separated | **PASS** observed 4.11/s vs contract 13/s |
| G9 | O_denom = max(windowMax over eligible) | **PASS** 1 |
| G10 | O_denom >= 1 | **PASS** |
| G11 | K_min = ceil(4120/O_denom) | **PASS** 4120 |
| G12 | R_required = 1+K_min | **PASS** 4121 |
| G13 | R_required <= 5120 | **PASS** |
| G14 | Terminal dependency = 0 | **PASS** |
| G15 | raw evidence persisted | **PASS** |

## 3. Campaign raw log / eligibility 相互参照

- `evidence/OdenomCampaign_console_2026-08-25.log` — raw console（11 windows, 全診断フィールド）
- `evidence/D102-C2-3-B-Campaign-Execution-Report.md:§6` — eligibility 表（11行×16列）
- `evidence/D102-C2-3-Numerical-Result.md:§5` — 最終報告テーブル（TBD なし）
- `evidence/D102-C2-3-A-Harness-Capability-Audit.md` — harness capability 事前監査

## 4. Production 0-diff CLOSE 時点再確認

```text
git diff HEAD -- src/audioengine → 0 lines
git diff HEAD --stat -- src/ → PublishPipelineIntegrationTests.cpp 21 lines (harness-only)
git diff --stat HEAD → CMakeLists.txt 1+ / ConvoPeq.md 2+- / PublishPipeline... 21+
```

## 5. Harness-only 差分の分離

| 区分 | ファイル | 内容 | production 影響 |
|---|---|---|---|
| harness-only | `src/tests/AudioEngineHarness/OdenomCampaignTests.cpp` (19KB, untracked) | campaign runner (`runOdenomCampaignDefault` 4/60/100/10+1) | なし |
| harness-only | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` 21 lines | `--odenom-campaign` dispatch | なし |
| harness-only | `CMakeLists.txt:1829` 1 line | Odenom 追加 | なし |
| production | `src/audioengine/*` / `src/DeferredDeletionQueue.h` | **0 diff** | — |

- Git commit 時の混在を避けるため、harness-only と evidence は production と分離してコミットすること

## 6. Observed rate 分離の維持

```text
contract rate = 13 events/s (固定・再定義しない)
observed rate = 4.11 events/s (40 pubs / 9.736s)
```

- observed を λ_prod_bound の根拠として昇格させていない

## 7. O_denom=1 の安全側昇格禁止の遵守

- `O_denom=1` は実測 campaign-wide maximum として採用
- `O_denom>=1` の構造的下界を safe bound として誤用していない
- `mean/median/P95/P99` は診断のみ、denominator に使用していない

## 8. Bounded compatibility 帰結

```text
O_denom>=1 ⇒ K_min<=4120 ⇒ R_required<=4121 < 5120
```

Terminal 容量の追加検討は不要だが、**Terminal-full 時の unsafe synchronous destruction は引き続き禁止**（D102-C2-4 で別途監査）

## 9. 次ゲート

```text
D102-C2-3 CLOSE PASS
  ↓
O_denom / K_min / R_required 固定
  ↓
D102-C2-4 (実装変更なし contract/evidence review) → bounded capacity と lifetime/ownership 整合性確認
```

---

*本 CLOSE は read-only であり、production source 変更 0、O_denom 事後選択 0、observed 昇格 0 で実施された。*
