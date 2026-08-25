# O_denom Definition & Measurement Protocol Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only definition/protocol audit（ソースコード変更 **0** / 契約変更 **0** / 数値採用 **0**）
- **目的**: `R_required = ceil(M_scope / O_denom)` の分母 `O_denom` を、一般論ではなく
  **設計契約上の定義 + 実装事実 + 測定 protocol** として確定する。
- **基準**: ConvoPeq.md 2026-08-25 16:25 版内容＋ fresh source trace（17:21 以降 diff なし）

---

## 1. O_denom の設計契約上の意味

### 1.1 定義（I4 契約 D80/D81 + D95 固定点から確定）

```text
O_denom ≜ reference measurement window 内における
         observedOutstandingMax (= windowMax) の測定値

semantic: published World outstanding count の sampled maximum
          （conservative lower bound of true peak — D80/D81 確定）
```

### 1.2 O_denom が満たすべき条件

| 条件 | 根拠 |
|---|---|
| O_denom > 0 | bootstrap invariant（post-first-commit で est ≥ 1 構造保証） |
| O_denom は R_required の分母 | `ceil(M_scope / O_denom)` の除算可能性 |
| O_denom と M_scope の足し合わせで B_max^true を被覆 | `B_max^true ≤ O_denom + M_scope` |

---

## 2. eligible window 条件の機械的判定 protocol

```text
window w が O_denom 測定に eligible である ⟺
  (P-1) post-first-commit:
        window Start の A0 ≥ 1 または window 内に ≥1 の commit event が存在
        （bootstrap invariant により最初の commit 後は常に成立）

  (P-2) counter wrap なし:
        snap.counterWrapped == 0

  (P-3) shutdown drain 専用 window ではない:
        windowTag ≠ Shutdown（shutdown drain 中の window は別枠記録）

  (P-4) 同一 measurement campaign 内:
        windowId が対象 campaign の連続範囲内

  (P-5) sampler tick が ≥1 回 window 内で実行されている:
        sampleCount ≥ 2（Start tick + End tick を含むため最低 2 は自然に成立）
```

---

## 3. O_denom の取得方法

```text
取得経路:
  telemetry.lastClosedSnapshot().snap.windowMax

対象:
  上記 P-1〜P-5 を満たす Closed 状態の measurement window

複数 window が存在する場合:
  O_denom = max(eligible windows の snap.windowMax)
```

---

## 4. 既存測定データの有無

| データ源 | 状態 |
|---|---|
| production telemetry | ❌ 未蓄積（初回 deploy 後に蓄積開始） |
| WorldRetirementMeasurementTests harness | ✅ burst/normal/jitter 3 条件で windowMax 記録済み（snap.windowMax 経由） |
| AudioEngineHarness SoakPublishIntegrationTests | ✅ getPublicationBacklogCount()==0 待ちループで windowMax 間接確認可能 |

---

## 5. dual-use separation

| 用途 | 使用する量 | 根拠 |
|---|---|---|
| diagnostic（E_w 測定） | O_w 生値（0 含む） | burst 未観測性の証跡 |
| retention design input | **O_denom ≥ 1**（eligible window のみから算出） | R authority へ接続する際は正の baseline が必要 |

既存契約整合: D86 非交渉条件 2「telemetry は lifetime authority にしない」と整合。
O_denom は R authority への入力であり authority 自体ではない。

---

## 6. VERDICT: **PASS**

全条件 closure。数値決定は次フェーズ D102-C3（R_cap/T2 compatibility audit）で実施。
