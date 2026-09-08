# D102-C2-5-D2-4 — D2 Closure / Evidence Record Audit

- **実施日**: 2026-08-26
- **作業種別**: **read-only / closure audit**（production 変更 0 / test 変更 0 / contract 変更 0）
- **根拠**: 最新 ConvoPeq.md（Generated 2026-08-26 20:06:47）＋ `evidence/d2_3_logs/` 実測ログ ＋ D2 系報告書一式

---

## 1. 証拠チェーン確認（D2-1 → D2-2 → D2-3）

```text
D2-1  Bounded Terminal Design Gate          … NO-GO（Phase A）
   │   根拠: 必要性未証明 + Candidate A/B が契約を破壊/無益
   │   残課題: 「持続的増加の実測」が起動条件として未検証 ← ここを D2-2/D2-3 が引き受けた
   ▼
D2-2  Telemetry Evidence Audit              … PASS-B
   │   根拠: telemetry の種類・経路・unit 正確性は十分（E1-E5 PASS）
   │   gap : 負荷下の時系列 capture データなし
   ▼
D2-3  Empirical Evidence Run                … Gate B-ENTRY 不成立
       既存 capture 機構（100ms OBS probe + T1/T3/T4 CLI）で実測
```

### チェーン整合性チェック

| チェック項目 | 判定 |
|---|---|
| D2-1 の保留条件と D2-2/D2-3 の検証内容が対応しているか | ✅ 一致 — D2-1 §12 の起動条件「terminalPeakResident 持続的増加＋health 裏付け」を、D2-2 が観測可能性（E5 分類）で、D2-3 が実測で検証した |
| D2-2 の分析規約が D2-3 で遵守されたか | ✅ 一致 — `T_store − T_drainEntry` 不使用、一次情報は `T_resident(t)`。さらに実測が規約の前提（drainAll 発火 0、epoch-gated drain で完全回収）を裏付け |
| D2-3 の結果が D2-1 の判定表のどれに該当するか | ✅ 「T resident ≈ 0」＋「一時的 spike → 0」＋「peak 上がるも resident 回収」→ **NO-GO 維持** 行に完全一致 |
| 矛盾する証拠の有無 | **なし** — 全 8 ランの生ログが `evidence/d2_3_logs/` に保存済み、再解析可能 |

**判定: 証拠チェーンに矛盾なし。D2 系の意思決定は閉じられる。**

---

## 2. D2-3 結論の確定（正式記録）

以下を実測確定事項として固定する（根拠: `evidence/d2_3_logs/` 8 ログ・全 `[D101_9_T5_OBS]` 時系列）:

```text
確定事項
├─ Terminal admission は異常負荷下（Q/E 枯渇）でのみ発生する        [T3-30s, T4-B/C]
├─ Q(1024)/E(512) 枯渇時、Terminal が大量 ownership を受領する       [peak 8632]
├─ reader recovery 後、T_resident == 0 に完全復帰する               [最終50サンプル min=max=0]
├─ terminalPeakResident 増加は transient spike として説明可能        [非ゼロ区間は stall 中のみ]
├─ sustained resident は未観測                                       [8/8 ランで不成立]
├─ sustained growth は未観測                                         [同上]
├─ memory-pressure 起因の持続的 growth の証拠なし                    [pressureLevel max=3 でも回収]
└─ よって K-terminal sizing の必要性は証明されていない
```

> **正式 disposition: D2-1 Phase B/C ＝ 保留継続（evidence-triggered hold）**
> 再開条件（不変）: 実運用/soak において `terminalPeakResident` の持続的増加が
> health telemetry（pressureLevel / retireEscalationCount）と相関して裏付けられた場合のみ。
> peak 単独の増加は起動条件にならない（lifetime high-water mark の性質上）。

---

## 3. 重要な正の観察 — P-4 設計意図の実測成立

T3-30s の `Q=1024 / E=512 / T_peak=8632` は bounded Terminal 採用の根拠では**ない**。
むしろ現行 architecture の核心契約が実運用条件で成立した実証として記録する:

```text
【実測成立した P-4 シナリオ】
D/Q/E bounded capacity exhausted (Q=1024 full, E=512 full)
      ↓
TerminalReclaimAuthority.store() が 8,632 件の ownership を全て受領
（growable vector — 1 件も拒否せず、caller ownership = 0 を維持）
      ↓
reader recovery → minEpoch 進行
      ↓
epoch-gated drain() が発動（drainAll() は 1 度も不要だった）
      ↓
T_resident = 0 完全排出・deleter exactly-once 実行
```

このシーケンスは以下の監査結論と相互整合する:
- D8-2-C C1（stored/estored/tstored チェーン）→ 実測で到達・完結を確認
- D8-2-C C2（無所有権 return 経路なし）→ 実測でも caller 保持は発生せず
- D8-2-B-2 T4（growable Terminal 受領テスト）→ 実負荷スケールで再現
- ISRRetireRouter.cpp:24-25 の契約コメント（store ALWAYS succeeds）→ 実証済み

---

## 4. D2 Series 最終 Disposition（固定）

| 項目 | 最終状態 | 根拠 |
|---|---|---|
| Terminal growable | **維持** | D2-1 NO-GO + D2-3 実測 |
| K_terminal | **未導入** | 同上 |
| `QueueFull` | **dead enum のまま維持** | D2-0 freeze 整合、D8-2-C C1/C2 |
| Candidate A (caller retains) | **NO-GO** | P-4 破壊 + ignoreUnused 契約崩壊 |
| Candidate B (shutdown-only bounded) | **NO-GO** | 便益ゼロ + INV-5 衝突 |
| Phase B (K sizing) | **保留（evidence-triggered hold）** | Gate B-ENTRY 不成立 |
| Phase C (implementation) | **保留** | Phase B 未着手 |
| telemetry instrumentation 追加 | **不要** | 既存 OBS probe + snapshot で十分（D2-2 E3） |
| ownership contract | **変更なし** | 通算 production 変更 0 |
| P-4 | **維持（実測で強化）** | §3 の正の観察 |
| I4 D15.2 | **変更なし** | 右辺追加は発生しなかった |
| `ignoreUnused(result)` | **維持** | D8-2-D D8 + 実測 |

---

## 5. D102-C2-5 残存 gate の棚卸し

### D 系列（QueueFull 仮想シナリオ系 — D0〜D7）の整理

D0〜D7 は「bounded Terminal 導入時に `QueueFull == caller retains` をどう閉じるか」という
**仮想シナリオの patch 設計**を目的としていた。D2 系 closure により以下の扱いとする:

| Gate | 当時の判定 | D2 closure 後の状態 |
|---|---|---|
| A | AUDIT GO / Implementation NO-GO（latent bug 指摘） | latent bug は D8-2-C/D で不存在を確認済み → **解決済み扱い** |
| D0/D1 | CONDITIONAL PASS（QueueFull 再定義の条件列挙） | 前提（QueueFull live 化）が消滅 → **moot** |
| D2-0 | PASS（semantic freeze） | **有効なまま**（QueueFull を消失理由にしない規約は将来も維持） |
| D3/D4/D5/D6/D7 | CONDITIONAL / NO-GO（patch 設計） | patch 不要が確定（D2-1 NO-GO + D8-2-C C2 全 return point 走査）→ **moot（棚上げ）** |
| D8-1 | PASS | 完了 |
| D8-2-A〜D | PREFLIGHT/PASS/PASS 9/9/PASS | 完了 |

> D0〜D7 の conditional 項目は「D2-1 Phase B が将来再開された場合にのみ意味を持つ設計メモ」であり、
> 現行契約下では発火条件が存在しない。削除はせず evidence に残置、再開時の入力とする。

### 残存作業

| 項目 | 状態 | 次アクション |
|---|---|---|
| **T9 / DSPLifetimeManager source audit** | **実質完了**（D8-2-D PASS — caller ownership trace / double-delete / handle-map / shutdown overlap を全て網羅） | 正式 closure 宣言のみ残す（下記） |
| T9 観察事項 1 件（non-blocking） | 記録済み | DSPGuard dtor true 分岐の理論 leak — 到達不能（commit 前 guard null 化 + DIAG jassert）。guard 扱い変更時の注意点として維持 |

### T9 正式クローズ宣言

D8-2-D（DSPLifetimeManager Caller Ownership Audit, PASS, violation 0 件）をもって、
元指示の T9 要件——

> 「DSPLifetimeManager が Router の ownership contract に対して追加 transfer を行っていない」

——は source-level で完全検証済みである。追加の抽象化・patch・テストは不要。
**T9 ＝ CLOSED（D8-2-D を根拠文書とする）。**

---

## 6. D102-C2-5 シリーズ総括

```text
完結:
  A (Terminal Ownership Caller)     → 解決済み（latent bug 不存在を後続監査が確認）
  D0-D7 (QueueFull 仮想 patch 系)    → moot（前提消滅・evidence 残置）
  D2-0 (Semantic Freeze)            → 有効規約として恒久維持
  D2-1 NO-GO / D2-2 PASS-B / D2-3 Gate 不成立 → 本報告で closure
  D8-1 PASS / D8-2-A〜D PASS        → 完了
  T9                                 → CLOSED（D8-2-D 根拠）

通算実績（本シリーズ全体）:
  production source change : 0（B-2 のテストファイル新規のみ）
  test source change       : 0（D2-3 は既存 harness 実行のみ）
  contract change          : 0

次工程候補:
  - D102-C2-5 外の残存 gate（D102-C3/C4 数値系等）の確認
  - または implementation gate の有無の洗い出し
```
