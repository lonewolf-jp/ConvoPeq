# D101-34-D — Contract Update 後の独立 Read-only Source Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0** / 契約文書変更 **0** / コミットなし）
- **判定**: **PASS**（BLOCKER 0 / REQUIRED 0。OBSERVATION 2件 — §6）
- **基準**: ConvoPeq.md **2026-08-25 16:06 再生成版**（ローカル実ソースから・以後 diff なし確認済み）

---

## 1. D-G01 基準化

```
python output_sourcecode_markdown.py   → ConvoPeq.md 再生成 (2026-08-25 16:06)
git status --short -- src CMakeLists.txt build.bat → 変更なし（I4 契約 md のみ未コミットで既知）
git diff --check                       → エラーなし
監査対象: I4_DESIGN_CONTRACT.md（14:24 commit f39fcd3 後の D101-34-C 更新版）+ 実ソース双方
```

---

## 2. D-G02 claim strength audit（契約が実装より強くないか）

| # | 契約主張 | 導出根拠 | 強度判定 |
|---|---|---|---|
| 1 | retirePublishedRuntimeWorldNonRt 実装・World entry 唯一生成 | h:3533/:3545 実在（full-repo grep） | ✅ 適切 |
| 2 | retireRejectedRuntimeWorldNonRt 実装・Generic 使用 | h:3549/:3563 実測 | ✅ 適切 |
| 3 | DeletionEntryType::World producer = 1 | full-repo grep（src/ 全域）で生成箇所は h:3545 のみ。他 10 ヒットは全て consumer 型比較 | ✅ 適切 |
| 4 | caller provenance（Published 6 / Rejected 2） | 全 caller のファイル:行番号付き分類（D101-34-A §3 を fresh grep で再現） | ✅ 適切 |
| 5 | INV-WORLD-TYPE ∀W 含意 | producer 単一性（構造）+ caller provenance 100%（監査）により導出。**ただし PRECONDITION は呼び出し規律であり compiler-enforced ではない** — 契約は「CONFIRMED (audit)」と記載しており過大主張ではない | ✅ 適切（O-2 参照） |
| 6 | INV-PUB-3 CODE-FIXED / formal closure 区分 | publication domain 閉包と whole-engine type-state pending を分離記載 | ✅ 境界維持（D-G07 も PASS） |
| 7 | INV-PUB-4 exactly-once proof complete | §4 の前提→遷移→terminal の再検査で成立（D-G05） | ✅ 適切 |
| 8 | RuntimeStore current==nullptr PROVED | mutation site 唯一性（§6）+ 8 終了経路列挙 + Q2 前提明示 | ✅ 適切（前提明示済み — D-G06 追加確認あり） |

**結論: コードから導出できない強い主張は存在しない。**

---

## 3. D-G03 stale statement propagation audit（文書全体）

パターン: `retireRuntimePublishWorldNonRt / DISPROVEN / NOT STARTED / NOT implemented / 3 terminal sites`

| 行番号帯 | 所属セクション | 分類 |
|---|---|---|
| 2676-2770 | **D45**（Design-27・2026-08-15 歴史記録） | 履歴 — 誤検出除外 |
| 2898-3010 | **D47/D48**（Design-29/30・歴史記録） | 履歴 — 除外 |
| 3230-3291 | **D50/D51**（Design-32/33・歴史記録） | 履歴 — 除外 |
| 4646-4722 | **D68/D69**（Design-50/51 telemetry・歴史記録） | 履歴 — 除外 |
| 4834-4993 | **D71/D74**（Design-53/56 control-flow 再突合・歴史記録） | 履歴 — 除外 |
| 5485-5858 | **D83〜D87**（Design-65〜69・歴史記録） | 履歴 — 除外 |
| 7078-7305 | **I4.D101**（本節） | ✅ 全て解消済み/OBSOLETE マーカー付きブロック内または更新注記内 |

方針どおり「歴史記録は保持し、現行主張と区別」。
**現行状態としての stale statement 残存 = 0。**

---

## 4. D-G04 7 onRelease sites 独立再列挙（grep 範囲漏れ再発防止・**src/ 全域**）

今回の教訓（A: 5 → B: 7 の訂正履歴）を踏まえ、`src/` リポジトリ全域を対象に再実施:

```text
$ grep -rn "DeletionEntryType::World" src/   → 11 hits（production 8 + tests 3）
$ grep -rn "onRelease()"            src/     → 9 hits（sites 7 + 定義 1 + 判定外 1）
```

| # | site | storage / direct | World branch 実在 |
|---|---|---|---|
| R1 | src/DeferredDeletionQueue.h:154 | D queue（reclaim・CAS dequeue） | ✅ :148 |
| R2 | src/DeferredDeletionQueue.h:204 | D queue（drainAllUnsafe・shutdown） | ✅ :199 |
| R3 | ISRRetireRouter.cpp:87 | T（TerminalReclaimAuthority::drain） | ✅ :82 |
| R4 | ISRRetireRouter.cpp:114 | T（drainAll） | ✅ :109 |
| R5 | ISRRetireRouter.h:128 | direct destruction（recordWorldReclaim — storage 非経由） | ✅（呼出元 cpp:524-525） |
| R6 | RetireQuarantineStore.h:145 | Q / E インスタンス（**drain** minReaderEpoch 版 ※訂正 — 旧記載名 reclaimBatch は不存在） | ✅ :140 |
| R7 | RetireQuarantineStore.h:182 | Q / E（drainAllUnsafe） | ✅ :177 |

- producer: AudioEngine.h:3545 のみ ✅
- テストデータ（TerminalTelemetryContractTests ×3）は production 外 ✅
- **grep 範囲漏れなし**（src/core を含む全域走査で 7 に確定・D101-34-A の 5 は core 漏れと判明済み）

⚠️ **訂正（O-1）**: R6 の実メソッド名は `RetireQuarantineStore::drain(minReaderEpoch, isOlderFn)`
（h:100）。旧契約 Layer 3 表の `handleRetire` および D101-34-B 記載の `reclaimBatch` という名称は
現行ソースに存在しない（古いリビジョンの名前）。site 数・行番号は正しいため影響は表記のみ。

---

## 5. D-G05 INV-PUB-4 proof soundness（前提 → 遷移 → terminal の順で再検査）

### 5.1 duplicate ownership

❌ 不可。ownership chain `D → Q → E → T` は排所有移動（cpp:20-22 ownership contract /
cpp:315 chain comment）。enqueueWithRetry は挿入成功段のみ ptr を保持。

### 5.2 double dequeue

❌ 不可。D の reclaim()/drainAllUnsafe() とも `compareExchangeAtomic(dequeuePos, ...)` 成功者
のみ破壊を実行し、直後に slot 無効化（ptr=nullptr / type=Generic / seq 前進）。
Q/E/T は mutex + swap 抽出のため同一 entry の二度抽出は不可能。

### 5.3 quarantine 二重取得

❌ 不可。reclaimBatch 相当（drain）と drainAllUnsafe は同一 `mtx_` 下での swap 抽出。
抽出後 size_=0。Q と E は別インスタンスで独立。

### 5.4 terminal と synchronous reclaim の重複

❌ 不可。R5（同期破壊）は handoff 時点で即破壊するため Terminal storage には挿入されない
（recordWorldReclaim は observer 通知のみ）。R3/R4 は storage 内 entry のみ処理。

### 5.5 onRelease 二重発火

❌ 不可。全 site とも `type` を deleter 実行**前に** capture（D86.1 順序）し、entry 破壊と
1:1 対応で発火。type != World では発火しない。

### 5.6 隠れ破壊経路の有無（追加検証）

deleter 実行点を全列挙（`deleter(` grep）: DDQ×2 / Router×3 / QuarantineStore×2 ループ =
**全て R1〜R7 のいずれかに対応**。onRelease を伴わない World 破壊経路は存在しない。

✅ **INV-PUB-4 proof soundness = 確認**。

---

## 6. D-G06 RuntimeStore shutdown proof 再検証

`current` の mutation site:

```text
RuntimeStore::current（private, friend Owner）
    └─ WriteAccess::publishAndSwap（exchangeAtomic acq_rel）… 唯一の write
         └─ WriteAccess は RuntimeWorldAuthority ctor で acquireWriteAccess() して唯一保持
              ├─ publish(): :249 publishAndSwap(next)
              └─ clearPublishedRuntimeSnapshotsNonRt(): :263 publishAndSwap(nullptr)
```

- RuntimeWorldAuthority 以外に WriteAccess 取得経路なし（INV-X4-3 明記・h:88-89）✅
- clear 後の新規 swap-in 不可能の根拠 = admission closed（closeAdmission 済み）+
  producer join（D101-33-C/D で確立）— **Q2 前提は契約 status block に明示済み** ✅
  （暗黇扱いになっていないことを確認）
- 8 終了経路列挙（D101-34-B §3.2）のうち、本監査で追加確認すべき点は見つからなかった

✅ **D-G06 = PASS**（前提の明示性含む）。

---

## 7. D-G07 / D-G08

| 項目 | 確認結果 |
|---|---|
| D-G07 CODE-FIXED ↔ formal closure の境界 | ✅ 契約 §INV-PUB-3 が「formal closure: publication domain complete / whole-engine type-state enforcement: pending」と分離記載。混同なし |
| D-G08 M-bound 誤波及 | ✅ Status block に「M NO-GO / Phase I NO-GO / D102 NO-GO」を維持。Tier 4 ブロックも「M-bound OPEN」分離。D101 #1 CLOSED ≠ M 証明済みである旨の記述整合 |

---

## 8. 最終判定

# VERDICT: D101-34-D = **PASS**

**BLOCKER 0 / REQUIRED 0 / OBSERVATION 2**

| # | OBSERVATION | 内容 |
|---|---|---|
| O-1 | R6 メソッド名の表記訂正 | 契約 Layer 3/旧記載の `handleRetire`・D101-34-B の `reclaimBatch` は现行ソースに不存在。正しくは `RetireQuarantineStore::drain(minReaderEpoch, isOlderFn)`（h:100、World branch h:140-149）。次回契約触媒時に表記修正推奨（site 数・行番号は正しいため実害なし） |
| O-2 | INV-WORLD-TYPE の強制力の位置づけ | ∀W 含意は「producer 単一性（構造）+ caller provenance 監査（実測）」による確認であり compiler-enforced ではない。将来 debug assertion / type-state 化の余地あり。契約側は audit-based CONFIRMED と記載済みで過大主張なし |

---

## 9. 終了条件と次フェーズ

D101-34-D **PASS** につき、指示の終了条件どおり:

```
D101-34-D PASS
      ↓
残余 GAP 再分類（完了 — BLOCKER/REQUIRED なし、OBSERVATION 2件）
      ↓
I4.D101.4 M-bound 復帰条件の確認
```

**M 導出フェーズへの復帰判断**: D101 #1 CLOSED により、M 導出の前提である
published-domain lifetime accounting 分離の形式化（I4.D101 Tier 4 本質項）は
code-fixed + structural proof まで到達。M 導入判断は I4.D101.4 判定条件
（reference completeness → state equation → sampler gap → acquire envelope → burst →
delayed release → shutdown/quarantine → finite M の順で一つずつ証明可能性確認）に従い、
次タスクとして段階的に開始することを推奨。

並行して未コミットの文書変更（I4_DESIGN_CONTRACT.md + evidence 報告書群）の
コミット判断が残存（tooling 5点は引き続き保留）。
