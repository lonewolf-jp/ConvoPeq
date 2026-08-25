# D102-C2-2A — Numerical Value Basis Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0** / 契約変更 **0** / 数値採用 **0**）
- **判定**: **PASS**（全入力の admissible basis 確定。数値決定は次フェーズ D102-C2-2 へ）
- **基準**: ConvoPeq.md 2026-08-25 16:25 版内容（17:21/最新再生成で同一致性確認済み）＋ fresh source trace

---

## Gate C2-2A-1: R_required semantics 確定

### 証明（C2-0-1 の再確認・確定）

```text
証明済み: B_max^true(w) ≤ O_w(w) + M_scope    （全 eligible window w）

要求条件: retention capacity R がピークを被覆する
    B_max^true ≤ R_required × O_denom

両式より:
    R_required × O_denom ≥ O_w(w) + M_scope ≥ O_denom + M_scope
    （O_denom ≜ max_w O_w(w) ≥ O_w(w)）

∴ R_required ≥ (O_denom + M_scope) / O_denom = 1 + M_scope / O_denom
   R_required = 1 + ceil(M_scope / O_denom)    （整数化・M_scope ≥ 0 で常に成立）
```

### semantics の切り分け

| 式 | 意味 | 使用場面 |
|---|---|---|
| `ceil(M_scope / O_denom)` | **gap coverage multiplier** — baseline 超過分のみの倍率。base 自体は含まない | gap 部分の capacity 追加分を論じる際 |
| **`1 + ceil(M_scope / O_denom)`** | **total retention multiplier** — base + gap coverage の合計倍率 | **R_cap/T2 authority 接続時の正式な要求値** ✅ 採用 |

### R_cap/T2 authority 接続時の意味

- `R_required` は「O_denom baseline に対する必要 retention 倍率」であり、絶対 retention 個数ではない
- `R_cap`（実装可能 capacity）との整合判断: `R_cap ≥ R_required × O_denom` を D102-C3 で検証
- `T2`（time authority）: G_bound 項が時間要素を吸収済みのため T2 変更は不要だが、
  shutdown drain duration が T2 の時間 budget に収まることの確認は D102-C3 scope

✅ **Gate C2-2A-1 = PASS**（訂正式確定・semantics 分離完了）。

---

## Gate C2-2A-2: λ 決定根拠の closure

### producer ごとの λ 決定方式

| producer | 発生契機 | λ 決定方式 | 測定値昇格 |
|---|---|---|---|
| P-4 Transition | DSP transition 完了 | workload contract: 「transition 完了率 ≤ W_trans/s」という運用上界をユーザー/仕様が定義 | ❌ |
| P-7 Orchestrator Path A | ユーザー/オートメーション publish 要求 | workload contract: 「publish 要求率 ≤ W_pub/s」をユーザー/仕様が定義 | ❌ |
| P-8 Recovery | quarantine 検出（build failure 連動） | workload contract: 「quarantine 発生率 ≤ W_rec/s」を運用仕様が定義 | ❌ |
| Timer #5 | timerCallback 内 fade 完了 | **λ スコープ外** — 構造 bound ≤10/s として N_timer 項と別枠管理（Model 2′） | — |

### lifecycle 系（P-1/P-2/P-5）の steady-state scope

```text
定常運転中（bootstrap commit 後〜shutdown clear 前）:
    P-1/P-2/P-5 は発生しない → λ への加算項 = 0 ✅

lifecycle 遷移時（prepareToPlay / releaseResources 実行中）:
    burst 的に発生し得るが、この期間は measurement window の対象外とするか、
    別フェーズとして扱うことを推奨（D101-35-A §1.3 の方針維持）
    → 契約上「measurement window は定常運転区間に限定」の但書きで対応
```

### 測定値昇格の禁止

```text
❌ D100.6 の実測 retire rate をそのまま λ_prod_bound に採用
   （実測は観測値であり、workload contract の上界とは性質が異なる）

✅ 正しい手順: ユーザー/運用仕様が「許容最大レート」を宣言 → それを契約値とする
   （実測値は妥当性検証の参考データとして併記可能）
```

✅ **Gate C2-2A-2 = PASS**（決定方式確定・具体値は未決定）。

---

## Gate C2-2A-3: G_bound / K_starve contract basis

### 3構成要素の保証者と契約区分

| 要素 | 値の出所 | 契約区分 | 誰が保証するか |
|---|---|---|---|
| `T_sampler = 100ms` | コード定数（startTimer(100)・Init.cpp:121 / kExpectedTickIntervalUs=100'000 Telemetry.h:311） | source constant | ✅ コードから確定 |
| `δ_processing` | samplerTick 内の atomic ops 処理時間 | **measurement-derived**（実測必要・微小予想） | 実測後に契約値へ（または微小として無視の根拠を文書化） |
| **K_starve** | **environment contract** — ホストアプリケーション環境が message thread starvation を K_starve 以下に抑えることの前提 | ユーザー/製品仕様 | **ユーザー決定必須** |

### K_starve の「例示値 vs 正式契約」の区別

C2-1 で「例: 5秒」として挙げた値は**あくまで例示**であり、正式な bounded-starvation
contract の値としては以下の検討が必要:

| 観点 | 検討内容 |
|---|---|
| 技術的整合性 | JUCE Timer の coalescing property 下、starvation が K_starve を超えないことが message thread の通常動作と矛盾しないこと |
| 検知可能性 | missedTickCount watchdog により K_starve 超過を検知できること（将来実装） |
| 製品要件 | modal dialog 等で message thread が停滞する製品シナリオでの K_starve 妥当性 |

→ **これらの検討はユーザー/製品設計者の意思決定事項であり、本監査では技術的境界の明示のみ実施。**

### JUCE framework premise と application environment premise の境界

| premise | 区分 | 内容 |
|---|---|---|
| JUCE Timer framework premise | **framework-level**（JUCE ライブラリの動作保証） | callback 間隔 ≥ 設定 interval（coalescing property）。遅延時も積み上げ発火せず次回1回に統合 |
| application environment premise | **application-level**（ホストアプリケーション環境の前提） | message thread starvation ≤ K_starve。OS scheduling delay ≤ OS premise |

→ 2階層に分離して文書化することで、どちらの前提が破られた場合に bound が失効するかを明確化。

✅ **Gate C2-2A-3 = PASS**（basis 区分完了・K_starve 値はユーザー決定事項として正しく保留）。

---

## Gate C2-2A-4: N_timer 式の依存関係 evidence 固定

```text
【依存チェーン】
JUCE Timer framework premise
    「callbacks は設定 interval（T_sampler=100ms）以上の間隔で発火する
      （coalescing property — 遅延時も積み上げ発火せず次回1回に統合）」
        ↓ 【framework-level 外部前提】
callback spacing bound
    任意の長さ L の閉区間に含まれる callback 回数 ≤ ⌊L/T_sampler⌋ + 1
        ↓ 【数学的帰結】
N_timer(G_bound) ≤ ⌊G_bound / T_sampler⌋ + 1
        ↓ 【Model 2′ 項として合流】
M₂′ ≤ K + λ_prod_bound × G_bound + ⌊G_bound/T_sampler⌋ + 1
```

**premise の分類**: JUCE framework-level external dependency。
我々のコードから導出不可・文書化可能・JUCE library の公開された Timer 動作仕様に基づく。

**式の意味不変性**: G_bound に任意の有限値を代入しても式の構造は不変
（線形関数 + floor 項）。将来の G 値変更時も式の再導出は不要。

✅ **Gate C2-2A-4 = PASS**（evidence として依存チェーン固定）。

---

## Gate C2-2A-5: O_denom measurement protocol

### eligibility の機械的判定 protocol

```text
window w が eligible である ⟺
  (E1) post-first-commit:
       window Start の A0 ≥ 1 または window 内の tick で estimate ≥ 1
       （機械判定: snap.startAcquire ≥ 1 or snap.windowMax ≥ 1）

  (E2) counter wrap なし:
       snap.counterWrapped == 0

  (E3) 定常状態含有:
       window が shutdown drain 専用でない
       （機械判定: window tag ≠ Shutdown または drain 完了後の通常 window）

  (E4) 同一 measurement campaign:
       windowId が対象 campaign の連続範囲内
```

### O_w = 0 の扱い（discard しない）

```text
O_w = 0 の window:
  - denominator 選択から除外（O_denom ≥ 1 precondition）
  - diagnostic record として保持（burst 未観測性の証跡）
  - E_w 測定の入力として使用可能（E_w = T_w − 0 = T_w）
```

### 取得方法の確定

```text
O_denom = max { O_w(w) | w は eligible }

取得経路: telemetry.lastClosedSnapshot() → snap.windowMax
          （Closed state のみ読み取り・window state 不変・D91 基準 10）

測定期間: post-first-commit 〜 shutdown clear までの連続 campaign
```

✅ **Gate C2-2A-5 = PASS**（protocol 機械化完了）。

---

## Gate C2-2A-6: 数値採用可否 最終表

| Input | Basis | 数値 | 採用可否 |
|---|---|---|---:|
| K | source constant | 4096 | ✅ 確定 |
| T_sampler | source/config + framework premise | 100 ms | ✅/要確認 |
| λ_transition | workload contract | TBD | ❌ ユーザー決定待ち |
| λ_user_publish | workload contract | TBD | ❌ ユーザー決定待ち |
| λ_recovery | workload contract | TBD | ❌ ユーザー決定待ち |
| K_starve | environment contract | TBD | ❌ ユーザー決定待ち |
| δ_processing | measurement-derived（微小） | TBD | △ 実測後確定 |
| G_bound | derived（K_starve + T_sampler + δ_processing） | TBD | ❌ 上記承認後 |
| N_timer(G_bound) | derived（⌊G/T⌋+1） | TBD | ❌ G_bound 確定後 |
| O_denom | eligible measurement | TBD | ❌ 実稼働測定必要 |
| M₂′ | derived | TBD | ❌ 全入力確定後 |
| R_required | derived | TBD | ❌ M₂′・O_denom 確定後 |

**全入力の basis が閉じた**: ✅（各入力の決定方法・保証者・依存チェーンが確定）。
**数値自体はまだ全て未決定** — 正しく次フェーズへ繰り越し。

---

## VERDICT: D102-C2-2A = **PASS**

（basis closure 完了。数値決定は D102-C2-2 でユーザー意思決定を経て実施。）

---

## 補足: read-skew の方向分析（sampleWindow の load 順序）

実コード確認（ISRWorldRetirementTelemetry.h:229-231）:

```cpp
const auto a = acquireObserved();   // ← 先に A
const auto r = releaseObserved();   // ← 次に R
estimate = a − r;
```

A→R 順の場合、2 load 間の release event は r に反映されず est が B を**過小評価**し得る。
ただし releaseObserved は cursor 方式（tick 間で変化しない）のため、同一 tick 内の
a/r load 間に新規 release は入り得ない（addReleaseObserved は tick 処理の step 1 で完了済み）。
∴ 同一 tick 内の a/r read skew による understate は**発生しない**。
tick 間の取りこなしは M_gap/M_burst 項で既に被覆済み。

この解析により、read-pair skew の formal caveat は **beginWindow/closeWindow の
Start/End snapshot についても同様に「同一 tick 内処理のため実質単一時点」** と結論付けられ、
D101-35-D §1.4 の残留 caveat は OBSERVATION レベルで妥当と再確認。
