# D102-C2-1 — Numerical Input Evidence & Contract Closure（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only contract/evidence closure（ソースコード変更 **0** / 契約ファイル変更 **0** /
  数値採用 **0** / R_cap·T2 非接続）
- **判定**: **PASS**（C2-1 全 12 条件充足 — §7）
- **基準**: ConvoPeq.md 2026-08-25 16:25 版内容（17:21/最新再生成で同一致性確認済み）

---

## 1. C2-1-1: λ_prod_bound producer contract closure

### 1.1 producer ごとの確定

| # | producer | call site | 発生契機 | publishes per operation | rate bound type |
|---|---|---|---|---|---|
| P-1 | PrepareToPlay publish #1 | PrepareToPlay.cpp:155 | prepareToPlay 呼出し（engine 初期化/re-init） | **2 回/call**（:155 + :277） | lifecycle contract |
| P-2 | PrepareToPlay publish #2 | PrepareToPlay.cpp:277 | 同上（同一呼出内の第2 publish） | （P-1 に含む） | 同上 |
| P-3 | Timer idle publish (#5) | Timer.cpp:994 | timerCallback 内 fade 完了時 | **≤ 1/callback** | **構造的 hard bound**: ≤ 10/s（startTimer(100)・Init.cpp:121）→ λ スコープ外・N_timer 項へ |
| P-4 | Transition (#6) | Transition.cpp:25 | DSP transition 完了時 | ≤ 1/transition | workload contract（transition 発生率の上界） |
| P-5 | ReleaseResources | ReleaseResources.cpp:175 | releaseResources 呼出し（shutdown sequence 中） | **1回/shutdown call** | lifecycle contract |
| P-6 | deferred resubmit | PublicationExecutor.cpp:53-66 | CoordinatorLoop 上の retry（waitForReceipt=false） | 元 submission の派生（独立 acquire を生成しない — 失敗 attempt は acquire 不発火） | **λ 加算から除外確定** |
| P-7 | Orchestrator Path A | submitPublishRequest 経由 executePublish | ユーザー/オートメーション publish 要求 | ≤ 1/request | workload contract 必要 |
| P-8 | Recovery publish | quarantine 検出時 | build failure 連動（外部要因） | ≤ 1/quarantine event | workload contract 必要 |

### 1.2 acquire accounting の正確性（P-6 二重計上問題の closed 証明）

```text
acquire event = onAcquireObserved() = commit 成功時のみ発火（Commit.cpp:406-409）
facade 失敗（admission reject / enqueue fail）→ acquire 不発火・token release 済み
resubmit 成功 → 当該 world で唯一の acquire event

∴ 各 committed World につき onAcquireObserved() = exactly 1（INV-PUB-4 と同一基盤）
   λ_prod_bound は「成功 commit レート」を契約すれば retry 由来の二重計上は構造的に発生しない
```

✅ **P-6 = λ から除外確定**（元 submission の workload contract に吸収）。

### 1.3 λ_ext 構成の確定

```text
λ_prod_bound = λ_lifecycle + λ_transition + λ_user_publish + λ_recovery

λ_lifecycle ≈ 0（定常状態では lifecycle event は発生しない）
              ※ re-init 時は burst になり得るが engine 再起動相当のため別枠
λ_transition = workload contract（transition 完了率の上界・ユーザー決定）
λ_user_publish = workload contract（Path A/直接 facade の要求率上界・ユーザー決定）
λ_recovery   = workload contract（quarantine 発生率の上界・build failure policy 依存）

加算の正当化: CoordinatorLoop 直列化により各 producer の acquire は排他的に処理され、
合成レートは個別上界の和を超えない（保守的正当上界）。
```

⚠️ **有限性の注意**: 上記 workload contract の値が未決の場合、λ_prod_bound 自体も未決。
しかし「各項が有限 workload contract で拘束可能」であること（producer inventory 完備 +
直列化構造）は確定 → **finite M への道は閉じていない**。

---

## 2. C2-1-2: G_bound contract closure

### 2.1 6 構成要素の分類

| # | 要素 | 分類 | 根拠 |
|---|---|---|---|
| (1) | nominal cadence 100ms | ✅ **必ず有限 bound に含める** | T_sampler 公称値 |
| (2) | callback scheduling delay | ✅ **含める** | message thread queue 待ち。通常運用では有限（OS scheduling premise 下） |
| (3) | missed/coalesced callback | ✅ **含める** | coalescing により gap が最大 1 interval 分延長 |
| (4) | OS scheduling delay | ✅ **含める** | thread priority/preemption。OS scheduling premise 下で有限 |
| (5) | **message-thread starvation** | 🔴 **要判断** | modal dialog / 長時間タスクで理論上無限遅延し得る |
| (6) | samplerTick 処理時間 | ✅ 含める（微小） | atomic ops のみ |

### 2.2 (5) starvation の formal closure

```text
unrestricted starvation を許す場合:
    G_bound = ∞ → M < ∞ の証明が失効 → D101 有限性証明が無意味化 ❌

解消策: bounded-starvation environment premise の導入

【environment premise（G_premise）】
    AudioEngine をホストするアプリケーション環境は、message thread の
    callback 実行を K_starve（契約定数・例: 5秒）を超えて遅延させないことを
   保証する。K_starve を超える遅延は environment contract violation として
    検知可能（missedTickCount watchdog による検知経路を将来実装可能）。

【G_bound の確定形】
    G_bound = K_starve + T_sampler + δ_processing
    （starvation 上限 + 1 interval + 処理時間）

    G_bound < ∞ は premise の下で成立 ✅
```

✅ **C2-1-2 Gate = PASS**: `G_bound < ∞` は bounded-starvation environment premise の下で成立。
premise の受容可否（(i) premise 採用 vs (ii) watchdog 実装による強制）はユーザー意思決定。

---

## 3. C2-1-3: N_timer lemma 形式閉鎖

### Lemma N_timer-bound

```text
前提:
  (a) JUCE Timer framework: callbacks は設定 interval（T_sampler = 100ms）以上の間隔で発火
      （coalescing property — 遅延時も積み上げ発火せず次回 1 回に統合）
  (b) 各 callback 内の #5 idle publish は高々 1 回（Timer.cpp:994 単一 site・guard 付き）

区間 [t_s, t_s + G_bound] 内の callback 回数:
  callbacks 間隔 ≥ T_sampler より、閉区間長 G_bound に含まれる最大回数 = ⌊G_bound/T_sampler⌋ + 1

各区間の #5 acquire 数 ≤ 1 より:
  N_timer(G_bound) ≤ ⌊G_bound / T_sampler⌋ + 1
```

### 訂正: 旧式 ⌈G/T⌉+2 からの改良

旧式 ⌈G/T⌉+2 は「区間端点の両側余剰 + 切り上げ」の三重保守であり、
spacing ≥ T の仮定（JUCE coalescing property）と組み合わせれば:

```text
最密充填: callbacks at t_s, t_s+T, t_s+2T, ..., t_s+kT where kT ≤ G_bound
  count = ⌊G_bound/T_sampler⌋ + 1
```

✅ **N_timer(G) = ⌊G_bound / T_sampler⌋ + 1 に訂正**（⌈G/T⌉+2 より厳密かつ正しい）。

目的確認: 後で G の具体値を代入しても式の意味は変わらない
（G_bound が抽象有限定数である限り N_timer も有限確定値関数）。✅

---

## 4. C2-1-4: O_denom 取得方法の固定

### eligible window の定義

```text
eligible window ≜ 以下をすべて満たす measurement window:
  (E1) post-first-commit: window Start または window 内に
       ≥1 の committed publish（A0 ≥ 1 or tick estimate ≥ 1）が存在
  (E2) counter wrap なし（counterWrapped flag = 0）
  (E3) 定常状態を含有（shutdown drain 専用 window は除外・別枠記録）
```

### O_w = 0 の扱い

```text
O_w = 0 の window は捨てない:
  - burst 未観測性の有効な観測結果として保持（diagnostic record）
  - E_w 測定の入力として使用（E_w = T_w − O_w で O_w=0 は E_w=T_w を意味）
  - denominator には使用しない（O_denom ≥ 1 precondition により排除）
```

### dual-use separation（measurement vs retention denominator）

| 用途 | 使用する量 | 根拠 |
|---|---|---|
| diagnostic（E_w 測定） | O_w 生値（0 含む） | burst 未観測性の証跡 |
| retention design input（D102-C3） | **O_denom ≥ 1**（eligible window のみから算出） | R authority に接続する際は正の baseline が必要 |

既存契約整合: telemetry は diagnostic observation であり R authority に直接接続しない方針
（D86 非交渉条件 2）と整合。O_denom は R authority への「入力」であり authority 自体ではない。

---

## 5. C2-1-5: 値決定方式の3種分離（最終固定）

| 入力 | 決定方法 | 具体値の出所 | 測定値昇格 |
|---|---|---|---|
| **λ_prod_bound** | **workload contract**（producer ごとの submission rate 上界の合計） | ユーザー/運用仕様 | ❌ 禁止 |
| **G_bound** | **environment worst-case contract**（bounded-starvation premise 含む） | ホスト環境仕様 | ❌ 禁止 |
| **O_denom** | **measurement-derived input**（eligible window の observedOutstandingMax max） | 実稼働 telemetry | ✅ measurement-derived が本設計の意図 |

| 定数 | 出所 |
|---|---|
| K | source constant = 4096（コード） |
| N_timer(G) | mathematical function = ⌊G/T_sampler⌋ + 1 |

---

## 6. C2-1-6: authoritative symbolic formula（最終固定）

```text
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║  M₂′ = K + λ_prod_bound × G_bound + N_timer(G_bound)      ║
║                                                            ║
║      K           = 4096                （source constant） ║
║      N_timer(G)  = ⌊G / T_sampler⌋ + 1                   ║
║                  （T_sampler = 100ms・JUCE coalescing）     ║
║                                                            ║
║  ∴ M₂′ = 4096 + λ_prod_bound × G_bound                    ║
║          + ⌊G_bound / 100ms⌋ + 1                          ║
║                                                            ║
║  R_required = 1 + ceil(M₂′ / O_denom)                     ║
║                                                            ║
║  O_denom = max(eligible windows の observedOutstandingMax) ║
║  O_denom ≥ 1（bootstrap invariant・post-first-commit）     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

**R_required の数値計算はまだしない**（λ/G/O_denom の値決定後）。

---

## 7. PASS 条件チェックリスト

```
[x] λ_ext の全 producer scope が closed        … §1.1（P-1〜P-8 分類完了）
[x] P-6 二重計上問題が closed                  … §1.2（commit-success ベース会計で構造的解消）
[x] λ の各項について finite upper-bound 根拠   … §1.3（workload contract 項はユーザー決定待ち・構造は確定）
[x] G_bound の environment premise が closed   … §2.2 bounded-starvation premise 確定
[x] starvation の扱いが closed                 … §2.2（K_starve premise + watchdog 将来強化）
[x] N_timer の +2 boundary term が closed      … §3 ⌊G/T⌋+1 に訂正（より厳密）
[x] O_denom の measurement protocol が closed  … §4 eligible window 定義
[x] O_w=0 を有効観測として保持                 … §4 dual-use separation
[x] O_denom>0 の eligibility が closed         … §4 post-first-commit 条件
[x] M₂′ symbolic formula が closed             … §6
[x] R_required = 1 + ceil(M₂′/O_denom) closed  … §6
[x] 数値はまだ採用していない                    … ✅
```

# VERDICT: D102-C2-1 = **PASS**

---

## 8. 訂正記録

| 項目 | 旧 | 新 | 理由 |
|---|---|---|---|
| N_timer 式 | ⌈G/T⌉ + 2 | **⌊G/T⌋ + 1** | spacing ≥ T_sampler の最密充填で floor+1 が正しい（閉区間端点含む）。⌈⌉+2 は過剰 |
| R6 メソッド名 | reclaimBatch | **drain(minReaderEpoch, isOlderFn)** | D101-34-D O-1 の表記訂正を本報告書でも反映 |

## 9. 次ステップ: D102-C2-2 — Numerical Value Determination

```text
D102-C2-1 PASS（本報告書 — 契約入力 closure）
      ↓
【ユーザー意思決定】
    ① λ_transition / λ_user_publish / λ_recovery の workload contract 値
    ② K_starve / G_bound の environment premise 承認
    ③ O_denom 測定期間の選択
      ↓
D102-C2-2: 数値適用
    M₂′ = 4096 + λ×G_bound + ⌊G_bound/100ms⌋ + 1
    R_required = 1 + ceil(M₂′ / O_denom)
      ↓
D102-C3: R_cap / T2 authority compatibility audit
      ↓
D102-C4: retention-capacity GO/NO-GO → Phase I NO-GO 解除判断
```
