# P1-5-IR-P2 — Step 5-W / P3-5-R15: Take-Boundary Accessor Design Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R15）
- **種別**: read-only design audit。実装・build・run なし。
- **目的**: R14-C の take 観測を R7–R10 と同じ設計基準で1案に絞る。
- **結論**: **R15-A（採用設計確定）**。次は R16＝実装承認待ち。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
F vehicle＋R1 diagnostic＋R10 accessor 保持／production／CMake／JUCE 0 diff
duplicate suppression・snapshot sealing の再評価なし（現行のまま）
```

## 2. take 境界の再トレース（実コード）

```text
submitRebuildIntent（:151-403）
  → requestRebuild(sr,bs)（MT直接・:343-363）
  → queue成功： pendingTask=task・hasPendingTask=true（:753-754・呼出スレッド）
  → rebuildBacklog_=1（:755）→ notify_all（:765）
  → worker loop wake（predicate :884-889）
  → ownership take（:906-909）
      task = pendingTask; pendingTask.currentDSP = nullptr; hasPendingTask = false;
  → rebuildBacklog_=0（:921）
  → build → validation → commit → publication
```

- take 実行箇所は :906-909 の**唯一箇所**である。
  他の `hasPendingTask` 書込みは queue（:754）・shutdown（:894）・
  lifecycle遷移（CtorDtor:183・PrepareToPlay:100・ReleaseResources:210）のみで、
  測定 window 内では take／queue のみが動作する。
- queued／taken を別イベントとして定義する：
  queued＝queue insertion accepted（:753-765完了）、
  taken＝worker が ownership を取得したこと（:906-909完了）。

## 3. accessor 目的の限定（1点のみ）

> worker が queued rebuild task を実際に取得したことを test vehicle が観測する

- build result／commit result／publication result／RuntimeWorld／health／
  pressure／fade／DSP ownership を混ぜない（指示どおり）。
- R10 `getActiveRuntimeSnapshot()` の対象外である（active 観測と混同しない）。

## 4. 候補比較（A–E）

| 案 | 内容 | 判定 |
| --- | --- | --- |
| **A monotonic counter** | worker-take 直後に fetch-add する累積カウンタ。差分で taken Δ を読む | **第一候補→採用（§6）** |
| B consume-observation | destructive read。履歴損失＋mutation（非const・exchange）が必要 | 棄却（A が lossless かつ non-mutating のため） |
| C dispatch diagnostics 拡張 | 既存 struct への take field 追加 | 棄却（dispatch 層と worker-take 層の混同。既存形の安定性を損なう） |
| D boolean hasWorkerTaken | stale 問題（reset protocol・race が要る） | 棄却 |
| E pendingTask／hasPendingTask getter | private worker state の露出大（mutex 保護構造・所有権含む） | 非推奨（指示どおり） |

## 5. 不変条件（採用案の充足）

```text
const：              読出しは const getter（atomic load）で充足
noexcept：           fetch-add＋load のみ。例外経路なし
POD／scalar only：   std::uint64_t カウンタ1点（＋同型 reader）
ownership exposure： 0（scalar のみ。pointer／handle／queue 非露出）
RuntimeWorld exposure： 0
DSP pointer exposure： 0
mutex exposure：     0（atomic のみ。lock なし）
queue object exposure： 0
RT path impact：     0（writer は worker thread のみ。audio thread 非関与）
publish／retire／rebuild side effect： 0（increment のみ）
```

- increment 位置は **ownership 取得直後**（:906-909 内）に限定する。
  queue 成功時（:753-765）に置いてはならない（queued==taken の再同一視になる）。
- lifetime：process 累積（reset なし・差分運用）。uint64 実用上 overflow なし。
- thread safety：単一 writer（worker）／複数 reader。release（加算）／
  acquire（読出し）を codebase idiom（publishAtomic／consumeAtomic 形）に従う。
- memory ordering の必要性：worker→main の可視性のため release／acquire を要する
  （x86 TSO 下でも idiom として明示する）。

## 6. counter semantics（明文化）

```text
queuedCount Δ：   queue insertion accepted（既存 :764）
takenCount Δ：    worker が queued task の ownership を取得（新設・§5位置）
published：       publication sequence 前進（既存）
```

- `queued Δ=1／taken Δ=0／published Δ=0` → queue 止まり（U6-pre 確定）。
- `queued Δ=1／taken Δ=1／published Δ=0` → U6-post（take 後 publish 未達）。
  build／commit 内訳は次段の観測対象となる（§7）。
- queued／taken の語義重複なし（§5 位置制約による）。

## 7. U4／U1 の staging（追加 instrument なし）

- U4：take 確定後に `takeあり＋publishなし` が出た場合のみ build-result 点を検討する。
  `catch` 内部への logger 追加方向には進まない（指示どおり）。
  よって現時点の結論は `U4 = still unresolved` のままである。
- U1：health／pressure／fading の instrument は commit-attempt 確立後。
  commit admission 入力リスト（R13 確定）は維持し、追加観測しない。

## 8. R15 要求表への回答

| 項目 | R15で確定する内容 |
| --- | --- |
| take境界 | :906-909（唯一・§2） |
| accessor対象 | take-count のみ（§3） |
| event semantics | queued／taken／published の三分離（§6） |
| candidate | A採用・B–E棄却理由付き（§4） |
| lifetime | process累積・差分運用（§5） |
| thread safety | 単writer／複数reader・release／acquire（§5） |
| memory ordering | 可視性のため必要・idiom準拠（§5） |
| RT影響 | なし（worker-only atomic・audio非関与）（§5） |
| ownership | 非露出（scalarのみ）（§5） |
| U6 | pre／post境界への効果（§6） |
| U4 | 不可分のまま（staged条件付き）（§7） |
| U1 | instrument保留（§7） |
| R16条件 | 実装承認（§9）。build-result点は take後残余時のみ |

## 9. R15-A/B/C 判定

```text
R15-A（採用設計確定）： ADOPTED
  take境界一意・最小accessor一意・scalar／POD・noexcept・ownership非露出・
  RT影響なし・side effectなし・queued／taken非重複・差分観測可・
  production変更範囲最小（counter＋reader）のすべて成立。
  → R16＝実装承認待ち。
R15-B（設計残件）： 非該当（lifetime／order／ownership は§5で閉じた）。
R15-C（想定外）： 非該当（単純 counter で安全に観測可能。authority 変更不要）。
```

停止位置：設計確定で停止。実装・build・F/R実行・比較・P3-1-D なし。
R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
