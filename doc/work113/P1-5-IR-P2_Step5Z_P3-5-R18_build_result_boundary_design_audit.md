# P1-5-IR-P2 — Step 5-Z / P3-5-R18: Build-Result Boundary Design Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R18）
- **種別**: read-only boundary audit。実装・build・run なし。
- **目的**: R17 の U6-post（taken=1／published=0）を分ける最小1観測点
  （build-result boundary）を確定する。
- **結論**: **R18-A**。次は R19（counter 実装＋build gate）へ。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
F vehicle＋R1 diag＋R10 accessor＋R16 take counter 保持
production 0 diff（R16承認範囲のみ）
ConvoPeq.md freshness： P1 TU qdelta のみ STALE（test-only・production basis 不変。
  本監査の production 経路には影響しない）
```

## 2. R17-A result inheritance

```text
U6-pre → 排除（全 window takeΔ>0）
U6-post → 確定（taken=1／published=0 全 window）
```

残課題は U6-post 内部（build／validation／commit-drop／exception／slow）の分離である。

## 3. Latest ConvoPeq.md source basis

R17 Step 0 再生成版（R16 実装含有）を基準とする。worker loop（:860-1422）、
RuntimeBuilder（:454-549）、PublicationAdmission、BuildErrorPolicy は
R2／R3／R12／R13 の監査済み内容を継承し、本 Step では境界確定に必要な差分のみ再読した。

## 4. Take → Build → Validate → Commit → Publish control-flow（実コード）

```text
take（:906-909・ownership移動・take counter位置）
  → deferred-publish handoff 消費（:913-920）
  → D135 latch drain（:932-936）
  → coordinator deferred／recovery 処理（:942-1175）
      ※ recovery は独自 generation で build→validate→commit しうる別 task 系。
         gap publishes（R11 seq 5→8）の候補経路の一つとして記録（断定なし）。
  → !wokeByPendingTask → continue（:1174-1175）
  → obsolete check（:1182-1193）
  → sealed check（:1196-1197）
  → warmup-retry rebind（:1198-1203）
  → RuntimeBuilder::build（:1207）
  → null check（:1211-1227・fail は diagLog のみ＋continue）
  → obsolete re-check（:1232-1241）
  → rebuildAllIRs（:1244-1262・IR有時のみ）
  → validateWarmup（:1272・fail は retry／exhausted／silent）
  → obsolete re-check（:1342-1352）
  → latency refresh・fade-in・[CONV_STATUS]（:1354-1371）
  → commit enqueue（:1408）
  → coordinator 消費 → admission → sequence bump（Commit.cpp:402）
```

## 5. Build-result boundary candidates（B0–B5 の source 対応）

```text
B0 take：                 :906-909（R16 counter 位置・確定済み）
B1 build invocation：     :1207（RuntimeBuilder::build 呼出し）
B2 build success：        :1211 null-check 通過直後（runtime 非null＋prepared）
B3 validation success：   :1272 validateWarmup None（＋WORK105 は build 内包のため B2 側）
B4 commit attempt：       :1408 enqueuePublicationIntent
B5 publication advance：  Commit.cpp:402 の sequence bump（観測：既存 sequence）
```

- 「build completed」と「publishable」を同一視しない（指示どおり）：
  B2 は usable-runtime 到達であり、validate／commit／admission を含まない。
- 最初の不可逆な観測境界は **B2** である。
  take→B2 間に外部可視の分岐はなく（obsolete :1182 のみ・同条件では不成立）、
  B2 以降の全 drop は B2 通過として記録される。

## 6. Candidate comparison（最小観測点の決定）

- 採用：**B2 直後（null-check 通過後・:1232 obsolete 再検査の前）**に1点。
  - `taken=1／buildResult=0／published=0` → build-stage cluster
    （InvalidInput／contract-refused／bad_alloc／catch-all／
     obsolete-pre-build／exception-in-build／slow-build）。
  - `taken=1／buildResult=1／published=0` → post-build cluster
    （obsolete-post／rebuildIR／validate-fail／commit-drop／exception-late／
     slow-late／commit-side health-pressure-fading）。
- B3（validate 後）配置は post 側を commit のみに狭める代わりに
  build-fail と validate-fail を一括りにするため不採用。
  build-fail／validate-fail の分離は B2 点で確保する（§8）。
- Candidate B（success／failure 複数 counter）・C（多状態 telemetry）は
  最小原則により不採用（指示どおり）。

## 7. Obsolete/drop analysis（§6B・確定）

- obsolete 判定は :1182（pre-build）・:1232（post-build）・:1342（post-warmup）の3点。
  predicate は newer-generation 要求であり、本 vehicle window 内に newer 源はない
  （自 intent 群は同一 task・loader 静止・retry／recovery 未発火・fault なし）。
- B2 点は :1232 より前のため、:1232 以降の drop は buildResult=1 側に正しく属する。
- `!task.runtimeBuildSnapshot.sealed → continue`（:1192-1193）は queue 時に
  seal されるため本経路では不発（通過見込み・記録のみ）。

## 8. Build failure analysis（§6A・列挙・計測追加なし）

```text
InvalidInput（sr／block不正）→ null＋error→ diagLog＋continue
IRRateMismatch／IRBlockMismatch（WORK105・convBypass 時のみ評価）→ 同上
ResourceUnavailable（bad_alloc）→ 同上
InternalError（catch-all）→ 同上
build内例外 → 上記 error-return に変換（build 自体 noexcept のため escape なし。
  escape は process death であり R11 clean exit と矛盾）
worker loop catch（:1410-1418）→ DBG のみ＋continue（thread 生存）
```

いずれも release vehicle 非出力（diagLog／DBG のみ）のため、
B2 点以前の内訳は本一点では分離しない（段階化の範囲内）。

## 9. Validation boundary（§6C・確定）

- `validateWarmup`（:1272）は B2 の下流に位置する。
  よって B2 点は validation 成否を含まず、build／validation の分離が成立する。
- WORK105 契約照合は `RuntimeBuilder::build` 内包（:491-519）のため B2 側に属する。
  g0／os1 条件（48k／1024 一致）では通過見込みである。

## 10. Commit boundary（§6D・確定・counter 実装なし）

- `enqueuePublicationIntentForRuntimeCommit`（:1408）→ coordinator 消費 →
  admission（shutdown／generation-stale／not-finalized／health／pressure／
  fading-defer）→ sequence bump。
- commit counter は本 Step では実装しない（R17 staged 計画の維持）。

## 11. Minimal observation design（1点）

```text
新規 counter ×1（take counter と同形：atomic uint64・process累積・差分運用）
increment 位置： B2（null-check 通過直後・:1232 の前）
reader： const noexcept scalar getter ×1（R15／R16 パターン踏襲）
RCU／mutex／allocation／ownership／World 露出： なし
audio／RT 影響： なし（worker-only atomic）
publish／retire／rebuild 副作用： なし
```

- queued／taken／published と合わせ `1／1／0 vs 1／0／0` の上位分離に加え、
  `taken=1` 時の build 到達有無が確定する。
- build-result 以降の内訳（validate／commit）は次段条件付きとする。

## 12. Production/test change boundary

- production： counter 宣言＋getter＋B2 increment の3 edits（R16 同形）。
  dispatch／commit／RCU／measurement／logger／CLI 不変。
- test： B2 値の差分読出し＋既存行追加のみ（承認後・本 Step では実施しない）。
- wait／sleep／settle／timeout／retry／health／pressure 計装なし。

## 13. R18-A/B/C gate

```text
R18-A： ADOPTED
  実コード経路確定／B2 境界確定／obsolete-drop 確認／exception-failure 確認／
  validation 位置確認／commit 位置確認／単一観測点／production 0／F-R 0 の
  すべて成立。次は R19（counter 実装＋build gate）。
R18-B／R18-C： 非該当。
```

## 14. STOP（本 Step 終了）

- 実装・build・F/R実行・比較・P3-1-D なし。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
