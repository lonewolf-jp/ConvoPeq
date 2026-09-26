# P1-5-IR-P2 — Step 5-AC / P3-5-R21: B2→Publish Boundary Design Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R21）
- **種別**: read-only design audit。実装・build・run なし。
- **目的**: R17 の U6-post（taken=1／published=0）を分ける最小追加観測点
  （commit-enqueue boundary）を確定する。原因特定はしない。
- **結論**: **R21-A**。次は commit-enqueue counter の実装承認ゲートへ。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
R16 take counter＋R19 B2 counter＋R10 accessor＋T6/T9＋delta観測 保持
ConvoPeq.md 再生成済み（本 Step 開始時）。R20行番号は信用せず再特定した。
production／CMake／JUCE 0 diff。
```

## 2. Latest ConvoPeq.md source reconciliation

- take（:906-909）→ B2 increment（null-check 直後）→ validate（:1281）→
  commit-enqueue（:1417・main site）→ coordinator → admission →
  sequence bump（Commit.cpp:402）の全経路を live source で再特定した。
- `enqueuePublicationIntentForRuntimeCommit` の呼出しは3箇所
  （main :1417＋recovery :1085／:1165）。関数自体は単一である。

## 3. R20 observation chain（継承）

```text
queued（既存）→ taken（R16）→ B2 reached（R19）→ published（既存 sequence）
```

R20 で pair2 両 pass が `taken=1／bld=0`（B2前）、pair1 が `taken=3／bld=1／seq+1`
（B2後混在）であることが確定している。

## 4. B2 → publish source trace（再特定）

```text
B2（null-check 通過）
  → obsolete re-check（:1232・newer-generation 要求）
  → rebuildAllIRs（:1244・IR有時のみ）
  → validateWarmup（:1272・fail は retry／exhausted／silent）
  → obsolete re-check（:1342）
  → latency refresh・fade-in・[CONV_STATUS]（:1354-1371）
  → commit enqueue（:1417・main site）
  → coordinator 消費 → admission evaluate
  → validators（Commit.cpp:186-229：semantic／monotonic／topology）
  → crossfade／timeout／recovery orchestration
  → sequence bump（Commit.cpp:402）
```

## 5. Candidate observation boundaries（比較）

| 候補 | 意味 | 判定 |
| --- | --- | --- |
| B2（既設） | build 成功境界 | 基準点 |
| validation reached | B2後の validation 到達 | 別点追加となり多段化。commit-enqueue 点が包含するため不採用 |
| **commit enqueue** | publish 要求の commit 側受渡し | **第一候補→採用（§7）** |
| coordinator／admission | Authority 到達以深 | 粒度過細。commit 点で残余が出た場合の次段に温存 |
| sequence++（既設） | 実 publish 到達 | 終端点 |

- commit-enqueue 点は validation 通過を含意する（:1272 の下流に位置）。
  よって1点で {B2→commit-enqueue間} と {commit-enqueue→publish間} を分離できる。
  これが最小 N=1 の根拠である。

## 6. Minimal-boundary comparison（採用理由）

- B2→commit-enqueue 間の残余（obsolete／rebuildIR／validate／exception／slow）と
  commit-enqueue→publish 間の残余（admission／health／pressure／fading／
  generation-stale／coordinator）を1点で切断できる。
- validation 単独点は commit 点に包含されるため不要（多段化の回避）。
- coordinator 以深は commit 点で残余が出た場合の次段に温存する（段階化の維持）。

## 7. commit-enqueue topology（§6B 回答）

- 関数は単一だが呼出しは3箇所（main＋recovery×2）。
- 主 task の運命追跡には **main-site（:1417）配置**を採用する。
  関数内配置は recovery enqueue を混入させ、U6-post 分離を汚染するため不採用。
- 配置は enqueue 呼出しの**到達**を示す（呼出し前置）。呼出し自体の失敗
  （例外等）は別途 worker catch で沈黙する既知経路であり、残余として記録する。

## 8. Admission/coordinator topology（§6C 回答）

- enqueue 後の reject 経路：shutdown／generation-stale／not-finalized／
  health Critical-Degraded／pressure throttle／fading-defer
  （PublicationAdmission＋Commit.cpp:186-229 validators）。
- reject／drop は `lastDroppedGeneration_` に記録される（既存・lifecycle 差分で読出可）。
  よって commit-enqueue 点と組合わせれば三者分離が成立する：
  `commit=1／droppedΔ>0／seqΔ=0`（commit 棄却）、
  `commit=1／droppedΔ=0／seqΔ=0`（coordinator 滞留・defer 継続）、
  `commit=1／seqΔ>0`（publish 到達）。
- したがって enqueue 点だけで「publish に向かった」ことは主張しない
  （指示どおり）。三点保持が条件である。

## 9. sequence++ relationship（§6D 回答）

- enqueue→sequence 間には coordinator queue・admission・validators・
  crossfade／timeout／recovery orchestration・ownership transition が介在する
  （§4）。failure／drop は各段に存在し、いずれも release vehicle 沈黙である。
- sequence bump（Commit.cpp:402）が唯一の publish 証拠であることに変わりはない。

## 10. Selected minimal observation point（1点のみ）

```text
新規 counter ×1（take counter 同形：atomic uint64・process累積・差分運用）
increment 位置： main-site commit-enqueue 呼出し到達時（:1417）
reader： const noexcept scalar getter ×1（R15／R16 パターン踏襲）
RCU／mutex／allocation／ownership／World 露出： なし
audio／RT 影響： なし（worker-only atomic）
publish／retire／rebuild 副作用： なし
```

- build-result 以降の追加 counter（validation／commit-attempt  doctrines）は
  本一点に含めない。必要になれば次段で個別に設計する。
- test 側は既存 delta パターン（pre／post＋WARN 行）で読む。
  新規 logger／CLI／wait／sleep／settle／retry なし。

## 11. Rejected alternatives（記録）

- validation-reached 点：commit 点に包含されるため不要。
- coordinator／admission 点群：粒度過細。commit 点残余時の次段に温存。
- 関数内配置（全呼出し計数）：recovery 混入のため不採用。
- DrainedCommand 復活・logger 追加・timeout 変更：禁止事項として不採用。

## 12. Remaining observability gap

- commit-enqueue 点と lastDroppedGeneration／sequence の三点でも、
  coordinator 滞留 vs defer 継続の内訳は不可分（次段の対象）。
- per-task attribution（generation linkage）は引き続きなし。
- gap publishes（R11 seq 5→8）の発行主体は本設計の対象外（別 track として記録のみ）。

## 13. Implementation scope for next step

- production： counter 宣言＋getter＋main-site increment の3 edits（R16 同形）。
- test： delta 読出し＋既存行追加のみ。
- G1–G7 相当の gate（R16 Step5X 踏襲）を適用する。
- 実装自体は次gate承認事項とし、本 Step では行わない。

## 14. STOP + R21 Gate

```text
R21-A： ADOPTED
  B2→publish 実コード経路の再特定／追加観測点の最小1点化（main-site commit-enqueue）／
  既存 authority 不変／原因帰属なし／production 変更なしのすべて成立。
  → 次は commit-enqueue counter の実装承認ゲートへ（本 Step では実装しない）。
R21-B／R21-C： 非該当。
```

- F/R 実行・比較・P3-1-D・limiter／stale／crossfade 帰属なし。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
