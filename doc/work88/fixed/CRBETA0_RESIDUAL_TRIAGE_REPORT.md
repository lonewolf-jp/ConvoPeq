# CR-β-0 — Phase-I Residual Candidate Triage（read-only Work Report）

```text
CR-β-0 — Phase-I Residual Candidate Triage

Type:  read-only audit
Date:  2026-09-01
Baseline: ConvoPeq.md Generated 2026-09-01 21:47:45（--check 実測 FRESH / NEWER_SRC_COUNT=0 / CHECK_EXIT=0）
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0
Forbidden 遵守: Episode layer 復活 0 / E_max・O_max・E×O≤32 再導入 0 / CR-α 再オープン 0 / source modification 0
```

## 総合判定

> ## **CR-β-0 = 完了。両候補とも IMPLEMENTATION CANDIDATE ではなく — Phase-I は新規実装を増やさない**
>
> - **候補 A（CW-8 pair-snapshot）= ALREADY COVERED（枠組み実装済み・production caller 未接続は保守的構成）**
>   — 16:20 棚卸し（PLAN_DOCS_UNIMPLEMENTED_INVENTORY）時点から状況が進展: **ND-01〜04 CW-8 作業で
>   `PublishedWorldObservation` 型 + factory + test（T-CW8-1〜7）が既に実装・検証済み**。
>   指示前提の「単一 acquire load 保証の未実装」は現行 snapshot（21:47:45）では**成立しない**。
> - **候補 B（build-error telemetry）= DEFER / NO TRIGGER**
>   — 機能欠陥ではない（CR-α で retry contract 成立済み）。観測性強化のみで、実要件・failure 証拠・
>   trigger が存在しない。RuntimeBuilder.h:120-124 の既決監査（E-NEXT-6 / Phase D2-0 NO-GO）が
>   trigger 条件まで明示している。
>
> **分岐: 「両方 DEFER / ALREADY COVERED」→ Phase-I は新規実装を増やさず、次の実運用検証へ。**

---

## 候補 A — CW-8 PublishedWorldObservation pair-snapshot strengthening

### A-1. 現行実装状態（実測 — 16:20 棚卸しからの進展を含む）

```text
src/audioengine/RuntimeWorldAuthority.h:
  :73-107   PublishedWorldObservation 型 — private member + private ctor + friend 構造遮断
            （aggregate init / 独立 pair 捏造不能・copy/move のみ・trivially copyable）
  :224-249  observePublishedObservation(const ReadToken&) — member function template factory
            （RuntimeState 完全型を呼び出し点で要求・依存型 const StateT* による遅延意味検査）
            → runtimeStore_.observe() **1 回 acquire load** のみから {world, &world->publication}
            を同時確定・world==nullptr 時は {nullptr, nullptr}
src/tests/ISRSemanticValidationTests.cpp :916-1066:
  testCW8_PublishedWorldObservation() — T-CW8-1（null + pair address identity）/
  T-CW8-2/3（publish N → observe → publish N+1 → observe・旧 identity 凍結）/ T-CW8-4（black-box
  bake == observed == identity）/ T-CW8-6/7（型レベル制約・non-aggregate / non-default-constructible /
  non-pair-constructible — T-CW8-7 = 「間違った pair は構造的に生成不能」の核心）
検証実績: ND-04 報告 — Debug/Release targeted build + standalone run PASS・static_assert 群は
  compile-time 成立（T-CW8-7）・その後 CR-α-3R/α-4 の full build + CTest 40/40 ×2 で regression 維持
```

### A-2. 既存 invariant / contract で十分か

十分。三層で成立:

| 層 | 内容 |
|---|---|
| 型構造 | private member + private ctor + friend = `{world N, identity N+1}` 混在ペアは**型システム上生成不能**（T-CW8-7 が compile-time で表明） |
| 単一物理 read | factory = `observe()` 1 回 acquire load（:245）のみ — 二段 read / 別 atomic は禁止（:225-227 コメント明記） |
| identity 導出 | `&world->publication` — 同一オブジェクト内部 pointer であり独立 storage が存在しない（CW-5: `RuntimeStore::current.identity == RuntimeState::publication.identity` と構造的に同根）+ INV-X4-6/7/A（bake-before-swap・単一 read source） |

### A-3〜A-5. Trigger / Safety / RT / Ownership

- **concrete trigger**: 不在。指示例の「単一 acquire load による read-contract 保証」自体が
  上述のとおり**実装済み**であり、強化の余地（新 atomic・新 topology・read path 変更）は残っていない。
- **safety / RT impact**: 型は非所有 borrow のみで publish/retire に参加しない（:81-89 契約明記）。
  production caller 未接続は「読む API が増えない = read path の不変」を意味する**保守的状態**。
- **ownership impact**: RuntimeWorldAuthority 内のみ・新 authority / 新 atomic なし（:75）。

### A-6. test 追加の必要性

なし。T-CW8-1〜7 が型契約・null 契約・世代進行・black-box 等価・構造遮断まで網羅（ND-04 検証済み）。
production caller が接続された時点で初めて「caller 側契約 test」が論理的に定義可能 — 現時点では
追加対象が存在しない。

### A-7. Disposition

> **ALREADY COVERED** — CW-8 read-side strengthening は ND-01〜04 で型 + factory + test として
> 実装・検証済み。単一 acquire load read-contract も factory 内に実装済み（:224-249）。
> production caller 未接続は保守的構成であり、**新規 CR 起票の必要なし**。
> 注記: 指示文・16:20 棚卸しの「未実装」記述は本日 16:20〜19:31 の ND-01〜04 作業より前の状態を
> 反映しており、現行 snapshot では陳腐化している（D159 規則: 旧記述を authority としない）。

## 候補 B — buildErrorCount_ / build-error telemetry strengthening

### B-1. 現行実装状態（実測）

```text
buildErrorCount_ : src 0 件（RuntimeBuilder.h:124 の将来拡張コメント言及のみ — 16:20 棚卸しと同値）
RetryScheduler   : rejectCount_（atomic uint64・RetryScheduler.h:58）+ pendingCount() 実装済み
Site 3（CR-α）   : diagLog による generation / attempt / limit / delayMs / error の逐次記録 + one-shot
                   exhaustion terminal log（CR-α-5 V-α5-09 で契約整合確認済み）
classify 経路    : RebuildDispatch.cpp:1197（build failure）/ :1273（warmup failure）の 2 箇所
MKLFailure / ConvolverFailure / PrepareFailure: enum + table + test のみ・production 生成経路 0
```

### B-2. 既存 contract で十分か

十分。観測に必要な情報は現行で成立:

1. **BuildError 分類**: `classifyBuildError()`（2 call sites）→ diagLog に error 名・classification・
   disposition を記録（:1198-1204 / :1255-1261）
2. **Retry 追跡性**: CR-α で Site 3 diagLog が generation / attempt / limit / delayMs / error /
   "(no further retry…)" を網羅（CR-α-5 V-α5-09 契約照合済み）
3. **Scheduler 側**: `rejectCount_` / `pendingCount()` が実装済み（retry drop の観測先が存在）
4. **counter 集約がなくても**: exhaustion は one-shot log、retry は attempt 付き逐次 log で
   復元可能（long-running 統計が必要になった時点が counter 導入の適時点）

### B-3〜B-5. Trigger / Safety / RT / Ownership

- **concrete trigger**: 不在。指示どおり**機能欠陥として扱わない**（retry contract は CR-α で成立済み）。
  RuntimeBuilder.h:119-124 の既決監査（E-NEXT-6 / Phase D2-0 2026-08-19 NO-GO）が trigger を明示:
  「将来 convolver/prepare の実 failure が観測可能になり subsystem 別 retry 判定が必要になる設計確定時」
  — 現在その要件は発生していない。
- **safety impact**: counter 追加自体は低リスクだが、`fetchAdd` 挿入点が非 RT 排他領域に増える。
  現行に failure/性能証拠がなく、このリスクを取る根拠がない。
- **RT impact**: RuntimeBuilder / RebuildDispatch は非 RT スレッド（RebuildThread / MessageThread）。
  RT への影響なし（追加しても）。ただし RT 影響ゼロであること自体が「今やる緊急性の根拠」ではない。
- **ownership impact**: BuildErrorPolicy.h への counter 追加は policy 純関数性を崩す恐れ（CR-α-5
  で純関数構造を audit PASS 済み）。TelemetryRecorder 経由にする場合は StateOwner-owned 境界
  （Authority Surface 規約）の確認が必要 — 現状その設計判断をする材料がない。

### B-6. test 追加の必要性

現時点で不要。counter が存在しないため test 対象がなく、cr-α test（checks=86）が policy/decision を
網羅済み。導入するなら用途（統計出力先・閾値アラート等）とともに test contract を同時設計する。

### B-7. Disposition

> **DEFER / NO TRIGGER** — 観測性強化として独立評価の対象だが、実要件・failure 証拠・D159 trigger
> 発生のいずれも不在。CR-α の retry contract とは分離して、将来「subsystem 別 retry 判定の設計確定」
> または「長時間運用での統計需要」が発生した時点で新規 CR として起票する。**現時点での実装禁止**。

## D159 freeze status / 整合性

| 項目 | 状態 |
|---|---|
| 候補 A（CW-8） | D159 freeze register 未登録（16:20 棚卸し時点）→ **本 triage で ALREADY COVERED 確定のため登録不要**（実装済み領域は register 対象外） |
| 候補 B（telemetry） | D159 freeze register 未登録 → **補助 trigger としての登録を推奨**（記録・監視のみ）: trigger =「subsystem 別 retry 判定の設計確定」または「運用統計の実需要」— D159 補助 trigger（RecoveryEpisodeId=D2 従属 等）と同形式 |
| Episode / E_max / O_max / E×O≤32 | Phase-II deferred のまま（I4 §7.5・D2/D3 凍結 — 本日も trigger 非発生・着手不適を再確認） |
| CR-α {10,80,2} / K=3 | CLOSED 固定（CR-α-6 §8 規約）— 本 triage で触れない |

## 判定後の分岐（指示の 4 択）

```text
★ 両方 DEFER / ALREADY COVERED → Phase-I は新規実装を増やさず、次の実運用検証へ
  （CW-8 のみ IMPLEMENTATION CANDIDATE → 該当せず）
  （telemetry のみ IMPLEMENTATION CANDIDATE → 該当せず）
  （NEW REQUIREMENT NEEDED → 該当せず）
```

## 禁止事項遵守（実測）

```text
Production source 変更: 0 / Test source 変更: 0 / CMake 変更: 0
Build: 0 / CTest: 0 / stress: 0
Episode layer 復活: 0（RecoveryEpisodeId production 0 件を現行でも再実測）
E_max / O_max / E×O≤32 再導入: 0（Phase-I invariant は L_logical_max=32 のまま）
CR-α 再オープン: 0（{10,80,2} / K=3 無変更）
本報告の artifacts: 本ファイルのみ
```

## Next

```text
CR-β-0 完了 → Phase-I 新規実装なし → 次の実運用検証（D116 系 operational validation 継続 or
ユーザー指定の次 work item）を待つ。候補 B の freeze register 補助 trigger 登録は
次の編集ウィンドウで（D159 補助 trigger 運用に準拠・監視のみ）。
```
