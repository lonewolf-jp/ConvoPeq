# P1-5-IR-P2 — Step 5-P / P3-5-R8: Minimal Accessor Implementation Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R8）
- **種別**: read-only／implementation design。production変更・実装・build・実行なし。
- **目的**: R7-A の概念（read-handle内部利用＋値snapshot返却）を
  production API 実装時の最小変更集合に具体化する。
- **結論**: **R8-A（Implementation-ready）**（§10）。次は実装承認ゲートへ。
  本 Step では実装しない。P3-1-D は HOLD 維持。

---

## 1. State Freeze

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
F vehicle＋R1 diagnostic     保持（f4723815…／a00d140d…）
production/CMake/JUCE        0 diff
```

## 2. 新しい型の要否（§1 判定）

- `RuntimeState`（＝RuntimePublishWorld）：copy／move とも delete
  （AudioEngine.h:162-165）。値返却不能。pointer／reference 返却は
  ownership・lifetime 露出となり §3 禁止に抵触する。
- `GlobalSnapshot`（observed 側）：coordinator 内部 view であり、
  reader 束縛つき・target field（routing／automation／dspProjection）を
  直接持たない。流用は不可。
- `dspProjection` 単体：copy 可能だが identity（generation／worldId／sequence・
  routing／automation）を欠く。単独では target 照合不能。
- よって既存型の安全なそのまま返却は不可能であり、
  **最小 POD snapshot の新設は所有権規則上の必要性による**（便宜ではない）。
  POD は plain bool／int／uint64／double のみとし、所有権型を含めない。

## 3. accessor 公開位置（§2 比較・4軸）

| 候補 | RCU | Ownership | Mutation | Surface | 判定 |
| --- | --- | --- | --- | --- | --- |
| A AudioEngine public accessor | 内部完結（§5） | 値のみ | 診断 counter 更新のみ（既定動作と同一） | 1 POD＋1 method | **採用候補** |
| B 既存 snapshot accessor 拡張 | 同等 | 同等 | 同等 | 既存 API の意味変更を伴う（不整合） | 不採用 |
| C diagnostic facade | 同等 | 同等 | 同等 | 新 type＋配線で A より大 | 不採用 |

- B の拡張対象として適格な既存 accessor は存在しない
  （lifecycle／dispatch 系は counter 専用。snapshot 系は handle 束縛）。
- C は同一機能に対する余剰 surface のため不採用。
- A の 1 method は test 以外からも診断用途で再利用可能な一般形とするが、
  用途限定の命名・文書化は実装承認 gate の事項とする（名・位置は本 Step で決めない）。

## 4. 許可経路（§3 固定）

```text
public accessor（AudioEngine member）
    ↓ messageThreadRcuReader（private・member 内利用のため変更なし）
    ↓ makeRuntimeReadHandle(Message channel)
    ↓ getRuntimeWorldFromReadHandle
    ↓ 必要 field を local POD へ copy
    ↓ handle 破棄（scope 終了・RAII release）
    ↓ snapshot return（value）
```

禁止の遵守（設計上）：`RuntimeWorld*／&／RuntimeReadHandle／
shared_ptr／weak_ptr` の返却なし。`wait／sleep／capture／rebuild／publish／
retire／crossfade` を accessor 内に入れない。
`RuntimeReadHandle` は move-only・private ctor（AudioEngine friend・:2332-2362）
であり、member 内利用に閉じる。

## 5. field 対応表（§4・全 field source 確定・STOP-6 非該当）

| snapshot field | source member | 型 | copy |
| --- | --- | --- | --- |
| generation | `world->generation`（:184） | uint64_t | 可 |
| worldId | `world->worldId`（:178） | uint64_t | 可 |
| publication.sequenceId | `world->publication.sequenceId`（PublicationSemantic :256） | PublicationSequenceId | 可 |
| routing.processingOrder | `world->routing.processingOrder`（RoutingSemantic :217） | int | 可 |
| routing.eqBypassed／convBypassed | 同（:218-219） | bool | 可 |
| automation.softClipEnabled | `world->automation.softClipEnabled`（AutomationSemantic :333） | bool | 可 |
| automation.saturationAmount | 同（:334） | double | 可 |
| automation headroom／makeup／trim | 同 inputHeadroomGain／outputMakeupGain／convolverInputTrimGain（:335-337・linear gain） | double | 可 |
| dspProjection.oversamplingFactor | `world->dspProjection.oversamplingFactor`（:232・RuntimeBuilder.cpp:243-250 投影） | int | 可 |
| dspProjection.irLoaded／irFinalized | 同（:229-230） | bool | 可 |
| dspProjection.structuralHash | 同（:231） | uint64_t | 可 |
| overlap.fadeTimeSec | `world->overlap.fadeTimeSec`（OverlapSemantic :285） | double | 可 |

対応不明 field なし。代替値使用なし。

## 6. const 問題（§5 判定：R8-Const-B）

- `makeRuntimeReadHandle` は token 管理（acquire＋observe 更新）のため非const。
  よって accessor は **非const observer** として設計する。
- ただし `非const ≠ RT unsafe` を分離する（指示どおり）：
  用途は main-thread test の瞬間 read のみであり、audio path に触れない。
  read 内容自体は const `RuntimePublishWorld*` 経由の読取り専用である。

## 7. Allocation／mutex／atomic write（§6 確認）

- allocation：POD 値返却のみ。heap 確保なし。
- mutex：RCU token acquire は lock-free（audio 毎 callback precedent により裏付け。
  実装時検証事項として残すが、STOP 要件（「必要」）には該当しない）。
- atomic write：make 側の observe 系 diagnostic counter 更新のみ（既定動作と同一）。
  accessor が**新しい counter を増やさない**こと（STOP 遵守）。
- 新規 string 構築・logger 呼出しを accessor 内に置かない
  （整形は test 側の既存 emitLine で行う）。

## 8. test-only 差分（§7 設計）

```text
P1PolyphaseGainCharacterization.cpp
  T6： accessor 呼出し → active_before として既存 [P1CHAR] 行へ field 追加
  T9： accessor 呼出し → active_after として同上
```

- 新規 logger／CLI／wait／sleep／settle／retry／normalization／measurement なし。
- `[P1CHAR]` 行への field 追加は test-log 変更として実装承認 gate の事項とする。
- F／R 対称：同一 schema・同一 accessor を両 order で使用する。

## 9. log schema（§8 固定）

- `requested`／`active_before`／`active_after` の三者分離を維持する。
- `waitWorldPublished()==true` からの target 断定コードを禁止する。
- `seqAfter > seqBefore` 単独での target 確定を禁止する。
- 証拠は `publication sequence ＋ active snapshot ＋ requested snapshot` の独立保持とする。

## 10. STOP-1〜10 照合＋R8判定

```text
STOP-1（既存型で隠蔽不能）： 新 POD は所有権規則上の必要性による → 通過
STOP-2（World露出要）： 値のみ・露出なし → 通過
STOP-3（handle返却要）： scope内完結 → 通過
STOP-4（mutex/allocation要）： 不要（audio precedent＋POD） → 通過
STOP-5（publish/retire/rebuild呼出し）： なし → 通過
STOP-6（field対応不明）： 全対応確定（§5） → 通過
STOP-7（新規診断logger要）： なし → 通過
STOP-8（T0–T9変更要）： なし → 通過
STOP-9（measurement semantics変更要）： なし（観測のみ） → 通過
STOP-10（surface最小化不能）： 1 POD＋1 method → 通過
```

```text
R8-A（Implementation-ready）: ADOPTED
  既存型または最小POD＋既存RCU read path＋value-only return＋
  production差分最小＋test-only差分最小のすべて成立。
  → 次に production accessor 実装承認ゲートへ（本 Step では実装しない）。
R8-B（再設計）: REJECTED（§5・§10 充足のため）
R8-C（R7前提不成立）: REJECTED（成立のため）
```

停止位置：R8-A 確定で停止。実装・build・F/R実行・比較・P3-1-D には進まない。
R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
