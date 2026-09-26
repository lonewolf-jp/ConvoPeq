# P1-5-IR-P2 — Step 5-Q / P3-5-R9: Accessor Implementation Approval Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R9）
- **種別**: read-only／承認監査。production変更・実装・build・実行なし。
- **目的**: R8-A の最小 POD＋public value accessor を production 適用前に最終確定する。
- **結論**: **R9-A APPROVED**（§12）。実装自体は別Step（identity／build gate）へ。

---

## 1. State Freeze

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
F vehicle＋R1 diagnostic     保持（f4723815…／a00d140d…）
production/CMake/JUCE        0 diff
```

## 2. 実装対象（固定・未実施）

```text
src/audioengine/AudioEngine.h
    └─ 最小 snapshot POD（§3）
    └─ public accessor（§4、1 method）

src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp
    └─ T6 active_before（§9）
    └─ T9 active_after（§9）
```

追加ファイルなし（R8「既存型に変更なし」の範囲を維持）。

## 3. POD 最終設計（R8 field表を実装仕様化）

```text
Identity:   generation／worldId／publicationSequence（uint64系）
Routing:    processingOrder（int）／eqBypassed／convBypassed（bool）
Automation: softClipEnabled（bool）／saturationAmount／
            inputHeadroomGain／outputMakeupGain／convolverInputTrimGain（double）
DSP proj:   oversamplingFactor（int）／irLoaded／irFinalized（bool）／
            structuralHash（uint64）
Overlap:    fadeTimeSec（double）
```

除外（R8維持）：sampleRate系／RuntimeWorld*／RuntimeState*／RuntimeReadHandle／
GlobalSnapshot*／shared_ptr／unique_ptr／string／logger state。
全 field の source 対応は R8 §5 で確定済み。STOP 事項（未確定 field の追加願望）なし。

## 4. accessor 実装経路（固定）

R7 の full-handle 経路を本監査で**縮小**する。既存 precedent
`hasPublishedRuntimeDSP()`（AudioEngine.h:2320-2325）にならい：

```text
public accessor（AudioEngine member）
    ↓ worldAuthority_.acquireReadToken()
    ↓ worldAuthority_.consumeWorldHandle(token)
    ↓ null check（§6）
    ↓ 必要 field を local POD へ copy
    ↓ token scope 終了（RAII release）
    ↓ POD value return
```

- `RuntimeReadHandle` クラス自体を使わない（observeCurrentRuntime の
  GlobalSnapshot copy＋observe diagnostic 更新を回避する最小形）。
  token の acquire／release は同 precedent と同一 pattern である。
- 返却禁止（RuntimePublishWorld*／&／RuntimeReadHandle／shared／weak）を遵守する。
- `wait／sleep／capture／rebuild／publish／retire／crossfade` を入れない。

## 5. observeCurrentRuntime の扱い（明確化）

- §4 経路では `observeCurrentRuntime()` を**呼ばない**。
  よって GlobalSnapshot copy・observe 系 counter 更新は発生しない。
- handle lifetime は accessor scope 内で完結する。
  POD への copy 完了後に token が破棄されることを実装レビューで確認する
  （precedent と同一 RAII 形）。
- 禁止：POD 内 pointer 残置／handle 返却／member 保存／static 保持／
  capture 区間保持（R6 §4 規約を継承）。

## 6. nullptr／no-world 契約（新 validity なしで確定）

- `consumeWorldHandle` が nullptr の場合：**zero-initialized POD を返す**。
- validity は caller 側で `(generation==0 && worldId==0 && sequenceId==0)` により判定する。
  根拠：`publicationSequenceCounter_{0}`＋fetch-add+1（:2392／:3621-3623）、
  id／generation 系 generator の next() 形より、0 は未発行値である。
- 新 validity counter／logger／state machine を追加しない（指示どおり）。
  STOP 事項（新 production state 要件）なし。

## 7. constness（R8-Const-B→Const-A へ改訂）

- precedent `hasPublishedRuntimeDSP()` は **`const noexcept`** で同一 token 経路を
  使用する。よって read 経路は const-compatible であり、accessor は
  **R8-Const-A（const accessor 可）**として設計する。
- `non-const method ≠ RuntimeWorld mutation` を分離記録する：
  accessor 内に World 変更操作は存在しない（§8）。

## 8. 副作用監査（追加は RCU read＋field copy＋return のみ）

```text
mutex／new-delete／filesystem／Logger／DBG／juce::String／
wait／sleep／rebuild／publish／retire／crossfade操作／新規atomic counter／
新規telemetry： すべてなし（§4 経路に存在しない）
```

- 既存 observe 系更新は §4 経路では発生しない（observeCurrentRuntime 不使用）。
  よって accessor 側の新規 counter 増加もない。
- R8 時点の「diagnostic counter 更新の許容」記載は本縮小経路では不要となる
  （緩和方向の変更であり、承認条件に影響しない）。

## 9. test-only 差分（T6／T9 のみ・未実施）

- T6：accessor 呼出し → `active_before` として既存 `[P1CHAR]` 行へ field 追加。
- T9：accessor 呼出し → `active_after` として同上。
- `requested == active_after` の成功判定コードを入れない。
- `seqAfter > seqBefore` 単独での target 断定をしない。
- 証拠は `requested snapshot ＋ publication sequence ＋ active_before／after` の
  独立保持とする（R6 §6／R8 §9 維持）。
- 新規 logger／CLI／wait／sleep／settle／retry／normalization／measurement なし。
- F／R 実行条件の変更なし（§10）。

## 10. F/R 実行条件（変更禁止の再掲）

```text
30s wait／waitWorldPublished()／sleepPump(800)／500ms post-settle／
measurement formula／DFT normalization／capture block／warmup／
IR generation／OS=1／g0／kP15FullMatrix=true／CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF
```

R9 は観測能力の追加承認のみであり、測定プロトコル改善ではない。

## 11. G1–G6 検証ゲート（実装後の確認項目として固定）

```text
G1 production diff： AudioEngine.h の POD＋accessor のみ
G2 test diff：       P1 TU の T6／T9 snapshot取得・出力のみ
G3 forbidden diff：  CMake／JUCE／RuntimeBuilder／Commit／Publish／Retire／
                     RCU実装／Rebuild dispatch／measurement formula／
                     sleep-wait／CLI／logger が0件
G4 API surface：     new POD＝1／new public method＝1 のみ
G5 ownership：       RuntimeWorld*／&／RuntimeReadHandle／GlobalSnapshot*／
                     shared_ptr／unique_ptr の露出0件
G6 runtime semantics：publish／retire／rebuild／crossfade／wait／sleep の呼出し0件
```

## 12. R9判定

```text
R9-A  APPROVED： 上記§2–§11のすべて成立。
  → production accessor 実装完了 → build／identity gate へ（別Step）。
R9-B（要改訂）／R9-C（STOP）： 該当なし。
```

R9-A 後も本 Step で停止する。build・F／R vehicle・比較・P3-1-D へは自動進行しない。
次は implementation identity／build gate を別Stepとして切り出す。
R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
