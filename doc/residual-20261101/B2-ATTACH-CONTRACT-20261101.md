# B-2 attach ライフサイクル契約・状態遷移・検証追加（2026-11-01）

- 状態: **契約資料のみ。製品コード・テスト・CI 定義は未変更。build / test / debugger / replay 未実行**
- 基準: `main` = `origin/main` = **`6bb7c8d5`**（ソースは `fd92c6e7` と同一）
- 目的: Owner 指示 5 項目（attach 契約、`bindDiagnostics` のアクセス制御と所有保証、状態遷移表、
  ASan 計画の追加、callback 以外の生産者の静止根拠）に回答する

## 0. 前回報告の訂正（Owner 指摘の受容）

前回「`store` は冪等」と書いたのは**値の冪等性**に過ぎず、**ライフサイクル上の安全性の証明ではない**。
本資料では契約を状態機械として定義し、**遷移の許否を API で強制**する形に改める。

## 1. attach ライフサイクル契約（Owner 指示 1）

### 1.1 状態

| 状態 | `sink_` | `owner_` | 意味 |
|---|---|---|---|
| **U**（Unbound） | `nullptr` | `nullptr` | どのエンジンにも束縛されていない |
| **B(E)**（Bound） | `&E.diagSink_` | `E` | エンジン E に束縛済み |
| （終了） | — | — | インスタンス破棄 |

### 1.2 契約（C1〜C6）

| # | 契約 | 強制方法 |
|---|---|---|
| **C1** | **owner は生涯固定**。最初に束縛したエンジンから変更できない | `owner_` を CAS（nullptr → E）で一度だけ設定。以後の設定要求は拒否 |
| **C2** | attach は **U → B(E) のみ**を許す | `sink_.compare_exchange_strong(expected=nullptr, s)` が成功した場合のみ true |
| **C3** | **同一 sink への再 attach は許容**（no-op） | CAS 失敗時に `expected == s` なら true（状態不変） |
| **C4** | **異なる sink への attach は拒否**（状態不変・false・Debug は assert） | CAS 失敗時に `expected != s` なら false |
| **C5** | **detach は「全生産者の静止後」に限る**（§1.4） | 呼び出し位置を `AudioEngine` の teardown に限定（private + 呼び出しは 2 箇所のみ） |
| **C6** | **detach 後の再 attach は、同一 owner かつ静止完了後に限り許容**（B(E) → U → B(E)） | C2 と同じ CAS。owner は C1 により E のまま |

### 1.3 なぜ単純な `store` では不十分か（Owner 指摘への回答）

`store(s, release)` は「最後に書いた値が残る」だけで、次の 3 つを区別できない。

| 危険 | `store` での挙動 | 契約（CAS）での挙動 |
|---|---|---|
| 既に B(E) の個体へ別エンジン F の sink を書く | **上書きしてしまう**（混線） | CAS 失敗 + `expected != s` → **拒否** |
| detach 直後に別 owner で再 attach | 上書き | **拒否**（C1 により owner 変更不可） |
| 二重 attach の検出 | 不可（常に成功に見える） | 「同一だから成功」と「別物だから失敗」を区別 |

### 1.4 detach と prepare の排他条件（C5）

| 事象 | 実行スレッド | 排他の取り方 |
|---|---|---|
| `prepare(owner, ...)`（＝attach） | **rebuild スレッド**（`RuntimeBuilder.cpp:474`）または **prepareToPlay の呼び出しスレッド**（`PrepareToPlay.cpp:272`） | — |
| `detach()` | **エンジン teardown**（`releaseResources` / `~AudioEngine`）または **DSPCore のデストラクタ**（drain 中） | — |
| 排他の根拠 | — | `~AudioEngine` は **`stopRebuildThread()`（`CtorDtor.cpp:171`）→ drain** の順。detach が起こる drain は rebuild スレッド停止後。**したがって prepare と detach は並行しない** |

**注意**: この排他は「rebuild スレッドを join してから drain する」という**既存の停止順序**に依存する。
実装時に `releaseResources` 経路でも同じ順序であることを確認する（未確定 §6-2）。

## 2. `bindDiagnostics()` のアクセス制御と所有関係（Owner 指示 2）

### 2.1 「owner が非 null」を所有関係の証明にしない

`owner != nullptr` は「引数が渡された」ことしか示さない。**所有関係は別に検査する**。

### 2.2 設計

```cpp
// AudioEngine（private・入れ子 DSPCore からのみ到達可能）
void bindDiagnostics(DSPCore& core) noexcept
{
    // 所有関係の検査: このコアの owner が *this であること
    assert(core.ownerForDiagnostics() == this);
    core.convolverRt().attachDiagSink(&diagSink_);   // ← 自分の sink しか渡さない
    core.eqRt().attachDiagSink(&diagSink_);
}
```

| 要求（Owner） | 担保 |
|---|---|
| 呼べるクラス・関数を限定 | `bindDiagnostics` は **`AudioEngine` の private**。`DSPCore` は `AudioEngine` の**入れ子クラス**であり、外側の private にアクセスできる（C++ 標準）。外部の自由関数・他クラスからは到達不可 |
| DSPCore の owner と bind 先の一致 | `assert(core.ownerForDiagnostics() == this)` を**必ず**通す |
| bind 後に owner を変更できない | `owner_` は CAS で一度だけ設定（C1）。変更要求は assert + 拒否 |
| 旧 sink への参照が残らない（再生成・再 prepare） | 再生成は**新しい DSPCore オブジェクト**（`sink_` は nullptr から）。再 prepare は**同一 owner** なので同一 sink への no-op attach |
| 別エンジンの sink を渡せない | `bindDiagnostics` は**引数で sink を受け取らない**（`&diagSink_` 固定） |

### 2.3 `ownerForDiagnostics()` の設定点

`DSPCore::prepare(owner, ...)` の**先頭**で `owner->bindDiagnostics(*this)` を呼ぶ前に、
`setOwnerOnce(owner)` を実行する（CAS で一度だけ）。異なる owner での再 prepare はここで拒否される。

## 3. 状態遷移表（Owner 指示 3）

### 3.1 単一 DSPCore のライフサイクル

| # | 現在 | 事象 | 遷移 | 許否 | 強制 |
|---|---|---|---|---|---|
| T1 | U（owner なし） | `prepare(E)` | **B(E)** | 許容 | owner CAS 成功 + sink CAS 成功 |
| T2 | B(E) | `prepare(E)`（同一 owner） | B(E)（不変） | 許容（no-op） | sink CAS 失敗 + `expected == s` |
| T3 | B(E) | `prepare(F)`（別 owner） | B(E)（不変） | **拒否**（assert + false） | owner CAS 失敗 |
| T4 | B(E) | `detach()`（静止後） | **U**（owner は E のまま） | 許容 | `store(nullptr)` |
| T5 | U（owner=E） | `attach(E.sink)`（再 attach） | **B(E)** | 許容（C6） | sink CAS 成功 |
| T6 | U（owner=E） | `attach(F.sink)` | U（不変） | **拒否** | owner 不一致で assert |
| T7 | B(E) | DSPCore 破棄（dtor） | （終了） | 許容 | dtor で `detach()` |
| T8 | B(E) | エンジン E の teardown（RT 静止後） | B(E) → detach → 破棄 | 許容 | §1.4 の順序 |
| T9 | U | `process()`（RT） | U（不変・push なし） | 許容 | null チェック |
| T10 | B(E) | エンジン E 破棄・コアが **leak** | B(E) のまま（dangling） | **許容（未使用）** | RT 不在。§5-3 で残余として明記 |

### 3.2 複数エンジン・再生成を含むケース

| # | シナリオ | 期待 |
|---|---|---|
| S-A | エンジン E1・E2 が同時に存在 | E1 のコアは `&E1.diagSink_`、E2 のコアは `&E2.diagSink_`。**混線しない**（グローバルが無い） |
| S-B | E1 のみ破棄（E2 は稼働継続） | E1 のコアは drain で detach → 破棄。**E2 のコアと sink は無影響** |
| S-C | 同一エンジンで rebuild を繰り返す（コア再生成） | 旧コアは破棄時に detach。新コアは prepare で新たに bind。**旧 sink 参照は残らない** |
| S-D | 同一コアで `prepare` を 2 回 | T2（no-op）。sink は不変 |
| S-E | 同一コアで別 owner の prepare | T3（拒否） |
| S-F | テストが DSPCore を生成して prepare を呼ばない | U のまま（`sink_ == nullptr`） |
| S-G | テストのローカル `ConvolverProcessor` | attach 経路が無い → U のまま |

## 4. callback 以外の生産者の静止根拠（Owner 指示 5）

### 4.1 全 14 箇所の実測（すべて RT 処理関数内）

| ファイル:行 | 関数（実測） | 実行スレッド |
|---|---|---|
| `AudioBlock.cpp:262/295/707/750` | `AudioEngine::processBlock`（float 経路） | **audio callback** |
| `BlockDouble.cpp:249/285/659/702` | `AudioEngine::processBlock`（double 経路） | **audio callback** |
| `DSPCoreFloat.cpp:86` | `logEqTime` ← `DSPCore::process` | **audio callback** |
| `DSPCoreDouble.cpp:46` | `logEqTime` ← `DSPCore::process` | **audio callback** |
| `DSPCoreDouble.cpp:650` | `DSPCore::processOutputDouble`（`:589` から） | **audio callback** |
| `DSPCoreIO.cpp:438` | `DSPCore::processOutput`（`:338` から） | **audio callback** |
| `ConvolverProcessor.Runtime.cpp:774/863` | `ConvolverProcessor::process`（`:238` から） | **audio callback**（エンジン所有コアの場外） |

### 4.2 結論: callback 以外の生産者は「存在しない」

| 候補 | 実測 | 判定 |
|---|---|---|
| `uiConvolverProcessor` | `uiConvolverProcessor.process` の呼び出し **0 件** | **生産者ではない** |
| テストのローカル `ConvolverProcessor`（23 個） | `process()` は呼ばれるが、B-2 後は `sink_ == nullptr` | **書き込みが発生しない** |
| rebuild スレッド | `RuntimeBuilder.cpp` に `process(` の呼び出し **0 件** | **生産者ではない** |
| メッセージスレッド | 診断 push を行う関数を呼ばない | **生産者ではない** |

**したがって「静止すべき生産者」は audio callback（＋将来追加された場合はそのスレッド）に限られる。**
静止の根拠は join / `audioCallbackLock`（`B2-LIFETIME-PROOF` §1.4）。

### 4.3 それでも「静止」を単独の根拠にしない

- 破棄順序（停止 → join/lock → カウント確認 → detach → 破棄）を**コードで固定**する。
- `audioCallbackActiveCount == 0` は**補助**であり、新規開始の抑止（停止/remove）と join/lock の代替にしない。
- デストラクタでの clear は**防御**であり、P3 の代替にしない（Owner 指摘）。

## 5. 残余と、証明できていないこと（正直な切り分け）

| # | 項目 | 状態 |
|---|---|---|
| 5-1 | エンジン破棄後に RT が push する経路 | **無い**（join/lock + 全生産者が RT） |
| 5-2 | 非所有インスタンスからの書き込み | **無い**（sink nullptr。実測で 23 個の生成経路を確認） |
| 5-3 | エンジンを leak した場合（`abandonEngine`） | コアの sink は dangling だが **RT 不在で未使用**。メモリは leak |
| 5-4 | 将来、診断 push を行う非 RT スレッドが追加された場合 | **契約違反**。「診断 push は RT スレッドのみ」を実装コメントとレビューで維持する |
| 5-5 | `prepare`/`detach` の排他が `releaseResources` 経路でも成立するか | **確認済み**（`ReleaseResources.cpp:236-238` で Coordinator/Builder を join してから `:444` の drain。同ファイル `:635-636` に順序不変条件の明記） |
| 5-6 | 並行アクセスの競合（データレース） | ASan の守備範囲外。**TSan は別軸**（本件の承認条件にはしない） |

## 6. ASan 検証計画の追加（Owner 指示 4）

### 6.1 追加する段

| 段 | 検証内容 | 判定基準 |
|---|---|---|
| **S5** | 同一エンジンで `prepareToPlay` を 2 回（再 prepare） | attach が no-op で状態不変。sink が変わらない |
| **S6** | DSPCore の再生成（rebuild 反復） | 旧コア破棄時に detach、新コアは新規 bind。**旧 sink への参照が残らない**（カウンタで確認） |
| **S7** | 片側エンジン破棄（E1 のみ破棄、E2 稼働） | E2 の sink とイベント数が**無影響** |
| **S8** | 異なる sink への誤 attach を試行 | **拒否**され、元の sink が維持される（assert / false） |
| **S9** | `sink_ == nullptr` のインスタンスで診断カテゴリ書き込みを試行 | push が **0 回**（カウンタで確認） |
| **S10** | 停止 → detach → 破棄の順序（ログで順序を検証） | 逆順が発生しない |

### 6.2 判定の扱い（Owner 指摘）

- **ASan でエラーが出ないことは、UAF が存在しない証明にはならない。** 「特定条件で検出されなかった」結果として記録する。
- S1/S2（B-2 適用前）で **UAF が報告されれば、原因の独立確認**になる（ダンプ 1 件に依存しない）。
- 並行アクセスの競合は ASan の守域外であり、**TSan は別軸の検証**として留意する（本件の承認条件にはしない）。

## 7. 実装前に残る確認 → **1 点は解決済み**

### 7.1 解決: `releaseResources` 経路でも「停止 → join → drain」の順序が成立する（実測）

`AudioEngine.Processing.ReleaseResources.cpp` の順序:

| 行 | 処理 |
|---|---|
| `:95` | `setShutdownPhase(StopAcceptingWork)` |
| `:153` | `setShutdownPhase(StopAudio)` |
| `:236-238` | `setShutdownPhase(StopWorkers)` → **`shutdownCoordinatorLoop()`（Coordinator を join）** → **`stopRebuildThread()`（Builder を join）** |
| `:267` | `setShutdownPhase(DrainRetire)` |
| `:339` / `:375` | `drainDeferredRetireQueues(...)` |
| `:444` | `drainAllQuarantineStore()`（**DSPCore が破棄される**） |
| `:450` | `retryDeferredQuarantineDestroy()` |

さらに同ファイル `:635-636` に**順序不変条件**が明記されている:
「`requestShutdown` → `shutdownCoordinatorLoop`（join）→ `stopRebuildThread`（join で Builder 完全終了）→ drain wait」。
`:829` にも「no active RT audio thread (post-stopRebuildThread)」とある。

**したがって `prepare`（rebuild スレッド）と `detach`（drain/破棄）は排他**である。
ハーネスも `stopAudioOnly()`（audio を join）→ `releaseResources()` の順で呼ぶ（`AudioEngineHarness.cpp:40-52`）。

### 7.2 残る確認（1 点）

`DSPCore` に `owner_`（8 B）と `sink_`（8 B）を追加する。**`#if` で囲まず常時存在**させ、
`sizeof(DSPCore)` を全 TU で一致させる（`AudioEngine.h:3984-3988` の ODR 教訓）。
実装時に `static_assert(sizeof(...))` で固定し、診断 ON/OFF で同一であることを確認する。

**この 1 点のみが残る。契約・状態遷移・検証計画・停止順序の根拠は揃った。**

