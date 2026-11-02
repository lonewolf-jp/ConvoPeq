# B-2 attach 契約の確定報告（2026-11-01 第 5 部）

- 基準: `main` = `origin/main` = **`6bb7c8d5`**（ソースは `fd92c6e7` と同一）
- **製品コード・テスト・CI 定義は未変更。build / test / debugger / replay 未実行**
- 詳細: `doc/residual-20261101/B2-ATTACH-CONTRACT-20261101.md`

## 1. 前回報告の訂正

「`store` は冪等」は**値の冪等性**に過ぎず、**ライフサイクル安全性の証明ではない**（Owner 指摘を受容）。
本報告では契約を**状態機械**として定義し、遷移の許否を API で強制する形に改めた。

## 2. attach ライフサイクル契約（指示 1）

状態: **U**（Unbound, `sink_ == nullptr`, `owner_ == nullptr`）/ **B(E)**（Bound, `sink_ == &E.diagSink_`）

| # | 契約 | 強制 |
|---|---|---|
| C1 | **owner は生涯固定**（変更不可） | `owner_` を CAS で一度だけ設定 |
| C2 | attach は **U → B(E) のみ** | `sink_.compare_exchange_strong(nullptr, s)` 成功時のみ true |
| C3 | 同一 sink への再 attach は**許容**（no-op） | CAS 失敗 + `expected == s` → true |
| C4 | **異なる sink への attach は拒否**（状態不変・false・Debug は assert） | CAS 失敗 + `expected != s` → false |
| C5 | **detach は全生産者の静止後のみ** | 呼び出しを `AudioEngine` の teardown に限定（private・2 箇所） |
| C6 | detach 後の再 attach は**同一 owner かつ静止完了後**に限り許容 | C2 と同じ CAS（owner は C1 により不変） |

### 単純 `store` では不十分な理由

| 危険 | `store` | 契約（CAS） |
|---|---|---|
| B(E) の個体へ別エンジン F の sink を書く | **上書きしてしまう** | **拒否** |
| detach 直後に別 owner で再 attach | 上書き | **拒否**（C1） |
| 二重 attach の検出 | 不可 | 「同一だから成功」と「別物だから失敗」を区別 |

### detach と prepare の排他（確定）

`ReleaseResources.cpp` の実測順序: `:95` StopAcceptingWork → `:153` StopAudio → `:236-238` StopWorkers
（**Coordinator join → Builder join**）→ `:267` DrainRetire → `:444` `drainAllQuarantineStore()`（コア破棄）。
同ファイル `:635-636` に**順序不変条件**が明記されている。
**したがって `prepare`（rebuild スレッド）と `detach`（drain）は排他**。

## 3. `bindDiagnostics()` のアクセス制御と所有保証（指示 2）

```cpp
// AudioEngine（private・入れ子 DSPCore からのみ到達可能）
void bindDiagnostics(DSPCore& core) noexcept {
    assert(core.ownerForDiagnostics() == this);       // ★ 所有関係の検査
    core.convolverRt().attachDiagSink(&diagSink_);    // ★ 自分の sink しか渡さない
    core.eqRt().attachDiagSink(&diagSink_);
}
```

| 要求 | 担保 |
|---|---|
| 呼べる範囲の限定 | `bindDiagnostics` は private。`DSPCore` は `AudioEngine` の**入れ子クラス**で外側の private に到達可能。外部からは不可 |
| owner と bind 先の一致 | `assert(core.ownerForDiagnostics() == this)` を必ず通す。**単に owner が非 null であることを証明にしない** |
| bind 後の owner 変更不可 | `owner_` は CAS で一度だけ設定（C1） |
| 旧 sink 参照の残存 | 再生成は新オブジェクト（nullptr から）、再 prepare は同一 owner で no-op |
| 別エンジンの sink を渡せない | 引数で sink を受け取らない（`&diagSink_` 固定） |

## 4. 状態遷移表（指示 3）

| # | 現在 | 事象 | 遷移 | 許否 |
|---|---|---|---|---|
| T1 | U | `prepare(E)` | B(E) | 許容 |
| T2 | B(E) | `prepare(E)` | B(E)（不変） | 許容（no-op） |
| T3 | B(E) | `prepare(F)` | B(E)（不変） | **拒否** |
| T4 | B(E) | `detach()`（静止後） | U（owner は E） | 許容 |
| T5 | U(owner=E) | `attach(E.sink)` | B(E) | 許容 |
| T6 | U(owner=E) | `attach(F.sink)` | U（不変） | **拒否** |
| T7 | B(E) | コア破棄（dtor） | 終了 | 許容 |
| T8 | B(E) | エンジン teardown（静止後） | detach → 破棄 | 許容 |
| T9 | U | `process()`（RT） | U（push なし） | 許容 |
| T10 | B(E) | エンジン破棄・コア leak | B(E) のまま（dangling） | 許容（未使用） |

**複数エンジン**: S-A（2 エンジン同時→混線なし）、S-B（片側破棄→他方は無影響）、S-C（rebuild 反復→旧参照なし）、
S-D（再 prepare→no-op）、S-E（別 owner→拒否）、S-F/G（テストの非所有→U のまま）。

## 5. callback 以外の生産者の静止根拠（指示 5）

**全 14 箇所を実測し、すべて RT 処理関数内であることを確認**:
`AudioBlock.cpp:262/295/707/750`、`BlockDouble.cpp:249/285/659/702`、`DSPCoreFloat.cpp:86`、`DSPCoreDouble.cpp:46/650`、
`DSPCoreIO.cpp:438`、`ConvolverProcessor.Runtime.cpp:774/863`（いずれも `process*` 系）。

| 候補 | 実測 | 判定 |
|---|---|---|
| `uiConvolverProcessor` | `process` 呼び出し **0 件** | 生産者ではない |
| テストのローカル conv（23 個） | B-2 後は `sink_ == nullptr` | 書き込みなし |
| rebuild スレッド | `process(` 呼び出し **0 件** | 生産者ではない |
| メッセージスレッド | 診断 push を行わない | 生産者ではない |

**「静止すべき生産者」は audio callback に限られる。** 根拠は join / `audioCallbackLock`。
ただし停止順序をコードで固定し、`audioCallbackActiveCount == 0` は**補助**、デストラクタの clear は**防御**として扱う
（P3 の代替にしない）。

## 6. ASan 検証計画の追加（指示 4）

| 段 | 検証 | 判定 |
|---|---|---|
| S1/S2 | B-2 適用前の反復 | UAF 再現の有無・再現率・スタックを記録 |
| S3 | スタンドアロン conv を含むテスト（B-2 後） | push 0 回 |
| S4 | 2 エンジン同時 | 各自の sink のみ |
| **S5** | 同一エンジンの再 prepare | attach no-op、状態不変 |
| **S6** | DSPCore 再生成（rebuild 反復） | 旧 detach・新 bind、旧参照なし |
| **S7** | 片側エンジン破棄 | 他方の sink・イベント数が無影響 |
| **S8** | 異なる sink への誤 attach | **拒否**され元の sink 維持 |
| **S9** | `sink_ == nullptr` での診断書き込み | push 0 回（カウンタ確認） |
| **S10** | 停止 → detach → 破棄の順序 | 逆順が発生しない |

**ASan 未検出を UAF 不存在の証明として扱わない。** 「特定条件で検出されなかった」結果として記録する。
並行アクセスの競合は ASan の守域外であり、**TSan は別軸**（承認条件にはしない）。

## 7. 残件

`DSPCore` に `owner_`（8 B）と `sink_`（8 B）を追加する。**`#if` で囲まず常時存在**させ、
`sizeof(DSPCore)` を全 TU で一致させる（`AudioEngine.h:3984-3988` の ODR 教訓）。実装時に `static_assert` で固定する。

**これ以外の契約・状態遷移・停止順序・検証計画は確定した。**

## 8. 遵守事項

製品コード・テスト・CI 定義変更 **なし**（追跡差分 0）／build・test・debugger・replay **未実行**／
U5 FAIL・既定 OFF、U3/U4 HOLD を維持／未追跡は既存 18 件を不変のまま保全（総数は `git status` で確認）。
