# B-2 attach 契約 第 2 版の確定報告（2026-11-01 第 6 部）

- 基準: `main` = `origin/main` = **`6bb7c8d5`**（ソースは `fd92c6e7` と同一）
- **製品コード・テスト・CI 定義は未変更。build / test / debugger / replay 未実行**
- 詳細: `doc/residual-20261101/B2-ATTACH-CONTRACT-V2-20261101.md`

## 0. 前版からの変更（Owner 指摘 4 点への対応）

| # | 前版 | 第 2 版 |
|---|---|---|
| 1 | `owner_` は `AudioEngine*`（実体） | **`uint64_t ownerId_`（識別情報。逆参照しない）** |
| 2 | `assert` で owner 一致 | **Release でも比較し、不一致なら bind しない** |
| 3 | T10「leak したコアは dangling を許容」 | **T10 を廃止**。teardown で leak 対象コアも detach |
| 4 | `DSPCore` に 16 B（owner_ + sink_） | **`DSPCore` の 16 B のみ**。`ConvolverProcessor` / EQ は **0 B**（sink を引数で受け取る） |
| 5 | 外側クラスが入れ子の private に触れる前提 | **規格どおり不可**。ロジックを入れ子側へ寄せ `friend` 不要に |

## 1. `drainAllQuarantineStore()` の対象範囲と、コアが長生きする経路（実測）

- `drainAllQuarantineStore()` = **Q + E + T**（`ISRRetireRouter.cpp:628-635`）。**D を含まない**。
- `~AudioEngine` は `activeReaderCount()==0` なら `drainAll()`（D+Q+E+T）、そうでなければ **D のみ + Q/E/T**（`CtorDtor.cpp:329-336`）。
- **`CtorDtor.cpp:338-359`: pin 残留のコアは「強制解放すると UAF になるため、リークを選ぶ」**。
  → **コアがエンジンより長生きする経路が実在する。**
- ただし leak されるコアは **`deferredDestroy_`（`std::vector<void*>`、`AudioEngine.h:5254`）に保持**され、**列挙できる**。

**→ teardown でこの vector を走査して detach できる。これが T10 廃止の根拠。**

## 2. 修正 1・2: owner は識別情報、Release でも強制

```cpp
// DSPCore（入れ子クラス）の prepare 内
const std::uint64_t id = owner->diagnosticsEngineId();     // 入れ子→外側の private は可
std::uint64_t expected = 0;
if (ownerId_.compare_exchange_strong(expected, id))     bound = true;   // 初回
else if (expected == id)                                bound = true;   // 同一 owner
else                                                    return;         // ★ 別 owner は Release でも拒否
if (bound) sink_.store(&owner->diagnosticsSink(), std::memory_order_release);
```

- **`assert` に依存しない**: 不一致は全ビルドで `return`（sink を書かない）。
- **入れ子クラスのアクセス規格**: 入れ子→外側は**可**（[class.access.nest]/1）、外側→入れ子は**不可**。
  本設計は**ロジックを入れ子側に置く**ため `friend` 不要（Owner 指摘の修正）。
- **owner は識別情報**なので、エンジン破棄後も dangling ポインタにならない。

## 3. 修正 3: owner/sink の整合性

| ID | 不変条件 |
|---|---|
| I1 | `sink_ != nullptr ⟹ ownerId_ != 0` |
| I2 | `ownerId_` は 0→E の一方向のみ |
| I3 | 書き込みは**単一の NonRT 制御経路**のみ |

- 「2 つの atomic の独立更新」でも、**I3（単一経路）＋ CAS（単一成功）＋ 合法な中間状態 U(ownerId)** で不整合にならない。
- **部分設定は合法**（sink 未設定の U(ownerId=E) は T4 と同じ状態）。I1 は「sink 非 null なら owner 非 0」であり逆は要求しない。
- 排他の根拠（実測）: `ReleaseResources.cpp:236-238` で Coordinator/Builder を join → `:444` で drain（コア破棄）。

## 4. 修正 4: T10 の廃止

| # | 現在 | 事象 | 遷移 | 許否 |
|---|---|---|---|---|
| T8 | B(E) | エンジン teardown | **detach → 破棄** | 許容 |
| **T10（改）** | B(E) | **エンジン破棄・コア leak** | **detach → U(id=E) → leak** | 許容（**sink は nullptr**） |

**実現方法**: `~AudioEngine` の member teardown 前に `deferredDestroy_` の各要素へ `detachDiagSink()` を呼ぶ。
→ **leak されたコアにも解放済み sink へのポインタが残らない。** ID（数値）だけが残るが参照されない。

### 再 attach（T5）の条件を明文化

| # | 条件 |
|---|---|
| R1 | 旧 sink を参照する生産者が存在しない（全生産者は RT、join 済み） |
| R2 | detach が完了している |
| R3 | 新しい sink が有効（同一 owner のエンジン生存中） |
| R4 | 別 owner からの attach は拒否（Release でも） |
| R5 | detach と再 attach が並行しない（単一経路） |

## 5. メンバ配置（sizeof の整理）

| クラス | 追加 | 増加 |
|---|---|---|
| **`DSPCore`** | `uint64_t ownerId_` ＋ `std::atomic<DiagSink*> sink_` | **16 B** |
| `ConvolverProcessor` | なし（`process(block, sink)` で受け取る） | **0 B** |
| `EQProcessor` | なし（`logEqTime(..., sink)` で受け取る） | **0 B** |

- `static_assert` は**サイズ一致の補助**であり ODR の完全証明ではない。ODR は**単一定義・`#if` なし**で担保する。
- 追加メンバは診断 ON/OFF で同一（`#if` を付けない）。

## 6. 生産者の範囲の補強（間接呼び出しの確認）

| 確認 | 結果 |
|---|---|
| `ConvolverProcessor::process` の呼び出し元 | **`DSPCoreDouble.cpp:390/454`、`DSPCoreFloat.cpp:274/339` の 4 箇所のみ** |
| `logEqTime` の呼び出し元 | `DSPCoreDouble.cpp:408/432`、`DSPCoreFloat.cpp:292` |
| `getNextAudioBlock` の呼び出し元 | `AudioEngineProcessor.cpp:88`（JUCE callback）のみ |
| 関数ポインタ・`std::function`・`std::bind` | **0 件** |

## 7. 検証計画の追加

| 段 | 検証 | 判定 |
|---|---|---|
| **S11** | leak 経路（pin を残してエンジン破棄） | leak コアの `sink_` が **nullptr** |
| **S12** | Release での誤 bind（別エンジンの prepare） | bind されない（assert 無効でも） |
| **S13** | 再 attach（detach → 同一 owner） | R1〜R5 を満たす |
| S8/S9（再掲） | 異なる sink の拒否 / nullptr での書き込み 0 回 | 拒否 / 0 回 |

**ASan 未検出を UAF 不存在の証明としない。** TSan は別軸。

## 8. 残る確認（実装時）

1. `deferredDestroy_` の `void*` から `DSPCore*` を安全に扱えるか（**格納時に detach 用コールバックを併せて保持**する案が安全）。
2. `DSPCore::prepare` の呼び出しスレッド（rebuild / message）の再確認。
3. `diagnosticsEngineId()` / `diagnosticsSink()` を private にし、入れ子 `DSPCore` から到達できることの実装確認。

## 9. 遵守事項

製品コード・テスト・CI 定義変更 **なし**（追跡差分 0）／build・test・debugger・replay **未実行**／
U5 FAIL・既定 OFF、U3/U4 HOLD を維持／既存の未追跡 18 件は不変。
