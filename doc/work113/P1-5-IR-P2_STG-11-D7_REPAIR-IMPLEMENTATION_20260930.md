# STG-11-D7 Repair Implementation Record

- Document: `doc/work113/P1-5-IR-P2_STG-11-D7_REPAIR-IMPLEMENTATION_20260930.md`
- Work item: **STG-11-D7-1** — session 復元時の未検証 enum cast による RT 配列 OOB
- Predecessor: `P1-5-IR-P2_STG-11-D7_REPAIR-CONTRACT-AUDIT_20260930.md`（GO）
- Date: 2026-09-30
- Commit / push: **未実施**（D1〜D6 を作業ツリーに保持）

---

## 1. Authority

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |

完了後の再生成値は Gate §1 参照。

---

## 2. 実装（Contract §10 どおり）

`src/audioengine/AudioEngine.StateIO.cpp` のみ。6 箇所に範囲ガードを追加
（B-3 と同一パターン。新規 authority / queue / worker / member なし。
RT 側の変更なし）：

| 箇所 | 範囲 | 既定維持値 |
| --- | --- | --- |
| `processingOrder` | `ConvolverThenEQ(0)..EQThenConvolver(1)` | ConvolverThenEQ |
| `analyzerSource` | `Input(0)..Output(1)` | Output |
| `convHCFilterMode` | `Sharp(0)..Soft(2)` | Natural(1) |
| `convLCFilterMode` | `Natural(0)..Soft(1)` | Natural(0) |
| `eqLPFFilterMode` | `Sharp(0)..Soft(2)` | Natural(1) |

`processingOrder` は 2 箇所の atomic publish を guard 内に移動しただけで意味不変。
正常 session（範囲内値）の動作は不変。唯一の挙動変更は範囲外値の扱い
（「不正値適用」→「デフォルト維持」＝期待動作 §4）。

### 変更範囲

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/audioengine/AudioEngine.StateIO.cpp` | 38 | 7 | **D7** |
| `src/tests/AudioEngineHarness/STG11D7StateEnumGuardTests.cpp` | 新規 | — | **D7**（untracked） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | +5 | 0 | D1〜D5 + **D7**（5 行） |
| `CMakeLists.txt` | +1 | 0 | D1〜D5 + **D7**（1 行） |
| `ConvoPeq.md` | 再生成 | — | — |

D1〜D5 の production 変更には触れていない（§Gate の scope check で実証）。

---

## 3. 不変条件

- RT / Publish / Retire / Epoch / ownership に影響なし（Message Thread 上の分岐追加のみ）。
- raw `std::atomic` 追加 0。mutex/alloc/new-auth 0（D7 diff grep で実証）。
- 性能影響なし（復元時のみの整数比較）。

---

## 4. テスト

`STG11D7StateEnumGuardTests.cpp`（harness サブテスト。新規 CTest target なし。
fixture は harness が所有）：

| Test | 内容 | 結果 |
| --- | --- | --- |
| D7-T1 | 破損 session（5 property に範囲外値）→ 全 mode がデフォルト維持 | PASS |
| D7-T2 | 有効 round-trip（Soft/Sharp 設定→保存→復元→一致） | PASS |
| D7-T3 | 境界値（min/max 適用、min-1/max+1 拒否、全 10 ケース） | PASS |
| Negative control | convHC ガードを一時除去 → `convHCFilterMode changed to 5` で **FAIL**（rc=1）。復元後 PASS | 実証済み |

---

## 5. 実施した検証

| 検証 | 結果 |
| --- | --- |
| Debug / Release build | BUILD_EXIT=0 両方 |
| D7 sub-test（Debug / Release） | rc=0 / 全 PASS 両方 |
| D3 / D4 / D5 sub-test | 全 PASS（両構成） |
| D1 / D2 回帰 | TD1-1 Q=304 / TD1-2 T=280、TD2-1 E=3 / TD2-2 T=3 / TD2-3（不変） |
| full Debug / Release CTest | **45/45 PASS 両方** |
| raw atomic / mutex / alloc audit | 0 件 |
| clang-tidy（StateIO.cpp） | D7 hunk 内 0 件。残存 1 件（line 8 の rounding）は hunk 外の既存事象 |
| cppcheck | project 135 件は既存ノイズ、**StateIO.cpp 0 件** |
| ASAN | 環境 block（`0xC0000139`）。別問題として記録 |

---

## 6. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**。Gate は `READY FOR COMMIT`。
