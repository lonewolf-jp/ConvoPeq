# STG-11-D10 Repair Implementation Record

- Document: `doc/work113/P1-5-IR-P2_STG-11-D10_REPAIR-IMPLEMENTATION_20261001.md`
- Work item: **STG-11-D10-1** — EQ band type/channel の未検証 enum cast による無言の band 無効化
- Predecessor: `P1-5-IR-P2_STG-11-D10_REPAIR-CONTRACT-AUDIT_20261001.md`（GO）
- Date: 2026-10-01
- Commit / push: **未実施**（D1〜D9 を作業ツリーに保持）

---

## 1. Authority

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| D9 マーカー反映 | 9 件（作業開始時に確認） |

完了後の再生成値は Gate §1 参照。

---

## 2. 実装（Contract §10 どおり）

`src/eqprocessor/EQProcessor.Parameters.cpp` のみ。2 setter に範囲ガードを追加
（D7 パターン。RT / ownership の変更なし。新規 authority なし）：

```cpp
// setBandType: 5 列挙値のいずれでもなければ return
// setBandChannelMode: 5 列挙値のいずれでもなければ return
```

### D10 監査の副産物（いずれも正常・修正なし）

実コード追跡により以下を確定した（Gate §4 に記録）：

- EQ band float（freq/gain/q）：`calcSVFCoeffs` 系の `isfinite` ガードが
  NaN を bypass 係数に中和。RT `processBand` の per-sample clamp＋reset が
  最終防衛。安全。
- mixedF 系：LoaderThread の `validateBuffer` が NaN 出力を拒否。安全。
- tailL1L2Multiplier：int-domain＋clamp のため有界。安全。
- tailMode：int-domain のため有界。安全。
- `copySnapshotToPendingUnlocked`：全 float 書き込み元が setter ガード済みか
  abs-gate 済み。setter 迂回の直接書き込みなし。

### 変更範囲

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/eqprocessor/EQProcessor.Parameters.cpp` | 15 | 0 | **D10** |
| `src/tests/AudioEngineHarness/STG11D10EQEnumGuardTests.cpp` | 新規 | — | **D10**（untracked） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | +5 | 0 | D1〜D9 + **D10**（5 行） |
| `CMakeLists.txt` | +1 | 0 | D1〜D9 + **D10**（1 行） |
| `ConvoPeq.md` | 再生成 | — | — |

D1〜D9 の production 変更には触れていない。

---

## 3. 不変条件

- RT / Publish / Retire / Epoch / ownership に影響なし（Message Thread 上の早期 return のみ）。
- raw `std::atomic` 追加 0。mutex/alloc/thread/new-auth 0。
- 性能影響なし。

---

## 4. テスト

`STG11D10EQEnumGuardTests.cpp`（harness サブテスト。新規 CTest target なし。
`getEQProcessor()` の実体を使用）：

| Test | 内容 | 結果 |
| --- | --- | --- |
| D10-T1 | 範囲外 type(99)/channel(-7) の setState → 変更前値維持 | PASS |
| D10-T2 | 有効 round-trip（全 5 type×全 5 channel） | PASS |
| D10-T3 | 境界値（min-1/max+1 拒否、min/max 適用） | PASS |
| Negative control | `setBandType` ガードを一時除去 → `OOB band type applied (99)` で **FAIL**（rc=1）。T3 も連動 FAIL。復元後 PASS | 実証済み |

---

## 5. 実施した検証

| 検証 | 結果 |
| --- | --- |
| Debug / Release build | BUILD_EXIT=0 両方 |
| D10 sub-test（Debug / Release） | rc=0 / 全 PASS 両方 |
| D3 / D4 / D5 / D7 / D8 / D9 sub-test | 全 PASS（両構成） |
| D1 / D2 回帰 | TD1-1 Q=304 / TD1-2 T=280、TD2-1 E=3 / TD2-2 T=3 / TD2-3（不変） |
| full Debug / Release CTest | **45/45 PASS 両方** |
| raw atomic / mutex / alloc audit | 0 件 |
| clang-tidy（Parameters.cpp） | D10 hunk（162/193）内 0 件。残存は hunk 外の既存事象 |
| cppcheck | project 135 件は既存ノイズ、**Parameters.cpp 0 件** |
| ASAN | 環境 block（`0xC0000139`）。別問題として記録 |

---

## 6. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**。Gate は `READY FOR COMMIT`。
