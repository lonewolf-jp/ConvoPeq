# STG-11-D11 Repair Implementation Record

- Document: `doc/work113/P1-5-IR-P2_STG-11-D11_REPAIR-IMPLEMENTATION_20261001.md`
- Work item: **STG-11-D11-1** — NaN totalGain の無条件 store による NaN RT 出力
- Predecessor: `P1-5-IR-P2_STG-11-D11_REPAIR-CONTRACT-AUDIT_20261001.md`（GO）
- Date: 2026-10-01
- Commit / push: **未実施**（D1〜D10 を作業ツリーに保持）

---

## 1. Authority

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| D10 マーカー反映 | 9 件（作業開始時に確認） |

完了後の再生成値は Gate §1 参照。

---

## 2. 実装（Contract §10 どおり）

`src/eqprocessor/EQProcessor.Parameters.cpp` のみ。`setTotalGain` の冒頭に
有限性ガードを追加（既存 `convo::numeric_policy::isFinite`、bit-pattern、
fp:fast 安全。RT / ownership の変更なし。新規 authority なし）：

```cpp
// ★ STG-11-D11: 非有限値の格納を拒否。jlimit は NaN を素通しするため、
//   破損 session の NaN が格納されると prepare 時の smoothTotalGain が
//   NaN 化し RT 出力が NaN 化する。以前値を維持する。
if (!convo::numeric_policy::isFinite(static_cast<double>(gainDb)))
    return;
```

### D11 監査の副産物（いずれも正常・修正なし）

実コード追跡により以下を確定した（Gate §4 に記録）：

- EQ band float（freq/gain/q）：`calcSVFCoeffs` 系の `isfinite` ガードが中和。
- `nonlinearSaturation` の NaN：`saturation > 0.0` が偽になり適用スキップ。
- mixedF 系：`validateBuffer` が NaN 出力を拒否。
- tailL1L2Multiplier／tailMode：int-domain＋clamp のため有界。
- `copySnapshotToPendingUnlocked`：書き込み元は setter ガード済み。迂回なし。

### 変更範囲

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/eqprocessor/EQProcessor.Parameters.cpp` | 21 | 0 | **D11** 6 行＋D10 15 行 |
| `src/tests/AudioEngineHarness/STG11D11TotalGainFiniteTests.cpp` | 新規 | — | **D11**（untracked） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | +5 | 0 | D1〜D10 + **D11**（5 行） |
| `CMakeLists.txt` | +1 | 0 | D1〜D10 + **D11**（1 行） |
| `ConvoPeq.md` | 再生成 | — | — |

D1〜D10 の production 変更には触れていない。

---

## 3. 不変条件

- RT / Publish / Retire / Epoch / ownership に影響なし（Message Thread 上の早期 return のみ）。
- raw `std::atomic` 追加 0。mutex/alloc/thread/new-auth 0。
- 性能影響なし。

---

## 4. テスト

`STG11D11TotalGainFiniteTests.cpp`（harness サブテスト。新規 CTest target なし。
`getEQProcessor()` の実体を使用）：

| Test | 内容 | 結果 |
| --- | --- | --- |
| D11-T1 | NaN `totalGain` の setState → 有限値維持（3.0） | PASS |
| D11-T2 | 有効 round-trip（+6dB→保存→復元→一致） | PASS |
| D11-T3 | +Inf / -Inf 拒否 | PASS |
| Negative control | ガードを一時除去 → `NaN totalGain stored (nan)` で **FAIL**（rc=1）。復元後 PASS | 実証済み |

---

## 5. 実施した検証

| 検証 | 結果 |
| --- | --- |
| Debug / Release build | BUILD_EXIT=0 両方 |
| D11 sub-test（Debug / Release） | rc=0 / 全 PASS 両方 |
| D3 / D4 / D5 / D7 / D8 / D9 / D10 sub-test | 全 PASS（両構成） |
| D1 / D2 回帰 | TD1-1 Q=304 / TD1-2 T=280、TD2-1 E=3 / TD2-2 T=3 / TD2-3（不変） |
| full Debug CTest | **45/45 PASS**（初回 44/45 は STG-8-D2 flake。再実行 2 回で 45/45） |
| full Release CTest | **45/45 PASS** |
| raw atomic / mutex / alloc audit | 0 件 |
| clang-tidy（Parameters.cpp） | D11 hunk（105）内 0 件。残存は hunk 外の既存事象 |
| cppcheck | project 135 件は既存ノイズ、**Parameters.cpp 0 件** |
| ASAN | 環境 block（`0xC0000139`）。別問題として記録 |

---

## 6. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**。Gate は `READY FOR COMMIT`。
