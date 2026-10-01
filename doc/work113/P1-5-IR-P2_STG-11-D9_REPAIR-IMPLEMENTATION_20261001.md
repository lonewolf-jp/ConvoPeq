# STG-11-D9 Repair Implementation Record

- Document: `doc/work113/P1-5-IR-P2_STG-11-D9_REPAIR-IMPLEMENTATION_20261001.md`
- Work item: **STG-11-D9-1** — NaN tail state の無条件 store による NaN DSP 出力／形状破壊
- Predecessor: `P1-5-IR-P2_STG-11-D9_REPAIR-CONTRACT-AUDIT_20261001.md`（GO）
- Date: 2026-10-01
- Commit / push: **未実施**（D1〜D8 を作業ツリーに保持）

---

## 1. Authority

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| D8 マーカー反映 | 9 件（作業開始時に確認） |

完了後の再生成値は Gate §1 参照。

---

## 2. 実装（Contract §10 どおり）

`src/convolver/ConvolverProcessor.Runtime.cpp` のみ。2 setter の store 前に
有限性ガードを追加（既存 `convo::numeric_policy::isFinite`、bit-pattern、
fp:fast 安全。RT / ownership の変更なし。新規 authority なし）：

```cpp
// setTailStartSec / setTailStrength の冒頭:
if (!convo::numeric_policy::isFinite(static_cast<double>(sec/strength)))
    return;  // 非有限は適用せず以前値を維持
```

### D9 監査の副産物（いずれも正常・修正なし）

実コード追跡により以下を確定した（Gate §4 に記録）：

- mixedF 系：LoaderThread の `validateBuffer`（有限＋energy check）が NaN 出力を拒否。
- tailL1L2Multiplier：int-domain＋clamp のため有界。
- D8 監査の安全確定項目（dB/saturation、dither validator、adaptive clampCoeff、
  learner、int-domain、cache clamp）は再確認のうえ維持。

### 変更範囲

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/convolver/ConvolverProcessor.Runtime.cpp` | 19 | 0 | **D9**（D8 9 行を含む累積） |
| `src/tests/AudioEngineHarness/STG11D9TailFiniteTests.cpp` | 新規 | — | **D9**（untracked） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | +5 | 0 | D1〜D8 + **D9**（5 行） |
| `CMakeLists.txt` | +1 | 0 | D1〜D8 + **D9**（1 行） |
| `ConvoPeq.md` | 再生成 | — | — |

D1〜D8 の production 変更には触れていない。

---

## 3. 不変条件

- RT / Publish / Retire / Epoch / ownership に影響なし（Message Thread 上の早期 return のみ）。
- raw `std::atomic` 追加 0。mutex/alloc/thread/new-auth 0。
- 性能影響なし。

---

## 4. テスト

`STG11D9TailFiniteTests.cpp`（harness サブテスト。新規 CTest target なし。
`getConvolverProcessor()` の実体を使用）：

| Test | 内容 | 結果 |
| --- | --- | --- |
| D9-T1 | NaN `tailStrength` / `tailStartSec` の setState → 有限値維持 | PASS |
| D9-T2 | 有効 round-trip（1.25/0.3 設定→保存→復元→一致） | PASS |
| D9-T3 | +Inf / -Inf 拒否 | PASS |
| Negative control | `setTailStrength` ガードを一時除去 → `NaN tailStrength stored (nan)` で **FAIL**（rc=1）。復元後 PASS | 実証済み |

---

## 5. 実施した検証

| 検証 | 結果 |
| --- | --- |
| Debug / Release build | BUILD_EXIT=0 両方 |
| D9 sub-test（Debug / Release） | rc=0 / 全 PASS 両方 |
| D3 / D4 / D5 / D7 / D8 sub-test | 全 PASS（両構成） |
| D1 / D2 回帰 | TD1-1 Q=304 / TD1-2 T=280、TD2-1 E=3 / TD2-2 T=3 / TD2-3（不変） |
| full Debug / Release CTest | **45/45 PASS 両方** |
| raw atomic / mutex / alloc audit | 0 件 |
| clang-tidy（Runtime.cpp） | D9 hunk（1172/1197）内 0 件。残存は hunk 外の既存事象 |
| cppcheck | project 135 件は既存ノイズ、**Runtime.cpp 0 件** |
| ASAN | 環境 block（`0xC0000139`）。別問題として記録 |

---

## 6. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**。Gate は `READY FOR COMMIT`。
