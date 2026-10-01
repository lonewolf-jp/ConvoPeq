# STG-11-D8 Repair Implementation Record

- Document: `doc/work113/P1-5-IR-P2_STG-11-D8_REPAIR-IMPLEMENTATION_20260930.md`
- Work item: **STG-11-D8-1** — NaN IR length の無条件 store による IR 破壊
- Predecessor: `P1-5-IR-P2_STG-11-D8_REPAIR-CONTRACT-AUDIT_20260930.md`（GO）
- Date: 2026-09-30 → 2026-10-01
- Commit / push: **未実施**（D1〜D7 を作業ツリーに保持）

---

## 1. Authority

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| D7 マーカー反映 | 10 件（作業開始時に確認） |

完了後の再生成値は Gate §1 参照。

---

## 2. 実装（Contract §11 どおり）

`src/convolver/ConvolverProcessor.Runtime.cpp` のみ。2 setter の store 前に
有限性ガードを追加（既存 `convo::numeric_policy::isFinite`、bit-pattern、
fp:fast 安全。RT / ownership の変更なし。新規 authority なし）：

```cpp
// setTargetIRLength / applyAutoDetectedIRLength の冒頭:
if (!convo::numeric_policy::isFinite(static_cast<double>(timeSec)))
    return;  // 非有限は適用せず以前値を維持
```

### D8 監査の副産物（いずれも正常・修正なし）

実コード追跡により以下が安全であることを確定した（Gate §8 に記録）：

- dB gain / saturation：jlimit＋abs-diff gate により NaN は格納されず、Inf は clamp。
- ditherBitDepth：publish 経路は `RuntimePublicationValidator`（{0,16,24,32}）が拒否。
- adaptive coefficients：DSP 適用時に `clampCoeff`（非有限→0、±0.85）が中和。
- learner settings：tanh 写像＋stability check（既定 ON）。
- oversamplingFactor：setter ガード＋validator。
- Convolver の int-domain setter：var→int 変換＋jlimit で有界。
- `setMaxCacheEntries`：[1,64] clamp。
- `setMix` / `setSmoothingTime`：store 自体が abs-diff gate。

### 変更範囲

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/convolver/ConvolverProcessor.Runtime.cpp` | 9 | 0 | **D8** |
| `src/tests/AudioEngineHarness/STG11D8IRLengthFiniteTests.cpp` | 新規 | — | **D8**（untracked） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | +5 | 0 | D1〜D7 + **D8**（5 行） |
| `CMakeLists.txt` | +1 | 0 | D1〜D7 + **D8**（1 行） |
| `ConvoPeq.md` | 再生成 | — | — |

D1〜D7 の production 変更には触れていない。

---

## 3. 不変条件

- RT / Publish / Retire / Epoch / ownership に影響なし（Message Thread 上の早期 return のみ）。
- raw `std::atomic` 追加 0。mutex/alloc/thread/new-auth 0。
- 性能影響なし。

---

## 4. テスト

`STG11D8IRLengthFiniteTests.cpp`（harness サブテスト。新規 CTest target なし。
`getConvolverProcessor()` の実体を使用）：

| Test | 内容 | 結果 |
| --- | --- | --- |
| D8-T1 | NaN `irLength` / NaN `autoDetectedIRLength` の setState → 有限値維持 | PASS |
| D8-T2 | 有効 round-trip（1.5s→保存→復元→一致） | PASS |
| D8-T3 | +Inf / -Inf 拒否 | PASS |
| Negative control | `setTargetIRLength` ガードを一時除去 → `NaN irLength stored (nan)` で **FAIL**（rc=1）。復元後 PASS | 実証済み |

---

## 5. 実施した検証

| 検証 | 結果 |
| --- | --- |
| Debug / Release build | BUILD_EXIT=0 両方 |
| D8 sub-test（Debug / Release） | rc=0 / 全 PASS 両方 |
| D3 / D4 / D5 / D7 sub-test | 全 PASS（両構成） |
| D1 / D2 回帰 | TD1-1 Q=304 / TD1-2 T=280、TD2-1 E=3 / TD2-2 T=3 / TD2-3（不変） |
| full Debug / Release CTest | **45/45 PASS 両方** |
| raw atomic / mutex / alloc audit | 0 件 |
| clang-tidy（Runtime.cpp） | D8 hunk（931/954）内 0 件。残存は hunk 外の既存事象 |
| cppcheck | project 135 件は既存ノイズ、**Runtime.cpp 0 件** |
| ASAN | 環境 block（`0xC0000139`）。別問題として記録 |

---

## 6. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**。Gate は `READY FOR COMMIT`。
