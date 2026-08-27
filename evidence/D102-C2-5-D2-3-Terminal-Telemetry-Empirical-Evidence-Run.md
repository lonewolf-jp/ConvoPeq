# D102-C2-5-D2-3 — Terminal Telemetry Empirical Evidence Run

- **実施日**: 2026-08-26
- **作業種別**: **measurement / analysis-only**（production source 変更 **0** / test source 変更 **0** / contract 変更 **0**）
- **測定環境**:
  - 実行バイナリ: 既存 `build-icx/Release/AudioEngineHarness.exe`（icx Release、`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` の既存ビルド構成を使用。ソース変更なし・target 再ビルドのみ）
  - 実行には oneAPI ランタイム環境（setvars.bat intel64）を使用（icx runtime DLL 解決のため）
  - AudioEngineHarness は simulated audio thread（実デバイス不要・headless）
  - 生ログ: `evidence/d2_3_logs/`（8 ファイル）

---

## Phase 1 — T1 Baseline（通常運転 60 秒）

| 指標 | 実測 |
|---|---|
| samples | 594（100ms 周期） |
| `T_store` total | **0** |
| `T_peak` max | **0** |
| `T_resident` max / final | **0 / 0** |
| `pendingRetire` max | 1 |
| `pressureLevel` max | 0 |

**結論**: 通常動作では Terminal への admission が **1 件も発生しない**。
Terminal growth は通常運転では存在しない。

---

## Phase 2 — T3 Long-stall（reader stall 段階負荷）

| stall | samples | pending_max | Q_max | E_max | pressure_max | **T_store** | **T_peak/T_resident max** | **final T_resident** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 s | 149 | 457 | 0 | 0 | 0 | **0** | **0** | 0 |
| 5 s | 228 | 2348 | 0 | 0 | 0 | **0** | **0** | 0 |
| 10 s | 326 | 4096 | 526 | 14 | 3 | **0** | **0** | 0 |
| **30 s** | 722 | 4096 | **1024 (full)** | **512 (full)** | 3 | **8632** | **8632 / 8605** | **0** ✅ |

### T3-30s 詳細時系列

```text
sample 0–138      : T_resident = 0（Q/E が滞留を吸収）
sample 139        : Q(1024 full) + E(512 full) 枯渇 → Terminal 到着開始
sample 139–325    : T_resident 増加（非ゼロ区間）、peak 8605
sample ~326       : reader recovery → minEpoch 進行 → epoch-gated drain 発動
sample 326 以降   : T_resident = **0 完全回収**（最終 50 サンプルすべて 0）
```

**drain メカニズムの実測確認**: `T_drainAll = 0`・`T_drainEntry = 0` のまま
8,632 エントリが完全回収された → 回収は **epoch-gated `drain()`** によるもの。
これは D2-2 E1 の caveat（drain() は cumulative counter を増加させない）を実測で裏付ける。

---

## Phase 3 — T4 Load Sweep（固定 30 秒 stall、実測 lambdaPublish 採用）

| Case | intervalUs | 実測 lambdaPublish | stallPublishes | Q_max | E_max | **T_store** | **T_resident max** | **final** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| T4-A | 5000 | **156.08/s** | 4683 | 577 | 65 | **0** | **0** | 0 |
| T4-B | 3333 | **187.38/s** | 5622 | 1024 | 512 | **502** | **483** | **0** ✅ |
| T4-C | 1000 | **448.26/s** | 13448 | 1024 | 512 | **8328** | **8316** | **0** ✅ |

負荷依存性が明確: 156/s → Terminal 不使用。187/s → 少量滞留。448/s → 大量滞留。
ただし **いずれのケースでも stall 終了＋recovery 後に T_resident は厳密に 0 へ回帰**。

---

## Phase 4 — 判定（transient / sustained 分類）

一次情報は `T_resident(t)`（D2-2 E1/E2 条件を継承 — `T_store − T_drainEntry` は不使用）。

| Run | 分類 | 根拠 |
|---|---|---|
| T1 baseline | **resident ≈ 0** | 全 594 サンプルで 0 |
| T3 1s/5s/10s | **resident ≈ 0** | Terminal 到達すらなし（D/Q/E で吸収） |
| T3 30s | **A. transient spike → 0** | 非ゼロ区間 [139..326] のみ、recovery 後全回収、再発なし |
| T4-A | **resident ≈ 0** | 全サンプル 0 |
| T4-B / T4-C | **A. transient spike → 0** | 同上（非ゼロ区間は stall 中のみ） |

**B. sustained resident / C. sustained growth: 該当なし。**
`T_resident(t2) > T_resident(t1)` の長時間継続および
`admission rate > effective drain rate` の持続は観測されなかった。

---

## Phase 5 — 判定表適用

| 実測結果 | D2-1 再開 | 本実測 |
|---|---|---|
| T resident ≈ 0 | NO-GO 維持 | ✅ **該当（T1/T3短/T4-A）** |
| 一時的 spike → 0 | NO-GO 維持 | ✅ **該当（T3-30s/T4-B/T4-C）** |
| peak は上がるが resident は回収 | NO-GO 維持 | ✅ **該当**（peak 8632 でも resident は 0 回帰） |
| resident 長時間非 0 plateau | 原因分析追加 | 該当なし |
| resident 継続的増加 | Phase B 再開候補 | **該当なし** |
| 複数負荷で sustained growth | Phase B 強く支持 | **該当なし** |
| growth + memory pressure | Phase B 起動条件成立 | **該当なし** |

**`terminalPeakResident` の増加単独では Phase B を起動しない**原則どおり、
peak 8632（T3-30s）/ 8328（T4-C）は記録されたが、いずれも人工的 30 秒全 reader stall という
極端な異常注入下での一時値であり、recovery で完全回収されている。

---

## Phase 6 — D2-1 Phase B 起動判定

### Gate B-ENTRY-1: `T_resident` の sustained non-zero / growth
→ **未達**。全 8 ランで recovery 後の T_resident は 0。最終 50 サンプル min=max=0。

### Gate B-ENTRY-2: 増加と publish load・memory pressure telemetry の相関
→ **判定不要**（ENTRY-1 未達のため）。参考として、負荷↔Terminal admission の相関自体は
観測された（156/s→0、448/s→8328）が、それは「stall 下で Q/E が満杯になると Terminal が
設計通り所有権を引き受ける」という P-4 契約の正常動作を示すものであり、leak/滞留ではない。

### 判定: **Gate B-ENTRY 不成立 → D2-1 NO-GO を実測により維持。Phase B/C 引き続き保留。**

---

## 学習事項（設計への含意）

1. **Terminal は「最後の authority」として設計どおり動作する**:
   Q(1024)+E(512) 枯渇という极端条件下でのみ登場し、reader 回復とともに
   lock-free 経路（epoch-gated drain）で完全排出される。drainAll は一度も発火しなかった。
2. **実測 K 参考値**: 30 秒全 reader stall @448 pub/s で peak 8,632 エントリ。
   将来 bounded 化を再審する場合の負荷条件データとして `evidence/d2_3_logs/` を保存。
3. **telemetry 分析規約の実証**: `T_drainEntry` は drainAll 専用であることが実測でも確認
   （8,632 回収に対し drainEntryCount==0）。D2-2 の分析規約を将来の soak 設計にも継承すること。

---

## 制限事項

- 単一マシン・単一ラン（各条件 1 回）。統計的ばらつきは未評価
- T4 の実測レートは目標（200/300/1000/s）に届かず（156/187/448/s）— harness の publish path
  処理能力による。ただし「低負荷→無滞留／高負荷→一時滞留→完全回収」の傾向判定には十分
- 60 秒 baseline は既定 600 秔より短い。より長期 baseline が必要になった場合は `--t1=600` で再実施可能

## 変更有無

```
production source change : 0（既存 CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON ビルド構成を使用、ソース無変更）
test source change       : 0（既存 T1/T3/T4 CLI をそのまま実行）
contract change          : 0
新規 telemetry 追加      : 0
```
