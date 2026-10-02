# STG-11-D13 Fresh Discovery — Session-State Boundary / Setter-Boundary Residual Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D13_FRESH-DISCOVERY_20261002.md`
- Work item: **STG-11-D13** — D12 クローズ後の残存 invalid-state 監査
- Date: 2026-10-02
- Authority: リポジトリルート `ConvoPeq.md`（監査 AI が参照する `ConvoPeq(20261002-115916).md` と同一ファイル。
  ファイル名の違いを理由に別世代として扱わない。Owner 確認済みの同一最新統合ソース）
  - baseline commit: `75cb76ffce66fd51d984d6cfccf88c1951accdc0`（D12-2 repair）
  - `Generated: 2026-10-02 20:56:11` / 5,928,120 B
  - SHA-256: `41E27B57205FB50E6F9CAA76EFE6E9F199A3E6571481FF87ED94ADDDFC82E52F`
  - `--check`: `NEWER_SRC_COUNT = 0` / **FRESH**（調査開始時に確認）
  - 過去資料・記憶・旧行番号は根拠として使用していない。結論はすべて現行ソースから導出。
- 本 work item は **read-only Fresh Discovery**。**production / test / build ファイルの変更は一切行っていない。**
- 本書は **未 commit**（Owner の判断待ち）

---

## 0. 判定

```
STG-11-D13 — NO-GO（concrete defect なし）
```

D12-1〜D12-4 の修正済み領域を前提に、現行 tree の session / setter 境界を field 単位で
`入力 → setter → 永続 state → snapshot → consumer → harm` まで追跡した。
Owner の 7 条件（concrete trigger / state-data flow / guard 不在 / concrete harm /
ISR-Authority 影響 / minimal repair / 再現 test）をすべて満たす新規 defect は存在しない。

D12-R1 §7 の継続調査事項（`setSaturationAmount` の NaN 経路・`oversamplingFactor` の UI 経路）も
本 R で実コードまで追跡し、結論を確定した（§2）。

---

## 1. 監査範囲と方法

D12 の再発見で終わらせず、以下の面を現行ソースで直接確認した。

| # | 面 | 方法 | 結果 |
| --- | --- | --- | --- |
| 1 | dB / saturation setter の NaN / Inf | setter 本体 + jlimit / abs-gate の意味論 | §2.1 NO-GO（機構付き） |
| 2 | DeviceSettings XML の double 経路 | sanitize helper の有無 | §2.2 NO-GO |
| 3 | learner settings（cmaesRestarts / coeffSafetyMargin） | consumer の loop / clamp | §2.3 NO-GO（observation） |
| 4 | adaptive 係数の NaN | bank → RT apply 経路 | §2.4 NO-GO（機構付き） |
| 5 | Validator bypass（requires 節） | Bridge 実装の列挙 | §2.5 NO-GO |
| 6 | queue overflow / silent loss | registry / coordinator の full 時動作 | §2.6 NO-GO（D1/D2 の範囲） |
| 7 | atomic wrapper 規約違反 | setter / restore ファイルの raw store/load 走査 | §2.7 NO-GO |
| 8 | RT 到達の allocation / lock | DSPCore process ファイル走査 | §2.8 NO-GO |
| 9 | setProcessingOrder / setOversamplingFactor | caller 列挙 + setter 本体 | §2.9 NO-GO |
| 10 | oversamplingType の無検証 cast + 無防備 setter | consumer（boolean 比較）と validator の突合せ | §2.10 observation（defect 計上なし） |

---

## 2. 追跡結果

### 2.1 dB / saturation setter の NaN（D12-R1 §7-4 の確定）

対象: `setInputHeadroomDb`（`Parameters.cpp:239-255`）、`setOutputMakeupDb`（`:262-274`）、
`setConvolverInputTrimDb`（`:291-303`）、`setSaturationAmount`（`:599-`）。

いずれも次の 2 段構造である。

```cpp
const float clampedDb = juce::jlimit(-12.0f, 0.0f, db);          // NaN は NaN のまま通過
if (std::abs(consumeAtomic(x, acquire) - clampedDb) > 1e-5f)     // NaN を含む比較は false
{
    publishAtomic(x, clampedDb, release);                         // ← 到達しない
    ...
}
```

`juce::jlimit` は NaN をそのまま返すが、続く変化検出 `std::abs(a - b) > eps` が
NaN を含む比較を `false` にするため、**NaN 入力は publish されず no-op になる**。
一度 clean な値が入っている限り、NaN が atomic に到達する経路は無い。
（current 値自体が NaN の場合は `NaN - NaN = NaN` で同じく `false` のため、
setter 経路では NaN が定着することもない。）

±Inf は `jlimit` が範囲端に clamp するため、有限値として正常に処理される。

したがって **D12-R1 §7-4 の懸念は解消する**。`setSaturationAmount` の NaN 経路は
defect ではない。setter 単独の無防備は事実だが、隣接する abs-gate が中和点として
機能している。なおこれは意図的な設計ではなく偶然の一致であるため、
将来 abs-gate を外す変更を行う場合は再評価が必要である（注意記録に留める）。

### 2.2 DeviceSettings XML の double 経路

`src/DeviceSettings.cpp:45-49` に次の helper が存在し、XML 復元で使用されている。

```cpp
double sanitizeFiniteClamped(double value, double fallback, double minValue, double maxValue) noexcept
{
    const double finite = sanitizeFiniteOrDefault(value, fallback);
    return juce::jlimit(minValue, maxValue, finite);
}
```

使用箇所（抜粋）: `:1225` inputHeadroomDb、`:1229` outputMakeupDb、`:946` sigma、
`:948` elapsedPlaybackSeconds、`:951` bestScore。XML double 経路は有限性と範囲の
両方を検査済みである。**D12-R1 §7-4 の XML 側の懸念も解消する。**

### 2.3 learner settings（cmaesRestarts / coeffSafetyMargin）

`StateIO.cpp:153-163` は範囲検査なしに `NoiseShaperLearnerSettings` へ格納する。
consumer は次のとおり。

- `cmaesRestarts`: `NoiseShaperLearner.cpp:798-799`
  `for (restartIdx = 0; restartIdx < restarts; ++restartIdx)`。
  負値は loop 不実行（benign）。巨大値は学習の長期化を招くが、
  `:801` で `stopRequested` / `stopToken` を毎反復検査するためユーザーが停止できる。
  学習自体が UI の明示操作でのみ開始される NonRT スレッドであり、
  取消不能の hang にはならない。concrete harm の基準に届かない。
- `coeffSafetyMargin`: `NoiseShaperLearner.cpp:659-661`
  `clampCoeff(tanhBuffer[i], safetyMargin)`。2 引数版 `clampCoeff`
  （`LatticeNoiseShaper.h:128-137`）は value の有限性を検査し NaN を 0.0 にする。
  margin 自身が NaN の場合は比較がすべて偽になり clamp が外れるだけであり、
  tanh 出力は有限のため NaN は注入されない。

いずれも defect 計上しない。observation として記録する。

### 2.4 adaptive 係数の NaN

`setAdaptiveCoefficientsForSampleRate`（`Learning.cpp:481-501`）および
`setAdaptiveCoefficientsForSampleRateAndBitDepth`（`:535-556`）は session の double を
有限性検査なしに bank へ格納する（`storeLearnedCoeffsToBank` も検査しない）。

しかし RT apply 経路は次のとおり中和されている。

```cpp
// DSPCoreIO.cpp:444 / DSPCoreDouble.cpp:638
adaptiveNoiseShaper.applyMatchedCoefficients(state.adaptiveCoeffSet->k, kAdaptiveNoiseShaperOrder);
// → LatticeNoiseShaper.h:62-65 → :56  coeffs[i] = clampCoeff(newCoeffs[i]);
// → :113-125  1 引数版 clampCoeff は isFinite を検査し NaN を 0.0 にする
```

bank に NaN が格納されても、DSP へ適用される時点で 0.0 に中和される。
**NaN audio には到達しない。** defect 計上しない。

### 2.5 Validator bypass（requires 節）

`src/core/RuntimePublicationCoordinator.h:118` の
`if constexpr (requires(Bridge bridge, const World& world) { bridge.validatePublicationNonRt(world); })`
は、Bridge 型に同名 method が無い場合に validation を黙って省略する構造である。

しかし production で同 Coordinator に渡される Bridge 型は
`AudioEngine::RuntimePublicationBridge`（`AudioEngine.h:3801`、method あり）のみであり、
他の Bridge 型の production 実装は存在しない（test 用 2 件も method あり）。
**live の bypass 経路は存在しない。** defect 計上しない。

### 2.6 queue overflow / silent loss

`PendingPublishRegistry::registerPublish`（`RuntimeWorldAuthority.h:43-49`）、
`SnapshotCoordinator` の queueFull 時の quarantine、retire routing は
STG-11-D1 / D2 の監査範囲であり、本 R で新規の証拠は得られていない。
D19 の failure boundary 走査（O-1〜O-7）も concrete harm なしで確定済みである。
再監査しない。

### 2.7 atomic wrapper 規約違反

setter / restore ファイル（`AudioEngine.Parameters.cpp`、`AudioEngine.StateIO.cpp`、
`AudioEngine.Learning.cpp`、`DeviceSettings.cpp`）に raw `.store()` / `.load()` /
`.exchange()` は存在しない。すべて `convo::publishAtomic` / `convo::consumeAtomic` /
`convo::fetchAddAtomic` 経由である。違反なし。

### 2.8 RT 到達の allocation / lock

`AudioEngine.Processing.DSPCoreIO.cpp` / `DSPCoreFloat.cpp` / `DSPCoreDouble.cpp` に
`new` / `malloc` / `std::mutex` / `std::lock_guard` / `std::scoped_lock` は存在しない。
違反なし。

### 2.9 setProcessingOrder / setOversamplingFactor

- `setProcessingOrder`（`Parameters.cpp:281-289`）自体に guard は無いが、
  型付き enum であり、C++ 呼び出し元はすべて有効な enumerator
  （`MainWindow.cpp:1372/1377`）か guard 済み restore（`StateIO.cpp:41-51`）である。
  `DeviceSettings` からは呼ばれていない。不正値の到達経路が無い。
- `setOversamplingFactor`（`Parameters.cpp`）は `1/2/4/8` のみ受理し他は 0 に正規化する。
  `DeviceSettings.cpp:1222` も同 setter 経由であり安全。

いずれも defect 計上しない。

### 2.10 observation: oversamplingType の無検証 cast + 無防備 setter（defect 計上なし）

```cpp
// DeviceSettings.cpp:1236-1237
int type = xml->getIntAttribute("oversamplingType", 0);
engine.setOversamplingType((AudioEngine::OversamplingType)type);   // 無検証 cast
// AudioEngine.Parameters.cpp:719-732
void AudioEngine::setOversamplingType(OversamplingType type)
{
    convo::publishAtomic(oversamplingType, type, release);         // 無条件（変化検査すら無い）
    ...
}
```

D12-2 と同型の構造的欠落である。**しかし defect として計上しない。** 理由は次の 3 点である。

1. consumer が boolean 比較である（`DSPCoreLifecycle.cpp:184/256`
   `(oversamplingType == OversamplingType::LinearPhase) ? LinearPhase : IIRLike`）。
   範囲外値（例: 7）は IIR 側に落ちる。OOB もクラッシュも無い。
2. validator が `oversamplingType` を検査しない（`validateResources` は
   oversamplingFactor のみ）。したがって D12-1 / D12-2 のような sticky-reject は
   発生しない。publish は継続する。
3. 到達経路が device-settings XML の改変に限定される（session 経路は B-3 guard 済み、
   UI 経路は `:586` の ternary で有効な enumerator のみ）。

効果は「IIR への silent fallback と UI/engine の乖離」に留まり、
ISR invariant の違反が無い。Owner の基準（concrete harm + ISR 影響 + minimal repair）に
照らして defect ではない。1 行の setter guard で閉じられる形であるため、
将来同型の sticky 構造が見つかった場合の候補として記録する。

---

## 3. D7〜D12-4 の再発確認（該当なし）

本 R の追跡で D7 / B-3 / D8 / D9 / D10 / D11 / D12-1 / D12-2 / D12-3 / D12-4 の
修正済み箇所に再発は無い。`setDitherBitDepth` / `setNoiseShaperType` の guard、
`updateBitDepthList` の集合制限、validator 契約はいずれも現行ソースで維持されている。

---

## 4. 本 R1 で確定しなかった事項（残置）

なし。本 R で D12-R1 §7 の 4 / 5（saturation NaN・oversamplingFactor UI）は確定した。
D12-R1 §7 の 1 / 2 / 3（blocked 範囲の厳密境界・D12-3 の UI 乖離・validator helper の最終扱い）は
D12-2 repair（`75cb76ff`）により解決または設計事項として切り分け済みである。

---

## 5. 本 D13 の作業記録（read-only 遵守）

```text
変更ファイル: 0 件（production / test / build / CMakeLists.txt / ConvoPeq.md すべて無変更）
追加ファイル: 本書 1 件（未 commit。Owner の判断待ち）

実施した操作:
  - 読み取り（git show / rg / Read）: 多数
  - ビルド / テスト: なし（ソース未変更のため不要）
  - reset / clean / rebase / amend / squash / force push: いずれも不使用

main checkout (C:\VSC_Project\ConvoPeq): 未改変（HEAD = bda43034）
作業 worktree: HEAD = 75cb76ff（= origin/main）、clean、未追跡 1 件（本書のみ）
```
