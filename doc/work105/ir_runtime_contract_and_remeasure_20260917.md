# WORK105 — D の構造的 containment（IR／ランタイム形状契約）と再測定

- **作成日**: 2026-09-17
- **種別**: Implementation（NonRT publish 前契約）+ Measurement（再測定による因果分離）
- **前工程**: WORK103 / WORK103-R2 / WORK104（`doc/work103/`, `doc/work104/`）
- **指示に基づく方針**: 「音を良くする修正」ではなく、まず D の rate/block mismatch を
  RuntimeWorld の publish 前契約として封じ込め、negative test で受入条件を固定する。
  優先順 F1 → F2 → F3 → 再測定 → F5 → F4 のうち、本作業は **F1/F2/F3 + 再測定**まで。

```text
判定: F1 IMPLEMENTED+VERIFIED / F2 IMPLEMENTED+TESTED / F3 PARTIAL
      再測定で D は buzz の原因ではないことが判明（否定的結果・重要）
      E（conv活性時のガベージ）が新たな未解決欠陥として残存 → 次工程
```

---

## 1. 実装した契約（NonRT・publish 前関門）

### 1.1 新規：IR形状契約（純粋関数・JUCE/MKL 非依存）

`src/audioengine/IRRuntimeContract.h`

```cpp
enum class IRRuntimeContractViolation { None, RateMismatch, BlockMismatch, UnknownSourceGeometry };
IRRuntimeContractResult checkIRRuntimeContract(irRate, irBlock, worldRate, worldBlock) noexcept;
```

- rate は相対 1e-9 で照合（double の表現揺れを吸収）、block は完全一致。
- 形状不明（旧経路: block==0 等）は **拒否せず許可**＋呼び出し側 loud log（後方互換）。
- RT へは一切持ち込まない（RT は Read→Execute→Output に限定。M-04 oversize gate は
  defense-in-depth として維持）。

### 1.2 BuildError 拡張と分類

`src/audioengine/BuildErrorPolicy.h` に `IRRateMismatch` / `IRBlockMismatch` を追加。
分類は `Infrastructure / RetryBackoff`（IR再構築で回復し得るため）。descriptor table と
`kBuildErrorNames` の size static_assert を 10 値に更新。

### 1.3 IRState への形状刻印（追跡可能化）

`src/ConvolverProcessor.h` の `IRState` に `int blockSize` を追加し、`updateIRState(...)` に
`irBlockSize` 引数を追加（既定 0＝不明）。書き込み元：

| 経路 | block 刻印 | ファイル |
|---|---|---|
| LoaderThread 非同期 finalize | `knownBlockSize`（= nextPowerOfTwo(bs)） | `ConvolverProcessor.LoadPipeline.cpp` |
| LoaderThread 同期 build | 同上 | `ConvolverProcessor.LoaderThread.cpp` |
| incremental finalize | 0（不明・loud log） | `ConvolverProcessor.Rebuild.cpp` |
| transferIRStateFrom | 伝搬（+ block/gen をログ出力） | `ConvolverProcessor.h` |

NonRT 読み出し API を追加：`getIRGeometry()`（hasIR/rate/block/generation）、
`getPreparedSampleRate()`、`getPreparedBlockSize()`。

### 1.4 publish 前の照合（Builder）

`src/audioengine/RuntimeBuilder.cpp` の `build()` が `prepare()` 直後に照合：

- `in.convBypassed` の world は convolver を実行しないため対象外。
- 一致 → 許可 / 不一致 → `BuildError::IRRateMismatch` または `IRBlockMismatch` で
  **publish 拒否**（`result.runtime` を返さない＝world は前回のまま）。
- 不明形状 → 許可＋`[IR_CONTRACT] unverifiable ...` を loud log。

### 1.5 F1：UI convolver の processing geometry 追従（本質的修正）

従来は `uiConvolverProcessor.prepareToPlay(hostSr, hostBs)`（`PrepareToPlay.cpp:325`）。
このため LoaderThread が **host rate** で IR を build し、DSP world（processing rate）へ
transfer されて rate mismatch → ガベージ（WORK104 の transfer ログ `len=31457 sr=192000`
に対し DSP 384k）。

修正：`AudioEngine::reprepareUiConvolverForProcessingGeometry(hostSr, hostBs)` を新設し、
`OversamplingPolicy::resolve()` で解決した processing 形状（rate×OS, bs×OS）で UI convolver を
prepare する。呼び出し点は 2 箇所：

1. `prepareToPlay`（prepare 時）
2. `setOversamplingFactor`（OS 変更時に追従。追従漏れは新形状との mismatch を生み、
   publish 拒否の連鎖になるため必須）

### 1.6 F3（部分）：M-1 再利用の形状ガード

`ConvolverProcessor.Lifecycle.cpp` の M-1 経路（旧 IR 配列を resample せず再利用する
re-init）に形状照合を追加。`storedSampleRate` / `storedKnownBlockSize` が新形状と
一致しない場合は再利用せず、旧 engine を維持して `rebuildAllIRsSynchronous` の完成を待つ
（無警告ガベージの経路を遮断）。NonRT のみ。

## 2. Negative test（受入条件）

### 2.1 新規：`IRRuntimeContractTests`（5/5 PASS）

`src/tests/IRRuntimeContractTests.cpp`（CMake 登録済み・JUCE/MKL 非依存）

| 検査 | 内容 | 結果 |
|---|---|---|
| match | 384k/2048 一致 → 許可 | PASS |
| rate mismatch | 192k vs 384k／逆方向／1%差 → **拒否** | PASS |
| rate tolerance | 1e-12 相対差 → 許可 | PASS |
| block mismatch | 2048 vs 4096／1024 vs 2048 → **拒否** | PASS |
| unknown | block=0／rate=0／負値 → 許可（Unknown） | PASS |
| toString | 4 値網羅 | PASS |

### 2.2 `BuildErrorClassificationTests`（100 checks / 0 fails PASS）

10 値化に追随（table/name/classifier 一致、defensive fallback、warmup retry policy 不変）。

### 2.3 実機系の受入（harness 実行）

WORK104 の D 実験（OS`2→4`、IR 再ロードなし）は、修正後 **publish されなくなった**
（`no world publish within 30000ms (seq stayed 13)`）＝「stale IR を使用しない」を満たす。
旧挙動は無警告でガベージ再生、新挙動は **拒否（＝無音維持）**。

## 3. 再測定（F1 適用後・192kHz／1024／OS×2）

### 3.1 F1 の効果（ログ実測）

```text
[CONV_IR] transferIRStateFrom: IR transferred ch=2 len=62914 sr=384000.0 block=2048 gen=8
```

旧: `len=31457 sr=192000.0`（DSP 384k と不一致）→ 新: **384000/2048 で完全一致**。
`[IR_CONTRACT] REFUSED` は **0 件**（bootstrap の IR ロード照合も通過）。

### 3.2 リグ妥当性（-6dBFS プローブ）

```text
SELFTEST C0 sine50(-6dBFS) ratio=0.8846 (=dither headroom 0.891相当) thd=-147.9dB
```

→ bypass 経路は透明（安全鎖非接触域）。測定系は正常。

### 3.3 主因Aの精密確定（0dBFS プローブ）

```text
C0(bypass) sine50(0dBFS) outPeak=0.8414 rms=0.5963 thd=-56.2dB ultra=-111.6dB
```

**出力が Limiter 閾値 0.8413951287507587 に完全一致**。A は「0dBFS 入力では dither
headroom 0.891 > limiter threshold 0.8414 のため常時リミッティング」という機序で確定
（WORK104-R2 の §2.3 予測どおり。低域ブーストで深化）。

### 3.4 **重要（否定的結果）: レート整合後もガベージは消えない**

形状を完全一致させた conv-only（C1）／conv+eq（C3）／makeup-6dB（C4）：

| row | outPeak | THD | ultra | 設計期待 |
|---|---|---|---|---|
| C1 sine50 | **0.0476** | **-16.5dB** | **-56.4dB** | ~0.128（scale込み）・clean |
| C3 sine50 | **0.0476** | **-16.5dB** | **-56.4dB** | ~0.13・limiter微接触 |
| C4 sine50 | **0.0476** | **-16.5dB** | **-56.4dB** | 同上 |
| C1 kick | 0.0399 | — | -49.5dB | — |
| C1 multi | 0.0820 | — | -69.4dB | — |

**rate/block を一致させても WORK104 と同一のガベージが残存**（0.0474→0.0476 と実質不変）。
したがって **D（レート不整合）は real defect だが buzz の原因ではない**。WORK104 の
因果候補のうち D は反証され、残る欠陥 E（conv 活性時のガベージ）が未解決の主因候補。

E の性質：レベルは設計の約 -20dB、THD -16.5dB、超音波 +82dB。**IR サンプルレートを
192k→384k と変えても性状がほぼ不変**という事実は、IR 内容そのものより
**convolver 実行系（partition/quantum/位相変換/scale 経路）** を示唆する。

### 3.7 フル行列（8 config × 5 信号 × 2.5s、rows=41）— E の範囲確定

| config | 内容 | sine50 outPeak / THD / ultra | 判定 |
|---|---|---|---|
| C0 | bypass-all | 0.8414 / -56.2dB / -111.6dB | limiter 閾値一致（A） |
| C1 | conv-only | **0.0476 / -16.5dB / -56.4dB** | **ガベージ（E）** |
| C2 | eq-only | 0.5274 / **-148.2dB** / -138.5dB | clean（EQ 経路は健全） |
| C3 | conv+eq | **0.0476 / -16.5dB / -56.4dB** | C1 と**bit 一致** |
| C4 | conv+eq makeup-6dB | **0.0476 / -16.5dB / -56.4dB** | makeup が**効いていない** |
| C5 | conv+eq softclip-off | **0.0950 / -16.5dB / -56.4dB** | ちょうど C1 の **2 倍** |
| C8 | conv+eq shaper-psycho | 0.0476 / -16.5dB / -56.4dB | shaper 無関係 |
| C6 | conv+eq phase-**AsIs** | 0.0477 / -16.5dB / -56.4dB | **Mixed と同一** |

確定した知見：

1. **Mixed 位相変換は無罪**（C6 AsIs が同一）→ 位相変換起因説は反証。
2. **EQ／shaper／makeup はガベージに影響しない**（C1=C3=C4=C8）。ただし C4 で
   makeup -6dB が効かないのは**別の二次所見**（手動ゲインステージングの伝搬）。
3. **softclip のみ 2 倍の差**（C5=0.0950）。384k 域の softclip がガベージを
   押さえ込んでいる＝ガベージは大振幅（入力に追従）。
4. 入力周波数依存：sine40 は **THD +2.8dB（残差が基本波より大きい）**＝出力は入力正弦では
   ない。ultra は -41〜-68dB の広帯域ハッシュ。
5. C2（EQ のみ）が clean である以上、**E は convolver 実行系に限定**される。

### 3.8 E の有力仮説（WORK106 の起点）

IR のエネルギーは先頭 ~70 サンプル（時間換算 0.18ms）に 99% 集中し、尾部は -100dB 級。
にもかかわらず出力が設計の約 -20dB で歪むという事実は、**「先頭（ピーク）が欠落し
尾部のみが畳み込まれている」**、または **「L0/L1/L2 の寄与が位相不整合で部分的に
打ち消し合っている」** ことを強く示唆する。根拠：

- `MKLNonUniformConvolver` の層構成（L0=即時 ring／L1/L2=遅延補償）と、
  IR パーティションを逆順に並べ替える最適化（`MKLNonUniformConvolver.cpp:1006-1032`）、
  L1/L2 の `outputDelaySamples`／`delayLineBuf`（同:1051-1066）のいずれかが破綻すると、
  先頭成分の喪失＋部分打消しが同時に起きる。
- ログには `[B13-GATE] L1: P=4096 o_L=6144 lead=6144 | I2(lead<=oL-B)=NG I2g(2P-B<=oL)=NG`
  という **NG 判定**が ctest 実行時に出現していた（本測定の 384k 構成とは別構成だが、
  同ゲートの整合性は要確認）。
- レート非依存（192k/384k で同一）であることは、**サンプル数固定の層境界／リング長**
  由来（幾何バグ）と整合する。

WORK106 の最初の一手は、**単一デルタ IR（後述の `writeBoostIrFile` から peaking を除いた
もの）** を投入し、出力が「clean な遅延デルタ」になるかを測ることで、
NUC 実行系の健全性を 1 ビットで判定すること。デルタが壊れれば幾何バグで確定する。


### 3.5 D の containment（受入）

```text
D-convonly: world seq 12 -> 13 committed
D-os4: no world publish within 30000ms (seq stayed 13)   ← 契約が拒否＝stale IR 不使用
D-os2restore: world seq 13 -> 14 committed
```

### 3.6 B2（据え置き・優先順どおり後回し）

```text
boost-IR readback getIrFreqPeakGainDb = 0.00 dB (file-side expected ~+9dB@50Hz)
```

WORK104 と同じく再現。指示の優先順（F5 は F1〜F3 の後）に従い本作業では未修正。

## 4. 変更ファイル一覧

| ファイル | 変更 |
|---|---|
| `src/audioengine/IRRuntimeContract.h` | **新規** 純粋契約関数 |
| `src/audioengine/BuildErrorPolicy.h` | +2 error値 / table / names |
| `src/audioengine/RuntimeBuilder.cpp` | +照合（publish前関門） |
| `src/audioengine/AudioEngine.h` | +`reprepareUiConvolverForProcessingGeometry` 宣言 |
| `src/audioengine/AudioEngine.Parameters.cpp` | +同実装 / `setOversamplingFactor` から追従 |
| `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | UI convolver を processing geometry で prepare |
| `src/ConvolverProcessor.h` | IRState.blockSize / PendingCommit.knownBlockSize / 形状API |
| `src/convolver/ConvolverProcessor.Lifecycle.cpp` | updateIRState に block 刻印 / M-1 形状ガード |
| `src/convolver/ConvolverProcessor.LoadPipeline.cpp` | knownBlockSize 伝搬 |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | 同期経路の刻印 |
| `src/convolver/ConvolverProcessor.Rebuild.cpp` | incremental 経路（block=0 不明） |
| `src/tests/IRRuntimeContractTests.cpp` | **新規** negative tests |
| `src/tests/BuildErrorClassificationTests.cpp` | 10値化 |
| `CMakeLists.txt` | +IRRuntimeContractTests |

production の DSP 音声経路（RT）への変更は **0**（NonRT の prepare/Builder/Loader のみ）。

## 5. 残課題（次工程 = WORK106 候補）

1. **E の隔離（最優先）**: §3.8 の候補を測定で切り分ける。
   - **単一デルタ IR プローブ**（最有力の 1 手。壊れれば幾何バグ確定）
   - L0/L1/L2 の寄与分離（`tailMode=Bypass` で L1/L2 を落として L0 単独を測る）
   - 層境界・ring 長・`outputDelaySamples` の実測値と理論値の突合
   - `MKLNonUniformConvolver` の IR 逆順ソート（:1006-1032）の正当性確認
2. **二次所見**: C4（手動 makeup -6dB）が出力に反映されない問題の切り分け
   （autoGain OFF 時の ProcessingPart 伝搬）。
3. F5（B2: irFreqPeakGainDb の NUC 経路伝搬）→ 再測定。
4. F4（limiter 運用点: lookahead / 天井）→ A の解消。
5. `--buzz-out` CSV が生成されない小不具合（stderr には全 row 出力済み）。
6. Dr.Memory による動的検査（build/Release に成果物あり）。

## 6. ツール使用（本作業）

- headroom proxy（transport）＋context-mode（batch/execute）＋rtk 常時。
- WSL: rg / ast-grep / fdfind / ag / fzf / sed / awk。AiDex（query・note）、cocoindex、
  graphify、semble。
- cppcheck（`--language=c++ --std=c++20 --enable=warning,performance,portability`）:
  新規 2 ファイル **指摘 0**。
- clang-tidy / tgrep / Obscura / DDGS / trafilatura / firecrawl / brave / context7 は
  WORK103/104 と同様に利用可能（本作業では cppcheck とビルド検証を優先）。
- ビルド: `vcvarsall x64` ＋`cmake --build --config Release --target ...`（`-j3`。
  `-j8` は icx/cl の C1060 ヒープ不足を誘発）。
- 検証: `IRRuntimeContractTests` 5/5 PASS、`BuildErrorClassificationTests` 100/0 PASS、
  `AudioEngineHarness` 既定スイート PASS、`ConvoPeq.exe` リンク成功。
