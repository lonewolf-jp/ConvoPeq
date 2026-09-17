# WORK102-PREV-01 — Contract Freeze / Implementation Gate（凍結文書）

- **作成日**: 2026-09-17
- **種別**: Contract Freeze（Pre-Audit の CONTRACT-READY 判定を受けた実装前凍結）
- **本段階の実装**: **0**（production source / tests / CMake / UI / commit / push すべて 0）
- **正本**: 本文書（Pre-Audit の調査結果を基に、実装で変更すべき箇所・例外境界・テスト観点・禁止範囲・完了条件を固定する）

## 0. Baseline

```text
HEAD        = 859718e47e6465f668e711e14532950f1b068d23   (859718e4 fix(convolver): bound IR load allocation and stream hashing)
origin/main = 同一（ahead/behind = 0/0）
WORK102     = CLOSED（859718e4 で commit/push 済み）

Source（authoritative）:
  ConvoPeq.md  Generated = 2026-09-17 09:13:37
  --check      NEWER_SRC_COUNT = 0 / STATUS = FRESH / CHECK_EXIT = 0

git status --short = WORK102 commit 後の unrelated items のみ
  （staged: .gitignore / build.bat / doc/work68/*、unstaged: README.md + evidence/*、
    untracked: doc/work101, Testing/, IDEA.md 等）
```

## 1. 引き継ぐ凍結済み契約（WORK102 CLOSED・変更禁止）

```text
kMaxIRLoadBytes             = 1073741824   // 1 GiB（FC-FORM-1）
kMaxIRLoadChannels          = 8u           // FC-FORM-2
kMaxFileLengthSamples       = 2147483647   // INT32_MAX（FC-FORM-3、既存 MAX_FILE_LENGTH 同値）
kMaxIRResampleOutputSamples = 2097153      // 2^21 + 1（FC-FORM-5）
FC-FORM-5 = trim 後の長さに対してのみ評価（R2-D4）
FC-8      = computeIRHash は streaming XXH64・O(1) 補助メモリ
FC-INV-9  = main loader の適用順序: FC-FORM-4 → 3 → 2 → 1 → [確保開始] → FC-FORM-6(hash) → … → FC-FORM-5(trim 後)
IRLoadAdmission.h は本 work で一切変更しない（値の追加・predicate の変更・文言変更のいずれも禁止）。
```

## 2. Preview 責務の意図仕様凍結（非 parity 項目を含む）

Preview は main loader の簡易版ではなく **read-only 解析器** である。WORK102 本体の契約を再変更せず、
以下を PREV-01 固有の意図仕様として凍結する。

| 項目 | 凍結内容 |
| --- | --- |
| hash | preview は **hash を計算しない**（FC-8 の対象外。`IRLoadPreview` に hash field は存在しない — ConvolverProcessor.h:105-115） |
| cancellation | **supersession-only**。途中打ち切り API は導入しない。`requestId` staleness guard（ConvolverControlPanel.cpp:1173-1174）で古い結果を棄却し、in-flight job は run-to-completion。admission により in-flight 1 job の最悪コスト（メモリ・時間）が有界化されることで許容する |
| 並列度 | `g_irPreviewThreadPool(1)`（ControlPanel.cpp:16）の逐次実行を維持。API 変更しない |
| engine publish | preview はエンジン IR を一切 publish しない。成功時の効果は length-sec 決定（`applyAutoDetectedIRLength`）＋ `requestConvolverPreset`（→ main loader = WORK102 authority 経由）のみ |
| completion owner | `finishAsyncIRLoadPreview`（callAsync :1150-1156 ＋ callAsync 失敗時 MessageManagerLock 直呼出し :1158-1163）。**worker が必ず `IRLoadPreview` を値として生成して渡すこと**を契約の中心にする |

## 3. Implementation Gate で変更してよい変更面（すべて）

```text
1. src/convolver/ConvolverProcessor.ResampleAndFallback.cpp
     loadImpulseResponsePreviewFile（:271-331）本体のみ。
     同 TU 内の resampleIR / applyAsymmetricTukey / estimateEffectiveIRLengthSamples /
     convertToMinimumPhase 本体は変更しない。
2. src/convolver/ConvolverProcessor.StateAndUI.cpp
     analyzeImpulseResponseFile（:455-544）関数のみ（no-throw 境界 + FC-FORM-5 挿入）。
     同 TU の他関数（main loader の applyNewState 等）は一切変更しない。
3. src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp
     runIRLoadPreviewAdmissionTests() を追加（実ファイル生成 runtime test）。
4. src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp
     宣言 + 呼出しのみ（WORK102 と同一パターン。新規 CTest target は作らない）。
5. ConvoPeq.md（再生成）／doc/work102（証跡文書）。
```

**scope note**: PREV-01 の対象は「ResampleAndFallback.cpp + ConvolverControlPanel.cpp」と宣言されていたが、
no-throw 境界と FC-FORM-5 の挿入点は `analyzeImpulseResponseFile`（StateAndUI.cpp:455）にある。
変更面は同関数 1 個に限定して拡張する（関数単位で明示し、TU 単位では拡大しない）。
`ConvolverControlPanel.cpp` は**変更しない**（worker/completion 構造・requestId 機構は現状維持）。

## 4. 凍結契約（5項目）

### FC-1（P0）Admission ordering — narrowing 前に FC-FORM-4 → 3 → 2 → 1 を適用

**変更箇所**: `loadImpulseResponsePreviewFile`（ResampleAndFallback.cpp:271-331）。
現状（defect 実測）: `:292` で `static_cast<int>(reader->numChannels)` が **admission より前に narrowing**、
`:293-299` に独自 constexpr `maxFileLength=2147483647`、`:301-305` に独自 `numChannels <= 0`、
byte bound なし、channel ≤8 なし。

**凍結する変更**（main loader `doLoadIRStep`（LoaderThread.cpp:388-431）と同一形状・同一述語）:

```text
reader 生成
 ↓
fileLength = reader->lengthInSamples        (int64)
rawChannels = reader->numChannels           (unsigned・narrowing 前)
 ↓
FC-FORM-4  admitChannelCountNonZero / admitFileLengthNonZero → diagnostic*
 ↓
FC-FORM-3  admitFileLengthRepresentable                     → diagnosticLengthLimit
 ↓
FC-FORM-2  admitChannelCount                                → diagnosticChannelLimit
 ↓
FC-FORM-1  admitByteBudget(rawChannels, fileLength)         → diagnosticByteLimit
 ↓
const int numChannels = static_cast<int>(rawChannels)       （narrowing は FC-FORM-2/4 の後）
 ↓
loadedIR.setSize(numChannels, (int)fileLength)              （確保は全 admission の後）
 ↓
チャンク読込（FC-2）
```

- 適用順序は FC-INV-9 と同一。失敗時は `errorMessage` を設定して `false` を返す（既存契約維持）。
- 独自 constexpr（R&F:293）は削除し、IRLoadAdmission.h の predicate に一本化する。
- `>8ch` は main loader と同一の決定論的拒否になる（互換契約: 1–3ch 既存挙動 / 4–8ch 許容 / >8ch 拒否）。

**例外境界**: 本関数自身は no-throw 契約としない（真の OOM は `loadedIR.setSize` から bad_alloc として
伝播しうる）。例外は FC-3 の no-throw 境界（`analyzeImpulseResponseFile`）で捕捉する。
チャンク確保（256 KB）は null 検査付きの既存形（R&F:315-319 / LoaderThread.cpp:445-449 と同じ）を維持。

**テスト観点**: 9ch 実ファイル（小長・例: 512 samples）→ 確保前に決定論的拒否（errorMessage =
`diagnosticChannelLimit(9)`）。8ch / 2ch / 1ch 実ファイル → accept。0-length / INT32_MAX+1 の発火は
実ファイル生成が非決定的なため **構造検証 S1/S2**（§5）で確認（executable PASS と誤表示しない）。
1 GiB 境界は既存 E/F predicate テスト（IRLoadAdmissionTests）が被覆。

**禁止範囲**: IRLoadAdmission.h 変更 / 4定数変更 / FC-INV-9 順序変更 / loader 以外への `convo::irload::`
横展開（S4 で確認） / `existsAsFile`・reader 生成・read 失敗時の文言変更（:276-311 は文面維持）。

**完了条件**: `loadImpulseResponsePreviewFile` 内に FC-FORM-1 より前の全量確保・narrowing が 0 件。
S1/S2 検査が述べる順序が実測行番号と一致。

### FC-2（P0）全量 `tempFloatBuffer` の廃止 → bounded / chunked read

**凍結する変更**: `tempFloatBuffer(numChannels, fileLength)`（R&F:307）と全量 `tempAlignedBuffer`（R&F:314）を
廃止し、main loader と同じ形状へ:

```text
constexpr int64 kStreamChunk = 256 * 1024;   ← LoaderThread.cpp:442 と同値の局所 constexpr
                                                （IRLoadAdmission.h への追加はしない — 変更禁止のため）
AudioBuffer<float> tempFloatBuffer(numChannels, (int)kStreamChunk);
auto tempAligned = makeAlignedArray<double>((size_t)kStreamChunk);   // null 検査 → OOM 時 graceful
loadedIR.setSize(numChannels, (int)fileLength);                      // FC-1 の admission 後
for (int64 offset = 0; offset < fileLength; offset += kStreamChunk) {
    chunk = min(kStreamChunk, fileLength - offset)
    reader->read(&tempFloatBuffer, 0, (int)chunk, offset, true, true)
    ch 毎: convertFloatToDoubleHighQuality → loadedIR.copyFrom(ch, (int)offset, tempAligned, (int)chunk)
}
```

- `jassert(offset + chunk <= 2147483647)` の belt-and-braces も main loader（LoaderThread.cpp:464）と同一で入れる。
- preview ピーク確保は `loadedIR`（ch×len×8 ≤ 1 GiB）＋ 256 KB×2 に帰着する。

**例外境界**: チャンク確保は null 検査で graceful。`loadedIR.setSize` / `copyFrom` の bad_alloc は
FC-3 境界で捕捉。conversion ループに新規 cancellation を導入**しない**（supersession-only §2）。

**テスト観点**: 小実ファイル（例: 44.1 kHz・1ch/2ch・数千 sample）で preview 出力を検証し、
チャンク化後もバッファ内容が読込と等価であることを回帰確認（trim/resample/estimate 下流が変わらないこと）。

**禁止範囲**: `kStreamChunk` を IRLoadAdmission.h へ移動 / LoaderThread.cpp 側の変更 / chunk 値の変更（262,144 固定）。

**完了条件**: `tempFloatBuffer(numChannels, fileLength)` 形状が ResampleAndFallback.cpp から消滅（S1）。

### FC-3（P0）no-throw failure boundary

**凍結する変更**: `analyzeImpulseResponseFile`（StateAndUI.cpp:455-544）の関数本体全体を no-throw 境界にする。
内部処理（loadImpulseResponsePreviewFile / trim / resampleIR / DCBlocker / Tukey / estimate）が投げうる
例外をすべて `IRLoadPreview.errorMessage` に変換して **必ず `preview` を返す**:

```text
catch (const std::bad_alloc&)  → errorMessage = "IR too large (Out of Memory)"    （LoaderThread.cpp:131 と同文言）
catch (const std::exception& e)→ errorMessage = "Error analyzing IR: " + e.what()
catch (...)                    → errorMessage = "Unknown error analyzing IR"
```

- `analyzeImpulseResponseFile` を唯一の例外境界とし、worker lambda（ControlPanel.cpp:1146-1164）は**変更しない**。
  （JUCE ThreadPool::runNextJob の catch(...) は例外を握り潰して completion を喪失させる（juce_ThreadPool.cpp:387-394）ため、
  境界を worker 側に置かないことが本契約の要点。）
- 完了性: どの failure 経路でも worker は `IRLoadPreview` を返す → callAsync（またはフォールバック）→
  `finishAsyncIRLoadPreview` → `setIRPreviewInProgress(false)` ＋ MessageBox。**completion 喪失経路が消滅する**。

**テスト観点**: 失敗注入（存在しないファイル・対応外形式）で success=false ＋ 非空 errorMessage を確認。
bad_alloc 等の例外注入は決定論的に不可のため **構造検証 S3**（§5）で確認。

**禁止範囲**: `finishAsyncIRLoadPreview` / `startAsyncIRLoadPreview` / `irPreviewRequestId` 機構 / MessageBox 配線の変更。
`analyzeImpulseResponseFile` のシグネチャ・`[[nodiscard]]` 変更。

**完了条件**: `analyzeImpulseResponseFile` が如何なる入力・メモリ条件下でも例外を caller（worker）へ
漏らさない。S3 検査が catch 群の存在と位置を実測。

### FC-4（P1）FC-FORM-5（trim 後 resample bound）

**凍結する変更**: `analyzeImpulseResponseFile` 内、trim 完了後・resample 実行前の **StateAndUI.cpp:496 相当**
（現 `resampleIR` 呼出し :497-499 の直前）に挿入:

```text
trimmedLength = loadedIR.getNumSamples()          (trim :468-495 の後)
if (!convo::irload::admitResampleOutput(trimmedLength, loadedSampleRate, processingSampleRate))
    → preview.errorMessage = convo::irload::diagnosticResampleLimit(...); return preview;
```

- raw fileLength では評価しない（R2-D4 凍結。trim が load 後に行われる preview の構造上、
  これは main loader の FC-FORM-5 と同一意味論になる）。
- `resampleIR` 本体（r8b 構築 R&F:54 / `getMaxOutLen` R&F:57）は無変更。admission 済み input を受け取る
  「呼出側 admission → resampleIR は既に bound 済み」の境界は main loader（LoaderThread.cpp:553→572）と同型。
- `fileSampleRate / processingSampleRate ≤ 0` の vacuous pass は predicate 実装に従う（IRLoadAdmission.h:99-104）。

**テスト観点**: trim 後に限界超過となる実ファイル → FC-FORM-5 reject で `diagnosticResampleLimit` 文言が
MessageBox に到達すること（WORK102 の H と同一観点の preview 版）。accept 側（G 相当: raw > hardMax だが
trim 後 accept）も preview で再確認。

**禁止範囲**: `resampleIR` 本体変更 / `admitResampleOutput` の trim 前（raw）適用への変更 / 2097153 値の変更。

**完了条件**: FC-FORM-5 の評価が trim 後・resampler 構築前に行番号で確認できる（S2 に含める）。

### FC-5（P1）diagnostic の single source 統一

**凍結する変更**: preview 独自の英文文字列を `convo::irload::diagnostic*` に統一:
- "IR file is too large (exceeds 2GB samples limit)."（R&F:297）→ `diagnosticLengthLimit(fileLength)`
- "Invalid channel count in IR file."（R&F:303）→ `diagnosticChannelLimit(rawChannels)`（0ch は同文言 — IRLoadAdmission.h:139-140 と一致済み）
- channel/byte/length/resample の各 rejection は対応する diagnostic 関数に一本化（FC-INV-8）。
- `file.existsAsFile()` 失敗（:278）・reader 生成失敗（:287）・read 失敗（:310）の文言は**維持**（ファイル形式系の
  文言は irload diagnostic の管轄外であり、現行文字列を凍結する）。

**テスト観点**: 拒否時に返る errorMessage が `convo::irload::diagnostic*` の出力と一致すること（9ch・length・resample）。

**禁止範囲**: IRLoadAdmission.h の文言変更（diagnostic 関数の改変） / 非拒否系文言（not found / unsupported /
read 失敗）の置換 / UI 側 MessageBox の文言ハードコード追加。

**完了条件**: `loadImpulseResponsePreviewFile` の admission rejection 3 経路（length/channel/byte）と
FC-FORM-5 経路がすべて diagnostic 関数を出力する。

## 5. 構造検証（PREV-01 版 J/K 相当・executable PASS と誤表示しない）

実行可能テストに決定論的に発火させられない項目は、WORK102 と同じく **ソース差分スコープの構造検証**
として報告し、PASS と偽装しない:

```text
S1  全量 tempFloatBuffer 形状の消滅      : `tempFloatBuffer(numChannels, fileLength)` および全量
    makeAlignedArray(fileLength) が ResampleAndFallback.cpp に 0 件。チャンク形は常駐 256 KB×2。
S2  admission ordering の実測            : loadImpulseResponsePreviewFile 内の行番号で
    FC-FORM-4 → 3 → 2 → 1 → narrowing(int化) → loadedIR.setSize の順を確認。narrowing が FC-FORM-2 前に
    現れないこと。loadedIR.setSize が FC-FORM-1 前に現れないこと。
S3  no-throw 境界の存在                  : analyzeImpulseResponseFile に bad_alloc / std::exception /
    catch(...) の捕捉群が存在し、worker lambda が無変更であること。
S4  横展開なし                           : `convo::irload::` の出現が loadImpulseResponsePreviewFile と
    analyzeImpulseResponseFile の 2 箇所に限定され、他 TU への波及が 0 件であること。
S5  IRLoadAdmission.h 無変更             : git diff -- src/convolver/IRLoadAdmission.h = 0。
```

## 6. 完了条件（Implementation Gate 以降の全工程で検証する基準）

```text
Build      : Release / Debug 全文ビルド BUILD_EXIT=0。変更 TU からのコンパイラ警告 0。
CTest      : Release 40/40・Debug 40/40（新規 CTest target を作らない・harness 内に統合）。
Runtime    : runIRLoadPreviewAdmissionTests
             ・9ch 実ファイル → 決定論的 reject（diagnosticChannelLimit 文言）PASS
             ・8ch / 2ch / 1ch 実ファイル → accept ＋ チャンク読込等価回帰 PASS
             ・FC-FORM-5: trim 後 reject（H 相当）/ trim 後 accept（G 相当）PASS
Structural : S1–S5 を「構造検証」として明示報告（executable PASS と誤表示しない）
Static     : clang-tidy 変更 TU 0 error / 0 warning。cppcheck 変更コードに新規指摘 0
             （pre-existing diagnostics は PREV-01 defect として再分類しない）。
ConvoPeq   : 再生成 → NEWER_SRC_COUNT = 0 / FRESH。
Arch       : atomic 追加 0 / RT process 変更 0 / Publish・Crossfade・Retire・Epoch・RuntimeWorld・
             LifetimeBudget・ISRShutdown 変更 0 / MAX_IR_LATENCY 変更 0 / DELAY_BUFFER_SIZE 変更 0 /
             UI 制御構造（startAsyncIRLoadPreview・finishAsyncIRLoadPreview・requestId）変更 0 /
             RT allocation・lock・wait・decision 追加 0。
Parity     : §Pre-Audit の matrix 全行が「Main loader と同等 or 意図仕様として凍結済み」に消化。
Boundary   : commit boundary = 変更面 §3 の 5種（ResampleAndFallback.cpp / StateAndUI.cpp /
             IRLoadAdmissionTests.cpp / PublishPipelineIntegrationTests.cpp / ConvoPeq.md）＋ 証跡文書。
             WORK102 既存 staged items（.gitignore / build.bat / doc/work68/*）・README.md・evidence/* の
             既存差分は混入させない。
```

## 7. 全工程共通の禁止範囲（Scope guard）

```text
ConvolverProcessor.Runtime.cpp / Lifecycle.cpp / MKLNonUniformConvolver.* / RuntimeWorld* /
Coordinator* / Publish・Crossfade・Retire・Epoch 経路 / LifetimeBudget* / ISRShutdown* /
MAX_IR_LATENCY / DELAY_BUFFER_SIZE : 変更禁止
IRLoadAdmission.h                  : 変更禁止（predicate 借用のみ）
LoaderThread.cpp / AllpassDesigner.cpp / IRLoadAdmissionTests.cpp の WORK102 部分 : 変更禁止
IS-7（trim copy-on-resize optimization）と simultaneous residency = 2 GiB の意味論 : 本 work に混ぜない
```

## 8. 以降の工程順

```text
Contract Freeze（本文書）
      ↓
Implementation Gate（本凍結に対する実装可否確認）
      ↓
Implementation（§3 の変更面のみ）
      ↓
Post Audit → Final Gate → commit/push（WORK102 と同一規律）
```
