# 2026-11-01 追加判断への対応報告（A / B / D-01 / D-02 / CI ゲート）

- 基準: `main` = `origin/main` = **`fd92c6e7`**
- 作業拠点: `C:\VSC_Project\ConvoPeq`（main-develop セッションのみ。他 worktree は読み取り専用）
- 状態: **A / B / D-01 / D-02 の調査・測定は完了**（B は原因特定まで。修正は未承認のため未実施）
- 製品コード: **未変更**（測定ドライバは **revert 済み・追跡差分 0**。未追跡 24 件は保全）

## 1. Owner 判断への対応状況

| # | Owner 判断 | 対応 | 状態 |
|---|---|---|---|
| 1 | D-01 製品 2 行は HOLD（影響調査のみ） | 最新ソースで影響調査を実施（§2） | **完了**（製品コード未変更） |
| 2 | D-02 測定を承認 | 条件を事前固定し、比較① と比較② を実行（§5） | **完了**（全 12 実行が有効） |
| 3 | A は調査承認・修正 HOLD | 競合条件の特定と修正案 3 種の比較（§3） | **完了**（修正なし） |
| 4 | B は調査承認 | 発生箇所の特定、再現、**原因の特定**（§4） | **完了** |
| 5 | 報告書 2 件の Git 追跡 | PR #102（doc-only、2 ファイルのみ）→ `389e8159` | **完了** |
| 6 | ローカル CI/CD の記録（CodeQL は未検証） | §6 に区別して記録 | **完了** |
| 7 | ASan 51/51 を SEGFAULT 解消と同一視しない | §4・§6 で明示 | **完了** |
| 8 | U5 = FAIL・既定 OFF、U3/U4 = HOLD、+50 µs と p99 基準は不変 | 変更なし | **維持** |
| 9 | 最新 ConvoPeq.md の再生成（生成元 commit・SHA-256） | §7 に記録 | **完了** |
| 10 | 作業拠点・セッションの限定 | 遵守 | **遵守** |

## 2. D-01 製品 2 行（HOLD 継続・影響調査）

詳細: `doc/residual-20261101/D01-IMPACT-ANALYSIS-LATEST-20261101.md`

### 2.1 調査結果

**最新ソースでは、当該フックは既に存在し、テストバイナリでは有効である。**
「製品 2 行」を適用してもテスト側の挙動は変わらず、変わるのは製品バイナリだけである。

| # | 事実 | 証拠 |
|---|---|---|
| 1 | `evaluate()` の fading 判定にフックが**既にある** | `src/audioengine/PublicationAdmission.cpp:51-56` |
| 2 | その形（`#if defined(CONVOPEQ_UNIT_TESTS)` ガード付き）は **2026-08-07 の `f0d6406f`** で導入 | `git blame -L 50,58` |
| 3 | `AudioEngineHarness` は `CONVOPEQ_UNIT_TESTS=1` でコンパイルされる | `CMakeLists.txt:2278-2280`、`AudioEngineHarness.vcxproj` の `PreprocessorDefinitions`（8 箇所一致） |
| 4 | `PublicationAdmission.cpp` はハーネスに**直接コンパイルされる** | `build\AudioEngineHarness.dir\Release\PublicationAdmission.obj` |
| 5 | friend とメンバは **ODR 回避のため無条件**、テスト限定なのは評価分岐だけ | `AudioEngine.h:3984-3988` |
| 6 | テストはフックを多用（32 箇所） | `STG8RecoveryObligationTests.cpp` |
| 7 | 製品ターゲットにテストは含まれない | `CONVOPEQ_ALL_SOURCES` に `src/tests` は **0 件** |
| 8 | フラグの書き手はテスト専用クラスのみ | `DeferredPublicationTestAccess.h:34` |

### 2.2 旧分析との差異（訂正）

`doc/residual-20261101/D01-STG8-ANALYSIS-20261101.md` §2 RC-1 は
「getter の呼び出し元は 0」「その分岐は現存しない」と結論していたが、
`git show a12e05f2:src/audioengine/PublicationAdmission.cpp` でも当該フックは存在しており、
**この結論は最新ソースと一致しない**。したがって「製品 2 行で STG-8-D2 の retention 検証が
実質化する」という前提も成立しない。

### 2.3 推奨（Owner 判断用）

1. **「製品 2 行」は適用しない**。目的（STG-8-D2 の retention 検証の実質化）は現行コードで達成済み。
2. 旧分析 RC-1 の訂正を記録する（RC-2 の単一書き込み者契約の指摘は独立であり、影響を受けない）。
3. STG-8-D2 に残る断続失敗があるなら、原因は「フックの欠落」ではなく**タイミング依存**として扱い、
   B の調査と共通化する。

## 3. A. Debug assert 違反（調査完了・修正 HOLD）

詳細: `doc/residual-20261101/A-ASSERT-RACE-ANALYSIS-20261101.md`

### 3.1 原因（競合の型）

`assert(prevState == DSPState::Quarantined)`（`ISRDSPHandle.cpp:529`）は **CAS（同 565 行）より前**にあり、
状態遷移を排他しているのは CAS である。観測と遷移の間に他スレッドが遷移させると、
**遷移は正しく 1 回に保たれたまま assert だけが発火**する。

```
スレッド X (commit 経路)                スレッド Y (reclaim tick)
--------------------------------------  ------------------------------------
destroyQuarantineSlot(slot, 0)
  state を読む → Quarantined
                                          retryDeferredQuarantineDestroy()
                                            → destroyQuarantineSlot(slot, 0)
                                            CAS 成功 → Reclaimed
  assert(prevState == Quarantined)  → **発火（Debug）**
  （Release は false を返して安全に終了）
```

**実害**: 二重解放は起きない（CAS が単独で遷移を成立させ、フリーリストへの push は
`freeListMutex_` 下で 1 回だけ）。Release は安全、Debug のみ abort する。

### 3.2 スレッド証拠

| 系統 | 実行スレッドの根拠 | 証拠 |
|---|---|---|
| reclaim | Timer コールバック＝メッセージスレッド | `AudioEngine.Timer.cpp:2018/2034`、`ConvolverProcessor.Lifecycle.cpp:226` |
| reclaim（テスト） | **テストスレッド** | `AudioEngine.h:5621-5625`、`V11EngineRetireProtectionTests.cpp:1443`（`std::thread drainer`） |
| commit | `commitRuntimePublication()` の呼び出し元に依存 | `AudioEngine.h:3884-3893` → `PrepareToPlay.cpp:179/309`、`ReleaseResources.cpp:223`、`Timer.cpp:1167` |

### 3.3 修正案の比較（実装は未承認）

| 案 | 内容 | 変更範囲 | 効果 | リスク |
|---|---|---|---|---|
| **A1** | 呼び出し側で `getSlotState()` を再確認してから破棄 | `Commit.cpp` / `ReleaseResources.cpp` の 2 箇所 | 限定的（TOCTOU の窓が残る） | 低 |
| **A2** | **CAS の結果に基づく検査へ移す**（assert は「CAS 成功時」または「CAS 失敗かつ状態が Quarantined のまま」に限定） | `ISRDSPHandle.cpp` の 1 関数（数行） | 競合の正常系を吸収し、真の異常は検出し続ける | 低 |
| **A3-a** | tick の対象を「pin gate で deferred 計上したスロット」に限定（グローバルカウンタ → スロット単位のマーク） | 小〜中 | 競合が構造的に消える。3 ストアの順序は不変 | 低〜中 |
| **A3-b** | tick が 3 ストアすべてを所有し、commit は要求のみ | 大 | 単一書き手が完全に成立 | 高（破棄が 1 tick 遅延、grace 条件の再評価が必要） |

**推奨**: **A2 を先行**し、併せて **A3-a** を設計評価する。**A1 単独は採用しない**（TOCTOU が残る）。
**assert の削除は提案していない**（A2 は検査の位置を遷移の所有者側へ移す案）。

## 4. B. 間欠 SEGFAULT（調査完了・原因特定）

詳細: `doc/residual-20261101/B-SEGFAULT-INVESTIGATION-20261101.md`

### 4.1 発生記録（実測）

| # | 日時 | 構成 | 結果 |
|---|---|---|---|
| 1 | 2026-11-01 14:03 | `build-diag` **Release**（`ENABLE_ASAN=OFF`） | PASS 298.30 s |
| 2 | 2026-11-01 14:03 | 同上 | **`***Exception: SegFault 40.00 sec`** |
| 3 | 2026-11-01 17:14 | `build-asan-local` RelWithDebInfo | PASS 359.9 s |
| 4 | 2026-11-01 17:20 | 同上 | PASS 360.5 s |

**重要**: 観測されたクラッシュは **ASan ビルドではない**。したがって
「ASan 51/51 PASS」は本件の解消を意味しない（Owner 指摘のとおり）。

### 4.2 発生箇所

クラッシュログ `tmp/f2_ctest2.log`（90,775 B）の最終出力は
`[L0_WRITE] slot=11 peak=0.500000 bin=19`（`MKLNonUniformConvolver.cpp:1068-1087` の L0 write 診断）で、
その直前まで `[I2T] T-I2-3`（`testRegisteredDSPFailurePreservesRegistration`）内の
**rebuild → IR ロード**（`[IR_RATE_GEN]` / `[IR_CHAIN]` / `[IR_LOAD]` / `[NUC_ALLOC]`）が続いていた。

| 項目 | 値 |
|---|---|
| 最後に完了したテスト | `[I2T] T-I2-3: gotWorld=1` |
| 実行中だった処理 | rebuild → IR ロード（`irLen=48000`、`Layers=2`、`L0Part=512`、`ringSize=4096`） |
| 最後の出力元 | `src/MKLNonUniformConvolver.cpp:1068-1087` |
| クラッシュ時刻 | 40.00 秒（PASS 時は 298.30 s） |

### 4.2.1 再現（2026-11-01 19:00・試行 5）

`build-diag` Release（**14:03 のクラッシュと同一バイナリ**、sha256 `0BE8F1B7…D13E9`）で
**同一シグネチャのクラッシュを再現**した。

| 項目 | 14:03 の観測 | 19:00 の再現 |
|---|---|---|
| 結果 | `***Exception: SegFault` | **`0xC0000005`（アクセス違反）** |
| 経過 | 40.00 秒 | **36.4 秒** |
| 最後の出力 | `[L0_WRITE] slot=11 peak=0.500000 bin=19` | **同一行**（`[L0_WRITE]` が 2 ブロック連続した直後） |
| 実行中だった処理 | `[I2T] T-I2-3` の rebuild → IR ロード | 同一（`irLen=48000`、`numIR=12`、`part=512`、`fft=1024`） |

**静的確認**: `[L0_WRITE]` の読み出しは `p < numPartsIR` で `irFreqReal + p*complexSize`、
確保は `numParts * complexSize`（`numParts = nextPowerOfTwo(numPartsIR) >= numPartsIR`）であり
（`MKLNonUniformConvolver.cpp:886-897` / `:1074-1091`）、**診断ループ自体は境界内**である。
クラッシュはその直後（`:1094` 以降）の可能性が高く、確定にはダンプを要する。

### 4.2.2 頻度（12 試行）

| 指標 | 値 |
|---|---|
| 試行数 | 12（観測窓 150 秒） |
| クラッシュ | **1**（試行 5） |
| クラッシュ率 | **8.3%** |
| クラッシュ時の経過 | 36.4 秒 |
| クラッシュなしの試行 | 11（すべて窓 150 秒を完走、最後のタグは `STG8-TRACE`） |

14:03 の観測（1/2）と合わせると、**IR ロードのタイミングに依存する間欠事象**と整理できる。
本バッチはハーネス・ビルドの並行プロセス無しで実施しており、14:03 の観測（他の作業と並行）とは
**負荷条件が同一ではない**。この差が頻度に影響している可能性は排除できない。

### 4.3 スタックトレースの取得（成功）

`build-diag\Release` はデバッグディレクトリが POGO のみで PDB が無いため、**`/DEBUG` 付きで再リンク**して
PDB を生成した（EXE 59,196,928 B / PDB 180,998,144 B、**ソースは不変**）。
そのうえで `procdump -ma -e -x` により未処理例外時のフルダンプを取得した（バッチ 2 の試行 5、40.1 秒）。

| 項目 | 値 |
|---|---|
| ダンプ | `tmp/b_dumps/AudioEngineHarness.exe_261101_195308.dmp`（**358.4 MB**、SHA-256 `03C521D7…C3C413`） |
| 例外コード | **`EXCEPTION_ACCESS_VIOLATION`** |
| 例外情報 | `[1, 0x1EC041AC800]` → **書き込み**、書き込み先 `0x1EC041AC800` |
| 例外アドレス | `0x00007FF7A9F5E21A`（RVA **`0x253E21A`**） |
| **関数** | **`ConvolverProcessor::process(juce::dsp::AudioBlock<double>&)`** |
| **ソース位置** | **`src/convolver/ConvolverProcessor.Runtime.cpp:774`** |
| 該当行 | `if (convDiagBuffer->push(event))` |
| 経過 | **40.1 秒**（14:03 の観測 40.00 秒と一致） |

シンボル化には `llvm-symbolizer --relative-address --obj=<exe> 0x253E21A` が必要だった
（`--relative-address` 無しでは `??:0:0` になる）。

### 4.4 原因（特定）

**診断用リングバッファへのポインタが、エンジンの寿命を追えていない。**

```cpp
// ConvolverProcessor.Runtime.cpp:91-104
LockFreeRingBuffer<DiagEvent, ...>* convDiagBuffer = nullptr;   // プロセス全体のグローバル
void setConvDiagBuffer(/*...*/) { convDiagBuffer = &db; ... }   // エンジンのメンバのアドレスを保存
```

```cpp
// AudioEngine.Init.cpp:116-119（AudioEngine の初期化時に毎回）
setEqDiagBuffer(diagBuffer, rtAuxMutable_.diagTickPushed, ...);
setConvDiagBuffer(diagBuffer, rtAuxMutable_.diagTickPushed, ...);
```

`diagBuffer` は **`AudioEngine` のメンバ**（`AudioEngine.h:3030`）で、エンジンとともに破棄される。
しかしそのアドレスは**プロセス全体のグローバル**に保存され、**エンジン破棄時にリセットされない**。

```
エンジン A 生成 → グローバル = &A.diagBuffer
エンジン A 破棄 → A.diagBuffer も破棄（グローバルは A を指したまま）
エンジン B 生成 → 途中までグローバルは A を指す
   ↑ この窓で RT スレッドが process() を実行 → 解放済みバッファへ push → アクセス違反
```

ハーネスは `AudioEngineHarness` を **114 回**構築しており、この窓に落ちたときにだけクラッシュする
（= 間欠的）。`/DEBUG` 再リンクで配置・タイミングが変わると発生率が変わることも整合する。

### 4.5 なぜ ASan 51/51 PASS が無関係か

| ビルドツリー | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` |
|---|---|
| `build-diag`（クラッシュする） | **ON** |
| `build-asan-local`（51/51 PASS） | **OFF** |
| `build`（Release） | **OFF** |

クラッシュ箇所は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の内側にある。
**ASan 構成ではこのコードがコンパイルされていない**ため、ASan が 51/51 PASS でも本件の検証にならない。
これは Owner の指摘「ASan 51/51 PASS と SEGFAULT 解消は同義ではない」の**具体的な根拠**である。

### 4.6 影響範囲

| 構成 | 影響 |
|---|---|
| 診断 OFF（製品既定・ASan・Release） | **コードが存在しないため無影響** |
| 診断 ON（`build-diag` など） | **間欠クラッシュの可能性** |
| 同型の `eqDiagBuffer`（`DSPCoreFloat.cpp:31/43`） | **同じ欠陥クラス**（未観測だが同様に起こり得る） |

### 4.7 修正案（**未承認・提案のみ**）

| 案 | 内容 | リスク | 評価 |
|---|---|---|---|
| B-1 | エンジン破棄時に `nullptr` でクリアし、ポインタを `std::atomic` 化 | 小 | 最小修正。TOCTOU は残る |
| **B-2** | グローバルを廃止し、`ConvolverProcessor` インスタンスが自分のバッファを持つ | 中 | **競合が構造的に消える**（推奨） |
| B-3 | 世代トークンで所有者を検証 | 中 | 安全だが複雑 |

**製品コードの変更は行っていない**（Owner 承認後に実施する）。

### 4.8 禁止事項の遵守

テストの削除・実行回数の削減・タイムアウトの緩和は**行っていない**。
再現試行は「観測窓」（1 試行あたり 150 秒で観測を打ち切る）を用いるが、これは
**テストのタイムアウトを変更するものではなく**、各試行の成否は「窓内でクラッシュしたか」として記録する。

## 5. D-02 測定（完了）

詳細: `doc/residual-20261101/D02-MEASUREMENT-RESULTS-20261101.md`（事前固定: `D02-MEASUREMENT-PROTOCOL-20261101.md`）

### 5.1 実施と条件の遵守

2026-11-01 17:32:52 〜 18:45:57 に **12 実行**（比較① 6、比較② 6）。**全 12 実行が有効**で、
無効条件（`verdict != OK`、`dropped/overflow != 0`、`n < 30,000`、ウォームアップのずれ、環境証拠の欠落）
には該当しなかった。電源プランは測定後にバランスへ復帰済み。

| 条件 | 実測 |
|---|---|
| mode 2（TimeOnly） | 全 12 実行 `mode=2` |
| 34,000 ブロック（ウォームアップ 2,000） | `n=32,000〜32,004`、`warmup_skipped=2000` |
| 93.75 blocks/s ペーシング | 適用（1 実行 ≒ 6 分 5 秒） |
| `HIGH_PRIORITY_CLASS` / 高パフォーマンス電源 | 全実行で設定・環境証拠で一致 |
| 交互 3 組 | 比較① A→B→A→B→A→B、比較② 0xC→0x4 交互 |

### 5.2 比較① 非干渉性 A/B（D-02 計装の有無）

| 組 | baseline p99 | current p99 | Δp99 |
|---|---|---|---|
| 1 | 1,263 µs | 1,258 µs | −5 |
| 2 | 1,261 µs | 1,231 µs | −30 |
| 3 | 1,268 µs | 1,282 µs | +14 |
| **中央値** | **1,263.0 µs** | **1,258.0 µs** | **−0.40%** |

**判定: PASS（基準「p99 差 1% 未満」）**。D-02 の 3 点計装は p99 に 1% 以上の影響を与えていない。

**注意**: アーム内のばらつき（current σ=25.5 µs、baseline σ=3.6 µs）が差（−5 µs）より大きい。
本結果は「許容幅 1% に収まった」という主張に留め、「計装が性能を改善した」とは解釈しない。

### 5.3 比較② 案2 E-1（mask 0xC vs 0x4）

| 組 | 0xC p99 | 0x4 p99 | Δp99 | 0xC max | 0x4 max | Δmax |
|---|---|---|---|---|---|---|
| 1 | 1,291 µs | 1,208 µs | −83 | 2,011 µs | 2,121 µs | +110 |
| 2 | 1,195 µs | 1,165 µs | −30 | 1,836 µs | 2,336 µs | +500 |
| 3 | 1,186 µs | 1,088 µs | −98 | 1,756 µs | 1,654 µs | −102 |
| **中央値** | 1,195.0 µs | 1,165.0 µs | **−83.0（σ=35.7）** | 1,836.0 µs | 2,121.0 µs | +110.0（σ=305.4） |

**記述統計のみ。合否判定は行わない（探索的比較）。**

- p99 は 3 組すべてで 0x4 が小さい（方向は一致）。
- ただし `max` は 3 組中 2 組で 0x4 が大きく、σ も 305.4 µs と大きい。**p99 の改善と max の悪化が併存**する。
- 組数 3 では検出力が無い。**この結果から affinity の既定値を変更しない。**

### 5.4 独立再パース

`verify_d02.py`（ランナーとは別実装）で生ログと環境証拠を再読し、比較①② の数値が**完全一致**することを
確認した（`problems: none`、exit 0）。バイナリ同一性も確認（比較① は 2 種、比較② は 1 種）。

### 5.5 この測定から言えないこと

- **U5 の合否**（承認範囲外。FAIL・既定 OFF を維持）
- 計装の push 側の費用（mode 2 では `t_push_p99=0`）
- affinity 既定値の妥当性（3 組・σ 大・max は逆方向）
- 製品の実時間性能（ハーネスの `audioLoop` による測定であり、実デバイスの callback ではない）


## 6. ローカル CI/CD の記録

`doc/work113/LOCAL-CI-GATE-20261101.md` の内容を維持する。**ローカルと GitHub Actions は別状態**として扱う。

| 検証 | 結果 | 区分 |
|---|---|---|
| 静的解析一式 | PASS | ローカルで実行 |
| 必須ビルド | PASS | ローカルで実行 |
| ASan 全ターゲットビルド | PASS | ローカルで実行 |
| ASan テスト 51/51 | PASS | ローカルで実行 |
| TSan | CI と同じ SKIP 分岐を確認 | ローカルで実行（実行ではなく分岐確認） |
| **CodeQL** | **未実行・非同等** | **GitHub Actions 依存の未検証項目** |

**ASan 51/51 PASS は、過去に確認された間欠 SEGFAULT の解消を意味しない**（§4.1）。
本報告では、両者を明確に区別して記録する。

## 7. ConvoPeq.md の再生成（指示 9）

| 項目 | 値 |
|---|---|
| 生成元コミット | **`fd92c6e7`**（`HEAD` = `origin/main`） |
| サイズ / blob | **6,436,359 B** / `dfe441b4035e325cbb39bfc9cf2ec6d59eb3f2b7` |
| SHA-256 | **`67cb8260a04eec6ac515eae9249da01c02bc0afd45cf15b98c40633eb146d58c`** |
| 作業ツリーとの一致 | **バイト一致** |
| 分割版 | 4 ファイル・合計 6,436,359 B・`identical=True`・全 2 MB 以下 |
| Dropbox | 9 ファイル上伝・**ALL MATCH** |

測定ドライバ（ローカル専用・非コミット）を revert した後、**素のソースから再生成して
同じ SHA-256 になること**を再確認する（先に共有した版にドライバが混入していないことの証明）。

### 7.1 revert 後の再確認（実測・2026-11-01 18:54）

| 項目 | 値 |
|---|---|
| `output_sourcecode_markdown.py --check` | `NEWER_SRC_COUNT: 0` / **`STATUS: FRESH`** / exit 0（155.7 秒） |
| blob | `dfe441b4035e325cbb39bfc9cf2ec6d59eb3f2b7` |
| サイズ | 6,436,359 B |
| SHA-256 | **`67cb8260a04eec6ac515eae9249da01c02bc0afd45cf15b98c40633eb146d58c`** |

ドライバ revert 後の素のソースに対して **FRESH** であり、SHA-256 は共有済みの値と**一致**した。
したがって、先に共有した `ConvoPeq.md` は**測定ドライバを含まない**（`fd92c6e7` のソースそのもの）。

## 8. 遵守事項の確認

| 項目 | 状態 |
|---|---|
| 未承認の製品コード変更 | **なし**（測定ドライバは **revert 済み・追跡差分 0**。`src/` の変更は残っていない） |
| 既存 HOLD / NO-GO の解除 | **なし**（D-01 HOLD、A 修正 HOLD、D-03 HOLD、D-04 NO-GO、U3/U4 HOLD） |
| U5 | **FAIL・既定 OFF を維持**（許容値 +50 µs と p99 基準は不変） |
| 外部公開 | **なし**（Dropbox への共有のみ。Owner 指示の範囲） |
| 他 worktree | **読み取り専用**（ベースライン worktree は測定用に新規作成し、測定後に削除） |

### 8.1 未追跡ファイルの扱い

Owner 承認により追跡したのは `doc/work113/REPORT-20261101-residual-phase.md` と
`doc/work113/LOCAL-CI-GATE-20261101.md` の **2 件のみ**（PR #102 → `389e8159`）。

本作業で追加した次の 6 件は**未追跡のまま保全**している（追跡の承認を受けていないため）。

| ファイル | 内容 |
|---|---|
| `doc/work113/REPORT-20261101-abc-investigation.md` | 本報告書 |
| `doc/residual-20261101/A-ASSERT-RACE-ANALYSIS-20261101.md` | A の競合分析 |
| `doc/residual-20261101/B-SEGFAULT-INVESTIGATION-20261101.md` | B の調査 |
| `doc/residual-20261101/D01-IMPACT-ANALYSIS-LATEST-20261101.md` | D-01 影響調査 |
| `doc/residual-20261101/D02-MEASUREMENT-PROTOCOL-20261101.md` | D-02 条件の事前固定 |
| `doc/residual-20261101/D02-MEASUREMENT-RESULTS-20261101.md` | D-02 測定結果 |

未追跡の総数は **24 件**（既存 18 ＋ 本作業で追加した 6）。既存の未追跡（`opencode.json.bak-*`、
`vc140.pdb`、`doc/work113/P1-5-IR-P2_STG-11-D1*_FRESH-DISCOVERY_20261001.md`、分割版 4 件など）には触れていない。

## 9. 次の判断ポイント

1. **B の修正案の採否**（推奨: B-2「診断バッファをインスタンス所有へ」、暫定緩和は B-1）。
   診断 ON のビルドのみに影響するが、**診断ビルドでの検証が不安定になる**ため、
   回帰の観点からも修正が望ましい。
2. **D-02 の測定結果の扱い**（比較① は PASS。比較② は探索的なので**既定値変更には使わない**）
3. **A の修正案の採否**（推奨: A2 先行 → A3-a の設計評価）
4. **D-01 の扱い**（推奨: 2 行は適用しない。旧分析 RC-1 の訂正を記録）
