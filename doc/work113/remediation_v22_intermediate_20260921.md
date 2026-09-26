# ConvoPeq 改修計画 再監査 中間保存（2026-09-21 夜・v2.2 起草用）

- **目的**: v2.1（`remediation_plan_20260921_v2.1_revised.md`）の再監査・未確定事項の確定・v2.2 作成の途中成果を、コンテキスト上限前に一時保存する
- **セッション**: ses_ffe5f3e41726dffeYLX2HMptQs
- **基準 HEAD**: `8f127bfe`（+ 未 push `c4a08171`）— 本セッション再実測で変更なし
- **authoritative モデル**: `doc/work113/model_polyphase_20260920.py`（再実行環境: WSL python3 **3.14.4 / numpy 2.5.3 / scipy 1.18.1**）
- **【2026-09-22 更新】D1 補正軸をモデルへ反映済み**（旧軸は `D1 OLD AXIS` として残す）。下準備 §4 完了。パッチ案 `remediation_v22_prep_patches_20260922.md`・Shadow 骨格 `PolyphaseGainCandidateRef.h` も追加（production 未適用）
- **【authoritative 計画】**: `doc/work113/remediation_plan_20260922_v2.3_revised.md`（R9-1〜R9-6）。本ファイルは中間保存のまま。矛盾する箇所は v2.3 優先
- **使用ツール**: rtk(WSL) / context-mode MCP（ctx_batch_execute・ctx_execute・ctx_search）/ serena MCP / AiDex MCP / semble / cocoindex (`ccc`) / headroom MCP stats / cppcheck (Windows CLI)

---

## 0. 結論サマリ（v2.2 へ引き継ぐ判断）

| 区分 | 判定 |
|------|------|
| B-1 defect（DC round-trip = 0.75^N） | **再確認・確定維持**（モデル 12 桁一致・production コードに `centerValue *= 2.0` 不在） |
| 案 E（candidate hypothesis） | **仮説のまま維持**（Phase 0 全 PASS + ユーザー GO まで確定表記禁止） |
| GO/HOLD | **変更なし**（B-1 Phase 0 GO / Phase 1 以降 HOLD / F-2 GO 候補 / F-1 GO 候補 / O-* はユーザー判断） |
| 新規確定 | **R8-1〜R8-12**（下記）— うち **R8-1（モデル D1 が旧軸のまま）** が最重要 |
| 新規 O 項目 | **O-14 `.mcp.json` +33/−23**（v2.1 O リストに不在だった） |
| 技術的未確定 | **解消済**（v2.1 §12.4 は全件確定済み。本セッションで新たに見つけた「モデル D1 軸未反映」も確定・対応方針を R8-1 で確定） |
| ユーザー判断待ち | O-1〜O-14 / G-0 / Phase 0 review GO / F-2 案 A 採用 / F-1 全実施 — **技術調査では解消しない** |

---

## 1. 本セッション再実測エビデンス

### 1.1 git / working tree（O-1 再監査）

```
HEAD = 8f127bfeea17831e049ef6ce569f3c570cb96a62
log: 8f127bfe docs(work113) / c4a08171 test(harness) / 06794615 / 32137e60 / a647d13b

git diff --numstat（実測・2026-09-21 夜）
  33  23  .mcp.json                          ★ v2.1 不在 → O-14 新設
   5   3  AGENTS.md                          ✓ R6-2 と一致
  73   2  ConvoPeq.md                        ✓ R6-2 と一致
   0   1  Testing/Temporary/CTestCostData.txt ✓ O-4
  71   1  doc/work113/residual_tasks_20260919.md ✓ R6-2 と一致
 634   2  src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp ✓
   8   0  src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp ✓
  -   -  tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc ✓ binary

production src/ の未 commit 差分（src/tests 以外）= 0 を再確認
untracked doc/work113/*.md = 30 件（計画書系列の増加。O-13 の件数更新が必要）
docs/tool-inventory-2026-09-20.md = 3102 bytes（O-9 継続）
build/Release/AudioEngineHarness.exe = 41,137,664 bytes（R4-11 継続）
```

**.mcp.json 差分の内容（要旨）**: MCP サーバ設定の書式・パス更新（serena 絶対パス `C:\Users\user\.local\bin\serena.exe` 化、`--context ide` 追加、aidex-mcp の node_modules パス修正など）。**environment/config 系**であり production DSP には無関係。推奨: **AGENTS.md（O-10）と同様に環境記録更新として commit 可**（ユーザー判断）。

### 1.2 閉形式モデル再実行（R6-1 の再検証 + R8-1/R8-2）

```
実行: python3 doc/work113/model_polyphase_20260920.py
比較: tr -d '\r' 正規化後
結果: exact_same = 91/96（exp=96 / rerun=96）
      v2.1 午前は 92/96 → 本セッションは 91/96（環境・実行時のフロア微差）
差分 5 行（すべて D2 フロア ±0.01 dB・GATE 影響なし）:
  L80/L81  511/140 D2 base/cand @0.05: -139.68 → -139.69
  L89      255/140 D2 cand @0.05/@0.35: -173.06→-173.07 / -79.15→-79.16
  L92/L93  1023/160 D2 base/cand @0.05: -150.23↔-150.22（base/cand 入れ替わり級のフロア差）
DC・§2.7.1 full-chain・§2.7.2 per-design・D3 は一致
```

**DC 実測（再現）**:

| 構成 | baseline DC | candidate DC | 0.75^N |
|------|-------------|--------------|--------|
| IIRLike r=2 / LP r=2 | 0.750000000000 | 1.000000000000 | 0.75 |
| r=4 | 0.562500000000 | 1.000000000000 | 0.75² |
| r=8 | 0.421875000000 | 1.000000000000 | 0.75³ |

係数検証: 全 6 design で FIRsum=1.0 / center=0.5 / convSum=0.5。

**【R8-1・最重要】results ファイルの D1 は旧軸のまま**:

```
results.txt:67  D1: up 単段出力（2x rate）で f̂ vs 0.5−f̂（v1.7 §2.7.3 相当定義）
results.txt:70  31/90 D1 base : 0.05:-9.75 0.1:-10.49 0.2:-9.97 0.3:11.51 ...   ← 旧軸
results.txt:71  31/90 D1 cand : 0.05:-35.94 0.1:-27.79 ...                      ← 旧軸
```

これは R7-1 が破棄した「トーン実ビンの 2 倍を読む軸」の値列と一致する。
**補正軸 D1（base −9.5424 dB 構造定数 / cand −84〜−116 dB）は v2.1 §2.7.3 表と `tmp/d1_full_check.py` 系スクリプト側にのみ存在し、authoritative モデル/結果ファイルには未反映。**

対応方針（v2.2 で確定）:
1. `model_polyphase_20260920_results.txt` の D1 セクションを **「旧軸・参考・解釈破棄済」** と明示
2. D1 の authoritative な補正軸表は **v2.1/v2.2 §2.7.3** とする
3. Phase 0-3 実装前に、測定ハーネス（C++ / モデル両方）で **tone bin = f̂·N（argmax 検証必須）** を実装する（教訓 §2.8.25）
4. GATE への影響なし（D1 は record-only / P0-E-2 は D2）

### 1.3 B-1 ソース再監査（production コード）

| 項目 | 実測 | v2.1 との一致 |
|------|------|----------------|
| `isLinearPhaseFIR` / `isSymmetricUpDown` | CustomInputOversampler.h:21-22 | ✓ |
| `Stage::centerCoeff = 0.5` | h:87 | ✓ |
| `convValue *= 2.0` のみ | cpp:**557**（`:558` denorm / `:564` center 出力） | ✓ |
| `centerValue *= 2.0` | **不在**（AiDex `centerValue *= 2.0` 0 hit / rg 確定） | ✓ 欠陥維持 |
| interpolate 出力 | `output[outBase+convParity]=convValue` / `+centerParity=centerValue` | ✓ |
| decimate center | `acc = stage.centerCoeff * centerSample`（×2 補償なし） | ✓ |
| `reset()` | :452-467（upHistory/downHistory clear + アトミック 3 個解除） | ✓ R4-3 |
| `clearAllStages()` | :469-484（履歴 clear + corruptionDetected のみ解除） | ✓ R4-3 |
| `processUp` hardFallback 透過 | :725 から | ✓ |
| prepareStage Kaiser β | A>50 → 0.1102(A−8.7) / A≥21 → 0.5842… / else 0（cpp:301-304 相当） | ✓ R7-3 |
| taps 表 | IIRLike {511,127,31} / LP {1023,255,63} | ✓ |
| Latency.cpp:30 | `groupDelaySamplesAtStageRate = taps[stage] - 1; // up + down` | ✓ R7-4 |
| static_assert | Latency.cpp:6-7（isLinearPhaseFIR && isSymmetricUpDown） | ✓ |
| JUCE E5 | up :185 `buf[N-1] = 2 * samples[i]` / down :228 `bufferSamples[i << 1]` | ✓ |
| SoftClip prepare | DSPCoreLifecycle.cpp:**188** と **261** `prepareSingleStage(31, 90.0, internalMaxBlock)` | ✓ |
| SoftClip process Float | DSPCoreFloat.cpp:405 processUp / :413 processDown | ✓ |
| SoftClip process Double | DSPCoreDouble.cpp:**505 processUp / :513 processDown** | ★ 行追加で確定（v2.1 は Float のみ明記） |
| kOutputHeadroom | DSPCoreDouble.cpp:**593** = 0.8912509381337456 | ✓ |

### 1.4 F-2 / F-4 / F-1 / R-2

**F-2（AGC 呼出順）**:
- `configureProbeFlatEQ` :1072 / 内 `setEQAGCEnabled(false)` :1081
- eq モード: **:1799 configureProbeFlatEQ → :1803 setAutoGainStagingEnabled(false)**（誤順）
- eqdiag パターン: **:1833 setAutoGainStagingEnabled(false) → :1834 configureProbeFlatEQ**（正順）
- `AudioEngine.h:1416-1425`: ON→OFF 遷移時のみ `getEQProcessor().setAGCEnabled(!enabled)`
- `autoGainStagingEnabled` 既定 true :2626
- `getEQProcessor` :1292 / `setEQAGCEnabled` :1306（同一 uiEqEditor）→ **案 A（呼出順逆）維持**

**F-4（ir / irwet）**:
- 未知モード fail-closed :1708-1710
- `ir`: :1745 `setConvolverBypassRequested(true)` 維持（dry ベースライン）
- `irwet*`: :1763 分岐 / **:1778 `setConvolverBypassRequested(false)`**（wet 対照）
- 用語定義 R5-8 維持

**F-1（PVIT / Validator）**:
- `src/audioengine/RuntimePublicationValidator.h` **105 行** / `.cpp` **211 行**（`src/core/` 不在）
- h:13 `enum class ValidationFailureReason` / h:23 errorMessage / h:24 failureReason / h:64 validatePublication / **h:92 private:** / **h:101 checkNoConflictingTransitions**
- cpp:8 validatePublication / :16-:41 検査順序と failureReason 設定 / :169 checkNoConflictingTransitions 本体
- PVIT: **519 行 / TEST 38 / 呼出 9 箇所**（:135/:246/:255/:265/:273/:281/:313/:324/:334）
- CMake add_executable **39 件**（PVIT 未登録）/ tools/build-debug.bat stale :29
- **Phase B' 主判定 = failureReason**（R5-7 維持）

**R-2（数値 parse fail-closed 仕様）**:
- BassBuzzMeasurement.cpp:
  - parseHcIdx :1558（stoi :1562）/ parseLcIdx :1564（stoi :1567）— try なし
  - :1582 stod(sr) / :1583 stoi(block) / :1590 stof(probe) / :1591 stoi(quiet) / :1592 stod(dur)
  - :1614 stod(flip-t) / :1615 系（flip 関連）
  - lambda 呼出: :1606 hc / :1607 lc / :1608 eqlpf / :1610 flip-hc / :1611 flip-lc
- PPIT: main :1085 / 前方宣言 :1116 / 呼出 :1223
  - stoi: **:1136 / :1146 / :1149 / :1157 / :1166**（**:1149 を追加検出** — v2.1 リストの 4 箇所に 1 つ増）
- 仕様: try/catch + `[BUZZ] FAIL: invalid numeric` + 非ゼロ終了 + regression test（B-2 fail-closed と同一パターン）

### 1.5 TruePeakDetector（R6-5/R7-4 再確認）

- `interpolateStage` のみ（**decimateStage 不在** → round-trip 構造なし）
- prepare: 2 段 2×2=4×、stageTaps = (i==0)? taps : **max(15, taps/2)**
- `kDefaultAttenuationDb = 100.0`
- 係数正規化後 `rawCoeffs[centerTap] = 0.5`、interpolate に ×2 補償なし
- 後続候補第三順位（案 D → tap 再設計 → TPD 型）維持

### 1.6 静的解析・ツール

- cppcheck（Windows `C:\Program Files\Cppcheck\cppcheck.exe`）`--enable=warning,performance,portability --std=c++20` → **exit=0・指摘 0 件**（再確認）
- graphify CLI 実在（`path` 等のサブコマンド）。本作業ではソース探索に AiDex/serena/semble/rg を優先し、graphify は補助
- serena: `get_symbols_overview` で CustomInputOversampler の全メソッド一覧取得（prepareStage/interpolateStage/decimateStage/processUp/processDown/reset/clearAllStages 等）
- AiDex: セッション開始・convValue 7 ヒット（:557 を含む）・`centerValue *= 2.0` 0 ヒット
- headroom MCP: session stats 取得可（本セッションは MCP compress 未使用・proxy 経由の自動圧縮が主）

---

## 2. v2.2 へ引き継ぐ R8 項目（確定）

| ID | 内容 | 確定 |
|----|------|------|
| **R8-1** | authoritative モデル/結果の **D1 測定が旧軸のまま**（R7-1 補正軸未反映）。results.txt:67 に「v1.7 §2.7.3 相当」と明記され、値列も旧軸。補正 D1 の authoritative は v2.1 §2.7.3 表。Phase 0 測定ハーネスで argmax 検証付き補正軸を実装必須 | **確定・方針明文化** |
| **R8-2** | モデル再実行 **91/96**（D2 フロア 5 行 ±0.01 dB）。v2.1 の 92/96 は同一現象の別実測。GATE 影響なし | **確定** |
| **R8-3** | **O-14 `.mcp.json` +33/−23** を新設（環境/MCP 設定更新・commit 可候補） | **確定** |
| **R8-4** | B-1/F-1/F-2/F-4/SoftClip/JUCE/Latency/CMake/PVIT の主要行番号を全件再実測し **v2.1 と一致** | **確定** |
| **R8-5** | cppcheck Windows 側 exit=0・指摘 0 を再確認 | **確定** |
| **R8-6** | work113 untracked md **30 件**（O-13 の件数を更新） | **確定** |
| **R8-7** | production `src/` 未 commit（tests 除く）= **0** | **確定** |
| **R8-8** | HEAD `8f127bfe` / 未 push 2（O-6 継続） | **確定** |
| **R8-9** | R-2 の PPIT stoi に **:1149 を追加**（計 5 箇所）。BassBuzz 側は v2.1 の通り | **確定** |
| **R8-10** | SoftClip の **Double 経路** processUp/Down も DSPCoreDouble.cpp:505/:513 に実在（Float だけでなく両ホストで同じ局所 OS） | **確定** |
| **R8-11** | TPD: attenuation 100.0 / stage1=max(15,taps/2) / 4×2 段 / decimate 不在 / center=0.5 | **確定（記録）** |
| **R8-12** | **技術的未確定事項は本セッションで解消**。残る「未確定」はユーザー意思決定（O-*・G-0・Phase GO・F 方針）のみ | **確定** |

---

## 3. 引き継ぎ手順（新しいチャットスレッド用）

1. 本ファイル `doc/work113/remediation_v22_intermediate_20260921.md` を読む
2. **authoritative は v2.3**: `doc/work113/remediation_plan_20260922_v2.3_revised.md`（R9 反映・下準備完了）
3. 参考: v2.2 `remediation_plan_20260922_v2.2_revised.md` / パッチ案 `remediation_v22_prep_patches_20260922.md` / Shadow `PolyphaseGainCandidateRef.h`
4. 必要ならモデル再実行: `wsl` 内で `python3 doc/work113/model_polyphase_20260920.py` + `tr -d '\r'` 比較
5. 作業指示: headroom / context-mode / rtk を常時使用。要調査・保留は「技術=確定 / 判断=ユーザー」に分離
6. 音響工学参照: https://asj-fresh.acoustics.jp/useful-links / https://acoustics.jp/journal/ / https://acoustics.jp/link/institutes/

---

## 4. 未使用だが推奨される次の技術作業（Phase 0 着手前の下準備）

**【2026-09-22 継続セッションで全件着手・production 未適用】**

- [x] `model_polyphase_20260920.py` の D1 測定関数を補正軸（tone bin = f̂·N、image = N−f̂·N、argmax 検証出力付き）へ更新し、結果ファイルを再生成（旧 D1 行は `D1 OLD AXIS` として残す）
  - 実測: base 構造定数 **−9.5424 dB が passband 24/24 一致**。cand は v2.2 §2.7.3 表と概ね一致（表が 1 桁丸めのため ±0.1 dB 級の差が大半。63/120@0.2 のみ ≈0.94 dB 差を記録）
  - 環境: WSL python3 3.14.4 / numpy 2.5.3 / scipy 1.18.1 / exit=0 / results 116 行
  - D1-ARGMAX: **base** は tone/image とも expected bin と一致。**cand** は tone 一致・image は近零域のため search 窓 ±8 bin 内で argmax が漂う（Phase 0 では argmax 採用 + bin 番号ログが正しい仕様）
  - DC は従前どおり base 0.75^N / cand 1.0
- [x] Shadow Reference ヘッダ `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` の骨格作成（production 変更 0・untracked test-only）
- [x] CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN` の差分パッチ案 → `doc/work113/remediation_v22_prep_patches_20260922.md` §1
- [x] R-2 fail-closed 実装スケッチ（parseIdxOrFail + PPIT :1136/:1146/**:1149**/:1157/:1166）→ 同 §3
- [x] F-2 案 A の 1 行差替パッチ（:1799/:1803 を :1833/:1834 パターンへ）→ 同 §2

**追加成果物**:
- `doc/work113/remediation_v22_prep_patches_20260922.md`（パッチ案・未適用）
- `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`（test-only 骨格）
- モデル/結果の D1 補正軸反映（R8-1 対応の技術側は完了。GATE 影響なし）

---

## 5. 判断待ちのまま残す項目（技術調査では解消しない）

| 項目 | 推奨（v2.2 継承） |
|------|-------------------|
| O-1 計装 commit | A: 最小実行可能資産 |
| O-2 台帳 | 独立 commit |
| O-3/O-12 ConvoPeq.md | commit しない |
| O-4 CTestCostData | 現状維持 |
| O-5 .opencode | 触らない |
| O-6 push | ユーザー手動 |
| O-8 pyc | 触らない → 将来 .gitignore |
| O-9 docs/tool-inventory | 触らない（ユーザー判断） |
| O-10 AGENTS.md | commit（環境記録） |
| O-11 residual_tasks 台帳 | commit |
| **O-13 work113 md 群** | 承認版+台帳を 1 commit（件数 30 に更新） |
| **O-14 .mcp.json** | **commit（環境/MCP 設定）候補** |
| G-0 DESIGN-CONTRACT-A | 明示承認（v2.1 版 E1〜E5） |
| B-1 案 E | 有力仮説として採用（確定表記禁止） |
| Phase 0 | GO（測定仕様 v2.1+v2.2 適用が条件） |
| Phase 1+ | HOLD（Phase 0 review のユーザー GO 前提） |
| F-2 | 案 A |
| F-1 | 全実施（A→B→B'→C） |
| F-3/F-4 | 記録のみ |

---

*本ファイルは中間保存であり、authoritative な改修計画本体ではない。本体は `remediation_plan_20260922_v2.3_revised.md`（v2.2 + R9）。*
