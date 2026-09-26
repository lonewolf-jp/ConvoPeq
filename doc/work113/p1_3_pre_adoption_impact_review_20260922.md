# P1-3 採用前影響範囲・既定値契約 read-only 監査（2026-09-22）

## 1. Scope

- **工程**: P1-1-A/B/C/D = PASS・P1-2（二重補償 review）= PASS → **P1-3（本 doc・read-only）** → P1-4 Adoption Decision Package（未着手）
- **目的**: 案E を ON にした場合に「polyphase gain convention の修正」以外の**既存ユーザー契約（既定値・可聴挙動）を意図せず変更していないか**を確認する。
- **規約**: production change = **0**・commit = **0**。追加測定は**既存 test harness の拡張のみ**（`src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp` に P1-3 用セクションを追加＝test-only・未 commit）。
- **分離の宣言**: 本 doc の PASS は **案E採用 / default ON の PASS ではない**。また Mimosa 完全監査は未実施であり、**P1-3 PASS ≠ project safety PASS**。

## 2. Source identity

| 項目 | 値 |
|------|----|
| HEAD | `1e9e63e3`（`d6995b2b` = P1-1 実装 / `eeeba325` = P1-0 audit） |
| production 差分 | `src/CustomInputOversampler.cpp` +3/−0（flag gate 済み 1 文）・`CMakeLists.txt` +8/−0（option + 3 target 伝播 + TU 登録）のみ。**P1-3 実施中は production diff = 0 を維持**（`git diff --name-only -- src/audioengine …CMakeLists.txt` が空であることを各段階で確認） |
| flag | `CONVOPEQ_CORRECT_POLYPHASE_GAIN` 既定 OFF・P1-3 測定中の cache 実値は OFF → ON → OFF（最終状態 OFF） |
| snapshot | `ConvoPeq.md` を P1-3 の test-only TU 拡張後に**再生成**（→ 本 doc 末尾の付録に identity を記録）。production source は P1-1 以降不変 |
| 測定識別 | 各 run のログ先頭に `[P1CHAR] flag_macro=0/1` を出力（in-binary attestation） |

## 3. Mathematical contract（数学契約）

案E が変更するのは **polyphase 往復の gain convention のみ**である。P1-1 の実測値に基づく分離:

| 項目 | OFF | ON | 判断対象 |
|------|-----|----|----------|
| OS round-trip DC | 0.75^N | 1.0 | 数学契約 |
| 1 kHz level | factor 依存（−2.4988·N dB） | factor-flat（0 dB） | 数学契約 |
| latency（reported / impulse argmax） | 255/287/290/582・256/287/291/583 | 同一 | 不変条件 |
| limiter 到達（os≥2・0 dBFS） | 0 | 119/4096 | 製品挙動 |
| SoftClip drive（clipEngMax, os8/−6） | 0.0018 | 0.0184 | 製品挙動 |
| hard clamp | 0 | 0 | 安全性 |
| default parameter（headroom/makeup/trim/OS factor） | 現行値 | **未決** | 採用判断 |
| calibration（出力校正値） | 現行（専用定数は存在しない） | **未決** | 採用判断 |

- 補足（P1-3 追加測定で確定）: **polyphase が経路に無い構成（OS=1 かつ SoftClip off）では flag の効果は厳密に 0.0000 dB**（下記 §6）。すなわち案E は「polyphase 段の利得規約」だけを変え、他の契約には触れない。

## 4. Existing default-value contract（既定値契約）

- **SOURCE**: `AudioEngine.h:2557/2612/2615-2616/2620-2626/2634-2635`（既定値）・`DeviceSettings.cpp:1017/1021-1022`（保存）・`:1217/1221/1225`（読込）・`:354-375`（UI ラベル）・`:1289`（"unity gain" コメント）。
- **OBSERVATION**:

| パラメータ | 既定値 | factor 依存 |
|------------|--------|-------------|
| `inputHeadroomDb` / Gain | −6.0 dB / 0.501187… | なし |
| `outputMakeupDb` / Gain | +12.0 dB / 3.9810717… | なし |
| `convolverInputTrimDb` / Gain | 0.0 dB / 1.0 | なし |
| `softClipEnabled` / `saturationAmount` | true / 0.1 | なし |
| `manualOversamplingFactor` / `oversamplingType` | 0（Auto → ≤96 kHz で 8x）/ IIR | —（factor を決める側） |
| `autoGainStagingEnabled` | true | なし |
| `eqBypassRequested` / `convBypassRequested` | false / false | なし |

  - 設定の永続化では makeup / headroom / trim / OS factor が**独立した属性として保存・読込**される（相互依存なし）。
  - UI ラベルは "Input Headroom:" / "Output Makeup:" の dB 表記のみで、**「0 dB = unity」という保証は UI に存在しない**（"unity gain" の語は `DeviceSettings.cpp:1289` のソースコメントにのみ現れる）。
  - **factor 依存の gain を符号化した既定値・校正値は存在しない**（factor 依存 droop は単一定数では表現できないため、構造的に「暗黙の前提」を作れない）。
- **INFERENCE**: 案E の採用は**既定値そのものを変更しない**が、既存プリセット・既定設定での**最終出力レベルを N に応じて +2.5〜+7.5 dB 押し上げる**（＝既定値の「意味」が変わる）。これは構造欠陥ではなく、**採用時に「既定値を見直すか／そのままにするか」を決めるべき論点**である。
- **CONTRACT IMPACT**: 停止条件「既定値が案E の gain convention を暗黙に前提としている」→ **非該当**（前提を持つ定数・分岐が存在しない）。

## 5. Downstream nonlinear impact（下流非線形への影響）

- **SOURCE**: `DSPCoreDouble.cpp:477-515`（makeup → SoftClip）・`:715-749`（limiter/clamp）・`:483-489`（θ/knee/asym = `sat` のみの関数）。
- **OBSERVATION**（P1-1 + P1-3 実測）: 非線形段の動作点は level に依存するため、案E の +2.4988·N dB は **SoftClip 駆動増**（clipEngMax が 5〜10 倍）と **limiter 到達量の増加**（0 → 119/4096 @ os≥2・0 dBFS）として現れる。**hard clamp は全 24 条件下で 0 件**（安全鎖は破綻しない）。
- **INFERENCE**: これは「数学修正の副作用」ではなく「入力レベルが正しくなった結果」である（OFF 側が 2.5〜7.5 dB 低かったことの裏返し）。したがって再調整の要否は**音質・校正ポリシーの問題**であり、DSP の正しさの問題ではない。
- **CONTRACT IMPACT**: 停止条件「SoftClip/limiter に未確認の gain compensation がある」→ **非該当**（閾値式に factor/0.75 項なし）。

## 6. Additional characterization（P1-3 追加測定）

条件: 48 kHz / block 512 / wet 経路（conv bypass + identity EQ）/ autoGain off / staging 0 dB（§6 末尾のみ staging −6 dB）/ 1 kHz / sat=1.0。

### 6.1 Δg = gainDb(ON) − gainDb(OFF)（exact 判定行のみ抜粋）

| os | n | ampDb | Δg sc0（SoftClip off） | 理論値 | 判定 |
|----|---|-------|------------------------|--------|------|
| 1 | 0 | −20 / −6 / 0 | **0.0000 / 0.0000 / 0.0000** | 0（polyphase なし） | **PASS（control）** |
| 2 | 1 | −20 / −6 | +2.4987 / +2.4987 | +2.4988 | PASS |
| 4 | 2 | −20 / −6 | +4.9975 / +4.9975 | +4.9975 | PASS |
| 8 | 3 | −20 | +7.4963 | +7.4963 | PASS |

- 追加の record 行: os≥2 の 0 dBFS は Δg = +2.0173 / +4.5049 / +7.0060 dB（limiter が吸収）、SoftClip on 行は clip 非線形のため理論値を仮定しない（Δg は +1.68〜+7.46 dB）。
- 参考（local OS のみの構成）: **os=1 + SoftClip on の −20 dBFS で Δg = +2.4989 dB**（局所 SoftClip OS = 1 stage の理論値 +2.4988 と一致）。
- **結果: exact 判定 8 行すべて PASS。**

### 6.2 limiter / hard clamp / clip engagement（24 条件の要約）

| 条件 | limiter 到達（OFF → ON） | hard clamp | clipEngMax（OFF → ON） |
|------|--------------------------|-----------|------------------------|
| os=1（−20/−6/0 dBFS） | 0→0 / 0→0 / **133→133** | 全 0 | 0.130→0.148 / 0.650→0.738 / 1.233→1.377 |
| os=2（−20/−6/0） | 0→0 / 0→0 / **0→119** | 全 0 | 0.000→0.000 / 0.0066→0.0182 / 0.0687→0.1472 |
| os=4（−20/−6/0） | 0→0 / 0→0 / **0→119** | 全 0 | 0.000→0.000 / 0.0018→0.0184 / 0.0255→0.1472 |
| os=8（−20/−6/0） | 0→0 / 0→0 / **0→119** | 全 0 | 0.000→0.000 / 0.0018→0.0184 / 0.0107→0.1473 |

- os=1 の 0 dBFS で limiter 到達が OFF/ON 同値（133）であることは、**flag が polyphase 段以外に作用していない**ことの傍証（§3 の control と同一の意味論）。

### 6.3 staging 非0（乗算独立性の確認）

`headroom = −6 dB`（staging = −6 dB）・os=8・1 kHz・−6 dBFS:

| sc | OFF gainDb | ON gainDb | Δg | 理論値 | 判定 |
|----|-----------|-----------|-----|--------|------|
| 0（SoftClip off・線形） | −14.5035 | −7.0072 | **+7.4963** | +7.4963 | **PASS（厳密）** |
| 1（SoftClip on・非線形） | −14.5032 | −6.9544 | +7.5488 | （適用外） | record |

- 結論: **既存 staging（headroom/makeup）と案E の gain correction は単純に乗算される**（§6.1 の staging 0 dB と同一の Δg）。limiter 到達・hard clamp は両 sc で 0。
- 目的は calibration の決定ではなく、**この乗算独立性の確認**である（calibration は P1-4 の判断対象）。

## 7. A / B / C classification

### A. 変更不要（構造上の根拠つき）

| 項目 | 構造上の根拠 |
|------|--------------|
| latency | OS 往復 gain は遅延と無関係（`estimateOversamplingLatencySamplesImpl` は taps/factor のみ）。P1-1 実測で OFF==ON（4 経路）。 |
| FIR / IIR の係数・topology | 係数生成（窓関数・halfband・taps/atten）は gain convention と独立。P1-1 の係数不変条件 gate（`R17-4(A)`）が両 build で PASS。 |
| limiter threshold | θ=0.8413951287507587・knee=0.108748 は固定 constexpr（OS/level 非依存）。 |
| SoftClip threshold/knee/asym | `sat` のみの関数（OS/factor/`kOutputHeadroom` を参照しない）。 |
| `kOutputHeadroom` | 固定 constexpr（−1.0 dBFS）。dither/NS/limiter/clamp がこれを共有するが、いずれも factor 非依存。 |
| default OFF（flag 契約） | `option(... OFF)` + 明示 ON のみで ON になる 2 重保証（P1-1 で OFF の 54/54 を 2 回再現）。 |
| bypass/wet/dry の構造 | dry は pre-OS 採取・fade は 0↔1 のみ（補償定数なし）。 |

### B. 再校正候補だが、変更はまだしない

| 項目 | 内容 |
|------|------|
| staging（`inputHeadroomDb`/`outputMakeupDb`） | 既定 −6/+12 dB（net +6 dB）。案E ON では出力が N に応じ +2.5〜+7.5 dB 上がるため、既定値の見直しは採用時の論点。 |
| default output level | 既存プリセット・既定設定での体感レベル（＋limiter 到達量）。 |
| SoftClip perceived drive | clipEngMax が 5〜10 倍に増える（＝より深い飽和）。 |
| limiter engagement | os≥2・0 dBFS で 0 → 119/4096。 |

**これらは「構造上必須の変更」ではない**（P1-2 の結論と一致）。本 P1-3 では変更しない。

### C. 実測不足（A へ格上げしない）

| 項目 | 不足している内容 |
|------|------------------|
| 複数実ユーザー設定での聴感 | listening test 未実施（本セッションでは実施不能な工程） |
| EQ boost + OS | identity EQ 以外（実 EQ カーブ・total gain・AGC）との組合せ |
| IR boost + OS | Convolver 有効（IR の gain profile・trim 非0）との組合せ |
| staging 非0 + 非線形 | §6.3 は 1 条件のみ（sweep 不足） |
| saturation sweep | `saturationAmount` を振った際の知覚差（0.1 → 1.0 の端のみ実測） |
| limiter engagement sweep | 入力レベルを細かく振った到達率カーブ（本 P1-3 は −20/−6/0 の 3 点） |
| 既存プリセットの level 変化 | 保存済みプリセット（makeup/headroom/trim/OS factor の組合せ）での再現確認 |

## 8. Unresolved questions

1. **既定値・校正の見直し要否**（B 群）: 案E 採用時に makeup/headroom 既定値やプリセットを調整するか、そのまま（レベルが上がる）を受け入れるか。→ P1-4 の判断対象。
2. **flag をどう扱うか**: 採用後も flag を残す（切替可能）か、削除して 1 本化するか。→ P1-4。
3. **C 群の測定**: 特に聴感評価と「EQ boost / IR boost との組合せ」は本監査の範囲外。
4. **release handling**: レベル変更を release note でどう告知するか（後方互換の説明責任）。→ P1-4。

## 9. Adoption decision boundary（採用判断の境界）

```text
P1-3 = PASS
  ≠ 案E採用 = PASS
  ≠ default ON = PASS
  ≠ calibration 変更 = 承認
  ≠ flag 削除 = 承認
  ≠ project safety = PASS（Mimosa 完全監査は未実施）
```

P1-3 が固定したのは「**影響範囲と既存契約の分離**」までである。採用・default ON・校正・release note の可否は **P1-4 Adoption Decision Package** で個別に判断する（本 doc は判断材料の提示のみ）。

## 10. PASS / HOLD

```text
P1-3 = PASS
```

**停止条件の確認（すべて非該当）**:

| 停止条件 | 実測・確認 |
|----------|-----------|
| 既定値が案Eの gain convention を暗黙に前提としている | 非該当（factor 依存の定数・分岐が存在しない・§4） |
| OS factor 依存の別補償が発見される | 非該当（`oversamplingFactor` は gain 計算に現れない・P1-2 §1） |
| staging/makeup に factor 依存がある | 非該当（`0/0/0`・`0/+10/−6`・`−6/+12/0` は mode 依存のみ・§4） |
| SoftClip/limiter に未確認の gain compensation がある | 非該当（閾値式は `sat`/固定 constexpr のみ・§5） |
| factor 変更で latency 以外の状態契約が変わる | 非該当（latency OFF==ON・他状態は P1-1/P1-2 で不変を確認） |
| float/double divergence | 非該当（単一 OS 実装の共有・P0-G ≤2.96e-8） |
| bypass/wet contract divergence | 非該当（dry pre-OS・補償定数なし） |
| 新たな production change が必要になる | 非該当（P1-3 中は production diff = 0 を維持。追加測定は test-only TU 拡張のみ） |

**次工程（P1-3 PASS の場合のみ）**: P1-4 Adoption Decision Package を作成し、`案E採用 / default ON / flag 維持 or 削除 / calibration / release note` を個別に判断する。

---

## 付録: 測定識別子・限界

- 追加測定は **test-only TU（`P1PolyphaseGainCharacterization.cpp`）の拡張**で実施（production 変更 0・未 commit）。P1-3 測定中の cache 実値は OFF → ON → OFF で、最終状態は **OFF**（既定状態に復帰済み）。復帰後の `PolyphaseGainFidelityTests` は **PASS=54 FAIL=0**。
- snapshot は test-only TU 拡張後に再生成（**header `Generated: 2026-09-22 10:28:33` / 5,488,742 B / `--check` FRESH・NEWER_SRC_COUNT=0**）。production source は P1-1 以降不変。
- 測定は 48 kHz / block 512 / 1 kHz / 単一 IR 無し（conv bypass）条件。頻度応答は P1-1 で 50 Hz〜10 kHz を確認済み。
- `limiter 入力 peak` は `getOutputLevel()`（peak 定義・pre-dither）、clip engagement は「clip on/off の出力差サンプル数」という proxy 定義（P1-0 で承認済み）。
- 聴感評価・複数プリセットでの level 再現は未実施（§7 C 群）。
