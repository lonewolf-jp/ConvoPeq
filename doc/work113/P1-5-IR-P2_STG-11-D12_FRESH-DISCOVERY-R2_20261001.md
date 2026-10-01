# STG-11-D12 Fresh Discovery — Session-State Boundary / Integer Domain（R2）

- Document: `doc/work113/P1-5-IR-P2_STG-11-D12_FRESH-DISCOVERY-R2_20261001.md`
- Work item: **STG-11-D12** — session/state 境界・整数ドメインの残存 invalid-state 監査
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（監査 AI が参照する `ConvoPeq(20261001-105209).md` と同一ファイル）
  - baseline commit: `7b799b493637737327af8046f0601dc39c8ffee1`（Fix-D11）
  - `Generated: 2026-10-01 21:30:09` / 5,892,101 B / 129,811 lines
  - SHA-256: `A455FDA78220DE2A3B57C08369D3EADAC5ACD2FE4E396B77FB60A377A0C3518F`
  - **FRESH 検証済み**: 同一 tree で再生成した内容と **129,811 行中 1 行（`Generated:` タイムスタンプ行）のみ** 相違。
    snapshot は現行ソースを正確に反映している（`--check` の STALE 表示は source と同一 mtime のタイによるもの）。
  - 過去資料・記憶・旧行番号は根拠として使用していない。結論はすべて現行ソースから導出。
- 本 work item は **read-only Fresh Discovery**。**production source の変更は一切行っていない。**
- 本 commit の内容: **audit document のみ**

> ### ★ 訂正記録（2026-10-01 / STG-11-D12-R1）
>
> 本書の §2.3.1 に事実誤認があった。`validatePresetStateTreeForDebug` には
> `ditherBitDepth` の検査が存在する（`AudioEngine.Parameters.cpp:132-137`、
> 条件 `bitDepth <= 0 || bitDepth > 64`）が、本書は「リストに無い」と記していた。
> 該当箇所は訂正済み。
>
> - 訂正の根拠、全 6 箇所の合法値集合の確定、defect chain の 8 hop 再証明、
>   UI 経路の新規所見、orphan テストの所見は
>   `doc/work113/P1-5-IR-P2_STG-11-D12-R1_REVERIFICATION_20261001.md` を参照。
> - **D12-1 の結論（defect として成立）は変わらない。** 検査が dead であるため
>   production 上の防御力は当初の記述と変わらず 0 である。
> - なお本書の §5.1 で「`saturationAmount` の NaN 経路は要 Owner 再評価」とした点も、
>   同じ dead な whitelist gate を唯一の根拠としており、同じ留保が必要である。

---

## 0. 判定

```
STG-11-D12 — GO（concrete defect 2 件: D12-1 / D12-2）
ただし実装には進まない。Owner 承認までは監査報告のみ。
```

前 Audit 記録（R1: `P1-5-IR-P2_STG-11-D12_FRESH-DISCOVERY_20261001.md`）は NO-GO であった。
本 R2 は現行ソースに対する再監査であり、**R1 の `ditherBitDepth` 判定を反転する**（§4）。

| 項目 | 結果 |
| --- | --- |
| production change | **0 件**（read-only） |
| 新規 concrete defect | **2 件**（D12-1 深刻、D12-2 同型） |
| D7〜D11 修正済み箇所の再発 | **該当なし**（D12-1 / D12-2 は別 defect class・別 field） |
| ISR / Authority への影響 | publish 判定（NonRT validation 層）のみ。RT path 変更不要 |
| 実装 | **未着手**（Owner 承認待ち） |

---

## 1. 監査範囲

R1 の結論を再利用せず、現行ソースで field 単位に
`session / XML 入力 → setter → 永続 atomic/state → build input → published world → consumer → harm`
を追跡した。重点は D12 方針どおり以下。

- Session-State Boundary（`requestLoadState` / `DeviceSettings` XML restore / `MainWindow` からの load）
- Integer Domain（enum ではなく**整数**の範囲）
- enum / mode cast（D7 の領域。再発確認のみ）
- integer / index / size / count / length
- sample rate / DSP topology
- restore → setter parity

---

## 2. D12-1: `ditherBitDepth` の session restore に範囲ガードが無い

### 2.1 concrete trigger

`ditherBitDepth` を含む session（"Preset" ValueTree）または device-settings XML を読み込む。
その値が合法集合 `{0, 16, 24, 32}` の外（例: `8`, `12`, `20`, `99`, `-1`）である。

到達経路は通常のユーザー操作であり、内部状態操作や異常注入を必要としない。

- 破損・truncated・手動編集された session / preset
- 将来バージョンまたは別 build が新しい bit depth を書き出した session
- device-settings XML（`ditherBitDepth` 属性）
- エントリポイント: `src/MainWindow.cpp:1652` `audioEngine.requestLoadState(state)`

**session に当該 property が実際に含まれることは確認済み**（往復 surface が存在する）:

- 保存側: `src/audioengine/AudioEngine.StateIO.cpp:239`
  `state.setProperty("ditherBitDepth", consumeAtomic(ditherBitDepth, acquire), nullptr);`
- 復元側: `src/audioengine/AudioEngine.StateIO.cpp:100-101`（無検証）

### 2.2 actual state / data flow（実コード行番号つき）

```text
(1) src/audioengine/AudioEngine.StateIO.cpp:100-101     ← session restore（無検証 cast）
    if (state.hasProperty("ditherBitDepth"))
        setDitherBitDepth(static_cast<int>(state.getProperty("ditherBitDepth")));

(2) src/audioengine/AudioEngine.Parameters.cpp:380-397   ← setter にも範囲ガードが無い
    void AudioEngine::setDitherBitDepth(int bitDepth)
    {
        if (consumeAtomic(ditherBitDepth, acquire) != bitDepth) {   // 「変化したか」だけを見る
            ...
            convo::publishAtomic(ditherBitDepth,         bitDepth, release); // :394 値をそのまま永続化
            convo::publishAtomic(m_currentDitherBitDepth, bitDepth, release); // :395
            submitRebuildIntent(Structural, ...);                        // :397 再構築を要求
        }
    }

(3) src/audioengine/AudioEngine.RebuildDispatch.cpp:46    ← 再構築のたびに atomic から読む
    snapshot.ditherDepth = consumeAtomic(engine.ditherBitDepth, acquire);
  → :605  task.buildInput.ditherBitDepth = paramSnapshot.ditherDepth;

(4) src/audioengine/RuntimePublicationValidator.cpp:123-126  ← publish 時に hard reject
    const int dd = resource.ditherBitDepth;
    if (dd != 0 && dd != 16 && dd != 24 && dd != 32)
        return false;

(5) src/core/RuntimePublicationCoordinator.h:118-126      ← world を破棄して拒否
    if (!bridge_.validatePublicationNonRt(*worldOwner))
    {
        auto* rejectedWorld = const_cast<World*>(worldOwner.release());
        bridge_.retireRejectedRuntimeWorldNonRt(rejectedWorld);
        return PublishStageResult::Rejected;              // runtime は更新されない
    }
```

合法集合の根拠:

- `src/audioengine/AudioEngine.h:14` `inline constexpr int kAdaptiveBitDepthValues[3] = {16, 24, 32};`
  （validator の `{0,16,24,32}` は「0 = auto/無効」＋ 3 段）
- `src/tests/PublicationValidatorIsolationTests.cpp:187` `world.resource.ditherBitDepth = 8;  // 0,16,24,32 以外`
  を invalid ケースとして固定済み。

### 2.3 existing guard が無いこと（3 経路すべて）

| 期待するガード | 実状 |
| --- | --- |
| `setDitherBitDepth` の範囲検証 | **無し**（`Parameters.cpp:380-395`。`!=` による変化判定のみ。clamp も assert も無い） |
| `requestLoadState` の範囲ガード | **無い**（`StateIO.cpp:100-101` は無検証 cast） |
| publish 経路のガード | validator が world を**拒否**するが、値は既に atomic に**永続化済み**であり、無害化しない |
| session まとめての whitelist gate | **実質的に無い**（§2.3.1） |

#### 2.3.1 whitelist gate が機能していないこと

`src/audioengine/AudioEngine.Parameters.cpp:74-145` に
`validatePresetStateTreeForDebug`（`[[maybe_unused]]`）が存在し、
ちょうど本件と同一クラス（整数域 / 有限性）を列挙している。

```cpp
hasIntRange("processingOrder", ...);            // :111
hasFiniteDouble("saturationAmount", 0.0, 1.0);  // :112
hasFiniteDouble("inputHeadroomDb", -12.0, 0.0); // :113
hasFiniteDouble("outputMakeupDb", 0.0, 12.0);   // :114
hasFiniteDouble("convolverInputTrimDb", ...);   // :115
hasIntRange("analyzerSource", ...);             // :116
hasIntRange("noiseShaperType", ...);            // :117
hasIntRange("oversamplingType", ...);           // :118
hasIntRange("convHCFilterMode", ...);           // :119
hasIntRange("convLCFilterMode", ...);           // :120
hasIntRange("eqLPFFilterMode", ...);            // :121
hasFiniteDouble("coeffSafetyMargin", 0.0, 2.0); // :122
hasIntRange("cmaesRestarts", 0, 1000);          // :123
// :125-130  oversamplingFactor: {0,1,2,4,8} の個別検査
// :132-137  ditherBitDepth: 1..64 の個別検査（★ 訂正: この検査は存在する）
```

- ~~**`ditherBitDepth` はこのリストに無い**（`noiseShaperType` 等の隣接項目は含むが、本項目は欠落）。~~
  **★ 訂正（D12-R1）**: この記述は誤りであった。`Parameters.cpp:132-137` に
  `ditherBitDepth` の個別検査が存在する（条件は `bitDepth <= 0 || bitDepth > 64`）。
  当初は 74-126 行までしか読まず 127-137 行を読み飛ばしていた。
- この validator は**呼び出し側がゼロ**。リポジトリ自身が
  `src/tests/AudioEngineHarness/STG11D7StateEnumGuardTests.cpp:13` に
  `//   The range validator (validatePresetStateTreeForDebug) exists but has zero callers.`
  と明記している（D7 の regression test ヘッダ）。
- したがって `requestLoadState` を防御する**能動的な whitelist gate は存在せず**、
  防御は `requestLoadState` 内に個別に書かれた inline ガード
  （`processingOrder` / `noiseShaperType` / `oversamplingType` / `analyzerSource` / 3 filter mode）に
  全面的に依存している。**`ditherBitDepth` はそのいずれにも属さない唯一の項目。**
- **★ 訂正による結論への影響**: なし。欠落ではなく「検査は存在するが caller がゼロで
  実行されない」ため、production 上の防御力は当初の記述と変わらず 0 である。
  むしろdead な検査の範囲 `1..64` は、`RuntimePublicationValidator` の契約
  `{0,16,24,32}` と矛盾しており（0 を却下し 8 や 64 を承認する）、
  gate として復活させるなら契約の是正が前提になる。

#### 2.3.2 同一ファイル内の restore 経路との対比

| field | 位置 | ガード |
| --- | --- | --- |
| `processingOrder` | `StateIO.cpp:41-51` | あり（範囲チェック） |
| `noiseShaperType` | `StateIO.cpp:107-113` | あり（範囲チェック） |
| `oversamplingType` | `StateIO.cpp:144-150` | あり（範囲チェック） |
| `analyzerSource` | `StateIO.cpp:173-` | あり（範囲チェック） |
| `convHCFilterMode` / `convLCFilterMode` / `eqLPFFilterMode` | `StateIO.cpp`（D7 追加分） | あり（範囲チェック） |
| **`ditherBitDepth`** | **`StateIO.cpp:100-101`** | **無し** |

### 2.4 concrete harm

1. **不正値が engine の内部 state に永続化される。**
   `setDitherBitDepth` は publish 判定より**先に** `ditherBitDepth` /
   `m_currentDitherBitDepth` を書き換えるため、拒否される値が残留する。
2. **以降の publish がすべて拒否される（stuck state）。**
   `RebuildDispatch.cpp:46` は再構築のたびに atomic から読むので、不正値を持つ world が
   作られ続け、`validateResources` が毎回拒否する。**拒否時にその値を戻す機構は存在しない。**
   → 結果として、復元した session の他の設定（IR / EQ / oversampling / processingOrder /
   softclip / noise shaper 等）と、**その後のユーザーパラメータ変更がすべて runtime に反映されない。**
   出力自体は旧 world のままなので「音が出ない」のではなく「設定を一切反映できない」状態になる。
3. **原因がユーザーに伝わらない。**
   publish 失敗は `[PUBLISH] commitRuntimePublication FAILED` のログと
   `emitValidationEvent(InvalidResources)`（`AudioEngine.h:3809`）までに留まり、
   原因が「dither bit depth が範囲外」であることが UI にもメッセージにも出ない。
   ユーザーが観測するのは「session を読み込んだのに何も変わらない」だけである。
4. **adaptive 係数バンクの誤バケット（独立の害ではない）。**
   `selectAdaptiveCoeffBankForCurrentSettings`（`AudioEngine.Learning.cpp:408-414`）から
   `getAdaptiveBitDepthIndex`（`Learning.cpp:377-382`）は
   `bitDepth <= 16` で 0、`<= 24` で 1、それ以外で 2 とバケット化する。
   戻り値は常に `[0, 2]` の範囲内で、`getAdaptiveCoeffBankForIndex`（`:392-398`）も index を clamp するため
   **OOB は発生しない**。また本害が成立している間は world が reject 済みで DSP が当該値で動作しないため、
   この経路は主要な害に吸収される。**独立した害としては計上しない。**

**到達性と回復性（誠実に記載）**

- 到達性: 通常の「session / preset / デバイス設定を読み込む」操作で到達する。特殊操作は不要。
- 回復性: ユーザーが dither bit depth を合法値（0/16/24/32）に選び直せば回復する。
  したがって恒久的な破損ではない。ただし原因に気付けない限り「何も反映されない」状態が続き、
  その間すべての publish が失敗する。
  UI 側のコントロールが合法値のみを返すことを本 R2 では未実地確認している（§6-4）。

### 2.5 ISR / Authority への影響

- 影響するのは **NonRT validation 層の publish 判定**のみ。
- RT path の変更は不要。mutex / allocation / delete / 新規 atomic / 新規 authority の追加は不要。
- 既存 authority（`RuntimePublicationValidator` / `RuntimeWorldAuthority` /
  `RuntimePublicationCoordinator`）の境界は一切変更しない。
- Practical Stable ISR Bridge Runtime の不変条件をすべて維持できる。

### 2.6 minimal repair の成立性（定義のみ・未実装）

guard の設置位置は 2 案ある。どちらも **NonRT のみ**で閉じ、RT 変更を伴わない。

| 案 | 設置位置 | 効果 | 欠点 |
| --- | --- | --- | --- |
| A | `StateIO.cpp:100-101` と `DeviceSettings.cpp:1136-1137` に範囲チェック | 既存の D7 / B-3 guard パターンを機械的に踏襲できる。最小差分 | UI 経路は守られない（UI が合法値のみ返すなら実害はない） |
| B | `setDitherBitDepth` の入口（`Parameters.cpp:380`）に guard | restore / XML / UI を一元化。restore → setter parity が真正に成立する | UI 側の不正値を黙って落とすため診断性がわずかに下がる |

**推奨は案 B**（D12 方針の「restore → setter parity」に合致し、将来 restore 経路が追加されても漏れない）。
ただし案 A でもテストと併せれば充足する。最終選択は Owner 判断。

### 2.7 想定されるテスト（D7 と同じ形）

- **D12-T1** 範囲外値（`8`, `-1`, `99`）の session restore → `ditherBitDepth` が合法値のまま維持され、
  後続の publish が reject されないこと。
- **D12-T2** 往復不変: `getCurrentState()` から `requestLoadState()` への合法値の round-trip が不変。
- **D12-T3** 境界値: `0`, `16`, `24`, `32` は適用され、`-1`, `8`, `15`, `17`, `33` は拒否。
- **Negative control**: guard を外すと D12-T1 が FAIL すること。
- 登録形態: `AudioEngineHarness` の sub-test（D7 と同じ形。standalone CTest target は追加しない）。

---

## 3. D12-2: `DeviceSettings.cpp` の `noiseShaperType` 無検証 cast（restore 経路 parity 不整合）

D12-1 と同型の別 instance。**別 defect として報告し、実装しない。**

```cpp
// src/DeviceSettings.cpp:1136-1141
int bitDepth = xml->getIntAttribute("ditherBitDepth", 0);
engine.setDitherBitDepth(bitDepth);                                   // :1137  ← D12-1

int shaperType = xml->getIntAttribute("noiseShaperType", 0);
engine.setNoiseShaperType((AudioEngine::NoiseShaperType)shaperType);  // :1141  ← D12-2
```

- 同じファイル内で、復元される `noiseShaperType` を**無検証 cast** している。
  一方 `requestLoadState` 側（`StateIO.cpp:107-113`）は同一 field に**範囲ガードがある**。
  → **restore 経路間の parity 不整合。**
- ガード側が存在しないことも確認済み:
  `AudioEngine.Parameters.cpp:439-448` の `setNoiseShaperType` は `!=` による変化判定のみで
  範囲正規化を持たない（D12-1 の `setDitherBitDepth` と同型）。
- `RuntimePublicationValidator.cpp:128-131` は `ns < 0 || ns > 3` で reject するため、
  **D12-1 と同一の「不正値の永続化 → publish が stuck 拒否」構造**になる。
- D7 / B-3 は `StateIO.cpp` の guard を対象としたものであり、`DeviceSettings.cpp` は対象外だった。
  `DeviceSettings.cpp` には B-3 由来の guard が 1 箇所も存在しない。→ **再発ではなく新規 defect。**

### 3.1 対応範囲の広さ（要 Owner 判断）

`DeviceSettings.cpp` は XML 属性を engine に流し込む境界であり、D12-2 と同型の無検証復元が
他にないかは未だ網羅走査していない。案は 2 つ。

- 案 1: 本件のみを独立 repair unit とする（最小・低リスク）
- 案 2: `DeviceSettings.cpp` の復元系を一度列挙し、同型をまとめて repair unit にする

D12-2 の深刻度は D12-1 と同等だが、**到達頻度は低い**（device-settings XML はユーザーが明示的に
設定保存操作を行わないと生成されず、session より改変されにくい）。優先度は D12-1 が上。

---

## 4. R1 記録の訂正

R1（`P1-5-IR-P2_STG-11-D12_FRESH-DISCOVERY_20261001.md`、本 R2 より未 commit）は
`ditherBitDepth` を次の根拠で **SAFE** と判定していた。

> `ditherBitDepth | S→無検証 store→atomic | publish は validator{0,16,24,32} が拒否 |
> placeholder prepare は bypass 付き | DSP 到達は validator 通過分のみ | SAFE`

この判断は **不完全**である。validator は world を拒否するが、
**拒否される値そのものが `setDitherBitDepth` によって engine の atomic に先に永続化されている**
（`Parameters.cpp:394-395`）ため、validator は「無効入力を無害化する filter」ではなく
**「不正値を永続化した状態で publish を stuck 拒否させる機構」** に変わる。
R1 は **DSP 到達のみ**を評価し、**不正値の永続性と publish の持続拒否**を評価していなかった。

本 R2 がこれを訂正する。

---

## 5. その他の audit 面

### 5.1 R2 で実コードを再追跡し、判断した項目

| field / 面 | 再追跡で確認した内容 | 判定 |
| --- | --- | --- |
| `saturationAmount` | `setSaturationAmount`（`Parameters.cpp:555-563`）は `jlimit(0,1)`。`jlimit` は NaN に対して NaN を返すため setter 単独では無防備。preset gate `:112` は `hasFiniteDouble(0.0,1.0)` で有限性と範囲を検査するが **caller ゼロ**（§2.3.1） | **要 Owner 再評価**（§6-3） |
| `inputHeadroomDb` | `setInputHeadroomDb` は `jlimit` で clamp（`Parameters.cpp:370-376`）。preset gate `:113` に `hasFiniteDouble` あり（ただし dead） | NO-GO（gate が dead である点は §6-3 の論点） |
| `outputMakeupDb` / `convolverInputTrimDb` | preset gate `:114-115` に `hasFiniteDouble` あり（dead） | NO-GO（同上） |
| `coeffSafetyMargin` / `cmaesRestarts` | preset gate `:122-123` に `hasFiniteDouble(0,2)` / `hasIntRange(0,1000)` あり（dead） | NO-GO（同上） |
| `oversamplingFactor` | `setOversamplingFactor`（`Parameters.cpp:571-577`）は `1/2/4/8` のみ許可し他は 0 に正規化。`validateResources:119-121` も 2 のべき乗 1〜16 を要求 | NO-GO |
| `getAdaptiveBitDepthIndex` | `Learning.cpp:377-382`。戻り値は常に `[0,2]`。`getAdaptiveCoeffBankForIndex`（`:392-398`）も index を clamp | NO-GO（OOB なし） |
| `processingOrder` / `analyzerSource` / `noiseShaperType` / `oversamplingType` / 3 filter mode（StateIO 経路） | inline ガードの存在を確認（§2.3.2） | NO-GO（D7 / B-3） |
| adaptive 係数の session 保存 / 復元 | `StateIO.cpp:260-270` の 10 SR bank ループ。`DeviceSettings.cpp:1143-1150` も `kAdaptiveBitDepthCount` でクランプ済み | NO-GO |

### 5.2 本 R2 で再追跡していない項目（R1 判定の暫定承継）

R1 が SAFE と判定した残りの項目（`phaseMode` / `tailMode` / `nucHC` / `nucLC` /
`filterStructure` / `tailL1L2Multiplier` 等）は、本 R2 では個別に再走査していない。
R1 の判定を**暫定的に承継する**が、以下の留保を付す。

- R1 の誤りが表面化した `ditherBitDepth` は、
  「session に含まれる」「publish が hard reject する」「その上で永続化される」の
  3 条件が揃う項目である。本 R2 は、この 3 条件を満たす別の session フィールドが
  残っていないことを確認していない。
- §5.1 で再追跡した項目の中に、preset gate（§2.3.1）が caller ゼロであるために
  setter 単独の無防備が残る例（`saturationAmount` の NaN 経路）が存在した。
  この型の残存が §5.2 の項目にないかは**未検証**である。

したがって **§5.2 は「NO-GO 確定」ではなく「R1 判定の暫定承継」** とし、
D13 以降または別 audit で §6-3 と併せて再走査する。

---

## 6. 未確定事項（要 Owner 判断 / 追加調査）

1. **guard の設置位置**（§2.6）。`setDitherBitDepth` 一元化（案 B）を推奨。
2. **D12-2 を別 repair unit とするか D12-1 と同じ unit に含めるか**、および
   `DeviceSettings.cpp` の復元系を網羅走査するか（§3.1）。
3. **`validatePresetStateTreeForDebug` が caller ゼロであることの本質的な扱い。**
   本 R2 の D12-1 / D12-2 は「個別に inline guard を足す」案を前提にしているが、
   本来は field 追加時の漏えいを構造的に防ぐ**能動的な whitelist gate** を復元するほうが
   根本的である可能性が高い。D7 の test ヘッダ（`STG11D7StateEnumGuardTests.cpp:13`）が
   この問題を既に指摘している。**これは D12 の defect ではなく設計上の未決事項**として扱う。
4. **UI 側の dither bit depth コントロールが合法値のみを返すことの実地確認。**
   本 R2 は session / XML 経路のみを追跡した。
5. **`kAdaptiveBitDepthValues`（`{16,24,32}`）と validator（`{0,16,24,32}`）の同期契約。**
   将来 bit depth を増やす際に両者が drift しないことを保証する仕組みは現状ない
   （コメントでの注記のみ）。本 audit の defect ではないため設計事項として保持。

---

## 7. 本 commit の内容

```text
src/ 配下の production / test / build ファイルの変更: 0 件
追加ファイル: audit document 2 件のみ
  - P1-5-IR-P2_STG-11-D12_FRESH-DISCOVERY_20261001.md     （R1。NO-GO 記録。§4 で訂正済み）
  - P1-5-IR-P2_STG-11-D12_FRESH-DISCOVERY-R2_20261001.md  （本 R2。現行ソースでの再監査・現行の判定）
ConvoPeq.md: 再生成しない（production 変更が無いため）
D13 以降 / D18-CLOSE / D19 / D20: 本 commit に含まれない
```
