# STG-11-D12-R1 — repair 前再監証（Session-State Boundary / Integer Domain）

- Document: `doc/work113/P1-5-IR-P2_STG-11-D12-R1_REVERIFICATION_20261001.md`
- Work item: **STG-11-D12-R1** — D12-1 / D12-2 の defect chain を現行ソースで再証明
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（監査 AI が参照する `ConvoPeq(20261001-105209).md` と同一ファイル）
  - baseline commit: `4dc1accc1d003142bfb2df5db8f64b18cb8c59db`（D12 audit commit）
  - `Generated: 2026-10-01 21:30:09` / 5,892,101 B / 129,811 lines
  - SHA-256: `A455FDA78220DE2A3B57C08369D3EADAC5ACD2FE4E396B77FB60A377A0C3518F`
  - D12-R1 は production 変更を伴わないため `ConvoPeq.md` も変更なし。
- 本 work item は **read-only**。**production / test / build ファイルの変更は一切行っていない。**
- 本書は **未 commit**（Owner の repair GO 判断待ち）

---

## 0. 結論

```
D12-1 : 再証明 OK（defect として成立）。ただし根拠の 1 点に事実誤認があった。
D12-2 : 再証明 OK（defect として成立）。
新規   : D12-3（UI が validator の合法集合外値を生成しうる）
        D12-4（合法集合を固定するテストが orphan でビルドされない）
```

**Owner のご指摘は正しかった。私の D12 audit 報告（`4dc1accc`）に事実誤認がある。**

誤りは §2.3.1 の次の主張である。

> 「**`ditherBitDepth` はこのリストに無い**（`noiseShaperType` 等の隣接項目は含むが、本項目は欠落）。」

実際は `src/audioengine/AudioEngine.Parameters.cpp:132-137` に次の検査が存在する。

```cpp
    if (state.hasProperty("ditherBitDepth"))
    {
        const int bitDepth = static_cast<int>(state.getProperty("ditherBitDepth"));
        if (bitDepth <= 0 || bitDepth > 64)
            return fail("Preset property out of range: ditherBitDepth");
    }
```

原因は私の査読不足である。`validatePresetStateTreeForDebug` を 74-126 行まで読んだところで
打ち切っており、続く 127-137 行（`oversamplingFactor` と `ditherBitDepth` の個別検査）を
読み飛ばしていた。推測で補完してはならない箇所で行を区切って残りを省略した。

副次的な訂正も 1 点。初めから最後まで読むべきだった表の記述が粗かった。

- 当初報告には「`ditherBitDepth` だけがチェックから漏れている」という含意があったが、
  `oversamplingFactor` も同じ関数で個別検査されている（`:125-130`）。
  「漏れている」ではなく「同関数の別個の if 節で検査されている」が正確な記述である。

**ただし、この誤りは D12-1 の結論を壊さない。むしろ真因を正確に言語化する。**
その根拠と全新 6 箇所の合法値集合を §1 に示す。

---

## 1. `ditherBitDepth` の合法値集合の確定（Owner 指示 2 / 3）

Owner の指示「合法値が `{0,16,24,32}` なのか、validator が許容する整数範囲が別なのかを
明確化せよ。推測で補完するな」に従い、現行ソースに存在する**全 6 箇所**の集合を抽出した。

| # | 出典 | 集合 | 実行状態 | 役割 |
| --- | --- | --- | --- | --- |
| S1 | `RuntimePublicationValidator.cpp:124-126` `validateResources` | `{0, 16, 24, 32}` | **LIVE**（publish 経路） | 唯一の実行される強制点。範囲外は `return false` |
| S2 | `DeviceSettings.cpp:727-741` `updateBitDepthList` および `:780` | `{0} ∪ {16,24,32} ∪ {device->getCurrentBitDepth()}` | **LIVE**（UI） | 値を生成する側。**S1 より広い** |
| S3 | `AudioEngine.h:14` `kAdaptiveBitDepthValues` | `{16, 24, 32}` | LIVE（データ） | adaptive 係数バンクの 3 段。S1 はこれに 0 を加えたもの |
| S4 | `AudioEngine.Parameters.cpp:132-137` `validatePresetStateTreeForDebug` | `1..64`（閉区間） | **DEAD**（caller ゼロ） | 実行されない。**0 を却下**し、**8 や 64 を承認**する |
| S5 | `AudioEngine.Processing.DSPCoreIO.cpp:364` `applyDither = (ditherBitDepth > 0)` | 「0 より大きい」すべて | LIVE（RT 判定） | 上限の notion が無い。RT は合法集合の概念を持たない |
| S6 | `SnapshotParams.h:34` / `GlobalSnapshot.h:44` / `AudioEngine.h:2575` / `AudioEngine.h:5214` | 24 / 24 / 0 / 24 | LIVE（既定値） | `m_currentDitherBitDepth` は 24、`ditherBitDepth` は 0（「未初期化」コメント） |

### 1.1 確定した authoritative 合法集合

**`{0, 16, 24, 32}`** を authoritative とする。根拠は 3 点。

1. これが**唯一の LIVE な強制点**（S1）であり、publish の受否を実際に決定している。
2. S3（adaptive バンク表）と整合する。0 は adaptive 無効の番兵、16/24/32 は 3 段。
3. S4 の dead 検査 `1..64` は S1 と**矛盾**している。0（Off / auto）を却下しながら、
   S1 が却下する 8 や 64 を承認する。**S4 は現行契約ではない。**

### 1.2 ただし S1 と S2 は矛盾する（新規所見 D12-3）

`DeviceSettings::updateBitDepthList` は UI 選択肢を次のように構築する。

```cpp
    supportedBitDepths.add(16);      // :730
    supportedBitDepths.add(24);      // :731
    supportedBitDepths.add(32);      // :732
    // さらに:
    int current = device->getCurrentBitDepth();                    // :738
    if (current > 0 && !supportedBitDepths.contains(current))
        supportedBitDepths.add(current);                            // :740
```

その後、`:780` で最大値を engine に適用する。

```cpp
    if (maxBitDepth > 0)
    {
        bitDepthComboBox.setSelectedId(maxBitDepth, juce::dontSendNotification);
        audioEngine.setDitherBitDepth(maxBitDepth);   // :780
    }
```

したがって **デバイスの `getCurrentBitDepth()` が 32 より大きい値（例: 64）を返した場合**、
`maxBitDepth` は 64 となり、通常のデバイス起動だけで `setDitherBitDepth(64)` が呼ばれる。
64 は S4（dead）では承認されるが S1 では却下される。
さらに UI コンボ自体に 64 が並ぶため、ユーザーが選べば同じ状況に入る。

**これは破損した session を介さない、通常の運用で到達しうる経路である。**
D12-1 の到達性は「session / XML の改変」だけに依存しない。

### 1.3 「既存 tests」についての訂正（Owner 指示 2 の最終項）

`src/tests/PublicationValidatorIsolationTests.cpp:187` は
`world.resource.ditherBitDepth = 8;  // 0,16,24,32 以外` として
S1 の却下契約を固定している。**しかしこのファイルは CMakeLists.txt から参照されておらず、
どのターゲットにもコンパイルされていない。**

実測:

```text
src/tests/*.cpp total      : 70
referenced in CMakeLists   : 68
NOT referenced             : 2
  ORPHAN: src/tests/MT-NUPC-Measurement.cpp
  ORPHAN: src/tests/PublicationValidatorIsolationTests.cpp
```

`build/build*.ninja` にも `PublicationValidatorIsolation` の参照は 1 件も無い。

したがって **`{0,16,24,32}` を固定する唯一のテストは dead である。**
実行されるテストのいずれにも、dither の合法集合を固定する記述は無い。
これは新規所見 **D12-4** として分離する（§6）。

---

## 2. D12-1 の defect chain 再証明（Owner 指示 1 / 4）

### 2.1 一本のデータフロー（現行ソースの行番号）

```text
[hop 1] setter 入口      AudioEngine.Parameters.cpp:380-397
        void AudioEngine::setDitherBitDepth(int bitDepth)
        {
            if (consumeAtomic(ditherBitDepth, acquire) != bitDepth)   // :382 変化判定のみ
            {
                ...
                convo::publishAtomic(ditherBitDepth,          bitDepth, release); // :394
                convo::publishAtomic(m_currentDitherBitDepth, bitDepth, release); // :395
                submitRebuildIntent(Structural, ...);                            // :397
            }
        }
        → 範囲検証・clamp・assert なし。渡された値を無条件に publish する。

[hop 2] authoritative storage
        AudioEngine.h:2575   std::atomic<int> ditherBitDepth { 0 };
        AudioEngine.h:5214   std::atomic<int> m_currentDitherBitDepth { 24 };
        → publish 判定より先に書かれる。却下されても残る。

[hop 3] snapshot 取り込み   AudioEngine.RebuildDispatch.cpp:43-59
        captureBuildParameterSnapshot(const AudioEngine& engine)
          :46  snapshot.ditherDepth = consumeAtomic(engine.ditherBitDepth, acquire);
        → :605  task.buildInput.ditherBitDepth = paramSnapshot.ditherDepth;
        ※ この関数は再構築のたびに呼ばれる。値の復元は行わない。

[hop 4] 新 runtime の prepare   RuntimeBuilder.cpp:474-480
        runtime->prepare(in.sampleRate, in.blockSize, in.ditherBitDepth,
                         in.oversamplingFactor, ...);
        → DSPCoreLifecycle.cpp:212 / :280  this->ditherBitDepth = bitDepth;
        → DSPCoreIO.cpp:364  const bool applyDither = (ditherBitDepth > 0);
        ※ この runtime は publish 失敗時に rollback される（hop 7 参照）。

[hop 5] world への write        RuntimeBuilder.cpp
        :257  const bool useSealedSnapshot = (sealedSnapshot != nullptr);
        :349-355  if (useSealedSnapshot)
                      worldOwner->resource.ditherBitDepth = sealedBuildInput.ditherBitDepth;  // :352
        :356-365  else
                      worldOwner->resource.ditherBitDepth = (current != nullptr)
                          ? current->ditherBitDepth : 0;                                       // :360-361
        ※ どちらの分岐でも不正値は world に到達する。
          sealed 経路の入力は hop 3 の atomic。
          非 sealed 経路の入力は DSPCore::ditherBitDepth で、それは hop 4 により atomic 由来
          （PrepareToPlay.cpp:267-273 の
            placeholderDSP->prepare(safeSampleRate, bufferSize,
                convo::consumeAtomic(ditherBitDepth, acquire), ...) で確認できる）。
        ※ Timer.cpp:1160-1165 は &currentBuildSnapshot_ を渡すため sealed 経路が live。
          currentBuildSnapshot_ は AudioEngine.Commit.cpp:818 で commit 時に更新される。

[hop 6] publish 時の validation   RuntimePublicationValidator.cpp:113-134
        if (!validateResources(world)) { result.isValid = false; ... return result; }   // :30-35
          :123-126
            // Dither: 0, 16, 24, 32 のみ許容（kAdaptiveBitDepthValues との整合性）
            const int dd = resource.ditherBitDepth;
            if (dd != 0 && dd != 16 && dd != 24 && dd != 32)
                return false;

[hop 7] 却下と rollback          src/core/RuntimePublicationCoordinator.h:118-126
        if (!bridge_.validatePublicationNonRt(*worldOwner))
        {
            auto* rejectedWorld = const_cast<World*>(worldOwner.release());
            bridge_.retireRejectedRuntimeWorldNonRt(rejectedWorld);
            return PublishStageResult::Rejected;     // world は破棄、runtime は更新されない
        }
        → PublicationExecutor.cpp:69-78
            if (!isCommitted(result.stage)) {
                Logger::writeToLog("[PUBLISH] commitRuntimePublication FAILED ...");
                return PublishResult::PublishFailed;   // ここで何の復旧も行わない
            }
        → 新 DSP は呼び出し元が DSPLifetimeManager::destroyRolledBackDSP で物理解放する
          （PrepareToPlay.cpp:317-321 が確認済みの rollback 実例）

[hop 8] publish failure 後の状態保持
        - hop 2 の atomic は範囲外値のまま。元の値へ戻す code パスは存在しない。
          setDitherBitDepth の呼び出し元は DeviceSettings.cpp:281/283/780/785/1137/1283 と
          MainWindow.cpp:694/876 のみで、いずれも別の値を渡す新規 publish であり、
          範囲外値を 0/16/24/32 に戻す経路は無い。
        - 既に publish 済みの world は hop 7 で変更されないため、音は旧設定で継続する。
```

### 2.2 blocked 範囲の正確な限定

「後続の正常 parameter change が全部止まる」は**正確ではない**。
runtime には 2 本の独立した経路がある。

| 経路 | publish validator を通るか | 不正値の影響 |
| --- | --- | --- |
| rebuild から publish（`submitRebuildIntent` 経由） | **通る**（hop 5 → 6） | **恒久的に却下**。IR / EQ / filter mode / oversampling 等の構造的変更が反映されない |
| RT snapshot（`SnapshotParams` 経由、`AudioEngine.Snapshot.cpp:40-53`） | 通らない | `ditherBitDepth` は `SnapshotParams` に含まれる（`SnapshotParams.h:34`）が、RT 側は参照しない。`SnapshotFactory.cpp:99/137` は変更検知と hash のみ。飽和量・headroom・bypass 等はこの経路で依然として到達しうる |

したがって害の正確な記述は次のとおり。

> `ditherBitDepth` に範囲外値が入り込むと、**rebuild / publish 経路が恒久的に閉じられる**。
> その間、IR を含む構造的パラメータの変更が反映されない。
> snapshot 経路で配送されるパラメータは影響を受けない。
> 出力は旧 world のまま継続し、原因を名づけるユーザー向けメッセージは無い。

**「何seisNovo 具体的に塞がるか」は repair 時に Owner が要求された
「invalid input を却下した後、後続の正常変更が正常に publish される」regression test で
確定させる。本 R1 では snapshot 経路と rebuild 経路の分離のみを確定した。**

### 2.3 §1.2 の経路 — 到達性の上振れ

`DeviceSettings.cpp:780` 経由なら、**改変された session なしに到達する**。
当初報告は破損 session を前提としたが、これにより到達条件は緩くなり、深刻度は上がる。

### 2.4 動的確認（Owner 指示 4 の「テストで証明」部分）

- **S1 の却下契約（hop 6）を固定するテストは dead**（§1.3）。実測 PASS の証拠は無い。
- 登録済みの関連テスト 2 件は PASS した（Owner 承認済み 45/45 の一部）。
  ```text
  ctest -C Debug -R 'PublicationValidatorIsolation|BuildInputSemantic|RuntimeWorldAuthorityProjection'
    RuntimeWorldAuthorityProjectionContract ... Passed
    BuildInputSemanticContract ............... Passed
    100% tests passed out of 2
  ```
  ただし要求した `PublicationValidatorIsolation` は未登録であり、実行されていない。
- **hop 1 から 5、および 7 から 8 は静的追跡のみ。** 既存テストに該当する検証が無い。
  動的証明は repair commit で追加するテストが担う（本 R1 は read-only のため未実施）。

---

## 3. D12-2 の再監証（Owner 指示 5）

### 3.1 二つの restore 経路の対比

| 項目 | `StateIO.cpp`（session / preset） | `DeviceSettings.cpp`（XML） |
| --- | --- | --- |
| 復元位置 | `:107-113` | `:1139-1141` |
| 読み出し | `state.getProperty("noiseShaperType")` | `xml->getIntAttribute("noiseShaperType", 0)` |
| 検査 | `raw >= Psychoacoustic(0) && raw <= Fixed15Tap(3)`。範囲外なら何もしない | **検査なし**。`(NoiseShaperType)shaperType` に無検証 cast |
| 保存側 | `:240` `setProperty(..., (int)consumeAtomic(noiseShaperType, acquire))` | `:1013` `setAttribute("noiseShaperType", (int)engine.getNoiseShaperType())` |
| 保存空間の整合 | enum 空間 | enum 空間（**combo id ではない**。§3.2 参照） |
| 同一保存値に対する semantic parity | 合法値では両者とも同一 enum を適用 | 同左 |
| 不正 enum の到達 | **到達しない**（guard が遮断） | **setter へ到達する** |

### 3.2 UI id と enum 値が異なる件（重要・誤解防止）

`DeviceSettings.cpp:293-306` の noise shaper コンボは、**id 空間が enum 空間と一致していない**。

```cpp
    noiseShaperComboBox.addItem("4th-order", 1);          // id 1
    noiseShaperComboBox.addItem("12th-order", 2);         // id 2
    noiseShaperComboBox.addItem("15th-order", 3);         // id 3
    noiseShaperComboBox.addItem("9th-order adaptive", 4); // id 4
    onChange = [this] {
        id == 1 -> NoiseShaperType::Fixed4Tap         (enum 1)
        id == 2 -> NoiseShaperType::Psychoacoustic   (enum 0)   <- id 2 が enum 0
        id == 3 -> NoiseShaperType::Fixed15Tap        (enum 3)
        id == 4 -> NoiseShaperType::Adaptive9thOrder  (enum 2)   <- id 4 が enum 2
    };
```

**保存と復元は combo id ではなく enum 値を書き出す**（`:1013` が `getNoiseShaperType()`、
`:1140` がそれを enum として読む）。したがって合法値での save / restore は正しく、
D12-2 の問題は「不正値が無防備」だけに限定される。
UI id と enum を混同した defect は無いことを明示的に確認した。

### 3.3 不正 enum から却下までの chain

```text
DeviceSettings.cpp:1140-1141   int shaperType = xml->getIntAttribute("noiseShaperType", 0);
                               engine.setNoiseShaperType((NoiseShaperType)shaperType);   // 無検証
→ AudioEngine.Parameters.cpp:439-448
    void AudioEngine::setNoiseShaperType(NoiseShaperType type)
    {
        if (consumeAtomic(noiseShaperType, acquire) != type)      // :441 変化判定のみ
        {
            publishAtomic(noiseShaperType, type, release);         // :443 無条件
            publishAtomic(m_currentNoiseShaperType, type, release);// :444
            publishAtomic(m_pendingNSChange, true, release);       // :445
            ...
        }
    }
    → 範囲正規化なし。D12-1 と構造的に同一。
→ RebuildDispatch.cpp:49  snapshot.noiseShaperType = consumeAtomic(engine.noiseShaperType, acquire);
→ RuntimeBuilder.cpp:353  worldOwner->resource.noiseShaperType = sealedBuildInput.noiseShaperType;
→ RuntimePublicationValidator.cpp:128-131
    const int ns = resource.noiseShaperType;
    if (ns < 0 || ns > 3) return false;
→ hop 7 と同じ却下と rollback。
→ atomic は範囲外値のまま残る（sticky）。
```

**`validateResources` の受理集合 `0..3` は `NoiseShaperType`（`src/core/Types.h:23-28`）と
完全に一致する。** したがって D12-1 と異なり合法集合の不整合はなく、
単に setter 境界に guard が無いという構造的欠落だけである。

### 3.4 D12-2 の判定

- **新規 defect である。** D7 と B-3 は `StateIO.cpp` の guard を対象としたものであり、
  `DeviceSettings.cpp` は対象に含んでいない（同ファイルに B-3 由来の guard は 1 箇所も無い）。
- 深刻度は D12-1 と同等。ただし**到達頻度は D12-1 より低い**
  （device-settings XML はユーザーが明示的に保存操作を行わないと生成されない）。
- 合法集合の不整合が無いので、合法集合の確定作業は不要で setter 境界 guard のみが課題になる。

---

## 4. `validatePresetStateTreeForDebug` の位置づけ（Owner 指示 6）

Owner の指示「caller zero だから直ちに repair 対象とはしない」に従い、
production correctness に**必要**な gate なのか **debug-only helper** なのかをコードで確定した。

### 4.1 事実

| 観点 | 実測 |
| --- | --- |
| 宣言 | `AudioEngine.Parameters.cpp:74` `[[maybe_unused]] bool validatePresetStateTreeForDebug(const juce::ValueTree& state, juce::String* reason = nullptr)` |
| 呼び出し元 | **0 件**。`rg validatePresetStateTreeForDebug src/` は宣言の 1 件のみ |
| `#if DEBUG` による囲み | **無い**（anonymous namespace 内の free function。debug 限定コンパイルではない） |
| 同名の兄弟 | `validateConvolverStateTreeForDebug`（`:13`）は **3 箇所から呼ばれている**（`:146` `:741` `:753`）。`ForDebug` 接尾辞は「caller ゼロ」を意味しない |
| 引数 | 失敗理由を返す `reason` 出力。診断用の性格 |
| 契約の鮮度 | `ditherBitDepth` を `1..64` としており、S1（`{0,16,24,32}`）・S3（`{16,24,32}`）・S5 と**いずれも矛盾**。0 を却下する |

### 4.2 判定

**現時点では dead な診断用 helper であり、production correctness の gate ではない。**

- 実行されないので、不正値の防止に何らかの影響も与えていない。
- `#if DEBUG` でも囲まれていないため、debug ビルドでのみ動くわけでもない。
  つまり dead code であり、限定的な debug 契約ではない。
- 同じファイル内の `validateConvolverStateTreeForDebug` が production 経路で使われていることから、
  この系統の helper が debug-only であるという慣習はコード中に存在しない。

### 4.3 したがって Owner 指示に従い、以下を別問題として切り分け、D12-1 / D12-2 の repair 対象から除外する

1. `validatePresetStateTreeForDebug` を production gate として復活させるか。
2. その場合、`ditherBitDepth` の範囲を `1..64` から S1 の契約に一致させるか。
3. 現状の `1..64` のまま gate 化すると、**Off（0）を含む合法 preset を拒否する**ため壊れる。

これらは D12-1 / D12-2 の repair と独立した設計事項である。
本 R1 では判定のみを確定させ、実装には進まない。

---

## 5. repair 方針の検討（実装は行わない。Owner 判断待ち）

### 5.1 D12-1

確定的要件は、authoritative 合法集合が `{0, 16, 24, 32}` であること（§1.1）。

Owner が第一候補として挙げた「`setDitherBitDepth` 入口での一元化」の評価:

| 観点 | 評価 |
| --- | --- |
| restore から setter への parity | **成立する**。`StateIO.cpp:101` と `DeviceSettings.cpp:1137` と `MainWindow.cpp:694/876` と UI が同じ判定を通る |
| `DeviceSettings.cpp:780`（UI が 64 を書く経路） | **まとめて保護される**。setter で弾けば publish の恒久的却下は起きない |
| 既存契約との整合 | **適合する**。D7 と D10 は「範囲外なら何もしない（現在値維持）」という同一の契約である（`EQProcessor.Parameters.cpp` の `if (type != ... ) return;` と同型） |
| 無効値を黙って落とすことによる診断性の低下 | 軽微。UI 経路が同じ guard を通るので「値が変わらないこと」自体が観測可能な症状になる |
| RT path への影響 | **無い**。setter は NonRT 側のみ |

**ただし §1.2（D12-3）が repair の範囲を広げる。**
setter だけで弾くと、UI コンボが 64 を選択状態に持ったまま engine 側は 32 等を保持し、
**UI と engine が乖離**する。したがって以下が併せて必要になる。

- **案 α**: `updateBitDepthList` で UI の選択肢を S1 の集合に制限する
  （device depth が 32 を超えた場合は 32 を上限とする、または当該値を Off / auto 側へ落とす）。
- **案 β**: UI が S1 に無い値を選ぶ状況を `updateBitDepthList` 内で正規化し、
  最終判断は `setDitherBitDepth` の一元 guard に委譲する。
- **案 γ**: 本 repair の範囲を setter 一元化に限定し、UI の乖離は別 ticket として残す。

Owner が挙げられた「`StateIO.cpp` 限定 guard」案は、**採用しない**。
setter 一元化の方が範囲と parity の両点で優れるという評価は §0 の訂正後も変わらない。

### 5.2 D12-2

Owner 方針「restore caller ごとに異なる validation semantics を持たせず、
可能な限り setter 境界で不正 enum を拒否する」に**そのまま適合**する。

- `setNoiseShaperType`（`Parameters.cpp:439`）の入口に範囲判定を置く。
- 受理集合 `0..3` は `NoiseShaperType` と完全一致（`Types.h:23-28`）するため、
  新しい集合定義は不要。
- **D7 の inline guard（`StateIO.cpp:107-113`）は削除しない。**
  Owner 指示「既存 D7 / B-3 / D10 / D11 の修正契約を変更しない」に従い、
  多重防御として温存する。contract test `STG11D7StateEnumGuardTests` の
  negative control も無傷である。
- 同様に `DeviceSettings.cpp:1141` の無検証 cast も、setter guard があれば到達不能になるため、
  **本 repair では書き換えない**（最小差分）。
  ただし「cast が残るが到達しない」状態をレビューで明確にするため、
  コメントの追記のみを検討対象とする。

### 5.3 テスト計画（Owner 指示のテスト要件への対応案）

| # | テスト | 形式 | 備考 |
| --- | --- | --- | --- |
| T1 | invalid dither（`8`, `-1`, `64`, `99`）を restore しても値が変わらない | harness sub-test | D7 test と同じ形 |
| T2 | 合法値（`0`, `16`, `24`, `32`）の round-trip | harness sub-test | |
| T3 | **invalid 投入後に正常 parameter change が publish できること** | harness sub-test | Owner の明示要求。hop 7 の sticky 性を直接固定する最重要テスト |
| T4 | invalid noiseShaper（`-1`, `4`）を XML 経路で投入しても値が変わらない | harness sub-test | D12-2 |
| T5 | valid noiseShaper enum（`0` から `3`）の round-trip | harness sub-test | D12-2 |
| T6 | D7 から D11 までの regression | 既存 sub-test | 既存 45/45 |
| T7 | **S1 の却下契約を実行可能なテストとして追加する** | 新規追加または orphan の登録 | D12-4。現状は合法集合がテストで固定されていない |
| T8 | Debug CTest 45/45（T7 で 46 になる見込み） | 全件 | Owner は 45/45 を要求。テスト数が変わるため Owner へ確認する |
| T9 | Release CTest と同数 | 全件 | 同上 |

**T3 が本 repair の核心である。** D12-1 の害は「不正値が永続化して publish が閉じること」なので、
「却下した後に後続の正常変更が publish できること」を証明できれば、根本的に閉じたとみなせる。

---

## 6. 本 R1 で確定した新規所見（D12-3 / D12-4）

### D12-3: UI が S1 の合法集合外値を生成しうる

- `DeviceSettings.cpp:738-740` が device の `getCurrentBitDepth()` を UI の選択肢に加える。
- `:780` がその最大値を engine に渡す。
- device depth が 32 を超える場合、`resource.ditherBitDepth` が S1 の集合外となり、
  破損のない通常運用で publish の恒久的却下に至る（§1.2 / §2.3）。
- これは §5.1 の repair 範囲に影響する（案 α / β / γ の選択）。

### D12-4: 合法集合を固定するテストが orphan

- `src/tests/PublicationValidatorIsolationTests.cpp` は 20,217 B で実在するが、
  CMakeLists.txt から参照されておらずビルドされない（§1.3）。
- このファイルは `world.resource.ditherBitDepth` に `8 / 0 / 16 / 24 / 32` を代入する
  一連のケースを持ち、S1 の契約を忠実に符号化している。
  CMakeLists.txt に登録するだけで S1 がテストで固定される。
- D12 audit（`4dc1accc`）はこの orphan テストを既存テストの証拠として引用していた。
  **引用先自体は正しかったが、「テストが passing である」と読める書き方だった。**
  本 R1 でその誤読の可能性を除去する。
- 同種の orphan はもう 1 件 `src/tests/MT-NUPC-Measurement.cpp` があるが、
  測定ツールであり本件とは別系統である。

### 参考（deferred。本 R1 の defect ではない）

- `AudioEngine.h:2575 ditherBitDepth { 0 }` と `:5214 m_currentDitherBitDepth { 24 }` の
  既定値不一致。`ditherBitDepth` 側のコメントが「0 = 未初期化」と性質付けているため
  意図された可能性がある。setter guard の判断材料となるが、defect として計上しない。
- `GlobalSnapshot::ditherBitDepth` は snapshot に載るが RT から参照されない（§2.2）。
  死重量であり、本 R1 の defect ではない。

---

## 7. 本 R1 で確定しなかった事項（要継続調査）

1. **blocked 範囲の厳密な境界**（§2.2）。どのユーザー可視パラメータが塞がるかは
   repair の T3 で確定させる。
2. **D12-3 の UI 乖離の正解**（案 α / β / γ）。Owner 判断。
3. **`validatePresetStateTreeForDebug` の最終扱い**（§4.3）。
   D12-1 / D12-2 の外側にある設計事項として保持する。
4. **`setSaturationAmount` の NaN 経路**（D12 audit §5.1 で暫定項目として指出した）。
   `juce::jlimit(0,1,NaN)` は NaN を返し、`validatePresetStateTreeForDebug` は dead であり、
   `validateResources` は saturation を検査しない。ここは再監証できていない。
   Owner の D12 テーマ（session-state / integer domain）から外れるが、
   whitelist gate が dead であることの帰結として未解決のまま残る。優先度は低い。
5. **`oversamplingFactor` の UI 経路**（§0 で訂正した点）。
   `StateIO.cpp:241-243` のコメントは oversampling を preset に保存しないと明記しているが、
   実際の getter と saver が一致しているかは未検証。

---

## 8. 本 R1 の作業記録（read-only 遵守）

```text
変更ファイル: 0 件（production / test / build / CMakeLists.txt / ConvoPeq.md すべて無変更）
追加ファイル: 本書 1 件（未 commit。Owner の repair GO 判断待ち）

実施した操作:
  - 読み取り（git show / rg / Read）: 多数
  - 動的確認: ctest -C Debug -R '...|BuildInputSemantic|RuntimeWorldAuthorityProjection' -> 2/2 PASS
  - ビルド: なし（ソース未変更のため不要）
  - reset / clean / rebase / amend / squash / force push: いずれも不使用

main checkout (C:\VSC_Project\ConvoPeq): 未改変（HEAD = bda43034）
作業 worktree: HEAD = 4dc1accc、未追跡 1 件（本書のみ）、commit 済み変更なし
```
