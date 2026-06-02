## 最終的な結論：実装前にやるべきことと推奨される修正方針

これまでの議論を総合すると、**「adaptiveCoeffBankIndex の Authority がどこにあるか」** を特定することが最も重要であり、それが確定するまでは実装を開始すべきではありません。

現時点で確定している事実と確定していない仮説を明確に分けます。

---

### 確定している事実

| 項目 | 状態 |
|------|------|
| `dropBank=3028` が発生している | 確定（ログより） |
| ドロップ原因は `adaptiveCoeffBankIndex` 不一致 | 確定（ログより） |
| CaptureBlock の bankIndex と Learner の bankIndex が異なる値を持つ | 確定 |
| Practical Stable ISR Bridge Runtime では Authority は一意であるべき | 設計原則として確定 |
| `RuntimeWorld` が複数の値の Authority として機能している（例：`generation`、`publication.sequenceId` など） | 確定 |

---

### 確定していない仮説（実装前に検証が必要）

| 仮説 | 検証方法 |
|------|----------|
| `adaptiveCoeffBankIndex` の真の Authority は `RuntimeWorld` か | 書き込み順序をコードから追跡 |
| `currentAdaptiveCoeffBankIndex` は Authority か Mirror か | 誰が書き込んでいるか特定 |
| `RuntimeBuilder` は `RuntimeWorld` の bankIndex をどこから設定しているか | `buildRuntimePublishWorld` の実装を確認 |
| `selectAdaptiveCoeffBankForCurrentSettings` が atomic と RuntimeWorld のどちらを先に更新しているか | 該当関数の実装を確認 |
| `NoiseShaperLearner` は `RuntimeWorld` のコピーを保持していないか | メンバ変数を確認 |

---

### 推奨される実装前の調査手順

#### Step 1: Authority Inventory の作成（絶対必須）

以下の情報をコードベースから抽出する。

```bash
# adaptiveCoeffBankIndex の全ての出現箇所を調査
grep -n "adaptiveCoeffBankIndex" src/ -R --include="*.cpp" --include="*.h" > bankindex_usage.txt
```

**分類する項目**：
- **WRITE**：`adaptiveCoeffBankIndex =` または `= newValue`
- **READ**：値の参照（`if`、`=`、関数引数など）
- **PUBLISH**：`RuntimeWorld` への代入
- **PROJECT**：`EngineParameterSnapshot` などへのコピー
- **MIRROR**：`currentAdaptiveCoeffBankIndex` などへのコピー

#### Step 2: 書き込み順序の特定

**最も重要な調査**：
```cpp
// バンク変更の流れを追跡
selectAdaptiveCoeffBankForCurrentSettings()
  → 新しい bankIndex を計算
  → 何を先に更新するか？
     ① currentAdaptiveCoeffBankIndex.store(newValue) ?
     ② それとも RuntimeBuilder に渡すだけ？
     ③ 両方？
```

この結果によって Authority が決まる。

#### Step 3: 仮説を検証する小さなテストコードの追加（推奨）

```cpp
// 一時的な診断コードを AudioEngine に追加
void AudioEngine::dumpBankIndexAuthority() const
{
    auto& world = getCurrentRuntimeWorld(); // 仮
    DBG("RuntimeWorld.bankIndex=" << world.coefficient.adaptiveCoeffBankIndex);
    DBG("currentAdaptiveCoeffBankIndex=" << currentAdaptiveCoeffBankIndex.load());
    DBG("EngineParameterSnapshot.bankIndex=" << getCurrentParameterSnapshot().adaptiveCoeffBankIndex);
}
```

これをバンク変更の直後に呼び出し、どの値が実際に一致しているかを観測する。

---

### Authority の特定結果に応じた修正方針

#### ケース A：RuntimeWorld が Authority である（確認できた場合）

**修正方針**（前回評価した案を採用）：
1. `currentAdaptiveCoeffBankIndex` を Mirror に降格
2. Mirror の更新は `onRuntimePublishedNonRt()` のみ
3. `EngineParameterSnapshot` に bankIndex を追加
4. CaptureBlock に `RuntimePublicationIdentity` を追加
5. バンク変更時の Structural Rebuild 要否を検証

#### ケース B：`currentAdaptiveCoeffBankIndex` が Authority である（確認できた場合）

**修正方針**：
1. `RuntimeWorld` は `currentAdaptiveCoeffBankIndex` の Mirror として動作するよう変更
2. `RuntimeBuilder` で世界を構築する際、Mirror から値をコピー
3. その他の手順はケース A と同じ（ただし Authority が atomic になる）

どちらのケースでも、**CaptureBlock に RuntimePublicationIdentity を追加する** ことは有効です。

---

### 実装すべき具体的なコード変更（Authority 確定後）

#### 1. CaptureBlock の拡張

```cpp
// LockFreeRingBuffer.h の AudioBlock 構造体
struct AudioBlock {
    // 既存メンバ...
    
    // 追加：Runtime Publication Identity
    uint64_t publicationSequenceId = 0;
    uint64_t publicationEpoch = 0;
    uint64_t publicationMappedGeneration = 0;
    
    // 簡易一致判定ヘルパー
    bool belongsToSameWorld(const RuntimePublishWorld& world) const noexcept {
        return publicationSequenceId == world.publication.sequenceId
            && publicationEpoch == world.publication.epoch
            && publicationMappedGeneration == world.publication.mappedRuntimeGeneration;
    }
};
```

#### 2. EngineParameterSnapshot への投影追加

```cpp
// AudioEngine.h
struct EngineParameterSnapshot {
    // 既存メンバ...
    int adaptiveCoeffBankIndex = 0;
    uint64_t publicationSequenceId = 0;
    uint64_t publicationEpoch = 0;
};

// AudioEngine.Processing.Snapshot.cpp 内
EngineParameterSnapshot AudioEngine::captureAudioThreadParameterSnapshot(
    const RuntimePublishWorld* world, ...) 
{
    snapshot.adaptiveCoeffBankIndex = world->coefficient.adaptiveCoeffBankIndex;
    snapshot.publicationSequenceId = world->publication.sequenceId;
    snapshot.publicationEpoch = world->publication.epoch;
    // ...
}
```

#### 3. CaptureBlock への書き込み

```cpp
// AudioEngine.Processing.DSPCoreIO.cpp
pushAdaptiveCaptureBlocks(..., snapshot.adaptiveCoeffBankIndex, 
                          snapshot.publicationSequenceId,
                          snapshot.publicationEpoch);
```

#### 4. Learner 側の受入判定強化

```cpp
// NoiseShaperLearner.cpp の drainCaptureQueue
if (block.publicationSequenceId != currentSession.publicationSequenceId
    || block.publicationEpoch != currentSession.publicationEpoch) {
    ++stats.droppedByWorld;
    continue;
}
```

---

### 結論

**現時点では「修正方針は妥当だが、Authority の事実認定が完了していない」** という状態です。

したがって、以下の順序で作業を進めることを強く推奨します。

1. **今すぐできること**：Authority Inventory の作成（grep 調査）
2. **次にやること**：書き込み順序の特定
3. **その次**：必要に応じて診断コードを追加
4. **Authority 確定後**：上記の修正方針を実施

この手順を踏めば、応急処置ではなく **Practical Stable ISR Bridge Runtime に準拠した恒久対策** を実装できます。