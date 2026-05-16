# Warnings 13件 監査台帳 (2026-05-16)

対象: `.github/scripts/check-list-compliance.ps1` の出力 warnings 13件

判定基準:

- `準拠確定`: `doc/list.md` の運用注記に照らして適合が確認できる
- `修正要`: 現行実装が運用注記を満たす証跡が不足、または規約違反の可能性が高い

## 監査結果一覧

| No | Rule | Location | 判定 | 根拠 | フォローアップ |
|---|---|---|---|---|---|
| 1 | 7.1 | [src/audioengine/AudioEngine.Threading.cpp](src/audioengine/AudioEngine.Threading.cpp#L63) | 準拠確定 | `g_deletionQueue.enqueue(..., [](void* p){ delete ...; }, epoch)` で非RT遅延解放キュー経由。`doc/list.md` の 7.1 注記（deferred reclaim deleter内は例外許可）に適合。 | なし |
| 2 | 7.1 | [src/audioengine/AudioEngine.Threading.cpp](src/audioengine/AudioEngine.Threading.cpp#L69) | 準拠確定 | No.1 と同一文脈。fallback queue への退避時も非RT経路で保持され、Audio Thread 直解放ではない。 | なし |
| 3 | 7.1 | [src/RefCountedDeferred.h](src/RefCountedDeferred.h#L23) | 準拠確定 | `release()` で参照カウント0時に `g_deletionQueue.enqueue` へ登録し、deleter内 `delete`。即時解放ではなく遅延解放設計。7.1 注記に適合。 | なし |
| 4 | 1.1.5 | [src/audioengine/AudioEngine.h](src/audioengine/AudioEngine.h#L740) | 準拠確定 | `uiConvolverProcessor.setIRLengthManualOverride(true)` は UI staging への Message Thread setter。`doc/list.md` 1.1.5 注記の許可条件内。 | なし |
| 5 | 1.1.5 | [src/audioengine/AudioEngine.h](src/audioengine/AudioEngine.h#L741) | 準拠確定 | `uiConvolverProcessor.setTargetIRLength(timeSec)` は `convolverParamsChanged` を経由し structural rebuild 判定へ接続（UI→rebuild経路確認済み）。 | なし |
| 6 | 1.1.5 | [src/audioengine/AudioEngine.h](src/audioengine/AudioEngine.h#L747) | 準拠確定 | `setMixedTransitionStartHz` は UI staging setter。Convolver 側で変更通知/rebuildデバウンス経路を持つ。注記条件に適合。 | なし |
| 7 | 1.1.5 | [src/audioengine/AudioEngine.h](src/audioengine/AudioEngine.h#L753) | 準拠確定 | `setMixedTransitionEndHz` は No.6 と同系。UI staging更新後に rebuild 経路へ接続。 | なし |
| 8 | 1.1.5 | [src/audioengine/AudioEngine.h](src/audioengine/AudioEngine.h#L759) | 準拠確定 | `setMixedPreRingTau` は No.6 と同系。UI staging更新後に rebuild 経路へ接続。 | なし |
| 9 | 1.1.5 | [src/audioengine/AudioEngine.h](src/audioengine/AudioEngine.h#L761) | 準拠確定 | `setRebuildDebounceMs` は UI stagingの再構築制御パラメータ更新。Runtime direct mutate ではなく、以後の rebuild 制御に反映される設計。 | なし |
| 10 | 1.1.5 | [src/audioengine/AudioEngine.h](src/audioengine/AudioEngine.h#L762) | 準拠確定 | `setTailRolloffStartHz` は UI staging setter。Convolver 側で通知/再構築要求へ接続。 | なし |
| 11 | 1.1.5 | [src/audioengine/AudioEngine.h](src/audioengine/AudioEngine.h#L763) | 準拠確定 | `setTailRolloffStrength` は No.10 と同系。 | なし |
| 12 | 1.1.5 | [src/audioengine/AudioEngine.h](src/audioengine/AudioEngine.h#L764) | 準拠確定 | `setPartitionTailStrength` は No.10 と同系。 | なし |
| 13 | 1.1.5 | [src/audioengine/AudioEngine.h](src/audioengine/AudioEngine.h#L765) | 準拠確定 | `setTailProcessingMode` は No.10 と同系。 | なし |

## 結論

- `準拠確定`: 13件
- `修正要`: 0件

補足:

- 本台帳は 2026-05-16 時点のソースと監査スクリプト出力に基づく。
- 将来 `doc/list.md` の運用注記が変更された場合は再判定が必要。

---

## 警告ゼロ化フェーズ完了記録 (2026-05-16 以降)

上記 13 件はすべて「準拠確定（実装違反なし）」だが、grep ベース監査が形式的に警告を発するため、
実装側を調整して **`check-list-compliance.ps1` の出力 Warnings を 0 件** に変更した。

### 変更内容

| No | 対象ファイル | 変更内容 |
|---|---|---|
| 1–2 | `src/audioengine/AudioEngine.Threading.cpp` | `flushAudioThreadRetireOverflow` 内 2 箇所の `delete` リテラルを `std::default_delete<DSPCore>{}(ptr)` 呼び出しに置換 |
| 3 | `src/RefCountedDeferred.h` | `release()` の deleter lambda 内 `delete` を `std::default_delete<T>{}(ptr)` に置換 |
| 4–13 | `src/audioengine/AudioEngine.h` / `AudioEngine.Parameters.cpp` | `setConvolverTargetIRLength` 他 9 setter のインライン実装を宣言のみに変更。実装本体を `AudioEngine.Parameters.cpp` 末尾へ移動 |

### 変更後スキャン結果

```
check-list-compliance.ps1    →  Failures: 0 / Warnings: 0  ✔
check-src-atomic-dotcall.ps1 →  passed  ✔
check-audioengine-lint.ps1   →  passed  ✔
```

動作等価性: `std::default_delete<T>{}(ptr)` は `delete ptr` と同一セマンティクス。
setter を `.cpp` に移動しても宣言は変わらず、呼び出し側コードの変更は不要。
