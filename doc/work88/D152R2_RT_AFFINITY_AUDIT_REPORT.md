# RT Affinity Audit — lifecycle W の ISR/AudioThread 非接触再確認（Work Report）

```text
Production/Test/CMake changes: 0 / Build: NOT RUN / CTest: NOT RUN
詳細: evidence/D152R2_RT_AFFINITY_AUDIT.md
```

**Status: RT affinity audit = PASS。T3c close candidate。**

## 結果

実装後ソースの全 lifecycle W アクセス点（20 サイト、ISRRuntimePublicationCoordinator.{h,cpp} に限定的）を列挙し、6 つの外部入口（E1-E6）の thread affinity を enclose 関数 + スレッド起動点で実測した:

| 入口 | スレッド |
|---|---|
| rebuildThreadLoop 内 postSignal ×4 / enqueuePublicationIntentForRuntimeCommit（Route A） | **RebuildThread** |
| runCoordinatorPhase（adjudicate / redrive / processIntent → submitRecoveryRequest・QuarantineIntentHandler） | **CoordinatorLoop**（juce::Thread "ConvoPeq.CoordinatorLoop"・ソース明記 Non-RT, never RT） |
| Orchestrator Route C（submitPublishRequest） | CoordinatorLoop / RebuildThread |
| shutdown close（discardRecoveryRequestsOnShutdown） | join 後の単一スレッド |

**Audio/RT パス（processBlock → getNextAudioBlock → DSPCore* 8 ファイル）からの recovery API 参照 = 0 件**。EQProcessor / ConvolverProcessor が保持する `retireCoordinator_` ポインタは prepareToPlay での配線のみ・**deref 0 件**（RT callback から coordinator メソッドを一切呼ばない）。

## 帰結

D154-F1 の「lock-free ではないが NonRT-only なので許容（D150 §12 既存承認の範囲内）」が、仮定ではなく**現行 topology の実測**として維持された。

- ST-1 PASS + RT affinity audit PASS → **T3c close candidate**
- 変質リスク注記: 将来 RT callback 内から retireCoordinator_ を deref する変更が入った場合は D152 §2 wrapper 案への切替が必須（D152-R2 §5 済み）

## 次工程

```text
RT affinity audit PASS（現在地）
   ↓
T3c close candidate（ユーザー判断）
   ↓
D154-F2（DSPHandle + T3c 側残存誤コメントの comment-only 別トラック — await go）
```
