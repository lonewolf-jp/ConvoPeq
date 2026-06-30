# Inputフェーズ解析と設計反映

## 調査結果の総括

### 確定したInputフェーズ(t0→t1)の内訳

| # | 処理 | 実装 | I/O? | 推定コスト |
|---|------|------|------|-----------|
| 1 | XRUN検出preamble (atomic読取+条件判定) | AudioBlock.cpp:436-500 | なし | <5μs |
| 2 | ACTIVATE検出 (generation比較) | AudioBlock.cpp:502-518 | なし | <1μs |
| 3 | CBSUMMARY更新 (atomic max) | AudioBlock.cpp:520-548 | なし | <1μs |
| 4 | サニティチェック | AudioBlock.cpp:104-121 | なし | <1μs |
| 5 | readAudioRuntimeView() (RCU enter) | RCUReader.h:43-79 | **なし** (lock-free) | <1μs |
| 6 | Worldポインタ解決 | AudioBlock.cpp:126 | なし | <1μs |
| 7 | **診断(A): drift計測 + [CPU_MIG] writeToLog** | AudioBlock.cpp:131-209 | **🔴 同期ファイルI/O** | **推定50-500μs** |
| 8 | 診断(H): CB_SEQ検出 | AudioBlock.cpp:183-209 | なし(稀) | <1μs |
| 9 | AudioCallbackAuthorityView構築 | AudioBlock.cpp:211 | なし | <5μs |
| 10 | Atomic increment | AudioBlock.cpp:213-214 | なし | <1μs |
| 11 | makeRTExecutionFrame | AudioBlock.cpp:216-226 | なし | <5μs |
| 12 | DSP解決 | AudioBlock.cpp:235 | なし | <1μs |
| 13 | 上限/レートチェック | AudioBlock.cpp:241-261 | なし | <1μs |
| 14 | **captureAudioThreadParameterSnapshot** | AudioEngine.h:3121-3141 | **なし** (memory read) | <10μs |
| 15 | **buildAudioThreadProcessingState** | AudioEngine.h:3178-3208 | **なし** (aggregate init) | <5μs |
| 16 | processCrossfadeDelayGateIfPending | AudioEngine.h:3267-3282 | なし(非遷移時) | <1μs |
| 17 | armCrossfadeIfPending | AudioBlock.cpp:291 | なし | <1μs |
| 18 | callbackSeq/cpu atomic store | AudioBlock.cpp:294-297 | なし | <1μs |

### 最重要発見: Audio Threadからの同期ファイルI/O

JUCE `FileLogger::logMessage()` の実装:
```cpp
void FileLogger::logMessage(const String& message) {
    const ScopedLock sl (logLock);       // ← Mutex Lock!
    DBG (message);
    FileOutputStream out (logFile, 256); // ← File Open!
    out << message << newLine;           // ← Sync Write!
}
```

**Audio Threadからの直接writeToLog呼び出し（27秒間の実測）:**

| ログ種別 | 呼び出し回数 | 呼び出し/秒 | 呼び出し元 |
|---------|------------|-----------|-----------|
| [CPU_MIG] | **29,980回** | **1,110回/秒** | AudioBlock.cpp:171 |
| [CALLBACK_STAGE] | 2,138回 | 79回/秒 | AudioBlock.cpp:605 |
| [CB_SEQ] | 2回 | 0回/秒 | AudioBlock.cpp:193 |

**合計: 32,120回の同期ファイルI/Oが27秒間にAudio Threadから実行されている（1,190回/秒）。**

### 他の操作の評価

| 操作 | 評価 | 根拠 |
|------|------|------|
| captureAudioThreadParameterSnapshot | ✅ 問題なし | RuntimeWorldからのmemory readのみ |
| buildAudioThreadProcessingState | ✅ 問題なし | aggregate initialization, 動的確保なし |
| processCrossfadeDelayGateIfPending | ✅ 問題なし | 非遷移時は即時false return |
| RCU Reader | ✅ 問題なし | lock-free CAS, スピン/ウェイトなし |
| CPU MigrationのDSP影響 | ✅ 軽微 (20μs) | 予算5.33msの0.4% |

---

## Inputフェース最適化: 設計変更案

### 提案1: [CPU_MIG]ログの非同期化または削減（優先度: 高）

**現状の問題:**
`[CPU_MIG]` ログが毎コールバック（87.6%の確率）で同期ファイルI/Oを実行している。1,110回/秒のwriteToLog。

**対策案 A: [CPU_MIG]をリングバッファ経由に変更（推奨）**
```
Audio Thread: LockFreeRingBuffer<CpuMigEvent,64> にpush
    ↓ (lock-free, non-blocking)
Timer Thread: pop() → diagLog() で一括出力
```
XRUNと同じパターン。InputフェースからファイルI/Oを排除。

**対策案 B: [CPU_MIG]の間引き**
```
CONVOPEQ_DIAG_SAMPLE_MASK と同様のサンプリングを適用
例: 1/16 に間引く → 約70回/秒に削減
```

**対策案 C: [CPU_MIG]の条件付き出力**
```
前回のCPUと同じ場合のみ出力をスキップ
→ 既にCPU_MIGは「変化した時のみ」出力しているが、
  全コールバックで変化しているため意味がない
→ 定期サンプリングに変更
```

### 提案2: CALLBACK_STAGEのLogger::writeToLogをTimer Thread委譲（優先度: 中）

**現状の問題:**
`[CALLBACK_STAGE]` もAudio Threadから直接writeToLogしている（79回/秒）。

**対策案:**
CALLBACK_STAGEもリングバッファ経由に変更。ただしCALLBACK_STAGEは診断データのため、リアルタイム性への影響はCPU_MIGより低い。

### 提案3: 診断ログ自体の出力抑制（優先度: 低）

**現状:**
CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1 の全ログが有効。リリースビルドでもデフォルトON。

**対策案:**
- リリースビルドでは CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=0 にする
- または診断有効時のみファイル出力し、通常時はDBGのみ

---

## 設計反映: 修正計画

### Step 1: [CPU_MIG]のリングバッファ化（本日中）

**変更ファイル:**
1. `AudioEngine.h` — `CpuMigEvent` 構造体＋リングバッファ追加
2. `AudioEngine.Processing.AudioBlock.cpp` — writeToLog → ring buffer push
3. `AudioEngine.Timer.cpp` — pop → diagLog に変更

**期待される効果:**
```
Inputフェース 2.8ms → 推定 1.5-2.0ms（-0.8〜1.3ms改善）
```

### Step 2: CALLBACK_STAGEのリングバッファ化（数日中）

**期待される効果:**
```
Inputフェース 2.8ms → 推定 1.2-1.5ms（追加-0.3〜0.5ms改善）
```

### Step 3: リリースビルドの診断ログ設定見直し

**対策案:**
- デフォルトリリース: CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1 のまま
- ただし [CPU_MIG] は常時OFF（デバッグ用途専用）
- 必要時に設定ファイルで有効化

---

## 残りの未確定事項（次フェーズ）

| 事項 | 状況 | 理由 |
|------|------|------|
| ホストcallback間隔8-12msの原因 | **未確定** | ConvoPeq内部では計測不可。OS/ホスト依存 |
| MMCSS有効時の挙動 | **未確定** | 未検証。環境依存のため実測が必要 |
| Block2048での効果 | **未確定** | 未検証。Inputの改善との組み合わせが必要 |
