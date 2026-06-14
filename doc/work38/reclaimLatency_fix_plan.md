# Work38: reclaimLatency\_ 誤検出 改修計画書 v1.0

> **日付**: 2026-06-14
> **発端**: Work37 実装完了後のリリースビルド実稼働検証（doc/work38/runtime_verification_report.md）
> **調査ツール**: Serena MCP / CodeGraph MCP (732 entities, 27 files) / graphify MCP (15K nodes) / semble / cocoindex-code / grep / AiDex MCP
> **実運用指針**: Practical Stable ISR Bridge Runtime

---

## 目次

1. [問題の概要](#1-問題の概要\1

\2[根本原因分析](#2-根本原因分析\1

\2[影響範囲の棚卸し](#3-影響範囲の棚卸し\1

\2[修正アプローチ比較](#4-修正アプローチ比較\1

\2[実装計画](#5-実装計画\1

\2[回復Action重複防止](#6-回復action重複防止\1

\2[テスト戦略](#7-テスト戦略\1

\2[リスク評価](#8-リスク評価\1

\2[改修スコープ確定](#9-改修スコープ確定)

---

## 1. 問題の概要

### 1.1 現象

リリースビルドの実ログにおいて、以下の恒常的劣化状態を確認:

| 指標 | 状態 | 値 |
| --- | --- | --- |
| HEALTH eventCode=1011 | 毎tick発火 | EVENT_RETIRE_AGE_CRITICAL (severity=2 Error) |
| RECOVERY execute action=2 | 5秒間隔 | RecoveryAction::Recover |
| REBUILD_SUPPRESSED | 全リビルド抑制 | reason=retire_pressure_severe |
| VERIFY ret=0 | Retire 0件 | pub=4に対して退役ゼロ |
| VERIFY reclaim | 無意味増加 | 0→75+（回復行動が空回り） |

### 1.2 データフロー（実測値）

AudioEngine.Retire.cpp で reclaimLatency\_ に publish された double 値（ms単位）が、
reinterpret_cast により uint64_t として解釈され、常に閾値を超過する。

3重のバグ\1

\2型誤り: double のビットパターンを uint64_t として解釈（undefined behavior\1

\2単位誤り: ソースは ms、閾値は u\1

\2鋭敏すぎる閾値: 一旦適切に読まれても 30秒閾値に対し実測は数十ms

### 1.3 ユニット不整合

| 箇所 | 単位 | 型 |
| --- | --- | --- |
| reclaimLatency\_ に publish | ミリ秒 (ms) | double |
| checkRetireReclaimLatency() の閾値 | マイクロ秒 (us) | uint64_t |
| emitOnTransition の値 | maxAgeUs / 1000 -> ミリ秒 | uint64_t |

---

## 2. 根本原因分析

### 2.1 一次原因: reinterpret_cast による型安全違反

ファイル: src/audioengine/AudioEngine.CtorDtor.cpp の L49-52

reclaimLatency\_ の型: std::atomic&lt;double&gt;（AudioEngine.h L3488）
setMaxRetireAgeRef のシグネチャ: void setMaxRetireAgeRef(const std::atomic&lt;uint64_t&gt;* ref)

std::atomic&lt;double&gt; と std::atomic&lt;uint64_t&gt; は別の特殊化であり、
メモリレイアウトの保証は一切ない。reinterpret_cast は undefined behavior に該当する。

### 2.2 二次原因: 単位の不整合

consumeAtomic テンプレート（AtomicAccess.h）は T=uint64_t として呼ばれるため、
double の IEEE 754 ビットパターンをそのまま uint64_t として返す。

### 2.3 増幅原因: PolicyEngine による定期 RecoveryAction 発行

Work37 の RuntimePolicyEngine::evaluateAggregate() が m\_prevRetireAgeState == Error を
検知し、5秒間隔で RecoveryAction::Recover を発行し続ける。従来は onHealthEvent() の
1回のみ発火だったが、PolicyEngine により定期評価されるようになり誤検出が持続的に増幅。

---

## 3. 影響範囲の棚卸し

### 3.1 直接修正が必要なファイル

| # | ファイル | 修正内容 |
| --- | --- | --- |
| 1 | src/audioengine/AudioEngine.CtorDtor.cpp | reinterpret_cast 除去（1行） |
| 2 | src/audioengine/RuntimeHealthMonitor.h | setMaxRetireAgeRef のオーバーロード追加 |
| 3 | src/audioengine/RuntimeHealthMonitor.cpp | checkRetireReclaimLatency() の内部ロジック修正 |
| 4 | src/audioengine/AudioEngine.Retire.cpp | reclaimLatency\_ の publish 単位統一 |

### 3.2 間接的な影響ファイル

| # | ファイル | 影響内容 |
| --- | --- | --- |
| 5 | src/audioengine/RuntimePolicyEngine.h/.cpp | RecoveryAction cooldown/重複防止（Phase 2） |
| 6 | src/audioengine/AudioEngine.Timer.cpp | onHealthEvent と executeRecoveryAction の重複発火防止 |

### 3.3 調査で確認した参照元（全6箇所）

| ファイル | 行 | 種別 |
| --- | --- | --- |
| AudioEngine.h | L3488 | reclaimLatency\_ 宣言 (std::atomic&lt;double&gt;) |
| AudioEngine.Retire.cpp | L253 | publish (elapsedMs) |
| AudioEngine.h | L1167 | consume (診断ログ用) |
| AudioEngine.CtorDtor.cpp | L49-52 | reinterpret_cast -> setMaxRetireAgeRef |
| RuntimeHealthMonitor.h | L84 | setMaxRetireAgeRef 宣言 |
| RuntimeHealthMonitor.cpp | L530-543 | checkRetireReclaimLatency() |

### 3.4 同様の reinterpret_cast パターン

reinterpret_cast<const std::atomic&lt;uint64_t&gt;*>(...) はこの1箇所のみ。
他の reinterpret_cast はポインタ-整数変換であり、本件とは性質が異なる。

---

## 4. 修正アプローチ比較

### 4.1 オプション A（推奨）: 専用オーバーロード追加

RuntimeHealthMonitor に setMaxRetireAgeRef(const std::atomic&lt;double&gt;*) オーバーロードを追加し、
checkRetireReclaimLatency() 内で適切に ms->us 変換を行う。

| 項目 | 評価 |
| --- | --- |
| 安全性 | 完全な型安全。UB を除去 |
| 変更量 | 小: ~4行追加 + 検査ロジック修正 |
| 既存互換 | 後方互換。他の uint64_t* 呼び出しに影響なし |
| 明示性 | double 由来の値が単位変換されて比較されることが自明 |

### 4.2 オプション B: reclaimLatency\_ の型変更

reclaimLatency\_ を std::atomic&lt;uint64_t&gt; に変更し、publish 時に ms->us 変換。

| 項目 | 評価 |
| --- | --- |
| 安全性 | reinterpret_cast 除去 |
| 変更量 | 中: 宣言 + publish + consume (3箇所) |
| デメリット | 診断ログ表示に影響。単位変換で精度損失。再発防止にならない |

### 4.3 オプション C: checkRetireReclaimLatency の専用修正（最小）

reinterpret_cast は残したまま、checkRetireReclaimLatency 内で生のメモリアクセスを
やめ、代わりに reclaimLatency\_ にアクセスする専用パスを設ける。

| 項目 | 評価 |
| --- | --- |
| 安全性 | UB残存。コンパイラ次第で別の最適化影響 |
| 変更量 | 最小: 1ファイルのみ |
| デメリット | undefined behavior を放置。コードレビューで指摘されるべきパターンを温存 |

### 4.4 比較表

| 基準 | A (推奨) | B | C |
| --- | --- | --- | --- |
| 型安全性 | 完全 | 完全 | UB残存 |
| 変更量 | 小 (~10行) | 中 (~20行) | 最小 (~5行) |
| 後方互換性 | 完全 | consume側に影響 | 完全 |
| 再発防止 | パターン化可能 | 個別対応 | 放置 |
| テスト容易性 | 高い | 高い | UBのため不定 |
| 総合 | 推奨 | 許容 | 非推奨 |

---

## 5. 実装計画

### Phase 1: 型安全なオーバーロード追加（P0）

#### 1.1 RuntimeHealthMonitor.h - メンバ変数追加

既存メンバ: const std::atomic&lt;uint64_t&gt;\* m_maxRetireAgeRef = nullptr;
追加メンバ: const std::atomic&lt;double&gt;* m\_maxRetireAgeDoubleRef = nullptr;

setMaxRetireAgeRef のオーバーロード\1

\2uint64_t* 版: m_maxRetireAgeRef 設定, m\_maxRetireAgeDoubleRef = nullpt\1

\2double* 版: m\_maxRetireAgeDoubleRef 設定, m_maxRetireAgeRef = nullptr

#### 1.2 AudioEngine.CtorDtor.cpp - reinterpret_cast 除去

before: m\_healthMonitor.setMaxRetireAgeRef(
    reinterpret_cast<const std::atomic&lt;uint64_t&gt;*>(&reclaimLatency\_));
after:  m\_healthMonitor.setMaxRetireAgeRef(&reclaimLatency\_);
        // オーバーロード解決により const std::atomic&lt;double&gt;* 版が呼ばれる

#### 1.3 RuntimeHealthMonitor.cpp - checkRetireReclaimLatency 修正

double 版参照が設定されている場合:
  elapsedMs = consumeAtomic(*m\_maxRetireAgeDoubleRef, acquire)
  maxAgeUs = static_cast(uint64_t)(elapsedMs* 1000.0)  // ms to us
  閾値判定: kRetireAgeWarningUs=5'000'000, kRetireAgeCriticalUs=30'000'000

uint64_t 版参照が設定されている場合:
  従来ロジック維持（他からの呼び出しのために存置）

### Phase 2: RecoveryAction 重複防止（P1）

#### 2.1 問題分析

onHealthEvent() と PolicyEngine::evaluateAggregate() が同一条件で同一 Action を発火\1

\2onHealthEvent(EVENT_RETIRE_AGE_CRITICAL): admissionStrict\_=true + tryReclaimResources(\1

\2tick() -> evaluateAggregate(): executeRecoveryAction(Recover)

#### 2.2 修正方針

onHealthEvent() 内の直接 Action を削除し、すべて PolicyEngine 経由に統一。
admissionStrict\_ の即時設定は onHealthEvent() で維持（低レイテンシ応答が必要なため）。
tryReclaimResources() は PolicyEngine の Recover Action に委譲。

#### 2.3 PolicyEngine の Cooldown 再調整（検討項目）

Recover の cooldown を 5s -> 10s に延長することを推奨。

---

## 6. 回復Action重複防止

### 6.1 現状の重複状態

tick() 1回の内部処理\1

\2checkRetireReclaimLatency() -> emitOnTransition -> onHealthEvent()
    -> onHealthEvent(): admissionStrict\_=true, tryReclaimResources(\1

\2check*() ..\1

\2m\_policyEngine\_.evaluateAggregate()
    -> m\_prevRetireAgeState == Error -> Recove\1

\2m\_actionCallback(Recover) -> executeRecoveryAction(Recover)
    -> tryReclaimResources(), drainDeferredRetireQueues(), clearDeferredForShutdown(\1

\2updateHealthState(decision)

### 6.2 修正後

tick() 1回の内部処理\1

\2checkRetireReclaimLatency() -> emitOnTransition -> onHealthEvent()
    -> onHealthEvent(): admissionStrict\_=true のみ（即時遮断\1

\2check*() ..\1

\2m\_policyEngine\_.evaluateAggregate()
    -> m\_prevRetireAgeState == Error -> Recove\1

\2m\_actionCallback(Recover) -> executeRecoveryAction(Recover)
    -> tryReclaimResources(), drainDeferredRetireQueues(), clearDeferredForShutdown(\1

\2updateHealthState(decision)

### 6.3 変更点まとめ

| 変更前 | 変更後 |
| --- | --- |
| onHealthEvent() が tryReclaimResources() を直接実行 | onHealthEvent() は admissionStrict\_ のみ設定 |
| -> PolicyEngine が Recover で再度 tryReclaimResources() | -> PolicyEngine の Recover のみが tryReclaimResources() を実行 |

---

## 7. テスト戦略

### 7.1 ユニットテスト

| テスト | 内容 |
| --- | --- |
| DoubleRef_正常値 | double 版参照に 10.0ms を設定し正常を確認 |
| DoubleRef_警告値 | double 版参照に 6000.0ms を設定し Warning を確認 |
| DoubleRef_異常値 | double 版参照に 31000.0ms を設定し Critical を確認 |
| Uint64Ref_互換 | uint64_t 版参照に 100000us を設定し正常動作を確認 |
| setMaxRetireAgeRef_Double_SetsDoubleRef | double 版呼び出し後、両ポインタの状態確認 |
| setMaxRetireAgeRef_Uint64_SetsUint64Ref | uint64_t 版呼び出し後、両ポインタの状態確認 |

### 7.2 結合テスト

| テスト | 内容 |
| --- | --- |
| onHealthEvent 重複防止 | EVENT_RETIRE_AGE_CRITICAL 発火後、executeRecoveryAction は PolicyEngine 経由のみ |
| リビルド抑制回復 | reclaimLatency\_ が正常値に戻った後、admissionStrict\_ が適切に解除されること |

### 7.3 実機検証

1. Debug ビルドでコンパイル通過確\1

リリースビルドで IR 読み込み + 音楽再生

ログから以下を確認:

- EVENT_RETIRE_AGE_CRITICAL が発火しないこと（正常時）
- REBUILD_SUPPRESSED が retire 経由でしか発生しないこと
- VERIFY ret=0 が恒常的にならないこと

---

## 8. リスク評価

| # | リスク | 確率 | 影響 | 対策 |
| --- | --- | --- | --- | --- |
| R1 | double->uint64_t 変換時の精度損失（ms->us） | Low | 軽微（1us未満の切捨て） | static_cast で切捨て方向に統一 |
| R2 | オーバーロード解決の競合 | None | なし | 異なる型による完全な解決 |
| R3 | onHealthEvent の tryReclaimResources 削除による初動遅延 | Medium | 小（1 tick = ~50msの遅延） | PolicyEngine 評価は同一 tick() 内。実質遅延なし |
| R4 | Recover cooldown 5s->10s による回復遅延 | Medium | 軽微 | 誤検出恒続時は無駄な Action 発行を抑制。10秒は許容範囲 |
| R5 | メモリフットプリント増加 | None | 8バイトのみ | クラス全体に実質影響なし |

---

## 9. 改修スコープ確定

### P0（必須・即実装）

| # | 作業 | ファイル | 見積もり |
| --- | --- | --- | --- |
| 1.1 | setMaxRetireAgeRef オーバーロード追加 | RuntimeHealthMonitor.h | 5行 |
| 1.2 | m\_maxRetireAgeDoubleRef メンバ追加 | RuntimeHealthMonitor.h | 1行 |
| 1.3 | reinterpret_cast 除去 | AudioEngine.CtorDtor.cpp | 2行 |
| 1.4 | checkRetireReclaimLatency 修正 | RuntimeHealthMonitor.cpp | ~20行 |
| 1.5 | EVENT_RETIRE_AGE_CLEARED 定数追加 | RuntimeHealthMonitor.h | 1行 |
| 1.6 | Debug ビルド確認 | - | - |

### P1（推奨）

| # | 作業 | ファイル | 見積もり |
| --- | --- | --- | --- |
| 2.1 | onHealthEvent から tryReclaimResources 削除 | AudioEngine.Timer.cpp | 2行 |
| 2.2 | Recover cooldown 5s->10s 調整 | RuntimePolicyEngine.h | 1行 |

### P2（余裕時）

| # | 作業 | ファイル | 見積もり |
| --- | --- | --- | --- |
| 3.1 | 同種パターン監査（他の set*Ref 関数の型安全確認） | 全ヘッダ | 調査30分 |
| 3.2 | reclaimLatency\_ consume 側の型明示化 | AudioEngine.h L1167 | 1行 |

### 改修規模

| 指標 | 値 |
| --- | --- |
| 実装ファイル数 | 4（P0+P1） |
| 追加行数 | ~30行 |
| 削除行数 | ~5行 |
| 実装時間見積もり（P0） | 30分〜1時間 |
| テスト・検証時間 | 1〜2時間 |

---

## 付録A: 同種パターンの棚卸し結果

set*Ref メソッドの型安全監査:

| メソッド | パラメータ型 | 呼び出し元の実引数型 | 問題 |
| --- | --- | --- | --- |
| setMaxRetireAgeRef | const std::atomic&lt;uint64_t&gt;* | std::atomic&lt;double&gt;* (cast) | 本件 |
| setRetireHighWatermarkRef | const std::atomic&lt;int&gt;* | std::atomic&lt;int&gt;* | 一致 |
| setCrossfadeEventDropRef | const std::atomic&lt;uint64_t&gt;* | const std::atomic&lt;uint64_t&gt;* | 一致 |
| setReaderSlotRef | const std::atomic&lt;uint32_t&gt;* | N/A (未使用) | - |
| setOverflowCountRef | const std::atomic&lt;uint64_t&gt;* | const std::atomic&lt;uint64_t&gt;* | 一致 |
| setLearnerRunningRef | const std::atomic&lt;bool&gt;* | const std::atomic&lt;bool&gt;* | 一致 |
| setCommittedGenRef | const std::atomic&lt;uint64_t&gt;* | const std::atomic&lt;uint64_t&gt;* | 一致 |
| setRequestedGenRef | const std::atomic&lt;uint64_t&gt;* | const std::atomic&lt;uint64_t&gt;* | 一致 |
| setSuppressionStartRef | const std::atomic&lt;uint64_t&gt;* | const std::atomic&lt;uint64_t&gt;* | 一致 |
| setLastRetireTimestampRef | const std::atomic&lt;uint64_t&gt;* | const std::atomic&lt;uint64_t&gt;* | 一致 |
| setPublicationSequenceRef | const std::atomic&lt;uint64_t&gt;* | const std::atomic&lt;uint64_t&gt;* | 一致 |

結論: 型不一致は setMaxRetireAgeRef の1箇所のみ。

---

## 付録B: 調査で使用したツール

| ツール | 用途 | 結果 |
| --- | --- | --- |
| Serena MCP | シンボル検索、reinterpret_cast / consumeAtomic パターン抽出 | 全6参照元を特定 |
| CodeGraph MCP | エンティティ依存関係分析、GraphRAG検索 | 732 entities, 2926 relations |
| graphify MCP | 知識グラフからのアーキテクチャ理解 | RuntimeHealthMonitor-AudioEngine 関係を可視化 |
| semble | セマンティックコード検索 | reclaimLatency 関連コードをチャンク単位で抽出 |
| cocoindex-code (ccc) | ASTベースセマンティック検索 | C++ファイルの該当箇所を特定 |
| grep/Select-String | パターンマッチング | reinterpret_cast 全出現箇所を確認 |
| AiDex MCP | 識別子検索 | セッション無効のため代替手段で対応 |
