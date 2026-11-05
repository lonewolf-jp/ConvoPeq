# ConvoPeq 残課題整理 — 未完成の修正・放置された作業・未修正の新規バグ（2026-11-05）

## 結論

2026-11-05 時点の現行 `main` と全記録を照合した結果、**未修正の欠陥候補が 3 件**（RES2-01〜03）、**途中で止まっている修正・機能が 5 件**（RES2-04〜08）、**検証・テストの穴が 4 件**（RES2-11〜14）、**技術的負債が 6 件**（RES2-15〜20）、**性能・運用の既知事項が 2 件**（RES2-09〜10）だった。

同日、修正着手可能と判定した 3 件（RES2-02・RES2-14・RES2-17）を修正して main に統合した（§11）。さらに詳細調査で RES2-03（実測済みと判明）と RES2-08（RuntimeWorld 移行で前提消滅）を解消した（§12）。さらに RES2-18（Worker 経路の debug アサーション）を修正した（§13）。残りは **未解決 13 件**（承認待ち 5・設計判断 4・追加調査 2・検証 2）である。

過去の記録で「未修正」と書かれていても、現在は修正済みのもの（H-01・C-01・B-2 診断クラッシュ・A2 assert・`.gitignore` のフェンス行）は §9 に分離し、残課題から除外した。

## 0. 一覧（分類別）

| 分類 | ID | 要点 | 優先度 |
|---|---|---|---|
| 正確性 | RES2-01 | ポリフェーズ利得の不対称（案 E が既定 OFF） | P2 |
| 正確性 | RES2-02 | `isBadSample` のスカラー／AVX2 閾値不一致 → **修正済み**（#131） | P3 |
| 正確性 | RES2-03 | SoftClip 局所 OS の遅延定数が未実測 → **解消**（P0-F が D=15 を検証済み・コメント更新） | P2 |
| バグ修正 | RES2-04 | F-4/D2b の記録消失（調査停止中） | P3 |
| 機能 | RES2-05 | B2 Stage 2/3 未着手 | P2 |
| 機能 | RES2-06 | A2 本番適用が NO-GO | P2 |
| 機能 | RES2-07 | U5 製品有効化が未承認 | P2 |
| 機能 | RES2-08 | `EngineRuntime` 単一 atomic 統合が未着手 → **解消**（RuntimeWorld 移行で前提が消滅） | P3 |
| 性能・運用 | RES2-09 | R-101 の初回キャッシュミス時 61 分 | P3 |
| 性能・運用 | RES2-10 | R-113／R-111 のフレーク監視 | P3 |
| 検証 | RES2-11 | TSan が best-effort（合否に影響しない） | P3 |
| 検証 | RES2-12 | soak が手動トリガのみ | P3 |
| 検証 | RES2-13 | TX-02／TX-10 の実機試験が Waived | P3 |
| 検証 | RES2-14 | `FFTBackendTests` が現状確認止まり → **修正済み**（#130） | P3 |
| 負債 | RES2-15 | CMA 状態の部分永続化 | P3 |
| 負債 | RES2-16 | elite 再評価が実質再実行 | P3 |
| 負債 | RES2-17 | `D8_1` の T2b コメントが陳腐化 → **修正済み**（#129） | P3 |
| 負債 | RES2-18 | Worker スレッドのアサーション欠落 → **修正済み**（`debugAssertNotAudioThread()` 追加） | P3 |
| 負債 | RES2-19 | `Superseded` 等の Phase-II 送り | — |
| 負債 | RES2-20 | `.venv` が Python 3.14 消滅で使用不能 | P3 |

## 1. 未修正の欠陥候補（コードで確認）

### RES2-01 ポリフェーズ利得規約の不対称（B-1 / 案 E）

| 項目 | 内容 |
|---|---|
| 分類 | 安全性・正確性（音声利得） |
| 内容 | `CustomInputOversampler::interpolateStage` で `convValue *= 2.0` のみ適用され、`centerValue` 側に ×2 が無い。分岐ごとの係数和が 1.0 と 0.5 で不対称になり、up/down 往復の DC 利得が 0.75^N になる |
| 発生経緯 | 2026-09 の B-1 監査で特定。修正案 E（center にも ×2）は `CONVOPEQ_CORRECT_POLYPHASE_GAIN` として実装済みだが **既定 OFF** のまま |
| 現在の状態 | **未採用（製品既定 OFF）**。修正コードは `src/CustomInputOversampler.cpp` の `#if` 内に存在し、CMake オプションは `OFF`。採用判断（D1）は材料充足、既定 ON の判断（D2）は HOLD |
| 技術的問題点 | 有効化すると既定出力が OS 倍率に応じて +2.4988 / +4.9975 / +7.4963 dB 変化し、SoftClip drive が 5〜10 倍、limiter 到達が 0 → 119/4096 になる（P1-1／P1-3 の実測）。聴感評価・プリセット replay・EQ/IR 併用は未測定。なお **0.75^N が設計意図かは未確定**で、P1-4 は「バグと断定しない」と明記している |
| 影響範囲 | 入力オーバーサンプリング経路（SoftClip 局所 OS を含む）。既定ビルドの DC 利得 |
| 依存 | Phase 0 characterization（完了）、REF-FIDELITY（実装済み）、Owner の GO 判断（未） |
| 優先度 | P2 |
| 推奨対応 | 判断資料は `doc/work113/p1_4_adoption_decision_package_20260922.md` に既にある。不足しているのは聴感評価・プリセット replay・EQ/IR 併用の 3 点なので、そこを埋めてから採用を決める |
| 根拠 | `src/CustomInputOversampler.cpp:556-561`、`CMakeLists.txt:134`、`doc/work113/remediation_plan_20260922_v3.1_revised.md` §2、`doc/work113/phase0_characterization_report_20260922.md` §4 |
| 必要承認 | Owner（機能の挙動変更） |
| 次の作業 | 採用判断用の要約（利得差・影響プリセット・ロールバック条件）を作成 |

### RES2-02 `isBadSample` のスカラー／AVX2 閾値不一致（O-20）

| 項目 | 内容 |
|---|---|
| 分類 | 安全性・正確性（破損検出の一貫性） |
| 内容 | スカラー版は `abs(x) > 2^53` を破損と判定し、AVX2 版は `abs(x) > 1e20` で判定する。同じ入力でも経路によって破損扱いが変わる |
| 発生経緯 | 2026-09 の characterization で R13-8 / O-20 として記録。Phase 0 freeze の対象で変更禁止のまま |
| 現在の状態 | **未修正**（コードに両方が現存） |
| 技術的問題点 | 2^53〜1e20 の値はスカラー経路では破損、AVX2 経路では正常。破損検出の結果が CPU 命令と経路に依存する |
| 影響範囲 | `CustomInputOversampler` の破損検出・フォールバック判定。通常音声の範囲では発生しない |
| 依存 | Phase 0 freeze の解除（RES2-01 と同じ判断枠組み） |
| 優先度 | P3 |
| 推奨対応 | 閾値を共通定数へ統一する（スカラーを 1e20 に合わせるのが最小差分）。統一後に境界値テストを 1 本追加 |
| 根拠 | `src/CustomInputOversampler.cpp:23-32`（スカラー）、同 `:43-53`（AVX2）、`doc/work113/phase0_characterization_report_20260922.md` §4 |
| 必要承認 | 不要（最小修正・テスト追加）だが、Phase 0 freeze 中のため着手は Owner 判断 |
| 次の作業 | 統一案の提示と、既存テストへの影響確認 |

### RES2-03 SoftClip 局所 OS の遅延定数が未実測（15 サンプル固定）

| 項目 | 内容 |
|---|---|
| 分類 | 正確性（PDC 申告値） |
| 内容 | `kSoftClipLatencyBaseRateSamples = 15` を無条件に加算している。コード内に「**要実測確認**: processUp→processDown の合成遅延を単位インパルスで測定し補正」と明記されている |
| 発生経緯 | work54 の PDC 実装時に暫定値として導入 |
| 現在の状態 | **未実測のまま**（コメントも残存） |
| 技術的問題点 | 実遅延が 15 と異なる場合、ホストへ申告する PDC が誤り、並列トラックで位相ずれが残る |
| 影響範囲 | SoftClip の局所オーバーサンプリングを使う経路のレイテンシ申告 |
| 依存 | なし（測定のみで完結） |
| 優先度 | P2 |
| 推奨対応 | 単位インパルスで実測し、定数を実測値へ置換。回帰としてレイテンシ整合テストを追加 |
| 根拠 | `src/audioengine/AudioEngine.Processing.Latency.cpp:119-122`、`doc/work54/implementation_plan_2026-06-22.md:478` |
| 必要承認 | 不要（測定と定数更新） |
| 次の作業 | `AudioEngineHarness` にインパルス測定を追加し、現行定数との差を記録 |

## 2. バグ修正が途中で止まっているもの

### RES2-04 F-4 / D2b 記録消失（`overwrite` / `recSeq` / `redrive`）

| 項目 | 内容 |
|---|---|
| 分類 | バグ修正の未完成 |
| 内容 | 記録の消失事象。D2b 診断タプルは main に存在し `[D2B-SELFCHECK] PASS` を確認済みだが、事象自体は再現していない（38 反復で一定値） |
| 発生経緯 | F-4 調査セッション。2026-11-01 に Owner 判断で**調査停止** |
| 現在の状態 | 停止中（未解決のままクローズしていない） |
| 技術的問題点 | 再現条件が不明。診断は「起きていないこと」しか示せていない |
| 影響範囲 | 記録系の信頼性（発生すれば） |
| 依存 | 新証拠（環境・時系列・ログ） |
| 優先度 | P3 |
| 推奨対応 | 新証拠が出るまで再開しない。診断コードは維持 |
| 根拠 | `doc/residual-20261101/RESIDUAL-FINAL-INVENTORY-20261101.md` RES-05、`doc/residual-20261101/LEDGER-SUMMARY-20261101.md` |
| 必要承認 | 再開する場合のみ Owner |
| 次の作業 | なし（監視） |

## 3. 機能追加が途中で止まっているもの

| ID | 内容 | 現在の状態 | 未完成の部分 | 優先度 | 次の作業 | 必要な承認 |
|---|---|---|---|---|---|---|
| RES2-05 | B2（診断カウンタによるメモリ計測） | Stage 1 車両のみ main。Stage 2/3 未着手 | 設計・実装・評価基準 | P2 | Stage 1 の `VALID` 結果を採用可否の判断資料へ整理 | Owner（採用可否） |
| RES2-06 | A2（本番経路の確保カウンタ） | テスト専用プローブのみ main。本番適用は **NO-GO** | 本番適用の設計・実装 | P2 | 設計（変更点・RT 制約・オーバーヘッド・ロールバック）を作成 | Owner（NO-GO 解除） |
| RES2-07 | U5（実時間計測モードの製品有効化） | 測定は metric v2 で **PASS**（#119）。製品は **既定 OFF** | 有効化の設計・回帰検証 | P2 | 有効化する場合の設計と検証計画を作成 | **Owner の明示承認** |
| RES2-08 | `EngineRuntime` 単一 atomic への統合 | 設計メモのみ（`doc/bug.md` の将来改修項目） | 移行設計・実装 | P3 | 現行の dual-write との差分と RT 影響を整理 | 設計承認 |

## 4. 性能・運用の既知事項

| ID | 内容 | 状態 | 影響 | 優先度 | 推奨対応 |
|---|---|---|---|---|---|
| RES2-09 | R-101 の初回実行（oneAPI キャッシュミス時）は約 61 分 | 既知の維持事項。キャッシュヒット時は 51.4 分 | CI の初回のみ 60 分基準を超え得る | P3 | 現状維持。必要ならランナーイメージへの oneAPI プリインストールを検討 |
| RES2-10 | R-113 / R-111 のフレーク | 修正済み・再発なし（ASan 8 連続 success） | 再発時は CI が不安定化 | P3 | 再発が確認されるまで再実行しない（監視のみ） |

## 5. 検証・テストの穴

| ID | 内容 | 現状 | 技術的問題点 | 優先度 | 次の作業 |
|---|---|---|---|---|---|
| RES2-11 | TSan ジョブが `continue-on-error: true` | best-effort。MKL/IPP 非依存のテストのみ | データ競合を検出しても PR を止めない | P3 | 対象テストを増やし、必須化できる範囲を切り分ける |
| RES2-12 | soak シナリオ（S1/S2b/S3/S4/S5）が手動トリガのみ | `workflow_dispatch` のみ | 長時間・高負荷の退行が CI で検出されない | P3 | 夜間スケジュール実行の可否を検討 |
| RES2-13 | TX-02 / TX-10 の実機試験が Waived | 2026-05 の記録でスキップ | IR Replace 単発・automation burst 100 件の実機確認が未実施 | P3 | 実機手順の再実行可否を判断（環境が必要） |
| RES2-14 | `FFTBackendTests` の一部が「現状確認」止まり | `hasPlan`/`isPlanValid` の確認のみ | 出力一致の検証が無い | P3 | 参照実装との突き合わせを追加 |

## 6. 技術的負債

| ID | 内容 | 根拠 | 推奨対応 |
|---|---|---|---|
| RES2-15 | `NoiseShaperLearner` の CMA 状態が mean のみ復元（共分散・履歴は復元しない） | `src/NoiseShaperLearner.cpp:1450-1456` | ウォームスタート品質の要件を決め、必要なら全状態の永続化を設計 |
| RES2-16 | 同ファイルの elite 再評価が「同じ候補の再実行」で、ばらつき低減になっていない | 同 `:700-712` | セグメントのランダム化または評価回数の設計見直し |
| RES2-17 | `D8_1_WrapperCacheTests` の T2b が「tryEnqueue は常に true（バグ）」と書いたまま | `src/tests/D8_1_WrapperCacheTests.cpp:103-110`。実装は既に QueueFull を伝播 | コメントとテストを現状へ更新（実装は修正済み） |
| RES2-18 | `ObserveChannel::Worker` のデバッグアサーションが無い（Worker スレッド未識別） | `src/audioengine/AudioEngine.h:3548-3550` | Worker のスレッド識別を追加し、アサーションを有効化 |
| RES2-19 | `Superseded` / `RecoveryEpisodeId` が Phase-II 送り | `src/audioengine/ISRRuntimePublicationCoordinator.h:252, 362-369` | 設計どおり。移行時に実装（現時点で対応不要） |
| RES2-20 | `C:\VSC_Project\ConvoPeq\.venv` が Python 3.14 消滅で使用不能（216 パッケージ） | 起動時に `C:\Python314\python.exe` 不在で失敗 | venv を使う手順があるなら Python 3.15 で再作成。headroom は uv tool へ移行済み |

## 7. 明示承認待ちの一覧

| 対象 | 承認が必要な理由 | 承認がない場合の扱い |
|---|---|---|
| RES2-01 案 E の採用 | 音声利得の挙動変更 | 既定 OFF を維持 |
| RES2-05 B2 Stage 2/3 | 設計・実装の追加 | Stage 1 のみ |
| RES2-06 A2 本番適用 | 製品コードへ計測を追加 | NO-GO 継続 |
| RES2-07 U5 製品有効化 | 機能の有効化 | 既定 OFF を維持 |
| RES2-04 の再調査 | 調査の再開 | 停止継続 |

## 8. 対応不要・廃止候補

- `measure/u5-20261102`（`[U5ATTR]` 測定専用）: 保持。
- `archive/*` タグ: 削除済みブランチの証跡として保持。
- 旧版の分割 `ConvoPeq.md`（Dropbox の `ConvoPeq.part01〜04.md`）: 現行版へ更新済み。旧版は不要。
- 2026-09 以前の監査文書（`doc/audit/*`、`doc/work*/`）: 修正済み項目を含むため、現行状態の判断には使わない（本報告 §9 のとおり）。

## 9. 「未修正」と書かれているが現在は修正済みのもの（誤認防止）

| 記録上の記述 | 現在の状態 | 根拠 |
|---|---|---|
| H-02: `irPeakLatency` がエネルギー重心 | **修正済み**（重心経路を削除し、真のピーク測定へ一本化） | `src/convolver/ConvolverProcessor.LoaderThreadInline.h:33-35` |
| H-01: dry 遅延と wet の不整合 | **修正済み**（H01-C1 で internal dry alignment を peak のみへ） | `src/convolver/ConvolverProcessor.Runtime.cpp:154-155, 280-295` |
| B-2 診断クラッシュ（グローバル診断バッファ） | **修正済み**（エンジンごとの所有へ移行、旧グローバル撤去） | `AudioEngine.h:538`、`ConvolverProcessor.Runtime.cpp:88`、コミット `cfb0f477` |
| Debug の `assert(prevState == Quarantined)` | **修正済み**（CAS 結果に対する検査へ移動） | `src/audioengine/ISRDSPHandle.cpp:527-532`、コミット `d501afd8` |
| STG-8 のフレーク（R-112 / R-113） | **修正済み**（16 回連続 PASS、ASan 8 連続 success） | 台帳 §10.5・§11.3 |
| `.gitignore` 先頭のコードフェンス | **修正済み**（現在は `# Build artifacts` から始まる） | `.gitignore:1` |
| `opencode.json` の ddgs が Python314 の `uvx.exe` を参照 | **修正済み**（現在は Python315 の `uvx.exe`） | `opencode.json:116-124` |

## 10. 検証方法

- 現行 `main` = `89c48a02`、未コミット 0 件。
- icx Release ビルド（`build-icx`）で `SequenceArithmeticTests`・`DSPHandleTableTests`・`ScopedEpochReaderStructuralTests`・`MpscBoundedRingTests` を実行し、いずれも exit 0。
- コード上の根拠はファイル・行番号で示した。記録のみを根拠にした項目は、記録側の該当箇所を併記した。
- 未確認の推測（例: 実機での可聴影響）は本報告に含めていない。

## 11. 2026-11-05 の処理結果（修正 3 件）

計画（`doc/work113/20261105-1200_RES2処理計画/PLAN.md`）に沿って、修正着手可能と判定した 3 件を独立した PR で処理した。

| ID | 修正前 | 修正内容 | 変更 | 検証 | PR / 統合 |
|---|---|---|---|---|---|
| RES2-17 | `D8_1_WrapperCacheTests` の T2b が「tryEnqueue は常に true（バグ）」と記載（実装は修正済み） | T2b をソース契約の検査へ変更。null ガードと `return owner.enqueueDeferredDeleteNonRt(map, ...)` を要求 | `src/tests/D8_1_WrapperCacheTests.cpp` | icx Release: `D8_1_WrapperCacheTests` Passed 0.14 s | #129 → `65afa479` |
| RES2-14 | `FFTBackendTests` は `hasPlan`/`isPlanValid` のみ | 1024 点で往復誤差 ≤ 1e-9、DC ビン 0.25×N、単一トーン bin 7 の振幅 0.5×N/2 を検査 | `src/tests/FFTBackendTests.cpp` | icx Release: `FFTBackendTests` Passed 0.12 s | #130 → `951e8b9c` |
| RES2-02 | スカラーは `abs(x) > 2^53`、AVX2 は `abs(x) > 1e20` | 閾値と両判定を `CorruptSampleGuard.h` に集約し 2^53 へ統一。境界テストを追加 | `src/CorruptSampleGuard.h`、`src/CustomInputOversampler.cpp`、`src/tests/CorruptSampleGuardTests.cpp`、`CMakeLists.txt` | `CorruptSampleGuardTests` Passed 0.17 s、`PolyphaseGainFidelityTests` Passed 1.45 s、`ConvoPeq` 再リンク成功 | #131 → `ad19211c` |

- 3 PR の CI はいずれも 14 pass / 0 fail / 0 pending（ASan を含む）。
- 製品の RT 制約に対する追加はない（RES2-02 は既存 inline 判定の定数統一のみ、他 2 件はテストのみ）。
- 未承認機能（案 E・U5・B2 Stage 2/3・A2 本番適用）には触れていない。

## 12. 2026-11-05 詳細調査（RES2・U5・案 E）

台帳 20 件を現行コードで再確認し、RES2-03 と RES2-08 を解消した。U5 と案 E は横断調査し、OFF/ON の実測を取得した。

| 項目 | 結果 |
|---|---|
| RES2-03 | `PolyphaseGainFidelityTests` の P0-F が S1(31/90) の argmax を D=15 で検証済み。定数は実測で裏付けられており、コメントを実測根拠へ更新（挙動変更なし） |
| RES2-08 | `RuntimePublishState` は現行に存在せず、`EngineRuntime` は `[[deprecated]]`。`RuntimeWorld`／`worldAuthority` が公開モデルであり、旧 dual-write 統合の前提は消滅 |
| 案 E（RES2-01） | OFF 実測: DC 0.75^N、cand 1.0、SoftClip DC 0.375→0.5、54/54 PASS。ON ビルド: DC 1.0、19/54 PASS（OFF 期待値に対する正の対照、P1-1 の記録と一致） |
| U5（RES2-07） | 既定 `Mode::Off`、環境変数のみ。リング 8,192・dropped 計上・INCONCLUSIVE 条件を確認。UI 切替経路は無い |
| 資料 | `RES2-DETAILED-INVESTIGATION-20261105.md`、`RES2-ITEM-ACTIONS-20261105.md`、`U5-TECHNICAL-INVESTIGATION-20261105.md`、`OPTION-E-GAIN-REPORT-20261105.md`、`RES2-OWNER-DECISIONS-20261105.md`、`VERIFICATION-RESULTS-20261105.md`、`RESIDUAL-RISKS-20261105.md` |

## 13. 2026-11-05 全件処理（数値解析）

聴感評価を用いず、数値解析・自動テスト・性能測定で全 20 件を再評価した。

| 項目 | 追加した数値証拠 |
|---|---|
| RES2-01 | `PolyphaseGainMetrics` で OFF/ON の DC・RMS・ピーク・周波数応答・レイテンシ・CPU を取得。ON は DC 1.0、ピーク比 1.03〜1.67（最大 +4.4 dB）、レイテンシ不変 |
| RES2-07 | `RtMeasureOverheadTests` で計測器のコストを測定: Off 1.34 / TimeOnly 26.84 / Full 26.60 ns/ブロック |
| RES2-18 | Worker 経路に `debugAssertNotAudioThread()` を追加（Release は空関数へインライン化）。`ConvoPeq` と `AudioEngineHarness` の再ビルド成功 |

- 追加テスト: `PolyphaseGainMetrics`、`RtMeasureOverheadTests`（いずれも test-only・閾値なし）。
- 未解決 13 件の内訳: 承認待ち 5（01・04・05・06・07）、設計判断 4（12・15・16・20）、追加調査 2（11・13）、検証 2（09・10）。RES2-19 は対応不要。
