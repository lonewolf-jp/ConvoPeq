# コンボルバーレビュー検証 — 相互評価レポート

**日付**: 2026-06-23
**対象**: `doc/work55/bug.md`（レビュー要旨）および `doc/work55/ConvoPeq_Convolver_Validation_Report.md`（添付詳細レポート）
**方法**: ソースコード静的解析（Serena MCP / CodeGraph MCP / semble / grep / AiDex / graphify）＋詳細レポートとの突き合わせ

---

## 0. 要旨

レビューの **6つの発見事項（F1〜F6）は全て本質的に妥当** であり、詳細レポートと私の独自検証は相互に補完的である。レポートは文献引用の質・数学的検証の厳密さで優れ、私の検証は影響範囲の正確な限定・用語の精緻化で補完する。

---

## 1. 各発見事項の評価

### F1: Mixed Phase クロスオーバー方向

| 項目 | 内容 |
| --- | --- |
| **レビューの主張** | 低域=Linear Phase / 高域=Minimum Phase は業界標準と逆の可能性 |
| **コード上の事実** | ✅ `ConvolverProcessor.MixedPhase.cpp` L303-311: `wLinear=1.0 (freq<Lo) → 0.0 (freq>Hi)` のコサインクロスフェードを確認 |
| **レポートの評価** | 「高確信度の疑義」— Dirac Research ホワイトペーパー・プロオーディオフォーラムの共通見解を引用し、方向逆転を主張 |
| **私の評価** | ⚠️ **レポートの文献調査により、当初の「議論の余地」評価から「より高い確信度の疑義」に上方修正**。特にデフォルトクロスオーバー帯域(200/1000Hz)が業界標準と一致する点は、実装意図と現在の重みが入れ替わっている強い状況証拠。ただし「確定したバグ」ではなく「要検証」とする留保は妥当。 |

### F2: `mixedPreRingTau` 無効

| 項目 | 内容 |
| --- | --- |
| **レビューの主張** | 音響的に無効なパラメータ |
| **コード上の事実** | ✅ `convertToMixedPhaseAllpass()` 内で `tau` はキャッシュキーのみで使用。`AllpassDesignerConfig` に `tau` フィールドなし。`convertToMixedPhaseFallback()` では `(void)tau;` |
| **評価** | ✅ **完全一致。確定した問題。** |

### F3: r8brain IRテイル切り捨てリスク

| 項目 | 内容 |
| --- | --- |
| **レビューの主張** | `IRDSP::resampleIR()` の `+2.0` マージンが不十分 |
| **コード上の事実** | ✅ `IRDSP.cpp` L18: `expectedLen = inLen * ratio + 2.0`。`CDSPResampler::getLatency()` は `return(0)`（CDSPResampler.h L486-488） |
| **レポートの評価** | Harris近似式 `N ≈ 140/(22·0.02) ≈ 318タップ` で定量的に裏付け |
| **私の評価** | ✅ **本質的に正しいが、影響範囲の正確な限定が必要**。問題が実際に影響するのは `IRDSP::resampleIR()` パス（`IRConverter.cpp` からの呼び出し）のみ。メインコンボルバーローダーは `ConvolverProcessorInternal::resampleIR()`（`ResampleAndFallback.cpp`）で `getMaxOutLen()` を正しく使用しており安全。レポートの「全てのIRロードに影響」は訂正が望ましい。 |

### F4: `computeMasteringSizing` 無効

| 項目 | 内容 |
| --- | --- |
| **レビューの主張** | 計算結果が NUC 構成に影響しない |
| **コード上の事実** | ✅ `StereoConvolver::init()` → `SetImpulse()` の経路で引数が存在せず、`storedMaxFFTSize`/`storedFirstPartition` に保存されるのみ。NUCのパーティションサイズは `nextPow2(max(blockSize, 64))` で独自計算 |
| **レポートの評価** | 「デッドコード（無効化された配線）」— 全243ファイル網羅検索を確認 |
| **私の評価** | ✅ **本質的に正しい。ただし用語の订正を提案**: 「デッドコード（dead code）」は未使用関数を連想させるが、`computeMasteringSizing` 自体は `Lifecycle.cpp`・`LoaderThread.cpp` で呼ばれており、結果は `clone()` 経路で `init()` に再投入される。より正確には **「orphan calculation（孤立計算）」** または **「無効化された配線」** と分類すべき。 |

### F5: `applyAllpassToIR` 未使用

| 項目 | 内容 |
| --- | --- |
| **レビューの主張** | 定義されているが呼び出し箇所なし |
| **コード上の事実** | ✅ `AllpassDesigner.h` L115 で宣言、`AllpassDesigner.cpp` L595 で実装。プロジェクト全体で呼び出しゼロ。MixedPhase変換はインラインFFT実装で代替 |
| **評価** | ✅ **完全一致。確定した問題。** |

### F6: レイヤースケジューリング（固定倍数8）

| 項目 | 内容 |
| --- | --- |
| **レビューの主張** | Garcia/Wefers の DP 最適解とは異なる Gardner 流ヒューリスティック。バグではない |
| **コード上の事実** | ✅ `MKLNonUniformConvolver.cpp` L604-608: `l1Part = l0Part * tailL1L2Mult`（デフォルト8、UI範囲2-16） |
| **レポートの評価** | Garcia(2002)から「最適解は倍々ではなく4倍以上」の直接引用を提示。ConvoPeqの「固定倍数8」が方向性として整合することを示唆 |
| **私の評価** | ✅ **妥当な評価。レポートの Garcia 直接引用により文献的裏付けが強化された。** |

---

## 2. 私の検証 vs 詳細レポート

### レポートが優れている点

1. **文献引用の質と網羅性**: Gardner(1995), Garcia(2002), Wefers(2015), Deczky(1972), Stockham(1966), Dirac Research ホワイトペーパー — 各主張に具体的な文献を明記
2. **FDL「2倍バッファ＋ミラー書き込み」の完全な数学的検証**: `Y[n] = ΣX[n-p]·H[p]` からの変形で正しさを証明
3. **Harris近似式によるr8brainテイル長の定量化**: `N ≈ 140/(22·0.02) ≈ 318タップ`
4. **周辺実装の網羅的健全性チェック**: denormal対策・メモリ管理・オートゲイン・Dry/Wetクロスフェード・RCU/スレッド安全性

### 私の検証が追加できる点

1. **F3の影響範囲限定**: 問題は `IRDSP::resampleIR()`（`IRConverter.cpp` パス）限定であり、メインローダー（`ConvolverProcessorInternal::resampleIR()`）は `getMaxOutLen()` 使用で安全
2. **F4の用語精緻化**: 「デッドコード」→「孤立計算（orphan calculation）」または「無効化された配線」
3. **F1のトーン調整根拠**: `convertToMixedPhaseFallback()` も同一のクロスオーバー方向であることを確認

---

## 3. 検証で使用したツール一覧

| ツール | 使用目的 | 成果 |
| --- | --- | --- |
| **Serena MCP** | コード検索・シンボル特定・依存関係分析 | 全キー関数の定義・呼び出し箇所を特定 |
| **CodeGraph MCP** | コミュニティ検索・エンティティ関連分析 | コードベース全体の構造把握 |
| **graphify MCP** | グラフ統計・近傍ノード探索 | ノード17,058、エッジ22,445のグラフ確認 |
| **semble** | セマンティックコード検索 | データフローの追跡確認 |
| **AiDex** | 識別子検索・ファイル構造確認 | セッション管理・コードナビゲーション |
| **grep/Select-String** | r8brainライブラリ内のシンボル検索 | `getLatency()`, `getMaxOutLen()` 等の動作確認 |
| **vscode-websearch** | 文献調査 | Mixed Phase 業界標準の確認 |

---

## 4. 優先度付き推奨アクション

| 優先度 | 項目 | アクション | 備考 |
| --- | --- | --- | --- |
| **高** | F1: Mixed Phase 方向 | クロスオーバー重み `wLinear`/`wMinimum` の入れ替えを検討。実機IR測定で確認後、修正判断 | レポートの文献調査が強力な裏付け |
| **高** | F3: r8brain テイル | `IRDSP::resampleIR()` のバッファサイズ計算を `getMaxOutLen()` ベースに修正 | 影響は `IRConverter.cpp` パスのみ |
| 中 | F2: `mixedPreRingTau` | パラメータを実際の設計ロジックに結合するか、UIから削除 | 既知事項 |
| 低 | F4: `computeMasteringSizing` | 関数削除、または NUC の `SetImpulse()` に引数追加して実際に反映 | リファクタリング時の対応で可 |
| 低 | F5: `applyAllpassToIR` | 削除、または使用箇所へ統合 | 保守性向上 |
| 情報 | F6: レイヤースケジューリング | 最適化余地として文書化 | Garcia/Wefers型の厳密最適化は将来課題 |

---

## 5. 参考文献（本検証で参照）

1. Gardner, W. G. (1995). *Efficient Convolution without Input-Output Delay*. J. Audio Eng. Soc., 43(3), 127–136.
2. Garcia, G. (2002). *Optimal Filter Partition for Efficient Convolution with Short Input/Output Delay*. AES 113th Convention. (特許 US6625629)
3. Wefers, F. (2015). *Partitioned convolution algorithms for real-time auralization*. PhD thesis, RWTH Aachen.
4. Stockham, T. G. (1966). *High-speed convolution and correlation*. AFIPS Conference Proceedings.
5. Deczky, A. G. (1972). *Synthesis of recursive digital filters using the minimum p-error criterion*. IEEE Trans. Audio Electroacoust.
6. Dirac Research AB. *On Room Correction and Equalization of Sound Systems* (technical whitepaper).
7. Oppenheim, A. V., & Schafer, R. W. *Discrete-Time Signal Processing*.
8. Vaneev, A. *r8brain-free-src* — GitHub: avaneev/r8brain-free-src.
9. プロジェクト内部資料: `doc/work55/bug.md`, `doc/work55/ConvoPeq_Convolver_Validation_Report.md`

---

*本レポートは2026-06-23時点の `lonewolf-jp/ConvoPeq` のソースコードに対する静的解析と文献調査に基づく。*
