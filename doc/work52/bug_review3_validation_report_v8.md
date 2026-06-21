# bug_review3.md 検証レポート v8（テスト結果反映版）

- **作成日**: 2026-06-21
- **対象**: `doc/work52/bug_review3.md`

---

## 0. 全テスト結果一覧

| # | テスト | 結果 | 判定 |
|:-:|:------|:----:|:----|
| 1 | PEQ-only | 正常 | — |
| 2 | Conv→Peq | **発生** | 必須条件 |
| 3 | Adaptive9th → Fixed4Tap | 発生 | NS固有では**ない** |
| 4 | Adaptive9th → Fixed15Tap | 発生 | NS固有では**ない** |
| 5 | **NoiseShaper完全OFF** | **発生** | **NS系統は全て除外** |
| 6 | SoftClip OFF | 発生 | SoftClip除外 |
| 7 | Saturation = 0 | 発生 | SoftClip内部非線形も除外 |
| 8 | OS = 1x | 発生 | Downsampler除外 |
| 9 | PEQ全バンド 0dB | 発生 | EQ相互作用除外 |
| 10 | IR長 86ms | 発生 | 長テール不要 |
| 11 | ボーン（単発）→ ジー | 確認 | 状態励振型 |
| 12 | ボンボンボン（連打）→ ジジジジ | 確認 | 状態蓄積型 |

### 除外された仮説（12テスト中11で否定）

| 仮説 | 除外理由 |
|:----|:---------|
| Adaptive9th 固有バグ（P7 advanceState） | Fixed系でも発生 |
| NoiseShaper 系統（全方式） | **NS完全OFFでも発生** |
| SoftClip | OFFでも発生 |
| Oversampling Downsampler | OS=1xでも発生 |
| PEQ 設定（EQブースト） | 全バンド0dBでも発生 |
| ゲインステージング（レベル差） | 設定変更で不変 |
| post-NS ハードクランプ | NS OFF時も clamp は常時動作→「低音のみ」の説明不可 |
| プリリンギング | 86ms短IR＋低音持続音で説明困難 |
| IRの長い残響テール | 86ms短IRでも発生 |

### 残った事実

**Convolver が動作していること** だけが唯一の必要条件。PEQモードでは絶対に発生しない。

---

## 1. ソースコードから見える Convolver の構造

### 1.1 処理フロー

```
Add(input, numSamples)
  ├─ processDirectBlock()  → 時間領域FIR（IRの先頭タップ）
  └─ processLayerBlock()   → 周波数領域（FFT重畳加算） → ringWrite()

Get(output, numSamples)
  ├─ ringRead(output)      → リングバッファからFFT処理結果を読出
  └─ addFallback()         → 直接畳み込み結果を加算
```

### 1.2 リングバッファの異常時動作

```cpp
// ringRead() - toRead == 0 時（リングバッファ空）
if (toRead == 0) {
    if (dst) memset(dst, 0, n * sizeof(double));  // ← ゼロ埋め！
    return 0;
}

// toRead < n 時（部分的不足）
if (toRead < n)
    memset(dst + toRead, 0, (n - toRead) * sizeof(double));  // ← 部分ゼロ埋め
```

リングバッファ不足時、出力がゼロ埋めされる。このゼロ→信号の急峻な遷移は後段にインパルスを与える。ただし低音特化性の説明は困難。

### 1.3 直接畳み込み（processDirectBlock）

```cpp
// 時間領域FIR（IRの先頭 tapCount タップを直接畳み込み）
m_directOutBuf[processed + n] = Σ_{k=0}^{tapCount-1} IR[k] × input[n+k];
// ブロック間履歴を m_directHistory で継承
```

標準的なFIR畳み込み。ブロック境界の連続性は `m_directHistory` で保証されている。

---

## 2. 現時点での最有力仮説

**全テストで消去法的に残った唯一の原因:** Convolver（IRとの畳み込み）そのもの。

### 2.1 IRの低域共振仮説（45%）

```
IRに含まれる特定周波数の共振（40-80Hzのroom mode等）
  ↓ 低音入力で共振が励起される
  ↓ 86msのIR内で共振が持続
  ↓ 共振周波数によっては可聴域の「ジー」に聞こえる
  ↓ 連打で共振が重なり「ジジジジ」
```

**根拠**:
- Convolverが唯一の必要条件
- 「ボーン→ジー」は共振の励起＋減衰で説明可能
- IRのエネルギー正規化（scaleFactor）はRMSベースであり、特定周波数のQ値を下げない

**反証**: 86ms IR内の共振が「ジー」という高域成分になるとは限らない。

### 2.2 FFTパーティション境界仮説（25%）

```
パーティション境界（Overlap-Saveの継ぎ目）
  ↓ 低域の長い信号で微細な不連続が発生
  ↓ この不連続が後段で「ジー」に変換
```

**根拠**: 低域ほどサンプル間相関が高く、境界の微細な誤差が目立つ。

### 2.3 Convolver＋DC Blocker相互作用（20%）

```
Convolver出力の低域成分
  ↓ output DC Blocker（2nd-order IIR）の内部状態を励振
  ↓ IIR状態の過渡応答 → 「ジー」
```

**根拠**: DC Blockerは全モード共通だが、Convolver経由で波形が変形されている場合、IIRの応答が異なる。

---

## 3. 推奨切り分けテスト

### テストA（最優先）: Convolver出力の直接確認

`processDirectBlock()` の出力と `Get()` の出力を比較。
Convolver単体で異常がないか確認。

実装案:
```cpp
// DSPCoreDouble.cpp の convolverRt().process() 直後に挿入（診断用）
// ブロック単位で max(abs(data)) をログ出力
```

### テストB: IRのDC/超低域除去

問題のIRをAudacity等で開き、**DCオフセット除去**＋**20Hzハイパスフィルタ**をかけたIRを作成してテスト。

改善する → IR由来の超低域/DC成分が原因の一部。

### テストC: 異なる種類のIRで比較

| IRタイプ | 期待 |
|:---------|:----|
| 非常に短いIR（〜10ms） | 共振が少ない → 改善するか |
| 合成IR（ホワイトノイズのFIR） | 特定共振なし → 改善するか |
| テスト用IR（単一インパルス） | 原音そのまま → 改善するか |

---

## 4. 結論

| 項目 | 判定 |
|:----|:-----|
| NS系統全般 | ✅ **除外**（完全OFFでも発生） |
| SoftClip | ✅ **除外** |
| OS Downsampler | ✅ **除外** |
| PEQ設定 | ✅ **除外** |
| ゲインステージング | ✅ **除外** |
| **Convolver（IR畳み込み）** | **残った唯一の原因** |
| うちIR低域共振 | 45%（最有力） |
| うちFFTパーティション境界 | 25% |
| うちDC Blocker相互作用 | 20% |

**残った唯一の原因領域は「Convolver（IRとの畳み込み）」である。これ以上のコード解析では確定できず、IRの実測とConvolver出力の直接確認が必要。**
