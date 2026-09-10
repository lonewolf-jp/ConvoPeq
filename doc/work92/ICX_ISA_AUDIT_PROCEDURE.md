# icx 生成 ISA 実測手順（work92 C-8 / AC-C8-4）

- **目的**: icx `/QxCORE-AVX2` でコンパイルした DSP コードが Intel 専用命令（AVX-512/AMX 依存命令等）を生成していないことを実測で確認し、AMD CPU 実行可否の判断材料を維持する。
- **初回実測**: 2026-09-09（PLAN v3 §5-0 G6 監査）— 結論: 代表ループは AVX2 サブセット内・Intel 専用命令 0 件。
- **再実行タイミング**: CMake の AVX2 系フラグ変更・icx バージョン更新・AVX-512/AMX 依存コードの新規追加時。

## 1. 対象ループの準備

`MKLNonUniformConvolver.cpp` の SIMD 積算パターン（aligned/unaligned 分岐 load → add → store）を単一ファイルに抽出する。実例:

```cpp
// isa_probe.cpp — G6 実測と同一パターン
#include <immintrin.h>

void addLoop(double* dst, const double* src, long long n, int aligned)
{
    long long i = 0;
    const bool isAligned = aligned != 0;
    for (; i + 4 <= n; i += 4)
    {
        const __m256d a = isAligned ? _mm256_load_pd(dst + i) : _mm256_loadu_pd(dst + i);
        const __m256d b = isAligned ? _mm256_load_pd(src + i) : _mm256_loadu_pd(src + i);
        if (isAligned)
            _mm256_store_pd(dst + i, _mm256_add_pd(a, b));
        else
            _mm256_storeu_pd(dst + i, _mm256_add_pd(a, b));
    }
    for (; i < n; ++i)
        dst[i] += src[i];
}
```

## 2. コンパイル（2 フラグで比較）

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 >nul

icx /O2 /QxCORE-AVX2 /FA /c isa_probe.cpp /Foisa_avx2.obj
icx /O2 /arch:AVX2  /FA /c isa_probe.cpp /Foisa_msvc.obj
```

- `/FA` でアセンブリ（.asm）を出力する。
- 対象が `ConvoPeq` 本体の場合は `build-icx` の link コマンドから該当 .obj の compile line を `impl-Release.ninja` から複製する（`/QxCORE-AVX2` と `/FA` を付与）。

## 3. 比較方法

```bat
fc isa_probe.asm isa_msvc.asm > diff.txt
```

確認項目:

1. **Intel 専用命令の有無**: `vpcompressd` / `zmm` レジスタ / `vgather` 系 / `kmov` / `valignd` / `vpconflict` 等が **0 件**であること
2. **命令セットの一致**: 両フラグで VADDPD / VMOVUPD(LOADUPD) / VZEROUPPER + スカラー命令のセットが同一であること
3. 差分が load/store のスケジューリング順のみであること（演算命令の集合変化がないこと）

## 4. 判定

| 結果 | 判定 |
|---|---|
| Intel 専用命令 0 件 & 命令セット同一 | **AVX2 サブセット内** — README の matrix（icx AMD = unsupported だが AVX2 subset 実測あり）を維持 |
| Intel 専用命令が生成された | matrix の「AVX2 subset verified」行を削除し、「AVX-512 依存の可能性あり（AMD 実行は動作保証外）」に更新。必要なら `/arch:AVX2` への切替を検討 |

## 5. 記録

再実測時は本ファイルに「実施日 / icx バージョン / 対象ループ / 結果」を追記する。

| 実施日 | icx | 対象 | 結果 |
|---|---|---|---|
| 2026-09-09 | 2026.1 (Build 20260617) | convolver load/add/store ループ | AVX2 サブセット内・Intel 専用命令 0 件（G6 PASS） |
