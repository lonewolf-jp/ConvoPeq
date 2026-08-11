# -*- coding: utf-8 -*-
"""
INTEGRATED_BUG_FIX.md の §2.2 設計判断表を現状（詳細設計 §3 の確定状態）と整合させる。

- 見出し: "設計判断（要精査）" → "設計判断（確定済み・一部要精査）"
- 各確定済み行の「設計判断点」列に **確定（§3-X）** を追記
- 未確定行（2-7）のみ「要精査」を明示
"""
import sys

PATH = r'c:\VSC_Project\ConvoPeq\doc\work88\big_bug\INTEGRATED_BUG_FIX.md'


def load_lines():
    with open(PATH, 'rb') as f:
        data = f.read()
    return data.decode('utf-8').split('\n')


def save_lines(lines):
    out = '\n'.join(lines)
    with open(PATH, 'wb') as f:
        f.write(out.encode('utf-8'))


# 置換対象: (行頭プレフィックス, 新しい行全体)
NEW_ROWS = {
    '| 1-4 |': '| 1-4 | `fastTanh` 3箇所独立複製 | DSPCoreFloat.cpp:146 / DSPCoreIO.cpp:76 | Medium | **確定（§3-4）**: 統合は NO-OP（係数ビット一致）。DSPCoreFloat/IO の fastTanh を SoftClipPadeApproxPolicy に統一 |',
    '| 1-7 |': '| 1-7 | ISRRetire Mutex | ISRRetire.cpp:94 | High | **確定（§3-7）**: リネーム + コメント強化。overflowRing ロックフリー化は将来 |',
    '| 1-8 |': '| 1-8 | LoaderThread OOM | LoaderThread.cpp:463 | High | **確定（§3-8）**: ストリーミング読み込み化 + MAX_FILE_LENGTH 維持 |',
    '| 1-9 |': '| 1-9 | int/size_t 混在 | MKLNonUniformConvolver.cpp:843 | High | **確定（§3-9）**: 割当サイト cast + :778 乗算順序修正（最小修正） |',
    '| 1-10 |': '| 1-10 | `m_pendingIRChange` 公開前クリア | Snapshot.cpp:95 | High | **確定（§3-10）**: クリアタイミング遅延（現状記載の技術的誤りを修正） |',
    '| 2-6 |': '| 2-6 | RCUReader ハッシュ衝突 | RCUReader.h:51,152 | High | **確定（§3-16）**: 前提不成立（severity 下方）。堅牢性改善のみ |',
    '| 2-7 |': '| 2-7 | atomic<DSPHandle> ロックフリー検証 | ISRDSPHandle.h:186 | Medium | **要精査（§3-17）**: Release でも abort 検証（案A/B/C 未決定） |',
    '| 3-1 |': '| 3-1 | AudioSegmentBuffer リングラップ競合 | AudioSegmentBuffer.h:50 | High | **確定（§3-21）**: 前提不成立（同一スレッド）。防御的強化 |',
    '| 3-6 |': '| 3-6 | SnapshotFactory NaN ハッシュ不一致 | SnapshotFactory.cpp:36 | Medium | **確定（§3-26）**: 両面対応（hash 正準化 + equivalence fail-closed） |',
    '| 3-9 |': '| 3-9 | /fp:fast 精度低下 | CMakeLists.txt | Medium | **確定（§3-29）**: ターゲット固有化（ConvoPeq 個別へ） |',
    '| 3-10 |': '| 3-10 | /QxCORE-AVX2 AMD 非互換 | CMakeLists.txt | Medium | **確定（§3-30）**: ターゲット固有化（MSVC と同型に） |',
    '| R-9 |': '| R-9 | CMAKE_CXX_FLAGS_RELEASE グローバル上書き | CMakeLists.txt | Medium | **確定（§3-31）**: target_compile_options 移行（3-9/3-10 と一体） |',
}


def find_line(lines, prefix, start=0, end=None):
    for i, ln in enumerate(lines[start:end] if end else lines[start:], start):
        if ln.rstrip('\r').startswith(prefix):
            return i
    return -1


def main():
    lines = load_lines()

    # 見出し更新
    h_idx = find_line(lines, '### 2.2 設計判断')
    if h_idx < 0:
        print('[FAIL] §2.2 heading not found')
        sys.exit(1)
    lines[h_idx] = '### 2.2 設計判断（確定済み・一部要精査）\r'
    print('[OK] §2.2 heading updated')

    sec_start = find_line(lines, '### 2.2 設計判断')
    sec_end = find_line(lines, '## A. 現状維持')
    if sec_end < 0:
        print('[FAIL] ## A. not found')
        sys.exit(1)
    print(f'[INFO] §2.2 range: {sec_start}..{sec_end}')

    replaced = 0
    for i in range(sec_start, sec_end):
        s = lines[i].rstrip('\r')
        for prefix, new_row in NEW_ROWS.items():
            if s.startswith(prefix):
                lines[i] = new_row + '\r'
                replaced += 1
                break
    print(f'[INFO] §2.2 rows updated: {replaced}')

    save_lines(lines)
    print('DONE: written')


if __name__ == '__main__':
    main()
