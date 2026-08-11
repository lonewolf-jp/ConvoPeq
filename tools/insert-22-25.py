# -*- coding: utf-8 -*-
"""
§2.1 即時改修表に 2-2 / 2-5 を挿入する（3回目・保存保証版）。
前回スクリプトは R-新規D 未発見で sys.exit(1) し、保存前に終了したため挿入が失われた。
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


def find_line(lines, prefix, start=0, end=None):
    for i, ln in enumerate(lines[start:end] if end else lines[start:], start):
        if ln.rstrip('\r').startswith(prefix):
            return i
    return -1


def main():
    lines = load_lines()

    sec_start = find_line(lines, '### 2.1 即時改修')
    sec_end = find_line(lines, '### 2.2 設計判断')
    if sec_start < 0 or sec_end < 0:
        print('[FAIL] §2.1 range not found')
        sys.exit(1)
    print(f'[INFO] §2.1 range: {sec_start}..{sec_end}')

    # (アンカー, 前/後, 挿入行)
    inserts = [
        ('| 2-3 | ユニットテストの矛盾条件', 'before',
         '| 2-2 | 入力 DC ブロッカー NaN/Inf 非対称 | DSPCoreIO.cpp:231,252,305 | Low | 入力側 DC 後スクラブ追加（§3-12）|'),
        ('| 2-9 | タイミング計算の uint64 underflow', 'before',
         '| 2-5 | LockFreeRingBuffer::size() データ競合 | LockFreeRingBuffer.h:76 | Medium | クランプ + コメント明記（§3-15）|'),
    ]

    inserted = 0
    for anchor, pos, new_row in inserts:
        idx = find_line(lines, anchor, sec_start, sec_end)
        if idx < 0:
            print(f'[FAIL] anchor not found in §2.1: {anchor}')
            sys.exit(1)
        token = new_row.split('|')[1].strip()
        dup = find_line(lines, f'| {token} |', sec_start, sec_end)
        if dup >= 0:
            print(f'[SKIP] already in §2.1: {token}')
        else:
            insert_at = idx if pos == 'before' else idx + 1
            lines.insert(insert_at, new_row + '\r')
            inserted += 1
            print(f'[OK] §2.1 inserted {pos}: {token}')

    if inserted == 0:
        print('[INFO] nothing to insert')
    save_lines(lines)
    print('DONE: written')


if __name__ == '__main__':
    main()
