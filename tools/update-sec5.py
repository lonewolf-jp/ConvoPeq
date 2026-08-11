# -*- coding: utf-8 -*-
"""§5 検証計画に 2-2（NaN スクラブ）の検証行を追加する。"""
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


def find_line(lines, prefix, start=0):
    for i, ln in enumerate(lines[start:], start):
        if ln.rstrip('\r').startswith(prefix):
            return i
    return -1


def main():
    lines = load_lines()
    sec_start = find_line(lines, '## 5. 検証計画')
    sec_end = find_line(lines, '## 6. 実施上の注意')
    if sec_start < 0 or sec_end < 0:
        print('[FAIL] §5 range not found')
        sys.exit(1)

    # 「データ競合」行の直後に「データ完全性」行を追加（既存チェック）
    anchor = '| データ競合 |'
    dup = None
    for i in range(sec_start, sec_end):
        if lines[i].rstrip('\r').startswith('| データ完全性 |'):
            dup = i
            break
    if dup is not None:
        print('[SKIP] データ完全性 row already present')
    else:
        idx = -1
        for i in range(sec_start, sec_end):
            if lines[i].rstrip('\r').startswith(anchor):
                idx = i
                break
        if idx < 0:
            print('[FAIL] データ競合 row not found')
            sys.exit(1)
        lines.insert(idx + 1, '| データ完全性 | NaN/Inf 入力を DC ブロッカー通過後に検出 | 2-2 |\r')
        print('[OK] inserted データ完全性 row')

    save_lines(lines)
    print('DONE: written')


if __name__ == '__main__':
    main()
