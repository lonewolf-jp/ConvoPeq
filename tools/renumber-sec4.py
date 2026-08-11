# -*- coding: utf-8 -*-
"""INTEGRATED_BUG_FIX.md の §4 実装順序の通し番号を振り直す（2 項目移動で欠番解消）。"""
import re
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
    sec_start = find_line(lines, '## 4. 実装順序')
    sec_end = find_line(lines, '## 5. 検証計画')
    if sec_start < 0 or sec_end < 0:
        print('[FAIL] §4 range not found')
        sys.exit(1)

    counter = 10  # フェーズ 1 は 10 から
    renumber = False
    updated = 0
    for i in range(sec_start, sec_end):
        s = lines[i].rstrip('\r')
        m = re.match(r'^(\d+)\.\s', s)
        if not m:
            continue
        num = int(m.group(1))
        if num == 10:
            renumber = True
        if renumber:
            new_s = re.sub(r'^\d+\.\s', f'{counter}. ', s)
            lines[i] = new_s + '\r'
            updated += 1
            counter += 1
    print(f'[INFO] renumbered: {updated} rows')

    save_lines(lines)
    print('DONE: written')


if __name__ == '__main__':
    main()
