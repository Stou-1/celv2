# Frames [first, first + n) of a y4m into another (C17 step 0b's dumps). usage: python -I y4mcut.py src dst first n
import sys
src, dst, first, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
f = open(src, 'rb'); head = f.readline()
w = int([t for t in head.split() if t.startswith(b'W')][0][1:]); h = int([t for t in head.split() if t.startswith(b'H')][0][1:])
size = w * h * 3 // 2
o = open(dst, 'wb'); o.write(head)
for i in range(first + n):
    tag = f.readline(); data = f.read(size)
    if len(data) < size: break
    if i >= first: o.write(tag); o.write(data)
