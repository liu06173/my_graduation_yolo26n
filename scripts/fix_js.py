import re

f = 'E:/claude/end001/my_graduation_yolo26n/webapp/templates/index.html'
t = open(f, encoding='utf-8').read()

# Fix all remaining stray backticks
# Pattern: trailing backtick-semicolon after single-quote string
t = t.replace("'+e+'</span>`;", "'+e+'</span>';")
t = t.replace("');`", "');")

# Fix any remaining backtick-semicolons
t = re.sub(r'`\)\s*;', "');", t)

# Count backticks in scripts
scripts = re.findall(r'<script>(.*?)</script>', t, re.DOTALL)
all_even = True
for i, s in enumerate(scripts):
    bt = s.count('`')
    ok = bt % 2 == 0
    print(f'Script {i}: {bt} backticks (even={ok})')
    if not ok:
        all_even = False
        lines = s.split('\n')
        running = 0
        for j, line in enumerate(lines, 1):
            b = line.count('`')
            if b > 0:
                running += b
                if running % 2 == 1:
                    clean = line.strip()[:120]
                    print(f'  ODD L{j}: {clean}')

if all_even:
    print('ALL BACKTICKS BALANCED!')
    open(f, 'w', encoding='utf-8').write(t)
else:
    print('STILL HAS ISSUES - not writing file')
