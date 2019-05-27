import os
import site
import sys


if sys.version_info[0] == '2':
    print(sys.version_info)
    up = site.getusersitepackages()
    if not os.path.exists(up):
        os.makedirs(up)

    with open(os.path.join(up, 'usercustomize.py'), 'w') as f:
        f.write('''import sys
sys.setdefaultencoding('utf-8')
''')
