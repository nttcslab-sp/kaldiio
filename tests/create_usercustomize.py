import os
import site
import sys


print(sys.version_info)
if sys.version_info[0] == '2':
    up = site.getusersitepackages()
    if not os.path.exists(up):
        os.makedirs(up)

    with open(os.path.join(up, 'usercustomize.py'), 'w') as f:
        f.write('''import sys
sys.setdefaultencoding('utf-8')
''')
