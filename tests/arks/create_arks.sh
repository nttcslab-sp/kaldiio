#!/bin/sh

# python -c 'import kaldiio as k;import numpy as np;k.save_ark("test.ark", {"test" + str(i): np.random.randn(10, 20).astype(np.float32) for i in range(3)})'
# type=CM
copy-feats --compress=true --compression-method=1 ark:test.ark ark:test.cm1.ark
# type=CM2
copy-feats --compress=true --compression-method=3 ark:test.ark ark:test.cm3.ark
# type=CM3
copy-feats --compress=true --compression-method=5 ark:test.ark ark:test.cm5.ark
copy-feats ark:test.ark ark,t:test.text.ark
