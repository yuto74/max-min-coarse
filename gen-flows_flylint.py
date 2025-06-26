#!/usr/bin/env python3
#
#
# Copyright (c) 2024, Hiroyuki Ohsaki.
# All rights reserved.
#
# $Id: gen-flows.py,v 1.1 2024/06/27 07:38:19 ohsaki Exp $
#

import sys
import random

from perlcompat import die, warn, getopts
import tbdump


n = int(sys.argv[1])
ratio = float(sys.argv[2])

for s in range(1, n + 1):
    for t in range(1, n + 1):
        if s == t:
            continue
        if random.random() < ratio:
            print(s, t)

# for s in range(1, n + 1):
#     for t in range(s, n + 1):
#         if s != t:
#             print(s, t)
