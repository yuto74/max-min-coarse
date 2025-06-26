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


orig = sys.argv[1]
coarse = sys.argv[2]

with open(orig) as f:
    with open(coarse) as g:
        while True:
            l1 = f.readline()
            l1 = l1.rstrip()
            if not l1:
                break
            l2 = g.readline()
            l2 = l2.rstrip()
            v1 = l1.split(' ')
            v2 = l2.split(' ')
            print(v1[2], v2[2])