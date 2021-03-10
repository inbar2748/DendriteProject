# #
#    Copyright (c) 2019  INBAR DAHARI.
#    All rights reserved.
#
#
# #
#

#
#

#
#  Copyright (c) 2019  INBAR DAHARI.
#  All rights reserved.
#

import random

import csv

csv.DictReader
GLOBAL = 360 # angle

arr = [0]* (GLOBAL +1)
map = dict()
counter = 0

while counter < 71:
    index = random.randint(1, GLOBAL)
    arr[index] += 1
    counter = counter +1

print (arr)
for i in range(1, len(arr)):
    if map.get(arr[i]) == None:
        map.update({arr[i]:[i]})
    else:
        map.get(arr[i]).append(i)

for key, value in map.items():
    print ('{0}: {1}'.format(key,value))
