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

from numpy import double

# recognize lines just without marking


def rawcount(s1):
    f = open(s1, 'rb')
    lines_num = 0
    buf_size = 1116*1126 # update the size of the image
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines_num += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines_num

res= rawcount('C:/Users/inbar/Desktop/DRGNFM1.png')
print(res)

