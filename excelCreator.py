#
#  Copyright (c) 2019  INBAR DAHARI.
#  All rights reserved.
#
import csv

from xlsxwriter import Workbook

# or leave it blank, default name is "Sheet 1"

def create_excel(data, headers, file_name, sheet_name):
    wb = Workbook(file_name)
    ws = wb.add_worksheet(sheet_name)

    first_row = 0
    for header in headers:
        col = headers.index(header)  # we are keeping order.
        ws.write(first_row, col, header)  # we have written first row which is the header of worksheet also.

    row = 1
    for line in data:
        for _key, _value in line.items():
            col = headers.index(_key)
            ws.write(row, col, _value)
        row += 1
    wb.close()
    # with open(file_name, 'wb') as outfile:
    #     writer = csv.writer(outfile)
    #     # to get tabs use csv.writer(outfile, dialect='excel-tab')
    #     writer.writerows(your_dictionary.iteritems())


# players = [{'dailyWinners': 3, 'dailyFree': 2, 'user': 'Player1', 'bank': 0.06},
# {'dailyWinners': 3, 'dailyFree': 2, 'user': 'Player2', 'bank': 4.0},
# {'dailyWinners': 1, 'dailyFree': 2, 'user': 'Player3', 'bank': 3.1},
# {'dailyWinners': 3, 'dailyFree': 2, 'user': 'Player4', 'bank': 0.32}]
#
# ordered_list=["user","dailyWinners","dailyFree","bank"] #list object calls by index but dict object calls items randomly
#
#
# create_excel(players, ordered_list, "sdfs", "Ssdfs")