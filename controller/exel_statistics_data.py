import csv


def exel_statistics_data(filepath, imagename, pthreshold1, mindistancetomerge, minangletomerge, linesnumber,
                         Sclassification, Mclassification, percentagePS, percentagePM, percentagePMS,
                         LongtermES):
    with open(f'{filepath}/.xlsx', 'w') as file:
        file_csv = csv.writer(file)
        s_col_name= []
        simulation_classification=[]
        for i in range(2, len(Sclassification)):
            s_col_name.append(i)
            simulation_classification.append(Sclassification[i])

        m_col_name= []
        measured_classification= []
        for i in range(1, len(Mclassification)):
            m_col_name.append(i + 1)
            measured_classification.append(Mclassification[i])

        file_csv.writerow(['Image name', 'Gui parameter value: p_threshold1', 'Gui parameter value: min distance to merge',
                       'Gui parameter value: min angle to merge', 'Identified numbers of lines: ',
                       'Random simulation classification:', s_col_name, 'Measured classification', m_col_name,
                       'Calculation of the percentage of parallelism in the simulation ',
                       'Calculation of the percentage of parallelism in the measured ',
                       'Calculation of the percentage of parallelism in relation to the simulation E/S',
                       'Long term parallels with weights calculation of E/S:'])

        file_csv.writerow([imagename, pthreshold1, mindistancetomerge, minangletomerge, linesnumber, None ,
                           simulation_classification, None , measured_classification, percentagePS, percentagePM, percentagePMS,
                           LongtermES])
        print(f"Data successfully written to {filepath}.")
