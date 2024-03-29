"""
@final project 2019 by Inbar DAHARI
"""

#
#

#
#  Copyright (c) 2019  INBAR DAHARI.
#  All rights reserved.
#
import math

from matplotlib.offsetbox import AnchoredText
from matplotlib_scalebar.scalebar import ScaleBar
import cv2 as cv
import os.path
from scipy import stats
from scipy.stats import binom
import seaborn as sns
from pylab import *

from controller.excel_creator import create_excel
from controller.excel_statistics_data import excel_statistics_data
from model.Dendrite import Dendrite
from model.Distance import DistanceBetweenLines
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter
from model.Range import Range
from model.Vector import Vector


def histogram(lis):
    hist = []
    m = max(lis)

    for i in range(1, m + 1):
        count = 0
        for k in range(len(lis)):
            if lis[k] == i:
                count = count + 1
        hist = hist + [count]
    return hist


def remove_duplication(ranges_with_dendrites):
    sorted_data = dict(sorted(ranges_with_dendrites.items(), key=lambda item: len(item[1]), reverse=True))
    for range, dendrite_data in sorted_data.items():
        for dendrite in dendrite_data:
            for query_range in sorted_data.keys():
                if query_range.dendrite.id == dendrite.id:
                    sorted_data[query_range].clear()
    for range, dendrite_data in sorted_data.items():
        if len(dendrite_data) == 0:
            ranges_with_dendrites.pop(range)

    return ranges_with_dendrites



class Interface:

    def __init__(self):
        super(Interface, self).__init__()
        self.p_file_path = None
        self.p_threshold1 = None
        self.min_distance_to_merge = 20
        self.min_angle_to_merge = 10
        self.excel_file = None
        self.preview_figure = None

    def get_lines(self, lines_in):
        if cv.__version__ < '3.0':
            return lines_in[0]
        return [l[0] for l in lines_in]

    def main(self):
        plt.close('all')
        p_file_path = self.p_file_path
        p_file_name = os.path.basename(p_file_path)
        print("file name:", p_file_name)


        src = cv.imread(p_file_path, cv.COLOR_BGR2HLS)

        # Check if image is loaded fine
        if src is None:
            print('Error opening image!')
            return -1

        blur = cv.GaussianBlur(src, (5, 5), 0)
        p_threshold2= self.p_threshold1*3 if self.p_threshold1 <= 85 else 255
        dst = cv.Canny(src, self.p_threshold1,p_threshold2, None, 3)
        # detect
        # Python: cv.Canny(image, edges, threshold1, threshold2, aperture_size=3) → None
        # threshold1 – first threshold for the hysteresis procedure

        # __________________primitive pic_______________________________________
        # Copy edges to the images that will display the results in BGR

        cdst2 = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst2)

        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(cdst2, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255, 255, 0), 3, cv.LINE_AA)

        # __________________regular pic_______________________________________

        (dendrite_list, img_merged_lines, _lines, merged_lines_all) = self.get_detected_picture(dst)

        print("information for each dendrite: ")
        dendrite_list.sort()
        range_map = dict()
        x = 5 if 180 / len(dendrite_list) < 36 else (180 / len(dendrite_list))   # angle to group
        for dendrite in dendrite_list:
            min_angle = dendrite.angle - x
            max_angle = dendrite.angle + x
            range_angle = Range(min_angle, max_angle, dendrite.id, dendrite.angle, dendrite.length, dendrite)
            if range_map.get(range_angle) is None:
                range_map.update({range_angle: []})
            print(dendrite)

        print('\n', "<--------------- classification of parallel groups: --------------->", '\n')

        for dendrite in dendrite_list:
            angle_another = dendrite.angle
            id_another = dendrite.id
            for key, value in range_map.items():
                if key.min <= angle_another <= key.max and id_another != key.id:
                    value.append(dendrite)
        modify_range_map = dict()
        not_parallel_map = dict()
        for key, value in range_map.items():
            if len(value) != 0:
                modify_range_map.update({key: value})
            else:
                not_parallel_map.update({key: value})

        print("Parallel:")
        for key, value in modify_range_map.items():
            print('Key: {0}: \nnumber of parallel lines: {1}'.format(key.id, len(value)), *value, sep='\n')

        print("\nNot Parallel:\n")
        for key, value in not_parallel_map.items():
            print(f'{key}: {value} number of parallel lines: {len(value)}\n')
            pass

        modify_range_map = remove_duplication(modify_range_map)

        print("Parallel after cleaning duplicate:")
        for key, value in modify_range_map.items():
            print('Key: {0}: \nnumber of parallel lines: {1}'.format(key.id, len(value)), *value, sep='\n')
        # combine
        combined_dict = dict()
        combined_dict.update(modify_range_map)
        combined_dict.update(not_parallel_map)

        # sorting results
        final_result = []

        for k in combined_dict.keys():
            d_id = k.id
            d_length = k.length
            d_v1 = str(k.dendrite.vector1)
            d_v2 = str(k.dendrite.vector2)
            d_angle = k.angle
            d_min = k.min
            d_max = k.max

            par_d = combined_dict[k]
            par_count = len(par_d)
            par_id = ' , '.join(list(map(lambda d: str(d.id), par_d)))
            final_result.append({"dendrite id": d_id, "length [\u03BCm]": d_length, "vector1": d_v1, "vector2": d_v2,
                                 "angle [\xb0]": d_angle, "min angle range [\xb0]": d_min,
                                 "max angle range [\xb0]": d_max,
                                 "number of parallel lines": par_count,
                                 "parallel lines id": par_id})

        # ----------------short distance between lines------------------

        distance_computed = dict()
        for key, value in modify_range_map.items():
            a1 = np.array([key.dendrite.vector1.x, key.dendrite.vector1.y, 1])
            a0 = np.array([key.dendrite.vector2.x, key.dendrite.vector2.y, 1])
            shortlist = []
            for parallel_dendirte in value:
                b0 = np.array([parallel_dendirte.vector1.x, parallel_dendirte.vector1.y, 1])
                b1 = np.array([parallel_dendirte.vector2.x, parallel_dendirte.vector2.y, 1])
                temp = DistanceBetweenLines.closest_distance_between_lines(a0, a1, b0, b1)
                shortlist.append(temp)
            distance_computed.update({key.id: shortlist})

        print('\n', "<--------------- The shortest distance between Parallel groups: --------------->", '\n')
        print('ID:   distance to parallel [\u03BCm]:')

        final_final_data = []

        for key, value in distance_computed.items():
            d_data = next(filter(lambda x: x["dendrite id"] == key, final_result))

            res = {"dendrite id": d_data["dendrite id"],
                   "length [\u03BCm]": d_data["length [\u03BCm]"],
                   "vector1": d_data["vector1"],
                   "vector2": d_data["vector2"],
                   "angle [\xb0]": d_data["angle [\xb0]"],
                   "min angle range [\xb0]": d_data["min angle range [\xb0]"],
                   "max angle range [\xb0]": d_data["max angle range [\xb0]"],
                   "number of parallel lines": d_data["number of parallel lines"],
                   "parallel lines id": d_data["parallel lines id"],
                   "distance to parallel [\u03BCm]": ' , '.join(map(str, value))}

            final_final_data.append(res)
            print('{0}: {1} \n'.format(key, value))

        final_final_data.sort(key=lambda item: item["dendrite id"])

        create_excel(final_final_data,
                     ["dendrite id", "length [\u03BCm]", "vector1", "vector2", "angle [\xb0]",
                      "min angle range [\xb0]", "max angle range [\xb0]",
                      "number of parallel lines", "parallel lines id",
                      "distance to parallel [\u03BCm]"],
                    self.excel_file, "data")

        # ----------------average of all the parallels lines:------------------

        def average(l):
            if len(l)==0:
                return 0
            sum_length = 0
            for key_, value_ in l.items():
                sum_length = sum_length + key_.length
            total = sum_length
            total = float(total)
            return round(total / len(l), 2)

        error_listp = []
        error_list_np = []
        for key_, value_ in modify_range_map.items():
            error_listp.append(key_.length)
        for key_, value_ in not_parallel_map.items():
            error_list_np.append(key_.length)
        if len(error_listp) == 0:
            error_listp.append(0)
        if len(error_list_np) == 0:
            error_list_np.append(0)

        def standard_error(group):
            if len(group) == 0:
                return ("Empty Result, No standard error found")
            return round(stats.sem(group, axis=None, ddof=0), 2)


        print('\nAverage of all the parallels lines:', (average(modify_range_map)))
        print('Standard Error of all the parallels lines:', standard_error(error_listp))
        print('\nAverage of all the NOT parallels lines:', (average(not_parallel_map)))
        print('Standard Error of all the NOT parallels lines:', standard_error(error_list_np))

        # ----------------------------------------------------------------------

        # ---------fig0-------------------------------------------------------------
        # first picture after the blurring and turning to binary

        # ---------fig1-------------------------------------------------------------

        plt.figure("Segmentation line workflow")
        # image = plt.imread(cbook.get_sample_data('prediction/merged_lines.jpg'))
        #
        gray()
        plt.subplot(131), plt.imshow(blur)
        # indicates that each pixel is equal to 0.167 micrometer.
        scalebar = ScaleBar(0.167, 'um')
        plt.gca().add_artist(scalebar)
        plt.xticks([]), plt.yticks([])
        # plt.gca().invert_yaxis()

        plt.subplot(132), imshow(cdstP)
        plt.xticks([]), plt.yticks([])

        plt.subplot(133), imshow(img_merged_lines)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()

        plt.show()

        # ---------fig2-------binomial_distribution of random grows:------------------

        self.binomial_distribution(dendrite_list, modify_range_map, merged_lines_all )

        # ------------fig3 -angles scatter plot of dendrites------
        x1 = [0] * (len(dendrite_list))
        y1 = [0] * (len(dendrite_list))

        i = 0
        for dendrite in dendrite_list:
            # x-axis values
            xAngle = dendrite.angle
            x1[i] = xAngle

            # y-axis values
            yId = dendrite.id
            y1[i] = yId

            i = i + 1

        # Get the color for each sample.
        colors = cm.rainbow(np.linspace(0, 1, len(x1)))

        plt.figure("Angular distribution of dendritic branches")
        # plot the data

        ax = plt.subplot(2, 1, 1)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, len(y1))
        ax.scatter(x1, y1, color=colors)
        # ax.set(xlabel="x - Angles")
        # ax.set(title="angles")

        for i, txt in enumerate(y1):
            plt.annotate(txt, (x1[i], y1[i]),fontsize=25)

        # tell matplotlib to use the format specified above
        ax.xaxis.set_major_locator(MultipleLocator(int(180 / len(dendrite_list))))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.grid(True, which='minor')
        ax.xaxis.set_minor_locator(MultipleLocator(int(180 / len(dendrite_list))))
        ax.tick_params(axis='x', rotation=70)
        # plt.yticks([])
        plt.yticks(fontsize=20)
        plt.ylabel('ID', fontsize=25,fontweight='bold')
        plt.xticks([])

        # plt.title('figure 3- Range of angles:')

        plt.subplot(2, 1, 2)
        plt.hist(x1, bins=int(len(dendrite_list)), range=[0, 180], rwidth=1, color='b', edgecolor='black',lw=0)
        ax = plt.gca()
        ax.set_xlim(0, 180)
        # plt.yticks(range(1,5))
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MultipleLocator(int(180 / len(dendrite_list))))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(int(180 / len(dendrite_list))))
        ax.tick_params(axis='x', rotation=70)
        # ax.set(xlabel="x - Angles", ylabel="# of dendrite")
        plt.xlabel('Angles [°]', fontsize=25,fontweight='bold')
        # ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))

        plt.ylabel('# of dendritic branches', fontsize=25,fontweight='bold')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


        # ------------fig4 -length scatter plot of dendrites------
        sns.set(color_codes=True)
        x2 = [0] * (len(modify_range_map))
        x3 = [0] * (len(not_parallel_map))

        i = 0
        for key, value in modify_range_map.items():
            # x-axis values
            xLength = key.length
            x2[i] = xLength
            i = i + 1
        j = 0
        for key, value in not_parallel_map.items():
            # x-axis values
            x3Length = key.length
            x3[j] = x3Length
            j = j + 1

        x2.sort()
        x3.sort()
        if len(x2) == 0 :
            x2.append(0)
        if len(x3) == 0:
            x3.append(0)
        if x2[len(x2) - 1] > x3[len(x3) - 1] :
            x_range = x2[len(x2) - 1]
        else: x_range = x3[len(x3) - 1]

        plt.figure("Length distribution of parallel vs. non-parallel dendritic branches")
        plt.subplot(2, 1, 1)
        plt.hist(x3, bins=int(len(dendrite_list)), range=[0, x3[len(x3) - 1]], rwidth=1, color='blue',
                 edgecolor='black', lw=1)
        ax = plt.gca()
        ax.set_ylim(0, len(x3))
        ax.set_xlim(0, x_range)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        ax.xaxis.set_major_locator(MultipleLocator(180 / len(dendrite_list)))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(180 / len(dendrite_list)))
        ax.tick_params(axis='x', rotation=45)
        # ax.xaxis.set_major_formatter(StrMethodFormatter(u" {x:.0f}\u03BCm"))
        # plt.xlabel('Length', fontsize=18)
        plt.ylabel('# of dendritic branches', fontsize=25,fontweight='bold')
        ax.set_facecolor('#d8dcd6')
        # plt.suptitle('figure 2- Range of length:', fontsize=14, fontweight='bold')
        # plt.title('NOT parallel groups vs. parallel groups')
        avg1 = (average(not_parallel_map))
        anchored_text = AnchoredText("Average of all 'NOT Parallels' groups: " + str("{:.1f}".format(float(avg1))) +
                                     u"\u03BCm" +
                                     "\nStandard Error of all 'NOT Parallels' lines: " +
                                     str("{:.1f}".format(
                                         float(stats.sem(error_list_np, axis=None, ddof=0)))) + u"\u03BCm",
                                     loc=1, prop={"size": 19})
        ax.add_artist(anchored_text)
        plt.grid(True)
        plt.tight_layout()


        plt.subplot(2, 1, 2)
        plt.hist(x2, bins=int(len(dendrite_list)), range=[0, x2[len(x2) - 1]], rwidth=1, color='blue',
                 edgecolor='black', lw=1)
        ax = plt.gca()
        ax.set_xlim(0, x_range)
        ax.xaxis.set_major_locator(MultipleLocator(180 / len(dendrite_list)))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='x', rotation=45)
        plt.xlabel('Length [\u03BCm]', fontsize=25,fontweight='bold')
        plt.ylabel('# of dendritic branches', fontsize=25,fontweight='bold')
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        # ax.xaxis.set_major_formatter(StrMethodFormatter(u" {x:.0f}\u03BCm"))
        ax.set_facecolor('#d8dcd6')
        avg2 = (average(modify_range_map))
        anchored_text = AnchoredText("Average of all 'Parallels' groups: " + str("{:.1f}".format(float(avg2))) +
                                     u"\u03BCm" +
                                     "\nStandard Error of all Parallels lines: " +
                                     str("{:.1f}".format(
                                         float(stats.sem(error_listp, axis=None, ddof=0)))) + u"\u03BCm",
                                     loc=1, prop={"size": 19})
        ax.add_artist(anchored_text)
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        # ------------------

        # ------------------

        #return merged_lines_all

    def binomial_distribution(self, dendrite_list, parallel_modify_range_map, merged_lines_all):
        # setting the values of n and p
        # defining the list of k values
        n = len(dendrite_list)
        p = 1 / n
        s_values = list(range(n))
        # obtaining the mean and variance
        mean, var = binom.stats(n, p)
        # list of pmf values
        # dist = [(binom.pmf(k, n, p)) * n for k in s_values]
        dist =[]
        for k in s_values:
            ans= (binom.pmf(k, n, p)) * n
            if float(ans) > 0.05:
                dist.append(int(math.ceil(ans)))
        # printing the table
        print('\n', "<--------------- Binomial distribution simulation: --------------->", '\n')
        print ("Random simulation Classification")
        for i in range(2, len(dist)):
            print(f'{i}: {dist[i]}')


        parallel_histlist = []
        for key, value in parallel_modify_range_map.items():
            parallel_histlist.append(len(value) + 1)
            # print(parallel_histlist)
        values_ = histogram(parallel_histlist)
        print ("Measured Classification")
        for i in range(1, len(values_)):
            print(f'{i+1}: {values_[i]}')

        print('\n', "<--------------- Calculation of the percentage of parallelism in relation to the simulation: --------------->", '\n')
        # simulation
        sum_all_simulation_lines = 0
        for i in range(2, len(dist)):
            sum_all_simulation_lines += i*dist[i]
        percentagePS = sum_all_simulation_lines/(len(merged_lines_all))
        print( "simulation: ", percentagePS)

        # measured
        sum_all_measured_lines = 0
        for i in range(1, len(values_)):
            sum_all_measured_lines += (i + 1) * (values_[i])
        percentagePM = sum_all_measured_lines / (len(merged_lines_all))
        print("measured: ", percentagePM)
        percentagePMS = sum_all_measured_lines / sum_all_simulation_lines
        print("E\S: ", percentagePMS )


        print('\n',"<---------------  Long - term  parallels of E\S: --------------->",'\n')
        # simulation with weights - in the long term
        sum_all_simulation_lines_weights = 0
        for i in range(2, len(dist)):
            sum_all_simulation_lines_weights += i * i * dist[i]

        # measured with weights - in the long term
        sum_all_measured_lines_weights = 0
        for i in range(1, len(values_)):
            sum_all_measured_lines_weights += (i + 1) * (i + 1) * (values_[i])
        LongtermES = sum_all_measured_lines_weights / sum_all_simulation_lines_weights
        print("Measured (E) with weights)\ Simulation (S) with weights): " , LongtermES)

        excel_statistics_data(filepath=os.path.dirname(self.excel_file),
                              imagename=os.path.basename(self.p_file_path),
                              pthreshold1=self.p_threshold1,
                              min_distance_to_merge=self.min_distance_to_merge,
                              min_angle_to_merge=self.min_angle_to_merge,
                              linesnumber=len(merged_lines_all),
                              Sclassification=dist,
                              Mclassification=values_,
                              percentagePS=percentagePS,
                              percentagePM=percentagePM,
                              percentagePMS=percentagePMS,
                              LongtermES=LongtermES
                              )

        # -------------------fig2 ---------------------

        if max(values_) > max(dist[2:len(dist)]):
            y_group_range = max(values_)
        else:
            y_group_range = max(dist[2:len(dist)])

        y_group_range = max(max(values_), max(dist))
        plt.figure("Classification of dendritic branch parallel growth vs random simulation")
        plt.subplot(1, 2, 1)
        plt.bar(x=list(range(1, len(values_)+1)),
                height=values_,
                color="blue", width=0.35)
        # plt.yticks(range(0, max(values_)))
        plt.ylim([0, y_group_range])
        # plt.yticks(np.arange(0, max(values_),step=2))
        plt.xticks(fontsize=20, rotation=45)
        plt.yticks(fontsize=20)
        plt.ylabel('frequency', fontsize=25,fontweight='bold')
        # plt.xlabel('parallel groups', fontsize=10)
        plt.xlabel('parallel groups', fontsize=25,fontweight='bold')
        plt.title('Measured', fontsize=26)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()


        plt.subplot(1, 2, 2)

        plt.bar(s_values[2:len(dist)],
                height=dist[2:len(dist)],
                color="blue", width=0.35)
        plt.ylim([0, y_group_range])
        # plt.yticks(np.arange(0, max(values_), step=2))
        plt.xticks(fontsize=20, rotation=45)
        plt.yticks(fontsize=20)
        plt.ylabel('frequency', fontsize=25,fontweight='bold')
        plt.xlabel('parallel groups', fontsize=25,fontweight='bold')
        plt.title('Simulation', fontsize=26)
        # plt.yticks(range(0, max(values_)))
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

        plt.show()

    def merge_lines_pipeline_2(self, lines):
        super_lines_final = []
        super_lines = []
        min_distance_to_merge = self.min_distance_to_merge
        min_angle_to_merge = self.min_angle_to_merge

        for line in lines:
            create_new_group = True
            group_updated = False

            for group in super_lines:
                for line2 in group:
                    if self.get_distance(line2, line) < min_distance_to_merge:
                        # check the angle between lines
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(
                                abs(math.degrees(orientation_i)) - abs(
                                    math.degrees(orientation_j)))) < min_angle_to_merge:
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))
                            group.append(line)

                            create_new_group = False
                            group_updated = True
                            break

                if group_updated:
                    break

            if (create_new_group):
                new_group = []
                new_group.append(line)

                for idx, line2 in enumerate(lines):
                    # check the distance between lines
                    if self.get_distance(line2, line) < min_distance_to_merge:
                        # check the angle between lines -finding the angle between start point and end point of each
                        # line
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(abs(math.degrees(orientation_i)) - abs(
                                math.degrees(orientation_j)))) < min_angle_to_merge:
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))
                            # #the difference of the angle between two lines

                            new_group.append(line2)

                            # remove line from lines list
                            # lines[idx] = False
                # append new group
                super_lines.append(new_group)

        for group in super_lines:
            super_lines_final.append(self.merge_lines_segments1(group))

        return super_lines_final

    def merge_lines_segments1(self, lines, use_log=False):
        # if there is just one line
        if (len(lines) == 1):
            return lines[0]

        line_i = lines[0]

        # orientation
        orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))
        # print(orientation_i)

        points = []
        for line in lines:
            points.append(line[0])
            points.append(line[1])

        # if ((orientation_i)*(180*math.pi) > 45) and ((orientation_i)*(180*math.pi)< (90 + 45)):
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):

            # sort by y
            points = sorted(points, key=lambda point: point[1])

            if use_log:
                print("use y")
        else:

            # sort by x
            points = sorted(points, key=lambda point: point[0])

            if use_log:
                print("use x")

        return [points[0], points[len(points) - 1]]

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        # https://stackoverflow.com/questions/32702075/what-would-be-the-fastest-way-to-find-the-maximum-of-all-possible-distances-betw

    def lines_close(self, line1, line2):
        dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
        dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
        dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
        dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])

        if (min(dist1, dist2, dist3, dist4) < 100):
            return True
        else:
            return False

    #
    def line_magnitude(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
        # https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
        # http://paulbourke.net/geometry/pointlineplane/

    def distance_point_line(self, px, py, x1, y1, x2, y2):
        # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        line_mag = self.line_magnitude(x1, y1, x2, y2)

        if line_mag < 0.00000001:
            distance_point_line = 9999
            return distance_point_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (line_mag * line_mag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = self.line_magnitude(px, py, x1, y1)
            iy = self.line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_line = iy
            else:
                distance_point_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_line = self.line_magnitude(px, py, ix, iy)

        return distance_point_line

    def get_distance(self, line1, line2):
        dist1 = self.distance_point_line(line1[0][0], line1[0][1], line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist2 = self.distance_point_line(line1[1][0], line1[1][1], line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist3 = self.distance_point_line(line2[0][0], line2[0][1], line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        dist4 = self.distance_point_line(line2[1][0], line2[1][1], line1[0][0], line1[0][1], line1[1][0], line1[1][1])

        return min(dist1, dist2, dist3, dist4)

    def get_detected_picture(self, dst):
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)  # turn to binary img
        """
            with the arguments:
            dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
            lines: A vector that will store the parameters ( x start ,y start,x end,y end) of the detected lines
            rho : The resolution of the parameter k in pixels. We use 1 pixel.
            theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
            threshold: The minimum number of intersections to "*detect*" a line
            minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
            maxLineGap: The maximum gap between two points to be considered in the same line.
            """
        lines = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        # It gives as output the extremes of the detected lines (x0,y0,x1,y1)

        # prepare
        _lines = []
        for _line in self.get_lines(lines):
            _lines.append([(_line[0], _line[1]), (_line[2], _line[3])])

        # sort
        _lines_x = []
        _lines_y = []
        for line_i in _lines:
            orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))
            if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):
                # if ((orientation_i)*(180*math.pi) > 45) and ((orientation_i)*(180*math.pi)< (90 + 45)):
                _lines_y.append(line_i)
            else:
                _lines_x.append(line_i)
        # sort the lines by the beginning  point- x and y
        _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
        _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

        merged_lines_x = self.merge_lines_pipeline_2(_lines_x)
        merged_lines_y = self.merge_lines_pipeline_2(_lines_y)

        # list of all the final merged lines
        merged_lines_all = []
        merged_lines_all.extend(merged_lines_x)
        merged_lines_all.extend(merged_lines_y)
        print("process groups lines", len(_lines)," -->", len(merged_lines_all), '\n')

        print("The number of lines identified in the image: ", len(merged_lines_all), '\n')
        img_merged_lines = cdst

        dendrite_list = []
        id_ = 0
        for line in merged_lines_all:
            cv.line(img_merged_lines, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (255, 255, 0), 3, cv.LINE_AA)

            # length of each line
            x0 = line[0][0]
            x1 = line[1][0]
            y0 = line[0][1]
            y1 = line[1][1]
            dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            # print('The length : ', dist)

            # finding the angle between start point and end point of line
            v1 = Vector(x0, y0)  # coordinate of the lines
            v2 = Vector(x1, y1)

            # the coordinate of the marks lines- Each line is represented by four numbers, which are the two
            # endpoints of the detected line segment print(merged_lines_all[line]) Vector.tuple(v1) Vector.tuple(v2)
            # _______________ __________________________________________
            radians = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))  # in radians
            turn_degrees = math.degrees(radians)

            # print(orientation_i)
            # orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))

            # in Radians(180 / math.pi) or in degrees(180 * math.pi)

            # print('angle:', radians, '\n')
            id_ = id_ + 1
            dendrite_list.append(Dendrite(id_, dist, v1, v2, turn_degrees % 180))  # %360
        return (dendrite_list, img_merged_lines, _lines ,merged_lines_all )



    def create_preview(self):
        if self.preview_figure != None:
            plt.close(self.preview_figure)
        # ---------fig0-------------------------------------------------------------
        # first picture after the blurring and turning to binary
        default_file = 'den.png'
        src = cv.imread(self.p_file_path, cv.COLOR_BGR2HLS)
        if src is None:
            print('Error opening image!')
            return -1
        blur = cv.GaussianBlur(src, (5, 5), 0)
        p_threshold2 = self.p_threshold1 * 3 if self.p_threshold1 <= 85 else 255
        dst = cv.Canny(src, self.p_threshold1, p_threshold2, None, 3)

        (dendrite_list, img_merged_lines, _lines, merged_lines_all) = self.get_detected_picture(dst)
        self.preview_figure = plt.figure("Preview segmentation line detection")
        ax = plt.gca()
        scalebar = ScaleBar(0.167, 'um')
        ax.add_artist(scalebar)

        textstr = "Identified lines: {} \nIdentified lines after merging : {} ".format(len(_lines), len(merged_lines_all))
        props = dict(facecolor='blue', alpha=0.2)
        ax.set_xlabel(textstr,bbox=props,fontsize=20)

        imshow(img_merged_lines)
        plt.xticks([]), plt.yticks([])
        plt.show()


    # The scipy.stats module contains various functions for statistical calculations and tests.
    # probability mass function(pmf), which is the total probability of achieving r success and n-r failure.


if __name__ == "__main__":
    Main = Interface()
    Main.main(sys.argv[1:])
