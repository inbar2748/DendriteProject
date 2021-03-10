

#
#  Copyright (c) 2019  INBAR DAHARI.
#  All rights reserved.
#
# https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments?rq=1

import numpy as np


class DistanceBL:
    def closestDistanceBetweenLines(self,a0,a1,b0,b1):

        """ Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
            Return the closest points on each segment and their distance
        """

        # If clampAll=True, set all clamps to True

        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

        # Calculate denominator
        A = a1 - a0
        B = b1 - b0
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)

        _A = A / magA
        _B = B / magB

        cross = np.cross(_A, _B);
        denom = np.linalg.norm(cross)**2

        # If lines are parallel (denom=0) test if lines overlap.
        # If they don't overlap then there is a closest point solution.
        # If they do overlap, there are infinite closest positions, but there is a closest distance
        if not denom:
            d0 = np.dot(_A, (b0-a0))

            # Overlap only possible with clamping
            if clampA0 or clampA1 or clampB0 or clampB1:
                d1 = np.dot(_A, (b1-a0))

                # Is segment B before A?
                if d0 <= 0 >= d1:
                    if clampA0 and clampB1:
                        if np.absolute(d0) < np.absolute(d1):
                            return np.linalg.norm(a0-b0)
                        return np.linalg.norm(a0-b1)

                # Is segment B after A?
                elif d0 >= magA <= d1:
                    if clampA1 and clampB0:
                        if np.absolute(d0) < np.absolute(d1):
                            return np.linalg.norm(a1-b0)
                        return np.linalg.norm(a1-b1)

            # Segments overlap, return distance between parallel segments
            return np.linalg.norm(((d0*_A)+a0)-b0)

        # Lines cris-cross: Calculate the projected closest points
        t = (b0 - a0) ;
        detA = np.linalg.det([t, _B, cross])
        detB = np.linalg.det([t, _A, cross])

        t0 = detA/denom ;
        t1 = detB/denom ;

        pA = a0 + (_A * t0) # Projected closest point on segment A
        pB = b0 + (_B * t1) # Projected closest point on segment B

        # Clamp projections
        if clampA0 or clampA1 or clampB0 or clampB1:
            if clampA0 and t0 < 0:
                pA = a0
            elif clampA1 and t0 > magA:
                pA = a1

            if clampB0 and t1 < 0:
                pB = b0
            elif clampB1 and t1 > magB:
                pB = b1

            # Clamp projection A
            if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
                dot = np.dot(_B, (pA-b0))
                if clampB0 and dot < 0:
                    dot = 0
                elif clampB1 and dot > magB:
                    dot = magB
                pB = b0 + (_B * dot)

            # Clamp projection B
            if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
                dot = np.dot(_A, (pB-a0))
                if clampA0 and dot < 0:
                    dot = 0
                elif clampA1 and dot > magA:
                    dot = magA
                pA = a0 + (_A * dot)

        return round(np.linalg.norm(pA-pB),2)


if __name__ == "__main__":
    a1 = np.array([547, 486, 1])
    a0 = np.array([604, 429, 1])
    b0 = np.array([374, 683, 1])
    b1 = np.array([322, 738, 1])
    dis = DistanceBL()
    print(dis.closestDistanceBetweenLines(a0, a1, b0, b1))
    # Result: (15.826771412132246, array([ 19.85163563,  26.21609078,  14.07303667]), array([ 26.99,  12.39,  11.18])) #
