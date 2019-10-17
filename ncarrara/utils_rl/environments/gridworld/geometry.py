
def computeMiddle(rectangle):
    xmin, ymin, xmax, ymax = rectangle
    x = xmin + (xmax - xmin) / 2.
    y = ymin + (ymax - ymin) / 2.
    return x, y


def inRectangle(s, rectangle):
    xmin, ymin, xmax, ymax = rectangle
    x, y = s
    return xmin <= x <= xmax and ymin <= y <= ymax


def isSegmentIntersectRectangle(segment, rectangle, nbstep):
    xmin, ymin, xmax, ymax = segment
    rxmin, rymin, rxmax, rymax = rectangle
    if (rxmin <= xmin <= rxmax and rymin <= ymin <= rymax) or (rxmin <= xmax <= rxmax and rymin <= ymax <= rymax):
        return True
    if xmin == xmax and ymin == ymax:
        return False
    s1 = (xmin, ymin)
    s2 = (xmax, ymax)
    A = (rxmin, rymin)
    B = (rxmax, rymin)
    C = (rxmax, rymax)
    D = (rxmin, rymax)
    return intersect(s1, s2, A, B) or intersect(s1, s2, B, C) or intersect(s1, s2, C, D) or intersect(s1, s2, D, A)

def ccw(A, B, C):
    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C
    return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)


# attention gere pas la colinearite entre segment mais c'est pas bien grave car on teste versus 4 segments et pas qu'un (si colineaire a l un, il le sera pas aux autres)
def intersect(A, B, C, D):
    res = ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    # print((A,B,C,D,res))
    return res


def inRectangles(s, rectangles):
    for rectangle in rectangles:
        if inRectangle(s, rectangle):
            return True
    return False

