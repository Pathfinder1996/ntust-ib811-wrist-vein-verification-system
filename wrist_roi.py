def line_intersection(p1, p2, p3, p4):
    """
    Calculate the intersection point of two lines (p1-p2) and (p3-p4)
    using the determinant method.
    """
    A1 = p2[1] - p1[1]
    B1 = p1[0] - p2[0]
    C1 = A1 * p1[0] + B1 * p1[1]
    
    A2 = p4[1] - p3[1]
    B2 = p3[0] - p4[0]
    C2 = A2 * p3[0] + B2 * p3[1]

    det = A1 * B2 - A2 * B1
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det

    return (x.astype(int), y.astype(int))

def scale_point(p, center, scale):
    """
    Scale a point `p` with respect to `center` by a scaling factor `scale`.
    """
    return (p - center) * scale + center