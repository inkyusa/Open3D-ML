import numpy as np

def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([
        t1 + t2 * t3, t7 - t9, t11 + t12, t7 + t9, t1 + t2 * t15, t19 - t20,
        t11 - t12, t19 + t20, t1 + t2 * t24
    ],
                 axis=1)

    return np.reshape(R, (-1, 3, 3))


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.
    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float): origin point relate to smallest point.
    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

def rotation_3d_in_axis(points, angles, axis=2):
    """Rotate points in specific axis.
    Args:
        points (np.ndarray, shape=[N, point_size, 3]]):
        angles (np.ndarray, shape=[N]]):
        axis (int): Axis to rotate at.
    Returns:
        np.ndarray: Rotated points.
    """
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError('axis should in range')

    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 1.0, 0.5)):
    """Convert kitti locations, dimensions and angles to corners.
    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray): Rotation_y in kitti label file with shape (N).
        origin (list or array or float): Origin point relate to smallest point.
            use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.
    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles)
    corners += centers.reshape([-1, 1, 3])
    return corners

def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above to surfaces that
    normal vectors all direct to internal.
    Args:
        corners (np.ndarray): 3D box corners with shape of (N, 8, 3).
    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces

def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """Check points is in 3d convex polygons.
    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of \
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3). \
            All surfaces' normal vector must direct to internal. \
            Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains \
            shape of (num_polygon).
    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    # num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons, ), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0] +
                    points[i, 1] * normal_vec[j, k, 1] +
                    points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_box(points, rbbox, origin=(0.5, 0.5, 0)):
    """Check points in rotated bbox and return indicces.
    Args:
        points (np.ndarray, shape=[N, 3+dim]): Points to query.
        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation.
        z_axis (int): Indicate which axis is height.
        origin (tuple[int]): Indicate the position of box center.
    Returns:
        np.ndarray, shape=[N, M]: Indices of points in each box.
    """
    # TODO: this function is different from PointCloud3D, be careful
    # when start to use nuscene, check the input
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def filter_by_min_points(pc, bboxes, min_points_dict):
"""Filter ground truths by number of points in the bbox."""
    filtered_boxes = []
    flat_boxes = np.array([box.to_xyzwhlr() for box in bboxes])
    num_points_in_box = points_in_box(points[:, :3], flat_boxes).sum(0)

    for i, box in enumerate(bboxes):
        if box.name in min_points_dict.keys():
            if num_points_in_box > min_points_dict[box.name]:
                filtered_boxes.append(box)
    
    return filtered_boxes
