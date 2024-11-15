import tator
import numpy as np
import math
from statistics import median
import cv2
import time
from pprint import pprint
from collections import defaultdict
import json

# Libraries to handle loading the matlab file
import gzip
import io
import base64
import tempfile
import uuid

from .pyStereoComp import pyStereoComp


def create_mask(polygon):
    # Create a blank mask
    # first infer image size from polygon bounds
    x_min = min([point[0] for point in polygon])
    x_max = max([point[0] for point in polygon])
    y_min = min([point[1] for point in polygon])
    y_max = max([point[1] for point in polygon])
    image_size = (y_max - y_min, x_max - x_min)
    mask = np.zeros(image_size, dtype=np.uint8)

    # normalize polygon to coordinates of this image
    polygon = [(point[0] - x_min, point[1] - y_min) for point in polygon]

    # Define the polygon points and fill the mask
    polygon = np.array(polygon, np.int32)
    cv2.fillPoly(mask, [polygon], 255)  # Fill the polygon with white

    return mask, x_min, y_min


def is_point_between_lines(point, line1, line2):
    # Unpack line points
    p1, p2 = line1
    p3, p4 = line2

    # Create vectors
    line1_vec = np.array(p2) - np.array(p1)
    line2_vec = np.array(p4) - np.array(p3)

    # Vector from line starts to the point
    vec_to_point1 = np.array(point) - np.array(p1)
    vec_to_point2 = np.array(point) - np.array(p3)

    # Calculate cross products to determine the relative position
    cross1 = np.cross(line1_vec, vec_to_point1)
    cross2 = np.cross(line2_vec, vec_to_point2)

    return (cross1 * cross2 < 0).all()  # Returns True if point is between the lines


# Function to calculate Euclidean distance
def euclidean_dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def extend_line_to_image_bounds(p1, p2, img_shape):
    height, width = img_shape[:2]

    # Calculate the slope and y-intercept of the line
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:  # Vertical line
        return [(x1, 0), (x1, height - 1)]
    if y1 == y2:  # Horizontal line
        return [(0, y1), (width - 1, y1)]

    # Line equation: y = mx + b
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Find intersections with image boundaries
    points = []

    # Intersect with left (x = 0) and right (x = width - 1)
    y_left = slope * 0 + intercept
    y_right = slope * (width - 1) + intercept
    if 0 <= y_left < height:
        points.append((0, int(y_left)))
    if 0 <= y_right < height:
        points.append((width - 1, int(y_right)))

    # Intersect with top (y = 0) and bottom (y = height - 1)
    x_top = (0 - intercept) / slope
    x_bottom = (height - 1 - intercept) / slope
    if 0 <= x_top < width:
        points.append((int(x_top), 0))
    if 0 <= x_bottom < width:
        points.append((int(x_bottom), height - 1))

    # Return two points at the boundary
    return points[:2]


def measure_classify(media_id, proposed_track_element, **args):
    api = tator.get_api(host=args["host"], token=args["token"])
    media = api.get_media(media_id)

    dimension = args.get("dimension", "both")
    method = args.get("method", "median")
    transform = args.get("transform", "none")
    scale_factor = args.get("scale_factor", 1.0)

    if dimension == "both":
        sizes = [
            (loc["width"] * media.width + loc["height"] * media.height) / 2
            for loc in proposed_track_element
        ]
    elif dimension == "width":
        sizes = [loc["width"] * media.width for loc in proposed_track_element]
    elif dimension == "height":
        sizes = [loc["height"] * media.height for loc in proposed_track_element]
    else:
        raise ValueError(
            f"Invalid dimension '{dimension}', must be one of "
            "'width', 'height', or 'both'!"
        )
    if method == "median":
        size = median(sizes)
    elif method == "mean":
        size = sum(sizes) / len(sizes)
    elif method == "center":
        centers = [
            (loc["x"] + loc["width"] / 2, loc["y"] + loc["height"] / 2)
            for loc in proposed_track_element
        ]
        dists = [
            math.sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) for x, y in centers
        ]
        nearest = np.argmin(dists)
        size = sizes[nearest]
    else:
        raise ValueError(
            f"Invalid method '{method}', must be one of "
            "'median', 'mean', or 'center'!"
        )
    size *= scale_factor
    if transform == "scallops":
        size = ((0.1 / 120) * (size - 40) + 0.8) * size
    elif transform == "none":
        pass
    else:
        raise ValueError(
            f"Invalid transform '{transform}', must be one of " "'scallops' or 'none'!"
        )
    return True, {args.get("size_attr", "Size (mm)"): size}


def average_point(points):
    # Calculate the average of a list of points
    return np.mean(points, axis=0)


def cluster_points(points, distance_threshold):
    clusters = []
    for point in points:
        for cluster in clusters:
            if np.min(np.linalg.norm(cluster - point, axis=1)) < distance_threshold:
                cluster.append(point)
                break
        else:
            clusters.append([point])
    return clusters


def line_intersection(line1, line2):
    # This function returns the intersection point of two lines (each defined by two points)
    # If the lines don't intersect, it returns None
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


MEMOIZED_VOID_POLY = None


def calculate_void_poly(void_base64):
    global MEMOIZED_VOID_POLY
    if type(MEMOIZED_VOID_POLY) is not type(None):
        return MEMOIZED_VOID_POLY

    data = base64.b64decode(void_base64)
    with io.BytesIO(data) as mem_file:
        with gzip.GzipFile(fileobj=mem_file, mode="rb") as f:
            void_mask = np.load(f)
            void_contours, _ = cv2.findContours(
                void_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            void_contour = np.squeeze(void_contours[0])
            MEMOIZED_VOID_POLY = void_contour.copy()
            return MEMOIZED_VOID_POLY


def get_length_depth(stereo, left, right, media_width):

    # Shift right line over to match the image coordinate space
    right[0] -= media_width // 2
    right[2] -= media_width // 2

    # Verify left_line and right_line are both in ascending point order (left to right and top to bottom)
    dot = np.dot(
        [left[2] - left[0], left[3] - left[1]],
        [right[2] - right[0], right[3] - right[1]],
    )
    if dot < 0.0:
        left = left[2], left[3], left[0], left[1]

    xyz0, xyz0a = stereo.triangulatePoint(
        np.array([[left[0]], [left[1]]]), np.array([[right[0]], [right[1]]])
    )
    xyz1, xyz1a = stereo.triangulatePoint(
        np.array([[left[2]], [left[3]]]), np.array([[right[2]], [right[3]]])
    )
    depth = abs((xyz0[2][0] + xyz1[2][0]) / 2)
    depth /= 1000.0
    length = np.linalg.norm(xyz0 - xyz1)
    length /= 1000.0
    return {"depth": depth, "length": length, "vector": np.abs(xyz1 - xyz0) / 1000.0}


def measure_classify_groundfish_poly(media_id, proposed_track_element, **args):

    track_id = int(args["track_id"])
    # Don't do anything with unassociated detections
    if track_id == -1:
        if args.get("drop_unassociated", False):
            return False, {}
        else:
            return True, {}
    api = tator.get_api(host=args["host"], token=args["token"])
    media = api.get_media(media_id)

    polys = [p for p in proposed_track_element if p.get("points", []) != []]
    boxes = [p for p in proposed_track_element if p.get("x") is not None]

    min_length = args.get("min_length", 0)
    max_length = args.get("max_length", 2000)  # both frames so 2000 is 1000
    length = max(len(polys), len(boxes))
    if length < min_length:
        return False, {"Label": "Too Short"}
    if length > max_length:
        return True, {"Label": "Too Long"}

    measurement_lines = []
    rotated_bbox = []
    visualizations = []

    # Load stereo cal out of b64 gz'd matlab file
    stereo_comp = pyStereoComp()
    stereo_data_b64 = args["stereo_calibration"]["matlab"]
    stereo_data_gz = base64.b64decode(stereo_data_b64)
    stereo_data = gzip.decompress(stereo_data_gz)
    with tempfile.TemporaryDirectory() as td:
        with open(f"{td}/stereo_calibration.mat", "wb") as f:
            f.write(stereo_data)
        stereo_comp.importCalData(f"{td}/stereo_calibration.mat")

    def visualize_line(p1, p2):
        x = p1[0] / media.width
        y = p1[1] / media.height
        u = (p2[0] - p1[0]) / media.width
        v = (p2[1] - p1[1]) / media.height
        visualizations.append((x, y, u, v, poly["frame"]))

    for poly in polys:
        contour = np.array(poly["points"])
        contour *= [media.width, media.height]
        contour = contour.astype(np.int32)
        # Reshape points to correct shape (n_points, 1, 2)
        contour = contour.reshape((-1, 1, 2))

        # Calculate the convex hull of the contour (to simplify the polygon)
        hull = cv2.convexHull(contour)

        # Find the two points with the maximum distance (major axis)
        max_dist = 0
        p1 = None
        p2 = None
        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                dist = euclidean_dist(hull[i][0], hull[j][0])
                if dist > max_dist:
                    max_dist = dist
                    p1, p2 = hull[i][0], hull[j][0]

        # DEBUG (longest line)
        # Visualize the longest line
        # visualize_line(p1, p2)

        # Calculate the rotated bounding box of the hull
        rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rect)

        # Make rotated bbox for visualization purposes
        normalized_box_points = []
        for point in box_points:
            # clip to image bounds
            point[0] = max(0, min(point[0], media.width))
            point[1] = max(0, min(point[1], media.height))

            normalized_box_points.append(
                [
                    max(0, min(1, point[0] / media.width)),
                    max(0, min(1, point[1] / media.height)),
                ]
            )

            # Clip to 0,1

        rotated_bbox.append(
            {
                "type": args.get("poly_type_id"),
                "version": args.get("version_id"),
                "media_id": media_id,
                "frame": poly["frame"],
                "points": normalized_box_points,
                "elemental_id": str(uuid.uuid4()),
                "attributes": {"Confidence": poly["attributes"]["Confidence"]},
            }
        )
        mid_top = (
            (box_points[0][0] + box_points[1][0]) // 2,
            (box_points[0][1] + box_points[1][1]) // 2,
        )
        mid_bottom = (
            (box_points[2][0] + box_points[3][0]) // 2,
            (box_points[2][1] + box_points[3][1]) // 2,
        )
        mid_left = (
            (box_points[0][0] + box_points[3][0]) // 2,
            (box_points[0][1] + box_points[3][1]) // 2,
        )
        mid_right = (
            (box_points[1][0] + box_points[2][0]) // 2,
            (box_points[1][1] + box_points[2][1]) // 2,
        )

        # Are top and bottom or left and right points further?
        if euclidean_dist(mid_top, mid_bottom) < euclidean_dist(mid_left, mid_right):
            bisecting_line = (mid_top, mid_bottom)
            translation_line = (mid_left, mid_right)
        else:
            bisecting_line = (mid_left, mid_right)
            translation_line = (mid_top, mid_bottom)

        # DEBUG visualize bisecting line
        # visualize_line(bisecting_line[0], bisecting_line[1])

        # Slide the bisecting line along the translation line in both directions by
        # 25% of the length of the translation line to isolate the head + tail of the fish
        translation_length = euclidean_dist(translation_line[0], translation_line[1])
        translation_vector = np.array(
            [
                (translation_line[1][0] - translation_line[0][0]) / translation_length,
                (translation_line[1][1] - translation_line[0][1]) / translation_length,
            ]
        )
        head_line = bisecting_line - (0.33 * translation_length * translation_vector)

        visualize_line(head_line[0], head_line[1])

        tail_line = bisecting_line + (0.33 * translation_length * translation_vector)

        visualize_line(tail_line[0], tail_line[1])

        tail_end_line = bisecting_line + (0.5 * translation_length * translation_vector)
        head_end_line = bisecting_line - (0.5 * translation_length * translation_vector)

        # create a mask of the fish
        mask, x_min, y_min = create_mask(contour.reshape(-1, 2))

        # Copy mask and blank out anything not within the tail_end_line and head_end_line
        tail_mask = np.zeros_like(mask)
        head_mask = np.zeros_like(mask)

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                # Determine if the point is between tail_line and tail_end_line or head_line and head_end_line
                norm_tail_end = tail_end_line - np.array((x_min, y_min))
                norm_tail = tail_line - np.array((x_min, y_min))
                norm_head_end = head_end_line - np.array((x_min, y_min))
                norm_head = head_line - np.array((x_min, y_min))
                if is_point_between_lines((x, y), norm_tail, norm_tail_end):
                    tail_mask[y, x] = mask[y, x]
                if is_point_between_lines((x, y), norm_head, norm_head_end):
                    head_mask[y, x] = mask[y, x]

        # Sanity check calculate centroid of mask + draw line to 1,1
        moments = cv2.moments(tail_mask)
        if moments["m00"] == 0:
            continue
        tail_centroid = [
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        ]

        moments = cv2.moments(head_mask)
        if moments["m00"] == 0:
            continue
        head_centroid = [
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        ]

        margin = 20
        with_margin_shape = (mask.shape[0] + margin, mask.shape[1] + margin)
        hc_margin = [head_centroid[0] + margin // 2, head_centroid[1] + margin // 2]
        tc_margin = [tail_centroid[0] + margin // 2, tail_centroid[1] + margin // 2]
        # calculate out the line if it were to extend for the shape of the mask
        extended_line = extend_line_to_image_bounds(
            hc_margin, tc_margin, with_margin_shape
        )

        contour_min = contour.reshape(-1, 2) - [x_min, y_min]
        contour_min += margin // 2

        line_bitmask = np.zeros(with_margin_shape, dtype=np.uint8)
        fish_bitmask = np.zeros(with_margin_shape, dtype=np.uint8)
        # Draw the outline of the contour
        # Use a thick line to handle sharp corners
        # We will filter out duplicate hits later
        fish_bitmask = cv2.polylines(
            fish_bitmask, [contour_min], isClosed=True, color=255, thickness=3
        )
        # Draw the extended line through this shape
        line_bitmask = cv2.line(
            line_bitmask,
            (int(extended_line[0][0]), int(extended_line[0][1])),
            (int(extended_line[1][0]), int(extended_line[1][1])),
            255,
            1,
        )

        line_finder = cv2.bitwise_and(fish_bitmask, line_bitmask)

        # find the points that are 255 which indicate the intersection points
        intersection_points = np.argwhere(line_finder == 255)
        if len(intersection_points) < 2:
            print(
                f"Intersection points frame={poly['frame']} {len(intersection_points)}"
            )
            debug = cv2.line(
                fish_bitmask,
                (int(extended_line[0][0]), int(extended_line[0][1])),
                (int(extended_line[1][0]), int(extended_line[1][1])),
                255,
                1,
            )
            # cv2.imwrite(f"line_finder_{poly['frame']}.png", debug)
            continue

        # sort by distance to the center of the mask (with_margin_shape)
        # pick two points on opposite sides of the center furthest from the center
        center = [with_margin_shape[1] // 2, with_margin_shape[0] // 2]

        down_or_left = []
        up_or_right = []
        for point in intersection_points:
            if point[0] >= center[0] or point[1] >= center[1]:
                up_or_right.append(point)
            else:
                down_or_left.append(point)

        down_or_left = sorted(
            down_or_left,
            key=lambda x: euclidean_dist(center, x),
            reverse=True,
        )

        up_or_right = sorted(
            up_or_right,
            key=lambda x: euclidean_dist(center, x),
            reverse=True,
        )

        if len(down_or_left) < 1 or len(up_or_right) < 1:
            continue

        # x-y are reversed for down_or_left and up_or_right
        p1 = down_or_left[0][::-1]
        p2 = up_or_right[0][::-1]

        boundary_hits = [p1, p2]

        # Use a set in case we hit a boundary exactly and get duped
        """
        boundary_hits = set()
        for idx in range(len(contour_min) - 1):
            p1 = contour_min[idx]
            p2 = contour_min[idx + 1]
            intersection = line_intersection([p1, p2], extended_line)
            # Check if intersection is along the segment specified by p1,p2
            if intersection is not None:
                if min(p1[0], p2[0]) <= intersection[0] <= max(p1[0], p2[0]) and min(
                    p1[1], p2[1]
                ) <= intersection[1] <= max(p1[1], p2[1]):
                    boundary_hits.add(tuple(intersection))

        if len(boundary_hits) != 2:
            print(f"Boundary hits = {len(boundary_hits)}")
            continue

        # convert to list to make mutable
        boundary_hits = [list(hit) for hit in boundary_hits]
        """
        boundary_hits[0][0] += x_min - (margin // 2)
        boundary_hits[0][1] += y_min - (margin // 2)
        boundary_hits[1][0] += x_min - (margin // 2)
        boundary_hits[1][1] += y_min - (margin // 2)

        x = boundary_hits[0][0] / media.width
        y = boundary_hits[0][1] / media.height
        u = (boundary_hits[1][0] - boundary_hits[0][0]) / media.width
        v = (boundary_hits[1][1] - boundary_hits[0][1]) / media.height
        measurement_lines.append((x, y, u, v, poly["frame"]))

        # DEBUG symmetry line

        # Symmetry line (isn't that great)
        # Calculate the line of symmetry of the hull
        # Find the centroid of the hull
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue
        centroid = [
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        ]
        centroid[0] += x_min
        centroid[1] += y_min
        contour_reshaped = contour.reshape((-1, 2))
        cov_matrix = np.cov(contour_reshaped, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Find the principal axis
        principal_axis = eigenvectors[np.argmax(eigenvalues)]
        # Convert to image coordinates
        principal_axis[1] *= -1

        length = 200
        p1 = centroid - (length * principal_axis)
        p2 = centroid + (length * principal_axis)

        visualize_line(p1, p2)

    # Take top-3 longest major axis lines
    # major_lines = major_lines[:3]

    attrs = {"Label": "Object"}

    new = []
    by_frame_measurement_lines = defaultdict(lambda: [])
    for line in measurement_lines:
        by_frame_measurement_lines[line[4]].append(line)

    measure_frames = []
    measure_depth = []
    measure_length = []
    measure_vectors = []
    for frame, lines in by_frame_measurement_lines.items():
        if len(lines) != 2:
            continue
        if lines[0][0] < lines[1][0]:
            left_line = lines[0]
            right_line = lines[1]
        else:
            left_line = lines[1]
            right_line = lines[0]

        # convert to abs coordinates for stereo comp
        left_line = [
            left_line[0] * media.width,
            left_line[1] * media.height,
            (left_line[0] + left_line[2]) * media.width,
            (left_line[1] + left_line[3]) * media.height,
        ]
        right_line = [
            right_line[0] * media.width,
            right_line[1] * media.height,
            (right_line[0] + right_line[2]) * media.width,
            (right_line[1] + right_line[3]) * media.height,
        ]
        results = get_length_depth(stereo_comp, left_line, right_line, media.width)

        vector = np.round(results["vector"], 2)
        measure_frames.append(frame)
        measure_depth.append(round(results["depth"], 3))
        measure_length.append(round(results["length"], 3))
        measure_vectors.append(vector)

    if len(measure_length) == 0:
        return False, {"Label": "No Measures"}

    # Calculate the median of the top-quartile of lengths
    # and find the closest line, which will be our selected measurement line.

    z_indices = []
    for vector in measure_vectors:
        z_indices.append(vector[2])
    z_indices = np.array(z_indices)

    bottom_limit = np.percentile(z_indices, 25)
    valid_idx = np.where(z_indices <= bottom_limit)[0]

    # Collect lengths only from valid_idx
    measure_length = np.array(measure_length)

    # Calculate the best length of the top quartile with the least amount of z-axis
    lengths = np.array(measure_length)[valid_idx]
    depths = np.array(measure_depth)[valid_idx]
    median_of_top_quartile = np.median(lengths[lengths >= np.percentile(lengths, 75)])
    closest_idx = np.argmin(np.abs(measure_length - median_of_top_quartile))

    # Calculate the min and max depth on all frames we got a detection
    min_depth = min(measure_depth)
    max_depth = max(measure_depth)
    # Comma seperated string of all lengths

    combined = [
        (length, depth, frame, vector)
        for frame, length, depth, vector in zip(
            measure_frames, measure_length, measure_depth, measure_vectors
        )
    ]

    # sort by frame
    combined = sorted(combined, key=lambda x: x[2])
    all_lengths = ",".join([f"{x[0]}@{x[1]}/{x[2]}|Vec={x[3]}" for x in combined])

    if min_depth > args.get("maximum_min_depth", 0.0):
        attrs["Label"] = "Bad Depth"
    maximum_length = measure_length[closest_idx]
    attrs["AllLengths"] = all_lengths
    attrs["Distance"] = measure_depth[closest_idx]
    attrs["Length"] = maximum_length
    attrs["MinDistance"] = min_depth
    attrs["MaxDistance"] = max_depth
    attrs["MeasurementFrame"] = measure_frames[closest_idx]
    attrs["Measurement Method"] = "Computer"

    for lines in by_frame_measurement_lines.values():
        for line in lines:
            measurement_spec = {
                "type": args.get("line_type_id"),
                "version": args.get("version_id"),
                "media_id": media_id,
                "frame": line[4],
                "x": max(0, min(1, line[0])),  # clip to 0-1
                "y": max(0, min(1, line[1])),
                "u": max(-1, min(1, line[2])),  # clip to -1 to 1
                "v": max(-1, min(1, line[3])),
                "elemental_id": str(uuid.uuid4()),
            }
            new.append(measurement_spec)

    if args.get("visualize", False):
        for line in visualizations:
            measurement_spec = {
                "type": args.get("line_type_id"),
                "version": args.get("version_id"),
                "media_id": media_id,
                "frame": line[4],
                "x": max(0, min(1, line[0])),  # clip to 0-1
                "y": max(0, min(1, line[1])),
                "u": max(-1, min(1, line[2])),  # clip to -1 to 1
                "v": max(1, min(1, line[3])),
            }
            new.append(measurement_spec)

    for box in rotated_bbox:
        new.append(box)

    return True, {**attrs, "$replace": new}


def measure_classify_poly(media_id, proposed_track_element, **args):
    api = tator.get_api(host=args["host"], token=args["token"])
    media = api.get_media(media_id)

    before = time.time()
    dimension = args.get("dimension", "both")
    method = args.get("method", "median")
    transform = args.get("transform", "none")
    scale_factor = args.get("scale_factor", 1.0)
    working_size = args.get("working_size", 512)
    kernel_size = args.get("kernel_size", 77)
    iterations = args.get("iterations", 10)
    min_size = args.get("min_size", 10)

    polys = [p for p in proposed_track_element if p.get("points", []) != []]
    boxes = [p for p in proposed_track_element if p.get("x") is not None]

    if len(boxes) < min_size:
        print("too small")
        return False, {}
    if "sgie_secondary_labels" in boxes[0]["attributes"]:
        class_labels = defaultdict(lambda: [])
        for box in boxes:
            sgie_labels = json.loads(box["attributes"]["sgie_secondary_labels"])
            for label in sgie_labels:
                for label, score in label.items():
                    class_labels[label].append(score)

        # Find the label with the highest score
        max_label = max(class_labels, key=lambda x: sum(class_labels[x]))

        # if max_label != "Scallop":
        #    print("Not a scallop")
        #    return False, {}

        for key, items in class_labels.items():
            avg_label = sum(items) / len(items)
            if avg_label > 0.3 and key != "Scallop":
                print("Probably not a scallop")
                return True, {"Label": key}

    box_centers = [
        (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2) for box in boxes
    ]
    box_dists = [
        math.sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) for x, y in box_centers
    ]
    box_nearest = np.argmin(box_dists)

    # If we have no polygons, we can't measure and it probably isn't a good track
    if polys == []:
        return False, {}

    new = []

    centers = [np.array(loc["points"]).mean(axis=0) for loc in polys]
    dists = [
        math.sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) for x, y in centers
    ]
    nearest = np.argmin(dists)
    window = 5
    min_idx = max(0, nearest - window)
    max_idx = min(len(polys), nearest + window)
    samples = 0
    hq_measurements = []
    lq_measurements = []
    box_measurements = []
    box_heights = []
    box_widths = []

    measure_box = boxes[box_nearest]
    box_measurement = (
        math.sqrt(
            (measure_box["width"] * media.width) ** 2
            + (measure_box["height"] * media.height) ** 2
        )
        * scale_factor
    )
    box_measurements.append(box_measurement)
    box_heights.append(measure_box["height"] * media.height * scale_factor)
    box_widths.append(measure_box["width"] * media.width * scale_factor)

    for idx, poly in enumerate(polys[min_idx:max_idx]):
        frame = poly["frame"]

        if samples > 2:
            continue

        contour = np.array(poly["points"])
        xmin = np.min(contour[:, 0])
        ymin = np.min(contour[:, 1])
        xmax = np.max(contour[:, 0])
        ymax = np.max(contour[:, 1])

        contour[:, 0] = (contour[:, 0] - xmin) / (xmax - xmin)
        contour[:, 1] = (contour[:, 1] - ymin) / (ymax - ymin)

        scaled_contour = contour * working_size
        image = np.zeros((working_size, working_size), dtype=np.float32)
        cv2.fillPoly(image, [scaled_contour.astype(int)], 255)

        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # Create a mask to exclude corners near the image edges
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[50:-50, 50:-50] = 255

        # Sharpen the image with a threshold function + get a contour of the sharpened blur
        thresh_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

        # Shrink and dilate the image to remove noise
        # Shrink the image to remove noise
        shrink_kernel = np.ones((5, 5), np.uint8)
        shrinked_image = cv2.erode(thresh_image, shrink_kernel, iterations=iterations)

        # Dilate the image to smooth corners
        dilate_kernel = np.ones((5, 5), np.uint8)
        thresh_image = cv2.dilate(shrinked_image, dilate_kernel, iterations=iterations)

        # Find the contour of the blurred image
        blur_contours, _ = cv2.findContours(
            thresh_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blur_contour = np.squeeze(blur_contours[0])

        # We only care about inside corners, not outside corners
        inside_corners = []
        window_size = 1

        hull = cv2.convexHull(blur_contour)

        hull = np.squeeze(hull)
        candidate_coords = []
        distances = []
        for index in range(len(blur_contour)):
            point = blur_contour[index].astype(np.float64)
            signed_distance = cv2.pointPolygonTest(hull, point, True)
            is_inside = signed_distance > 0
            # Sanity check (need some distance from hull to even consider)
            if signed_distance > 10:
                distances.append(signed_distance)
                candidate_coords.append((point, signed_distance))

        # Sort the candidate corners based on their signed distances
        sorted_corners = sorted(candidate_coords, key=lambda x: x[1], reverse=True)

        top_percent = int(len(sorted_corners) * 0.25)
        candidate_coords = [corner[0] for corner in sorted_corners[:top_percent]]

        inside_corners = np.array(candidate_coords)

        centroid = average_point(blur_contour)

        # cluster the points to remove duplicates
        grid_size = 5
        inside_clusters = cluster_points(inside_corners, grid_size)
        while len(inside_clusters) > 2 and grid_size < 300:
            inside_clusters = cluster_points(inside_corners, grid_size)
            grid_size += 10

        if len(inside_clusters) != 2:
            continue

        # Find the point in each cluster that is furthest from the hull
        final_inside_corners = []
        for cluster in inside_clusters:
            max_distance = -1
            farthest_point = None
            for point in cluster:
                distance = cv2.pointPolygonTest(hull, tuple(point), True)
                if distance > max_distance:
                    max_distance = distance
                    farthest_point = point
            final_inside_corners.append(farthest_point)

        final_inside_corners = np.array(final_inside_corners)

        handle_length = np.linalg.norm(
            final_inside_corners[1] - final_inside_corners[0]
        )

        # Don't let small handles in
        if handle_length < 0.5 * working_size:
            continue

        handle_center = average_point(final_inside_corners)

        # Define the line
        dx = centroid[0] - handle_center[0]
        dy = centroid[1] - handle_center[1]

        line = [
            (handle_center[0] - 1000 * dx, handle_center[1] - 1000 * dy),
            (handle_center[0] + 1000 * dx, handle_center[1] + 1000 * dy),
        ]

        # Initialize variables to store the closest intersection points and their distances
        closest_intersection_before = None
        closest_distance_before = float("inf")
        closest_intersection_after = None
        closest_distance_after = float("inf")

        # Iterate over the segments of the contour
        for i in range(len(hull)):
            segment = (hull[i - 1], hull[i])
            intersection = line_intersection(line, segment)
            if intersection is not None:
                # Check if the intersection point lies within the line segment
                if min(segment[0][0], segment[1][0]) <= intersection[0] <= max(
                    segment[0][0], segment[1][0]
                ) and min(segment[0][1], segment[1][1]) <= intersection[1] <= max(
                    segment[0][1], segment[1][1]
                ):
                    # Calculate the distance to the handle_center
                    distance = np.sqrt(
                        (intersection[0] - handle_center[0]) ** 2
                        + (intersection[1] - handle_center[1]) ** 2
                    )
                    # Update the closest intersection point and distance
                    if (
                        intersection[0] < handle_center[0]
                        and distance < closest_distance_before
                    ):
                        closest_intersection_before = intersection
                        closest_distance_before = distance
                    elif (
                        intersection[0] > handle_center[0]
                        and distance < closest_distance_after
                    ):
                        closest_intersection_after = intersection
                        closest_distance_after = distance

        if closest_intersection_before is None or closest_intersection_after is None:
            continue

        closest_intersection_before = np.array(closest_intersection_before)
        closest_intersection_after = np.array(closest_intersection_after)

        measurement_line = [closest_intersection_before, closest_intersection_after]

        # Calculate the dot product of the two vectors
        dot_product = np.dot(
            measurement_line[1] - measurement_line[0],
            final_inside_corners[1] - final_inside_corners[0],
        )

        # Calculate the magnitudes of the two vectors
        magnitude1 = np.linalg.norm(measurement_line[1] - measurement_line[0])
        magnitude2 = np.linalg.norm(final_inside_corners[1] - final_inside_corners[0])

        # Calculate the cosine of the angle between the two vectors
        cosine_angle = dot_product / (magnitude1 * magnitude2)

        # Calculate the angle in degrees
        angle = np.arccos(cosine_angle) * 180 / np.pi

        if np.abs(angle - 90) > 5:
            continue

        # Now that is a confirmed HQ Measurement we can add it to the list
        hq_measurement_line = np.array(measurement_line)
        hq_measurement_line = hq_measurement_line / working_size
        hq_measurement_line[:, 0] = hq_measurement_line[:, 0] * (xmax - xmin) + xmin
        hq_measurement_line[:, 1] = hq_measurement_line[:, 1] * (ymax - ymin) + ymin
        hq_measurement_line[:, 0] *= media.width
        hq_measurement_line[:, 1] *= media.height
        hq_measurement_line = hq_measurement_line * scale_factor
        hq_measurement_length = np.linalg.norm(
            hq_measurement_line[1] - hq_measurement_line[0]
        )
        hq_measurements.append(hq_measurement_length)

        # Use the boolean value is_90_degrees as needed

        # Convert to image abs coordinates
        final_inside_corners = final_inside_corners / working_size
        final_inside_corners[:, 0] = final_inside_corners[:, 0] * (xmax - xmin) + xmin
        final_inside_corners[:, 1] = final_inside_corners[:, 1] * (ymax - ymin) + ymin

        closest_intersection_before = closest_intersection_before / working_size
        closest_intersection_before[0] = (
            closest_intersection_before[0] * (xmax - xmin) + xmin
        )
        closest_intersection_before[1] = (
            closest_intersection_before[1] * (ymax - ymin) + ymin
        )
        closest_intersection_after = closest_intersection_after / working_size
        closest_intersection_after[0] = (
            closest_intersection_after[0] * (xmax - xmin) + xmin
        )
        closest_intersection_after[1] = (
            closest_intersection_after[1] * (ymax - ymin) + ymin
        )

        handle_spec = {
            "type": args.get("line_type_id"),
            "version": args.get("version_id"),
            "media_id": media_id,
            "frame": frame,
            "x": final_inside_corners[0][0],
            "y": final_inside_corners[0][1],
            "u": final_inside_corners[1][0] - final_inside_corners[0][0],
            "v": final_inside_corners[1][1] - final_inside_corners[0][1],
            "attributes": {"Type": "Handle"},
        }

        measurement_spec = {
            "type": args.get("line_type_id"),
            "version": args.get("version_id"),
            "media_id": media_id,
            "frame": frame,
            "x": closest_intersection_before[0],
            "y": closest_intersection_before[1],
            "u": closest_intersection_after[0] - closest_intersection_before[0],
            "v": closest_intersection_after[1] - closest_intersection_before[1],
            "attributes": {"Type": "Measure"},
        }
        new.extend([handle_spec, measurement_spec])
        samples += 1

    attrs = {}
    measure_type_name = args.get("measure_type_attr", None)
    if hq_measurements:
        attrs[args.get("size_attr", "Size (mm)")] = median(hq_measurements)
        if measure_type_name:
            attrs[measure_type_name] = "HandleCentroid"
    else:
        average_dim = (box_heights[0] + box_widths[0]) / 2
        attrs[args.get("size_attr", "Size (mm)")] = average_dim
        if measure_type_name:
            attrs[measure_type_name] = "BoxDimMean"

    # print(f"Time: {time.time() - before}")
    attrs.update({"$new": new})
    return True, attrs


if __name__ == "__main__":
    """Unit test"""
    import tator
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="https://cloud.tator.io")
    parser.add_argument("--token")
    parser.add_argument("--scale-factor", type=float, default=0.4289)
    parser.add_argument("state_id", type=int)
    args = parser.parse_args()
    api = tator.get_api(host=args.host, token=args.token)
    state_obj = api.get_state(args.state_id)
    state_type_obj = api.get_state_type(state_obj.type)
    localizations = api.get_localization_list_by_id(
        state_type_obj.project, {"ids": state_obj.localizations}
    )

    extended_attrs = measure_classify_poly(
        state_obj.media[0],
        [l.to_dict() for l in localizations],
        host=args.host,
        token=args.token,
        scale_factor=args.scale_factor,
    )
    print(f"Source (ID={args.state_id})")
    pprint(state_obj.attributes)
    print("==========Transforms into======")
    pprint(extended_attrs)
