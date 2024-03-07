import tator
import numpy as np
import math
from statistics import median
import cv2


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


def measure_classify_poly(media_id, proposed_track_element, **args):
    api = tator.get_api(host=args["host"], token=args["token"])
    media = api.get_media(media_id)

    dimension = args.get("dimension", "both")
    method = args.get("method", "median")
    transform = args.get("transform", "none")
    scale_factor = args.get("scale_factor", 1.0)

    polys = [p for p in proposed_track_element if p.get("points", []) != []]

    # If we have no polygons, we can't measure and it probably isn't a good track
    if polys == []:
        return False, {}

    new = []

    centers = [np.array(loc["points"]).mean(axis=0) for loc in polys]
    dists = [
        math.sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) for x, y in centers
    ]
    nearest = np.argmin(dists)
    min_idx = max(0, nearest - 20)
    max_idx = min(len(polys), nearest + 20)
    samples = 0
    for poly in polys[min_idx:max_idx]:
        if samples > 5:
            break
        frame = poly["frame"]
        contour = np.array(poly["points"])
        xmin = np.min(contour[:, 0])
        ymin = np.min(contour[:, 1])
        xmax = np.max(contour[:, 0])
        ymax = np.max(contour[:, 1])

        contour[:, 0] = (contour[:, 0] - xmin) / (xmax - xmin)
        contour[:, 1] = (contour[:, 1] - ymin) / (ymax - ymin)

        scaled_contour = contour * 2000
        image = np.zeros((2000, 2000), dtype=np.float32)
        cv2.fillPoly(image, [scaled_contour.astype(int)], 255)

        image = cv2.GaussianBlur(image, (301, 301), 0)

        # Create a mask to exclude corners near the image edges
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[50:-50, 50:-50] = 255

        # Sharpen the image with a threshold function + get a contour of the sharpened blur
        thresh_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

        # Shrink and dilate the image to remove noise
        # Shrink the image to remove noise
        shrink_kernel = np.ones((5, 5), np.uint8)
        shrinked_image = cv2.erode(thresh_image, shrink_kernel, iterations=10)

        # Dilate the image to smooth corners
        dilate_kernel = np.ones((5, 5), np.uint8)
        thresh_image = cv2.dilate(shrinked_image, dilate_kernel, iterations=10)

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
        for i in range(len(blur_contour)):
            segment = (blur_contour[i - 1], blur_contour[i])
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

        final_inside_corners = final_inside_corners / 2000
        final_inside_corners[:, 0] = final_inside_corners[:, 0] * (xmax - xmin) + xmin
        final_inside_corners[:, 1] = final_inside_corners[:, 1] * (ymax - ymin) + ymin

        # Lets do the same for centroid
        centroid = centroid / 2000
        centroid[0] = centroid[0] * (xmax - xmin) + xmin
        centroid[1] = centroid[1] * (ymax - ymin) + ymin

        closest_intersection_before = closest_intersection_before / 2000
        closest_intersection_before[0] = (
            closest_intersection_before[0] * (xmax - xmin) + xmin
        )
        closest_intersection_before[1] = (
            closest_intersection_before[1] * (ymax - ymin) + ymin
        )
        closest_intersection_after = closest_intersection_after / 2000
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

    return True, {"$new": new}


if __name__ == "__main__":
    """Unit test"""
    import tator
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="https://cloud.tator.io")
    parser.add_argument("--token")
    parser.add_argument("--scale-factor", type=float, default=0.4242)
    parser.add_argument("state_id", type=int)
    args = parser.parse_args()
    api = tator.get_api(host=args.host, token=args.token)
    state_obj = api.get_state(args.state_id)
    state_type_obj = api.get_state_type(state_obj.type)
    localizations = api.get_localization_list_by_id(
        state_type_obj.project, {"ids": state_obj.localizations}
    )
    extended_attrs = measure_classify(
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
