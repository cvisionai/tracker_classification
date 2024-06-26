import tator
import numpy as np
import math
from statistics import median
import cv2
import time
from pprint import pprint
from collections import defaultdict
import json


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

        if max_label != "Scallop":
            print("Not a scallop")
            return False, {}

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
