import tator
import numpy as np
import math
from statistics import median


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


def measure_classify_poly(media_id, proposed_track_element, **args):
    api = tator.get_api(host=args["host"], token=args["token"])
    media = api.get_media(media_id)

    dimension = args.get("dimension", "both")
    method = args.get("method", "median")
    transform = args.get("transform", "none")
    scale_factor = args.get("scale_factor", 1.0)

    polys = [p for p in proposed_track_element if p.get("points", None) != None]
    major_lines = []
    minor_lines = []
    for poly in polys:
        # For each polygon there is a major/minor axis. The major axis is the longest axis in the polygon and the minor axis is perpendicular to the major axis.
        # The major axis is calculated by finding the distance between the two points that are farthest apart in the polygon.
        # The minor axis is calculated by finding the distance between the two points that are farthest apart in the polygon that are perpendicular to the major axis.
        points = poly["points"]

        # Calculate the major axis
        distances = [
            (p1, p2, np.linalg.norm(np.array(p1) - np.array(p2)))
            for p1 in points
            for p2 in points
        ]
        major_axis_points = max(distances, key=lambda x: x[2])[:2]
        major_lines.append(major_axis_points)

        # Calculate the vector of the major axis
        major_vector = np.array(major_axis_points[1]) - np.array(major_axis_points[0])

        # Calculate the minor axis
        minor_axis_points = None
        max_minor_distance = 0
        for p1 in points:
            for p2 in points:
                # Calculate the vector from p1 to p2
                vector = np.array(p2) - np.array(p1)
                # Calculate the dot product of the vector and the major axis vector
                dot_product = np.dot(vector, major_vector)
                # If the dot product is 0, the vectors are orthogonal
                if abs(dot_product) < 1e-6:
                    dist = np.linalg.norm(vector)
                    if dist > max_minor_distance:
                        max_minor_distance = dist
                        minor_axis_points = (p1, p2)

        minor_lines.append(minor_axis_points)

    centers = [np.array(loc["points"]).mean(axis=0) for loc in proposed_track_element]
    dists = [
        math.sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) for x, y in centers
    ]
    nearest = np.argmin(dists)
    rel_measurement_line = minor_lines[nearest]
    rel_major_line = major_lines[nearest]

    # Measurement line is in relative coordinates, we need to convert to absolute coordinates before applying scale factor
    measurement_line = [
        (
            rel_measurement_line[0][0] * media.width,
            rel_measurement_line[0][1] * media.height,
        ),
        (
            rel_measurement_line[1][0] * media.width,
            rel_measurement_line[1][1] * media.height,
        ),
    ]
    length_of_line = np.linalg.norm(
        np.array(measurement_line[0]) - np.array(measurement_line[1])
    )
    size *= length_of_line * scale_factor
    return True, {
        args.get("size_attr", "Size (mm)"): size,
        "$new": [
            {
                "type": args.get("line_type_id"),
                "version": args.get("version_id"),
                "x": rel_measurement_line[0][0],
                "y": rel_measurement_line[0][1],
                "u": rel_measurement_line[1][0],
                "v": rel_measurement_line[1][1],
                "media_id": media_id,
            },
            {
                "type": args.get("line_type_id"),
                "version": args.get("version_id"),
                "x": rel_major_line[0][0],
                "y": rel_major_line[0][1],
                "u": rel_major_line[1][0],
                "v": rel_major_line[1][1],
                "media_id": media_id,
            },
        ],
    }


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
