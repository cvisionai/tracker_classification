import tator
import numpy as np
import math
from statistics import median


def measure_classify(media_id, proposed_track_element, **args):
    api = tator.get_api(host=args["_host"], token=args["_token"])
    print("Media id is: ", media_id)
    media = api.get_media(media_id)

    dimension = args.get("dimension", "both")
    method = args.get("method", "median")
    transform = args.get("transform", "scallops")
    scale_factor = args.get("scale_factor", 1.0)

    if dimension == "both":
        sizes = [
            (loc["width"] * media.width + loc["height"] * media.height) / 2
            for loc in proposed_track_element
        ]
    elif dimension == "width":
        sizes = [loc["width"] * media.width for loc in locs]
    elif dimension == "height":
        sizes = [loc["height"] * media.height for loc in locs]
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
            (loc.x + loc["width"] / 2, loc.y + loc["height"] / 2) for loc in locs
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
        _host=args.host,
        _token=args.token,
        scale_factor=args.scale_factor,
    )
    print(f"Source (ID={args.state_id})")
    pprint(state_obj.attributes)
    print("==========Transforms into======")
    pprint(extended_attrs)
