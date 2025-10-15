import requests
import json

WORMS_API_URL = "https://marinespecies.org/rest"


def _round_to_bin(float_val):
    """convert to enum-based confidence"""
    if float_val < 0.25:
        return "0"
    elif float_val < 0.75:
        return ".5"
    else:
        return "1"


def _worms_rank_to_portal(worms_rank):
    """Not all worms ranks exist in portal. This truncates to the neartest rank"""
    portal_ranks = [
        "Object type",
        "Kingdom",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
        "null",
        "Label",
    ]
    portal_rank = "null"
    if worms_rank in portal_ranks:
        portal_rank = worms_rank
    elif worms_rank in ["Subclass", "Superorder"]:
        portal_rank = "Class"
    elif worms_rank in ["Suborder"]:
        portal_rank = "Order"
    elif worms_rank in ["Subfamily", "Tribe"]:
        portal_rank = "Family"
    elif worms_rank in ["Variety"]:
        portal_rank = "Species"
    else:
        print(f"WARNING: Unhandled taxonomic level '{worms_rank}'!")

    return portal_rank


def worms_classify(media_id, proposed_track_element, **args):
    """Update portal attributes based on existing labels"""
    box_label_attr = args.get("box_label_attribute", "Label")
    box_confidence_attr = args.get("box_label_attribute", "Confidence")
    label = proposed_track_element[0]["attributes"].get(box_label_attr, "Unknown")
    response = requests.get(url=f"{WORMS_API_URL}/AphiaIDByName/{label}", timeout=30)
    aphia_id = 0
    try:
        aphia_id = json.loads(response.content)
    except Exception as e:
        print(e)

    sum_conf = 0.0
    for x in proposed_track_element:
        sum_conf += x["attributes"].get(box_confidence_attr, 0)
    avg_conf = sum_conf / len(proposed_track_element)

    extended_attrs = {
        "LabelRank": "Label",
        "Object type": "object",
        "Object-type_confidence": _round_to_bin(avg_conf),
    }
    if aphia_id > 0:
        response = requests.get(
            url=f"{WORMS_API_URL}/AphiaRecordByAphiaID/{aphia_id}", timeout=30
        )
        if response.status_code == 200:
            aphia_record = json.loads(response.content)
            rank = _worms_rank_to_portal(aphia_record["rank"])
            extended_attrs["LabelRank"] = rank
            extended_attrs["Object type"] = "biota"
            if rank == "Species":
                species = (
                    aphia_record["scientificname"]
                    .replace(aphia_record["genus"], "", 1)
                    .strip()
                )
                extended_attrs["Species"] = species
            for x in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]:
                record = aphia_record.get(x.lower(), "")
                if record:
                    extended_attrs[x] = record
                    extended_attrs[f"{x}_confidence"] = _round_to_bin(avg_conf)
        else:
            print(f"ERROR: {label} not found in WoRMs API")

    return True, extended_attrs


if __name__ == "__main__":
    """Unit test"""
    import tator
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--token")
    parser.add_argument("state_id", type=int)
    args = parser.parse_args()
    api = tator.get_api(host=args.host, token=args.token)
    state_obj = api.get_state(args.state_id)
    state_type_obj = api.get_state_type(state_obj.type)
    localizations = api.get_localization_list_by_id(
        state_type_obj.project, {"ids": state_obj.localizations}
    )
    extended_attrs = worms_classify(
        state_obj.media, [l.to_dict() for l in localizations]
    )
    print(f"Source (ID={args.state_id})")
    pprint(state_obj.attributes)
    print("==========Transforms into======")
    pprint(extended_attrs)
