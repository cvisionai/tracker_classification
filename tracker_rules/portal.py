import requests
import json

wormsApi = {
  "aphiaLookup": "https://database.fathomnet.org/worms/names/aphiaid/",
  "queryContains": "https://database.fathomnet.org/worms/query/contains/", 
  "synonyms": "https://database.fathomnet.org/worms/synonyms/", 
  "queryStartsWith": "https://database.fathomnet.org/worms/query/startswith", 
  "taxaAncestors": "https://database.fathomnet.org/worms/taxa/ancestors", 
  "taxaChildren": "https://database.fathomnet.org/worms/taxa/children", 
  "taxaStartsWith": "https://database.fathomnet.org/worms/taxa/query/startswith", 
  "taxaContains": "https://database.fathomnet.org/worms/taxa/query/contains",
}

taxaFields = [
    "Kingdom",
    "Subkingdom",
    "Phylum",
    "Subphylum",
    "Infraphylum",
    "Parvphylum",
    "Gigaclass",
    "Megaclass",
    "Superclass",
    "Class",
    "Subclass",
    "Infraclass",
    "Subterclass",
    "Superorder",
    "Order",
    "Suborder",
    "Infraorder",
    "Parvorder",
    "Section",
    "Subsection",
    "Superfamily",
    "Epifamily",
    "Family",
    "Subfamily",
    "Supertribe",
    "Tribe",
    "Subtribe",
    "Genus",
    "Subgenus",
    "Species",
    "Subspecies",
    "Natio",
    "Variety",
    "Subvariety",
    "Forma",
    "Subforms",
    "Mutatio",
    "Label",
]


def _round_to_bin(float_val):
    """convert to enum-based confidence"""
    if float_val < 0.25:
        return "0"
    elif float_val < 0.75:
        return ".5"
    else:
        return "1"


def _worms_rank_to_portal(worms_rank):
    portal_rank = "null"
    if worms_rank in taxaFields:
        portal_rank = worms_rank
    else:
        print(f"WARNING: Unhandled taxonomic level '{worms_rank}'!")

    return portal_rank


def _parse_tree_children(children, data):
    """Recursively parse nested tree structure from WoRMS API"""
    if not children or len(children) == 0:
        return data
    
    child = children[0]
    
    # Set rank-specific data
    if child.get('rank'):
        data[child['rank']] = child['name']
    
    data["Label"] = child['name']
    data["AphiaID"] = child.get('aphiaId', '')
    data["LabelRank"] = child.get('rank', 'Label') if child.get('rank') else 'Label'
    
    # Handle alternate names (common names)
    if 'alternateNames' in child and isinstance(child['alternateNames'], list):
        data["Common name"] = ", ".join(child['alternateNames'])
    
    # Recurse into children
    if child.get('children') and len(child['children']) > 0:
        return _parse_tree_children(child['children'], data)
    
    return data


def _nested_tree_to_json(tree_data):
    """Convert nested tree structure to flat JSON attributes"""
    new_data = {}
    
    # Determine object type
    if tree_data.get('children') and len(tree_data['children']) > 0:
        first_child_name = tree_data['children'][0].get('name', 'object')
        new_data["Object type"] = first_child_name.lower()
    else:
        new_data["Object type"] = "object"
    
    # Parse the tree based on object type
    if new_data["Object type"] == "biota":
        new_data = _parse_tree_children(tree_data.get('children', []), new_data)
    else:
        # Non-biota objects (equipment, substrate, etc.)
        new_data["LabelRank"] = "Label"
        new_data = _parse_tree_children(tree_data.get('children', []), new_data)
    
    return new_data


def worms_classify(media_id, proposed_track_element, **args):
    """Update portal attributes based on existing labels"""
    box_label_attr = args.get("box_label_attribute", "Label")
    box_confidence_attr = args.get("box_label_attribute", "Confidence")
    label = proposed_track_element[0]["attributes"].get(box_label_attr, "Unknown")
    # print(f"WORMS query for label '{label}'")
    response = requests.get(url=f"{wormsApi['taxaStartsWith']}/{label}", timeout=30)
    aphia_id = 0
    results = []
    try:
        results = json.loads(response.content)
        # print(f"WORMS query for '{label}' returned {results}")
        # Get the first matching result's aphia ID if available
        if results and len(results) > 0:
            aphia_id = results[0].get('aphiaId', 0)
            aphia_record = results[0]
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
            url=f"{wormsApi['taxaAncestors']}/{aphia_record.get('name')}", timeout=30
        )
        if response.status_code == 200:
            tree_data = json.loads(response.content)
            
            # Parse the nested tree structure
            parsed_data = _nested_tree_to_json(tree_data)
            
            # Update extended attributes with parsed taxonomy
            rank = parsed_data.get("LabelRank", "Label")
            extended_attrs["LabelRank"] = rank
            extended_attrs["Object type"] = parsed_data.get("Object type", "biota")
            
            # Fields that should have confidence values
            confidence_fields = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
            
            # Add all taxonomy fields from parsed data
            for field in taxaFields:
                if field in parsed_data:
                    extended_attrs[field] = parsed_data[field]
                    # Only add confidence for specific fields
                    if field in confidence_fields:
                        extended_attrs[f"{field}_confidence"] = _round_to_bin(avg_conf)
            
            # Add common name if present
            if "Common name" in parsed_data:
                extended_attrs["Common name"] = parsed_data["Common name"]

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
