import requests
import json
import os
import re
import time

# Link to Fathomnet WoRMs GitHub
# https://github.com/fathomnet/worms-server

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
    "AphiaID",
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
    data["AphiaID"] = str(child['aphiaId']) if child.get('aphiaId') else ''
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


def _parse_override_labels(value):
    """Parse override labels into a normalized set."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        parts = value
    else:
        parts = str(value).split(",")
    return {str(x).strip().lower() for x in parts if str(x).strip()}


# ---------------------------------------------------------------------------
# FathomVerse mission classification
#
# fv_obj_* models emit CMS concept (mission) IDs as their class names. This
# rule resolves each ID through the FathomVerse CMS and mirrors the label
# mapping in portal-web-ui/server/server.js (missionAttributes /
# wormsTreeToAttributes):
#   - includedTaxa of length 1 is authoritative: expand through the WoRMS
#     ancestors endpoint when the tree's LEAF matches the taxon exactly
#     (case-insensitive); otherwise apply the taxon verbatim.
#   - any other length falls back to the mission display name, which is
#     colloquial and deliberately NOT WoRMS-resolved -- with one rename:
#     "medusa" becomes "medusae" (Medusa is a real, unrelated genus).
#
# CMS auth: FATHOMVERSE_TOKEN env var (ApiKey) is exchanged for a short-lived
# bearer. The token comes ONLY from the environment (a k8s secret on the
# worker pod), never from classify_args -- strategy blobs are logged.
# ---------------------------------------------------------------------------

CMS_AUTH_URL_DEFAULT = "https://cms-stage.fathomverse.game:448/auth/v1/token"
CMS_MISSIONS_URL_DEFAULT = "https://cms-stage.fathomverse.game:448/portal/v2/missions"

_cms_bearer_cache = {"token": None, "expires_at": 0.0}
_mission_attr_cache = {}
_worms_exact_cache = {}


def _merge_classify_args(args):
    """Merge the optional nested classify_args blob into args (top-level wins)."""
    classify_args = args.get("classify_args")
    if classify_args:
        if isinstance(classify_args, str):
            try:
                classify_args = json.loads(classify_args)
            except Exception:
                classify_args = None
        if isinstance(classify_args, dict):
            for k, v in classify_args.items():
                args.setdefault(k, v)
    return args


def _get_cms_bearer(auth_url):
    now = time.time()
    if _cms_bearer_cache["token"] and now < _cms_bearer_cache["expires_at"] - 30:
        return _cms_bearer_cache["token"]
    api_key = os.environ.get("FATHOMVERSE_TOKEN", "").strip()
    if not api_key:
        raise RuntimeError(
            "FATHOMVERSE_TOKEN env var is not set (expected from the "
            "fathomverse-token k8s secret on the worker pod)"
        )
    response = requests.post(
        auth_url, headers={"Authorization": f"ApiKey {api_key}"}, timeout=30
    )
    response.raise_for_status()
    data = response.json()
    _cms_bearer_cache["token"] = data["access_token"]
    try:
        ttl = float(data.get("expires_in") or 300)
    except (TypeError, ValueError):
        ttl = 300.0
    _cms_bearer_cache["expires_at"] = now + ttl
    return _cms_bearer_cache["token"]


def _worms_exact_attributes(taxon_name):
    """taxa/ancestors lookup accepted only when the tree LEAF is exactly the
    requested taxon. Definitive answers (parsed tree or 404) are cached;
    transient failures are not. Returns attrs dict or None."""
    key = str(taxon_name).lower()
    if key in _worms_exact_cache:
        return _worms_exact_cache[key]
    try:
        response = requests.get(
            url=f"{wormsApi['taxaAncestors']}/{taxon_name}", timeout=30
        )
    except requests.RequestException as err:
        print(f"WARNING: WoRMS lookup failed for '{taxon_name}': {err}")
        return None
    if response.status_code == 404:
        _worms_exact_cache[key] = None
        return None
    if response.status_code != 200:
        print(f"WARNING: WoRMS lookup for '{taxon_name}' returned {response.status_code}")
        return None

    tree = response.json()
    children = tree.get("children") or []
    node = children[0] if children else None
    if node is None:
        _worms_exact_cache[key] = None
        return None
    attrs = {"Object type": str(node.get("name") or "object").lower()}
    is_biota = attrs["Object type"] == "biota"
    leaf = None
    while node:
        rank = node.get("rank")
        if is_biota and rank:
            if rank in taxaFields:
                attrs[rank] = node.get("name")
            else:
                print(f"WARNING: Unhandled taxonomic level '{rank}'!")
        leaf = node
        node_children = node.get("children") or []
        node = node_children[0] if node_children else None
    if (
        not leaf
        or not leaf.get("name")
        or str(leaf["name"]).lower() != str(taxon_name).lower()
    ):
        _worms_exact_cache[key] = None
        return None
    attrs["Label"] = leaf["name"]
    attrs["LabelRank"] = leaf.get("rank") or "Label"
    attrs["AphiaID"] = str(leaf["aphiaId"]) if leaf.get("aphiaId") else ""
    alternate = leaf.get("alternateNames")
    if isinstance(alternate, list) and alternate:
        attrs["Common name"] = ", ".join(alternate)
    _worms_exact_cache[key] = attrs
    return attrs


def _fathomverse_mission_attributes(concept_id, auth_url, missions_url):
    """Resolve a CMS mission id to observation attributes, or None."""
    key = str(concept_id)
    if key in _mission_attr_cache:
        return _mission_attr_cache[key]
    try:
        bearer = _get_cms_bearer(auth_url)
        response = requests.get(
            f"{missions_url}/{concept_id}",
            headers={"Authorization": f"Bearer {bearer}"},
            timeout=30,
        )
    except (requests.RequestException, RuntimeError) as err:
        print(f"WARNING: mission {concept_id} fetch failed: {err}")
        return None
    if response.status_code != 200:
        print(f"WARNING: mission {concept_id} fetch returned {response.status_code}")
        return None
    mission = response.json()

    taxa = mission.get("includedTaxa")
    if isinstance(taxa, list):
        taxa = [t.strip() for t in taxa if isinstance(t, str) and t.strip()]
    else:
        taxa = []
    if len(taxa) == 1:
        attrs = _worms_exact_attributes(taxa[0])
        if attrs is None:
            attrs = {"Label": taxa[0], "LabelRank": "Label", "Object type": "object"}
    else:
        name = str(mission.get("Name") or mission.get("name") or "").strip()
        if re.fullmatch("medusa", name, re.IGNORECASE):
            name = f"{name}e"
        if not name:
            _mission_attr_cache[key] = None
            return None
        attrs = {"Label": name, "LabelRank": "Label", "Object type": "object"}
    _mission_attr_cache[key] = attrs
    return attrs


def fathomverse_classify(media_id, proposed_track_element, **args):
    """Map a track's FathomVerse concept-id label to portal attributes."""
    args = _merge_classify_args(args)
    box_label_attr = args.get("box_label_attribute", "Label")
    box_confidence_attr = args.get("conf_attribute", "Confidence")
    concept_id = proposed_track_element[0]["attributes"].get(box_label_attr, "Unknown")

    sum_conf = 0.0
    for x in proposed_track_element:
        sum_conf += x["attributes"].get(box_confidence_attr, 0)
    avg_conf = sum_conf / len(proposed_track_element)
    object_type_confidence = _round_to_bin(avg_conf)

    auth_url = args.get("cms_auth_url") or os.environ.get("CMS_AUTH_URL") or CMS_AUTH_URL_DEFAULT
    missions_url = (
        args.get("cms_missions_url") or os.environ.get("CMS_MISSIONS_URL") or CMS_MISSIONS_URL_DEFAULT
    )

    attrs = _fathomverse_mission_attributes(str(concept_id).strip(), auth_url, missions_url)
    if attrs is None:
        # Keep the track visible rather than dropping it silently.
        print(f"WARNING: no mission mapping for concept '{concept_id}'; keeping raw label")
        attrs = {"Label": str(concept_id), "LabelRank": "Label", "Object type": "object"}

    extended_attrs = {
        "LabelRank": attrs.get("LabelRank", "Label"),
        "Object type": attrs.get("Object type", "object"),
        "Object-type_confidence": object_type_confidence,
    }
    confidence_fields = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    for field in taxaFields:
        if field in attrs:
            extended_attrs[field] = attrs[field]
            if field in confidence_fields:
                extended_attrs[f"{field}_confidence"] = object_type_confidence
    for field in ("Label", "AphiaID", "Common name"):
        if field in attrs:
            extended_attrs[field] = attrs[field]

    return True, extended_attrs


def worms_classify(media_id, proposed_track_element, **args):
    """Update portal attributes based on existing labels"""
    box_label_attr = args.get("box_label_attribute", "Label")
    box_confidence_attr = args.get("conf_attribute", "Confidence")
    label = proposed_track_element[0]["attributes"].get(box_label_attr, "Unknown")
    # print(f"WORMS query for label '{label}'")

    sum_conf = 0.0
    for x in proposed_track_element:
        sum_conf += x["attributes"].get(box_confidence_attr, 0)
    avg_conf = sum_conf / len(proposed_track_element)
    object_type_confidence = _round_to_bin(avg_conf)

    # Optional nested config blob passthrough.
    # If present, merge it with top-level args (top-level wins).
    classify_args = args.get("classify_args")
    if classify_args:
        if isinstance(classify_args, str):
            try:
                classify_args = json.loads(classify_args)
            except Exception:
                classify_args = None
        if isinstance(classify_args, dict):
            for k, v in classify_args.items():
                args.setdefault(k, v)

    if args.get("fathomverse_missions", False):
        return fathomverse_classify(media_id, proposed_track_element, **args)

    override_labels = _parse_override_labels(args.get("classify_override_labels"))
    if args.get("skip_worms_lookup", False) or override_labels:
        
        if override_labels:
            if str(label).strip().lower() in override_labels:
                extended_attrs = {
                    "LabelRank": "Label",
                    "Object type": label,
                    "Object-type_confidence": object_type_confidence,
                }
                return True, extended_attrs
            else: 
                extended_attrs = {
                    "LabelRank": "Label",
                    "Object type": "object",
                    "Object-type_confidence": object_type_confidence,
                }
                return True, extended_attrs
        else:
            extended_attrs = {
                    "LabelRank": "Label",
                    "Object type": label,
                    "Object-type_confidence": object_type_confidence,
            }
            return True, extended_attrs
        

    print(f"WORMS query for label '{label}'")

    response = requests.get(url=f"{wormsApi['taxaStartsWith']}/{label}", timeout=30)
    aphia_id = 0
    results = []
    try:
        results = json.loads(response.content)
        # Get the first matching result's aphia ID if available
        if results and len(results) > 0:
            aphia_id = results[0].get("aphiaId", 0)
            aphia_record = results[0]
    except Exception as e:
        print(e)

    extended_attrs = {
        "LabelRank": "Label",
        "Object type": "object",
        "Object-type_confidence": object_type_confidence,
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
                        extended_attrs[f"{field}_confidence"] = object_type_confidence
            
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
