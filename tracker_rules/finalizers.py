#!/usr/bin/python3

# Finalizes are used on the media file after all other processing is complete
import tator


def qr_code_foi(media_id, **args):
    api = tator.get_api(host=args["host"], token=args["token"])
    state_type_id = args["state_type_id"]

    media = api.get_media(media_id)
    qr_codes = api.get_state_list(
        media.project, media_id=[media_id], type=state_type_id
    )
    print(f"Found {len(qr_codes)} QR codes")
    attrs = {"QR Count": len(qr_codes)}
    if len(qr_codes) == 2:
        sorted_codes = sorted(qr_codes, key=lambda x: x.segments[0][0])
        end_of_first = sorted_codes[0].segments[-1][1]
        start_of_second = sorted_codes[1].segments[0][0]
        if abs(end_of_first - start_of_second) > 50:
            print(f"foi = {end_of_first} to {start_of_second}")
            attrs["fois"] = f"{[[end_of_first, start_of_second]]}"

    api.update_media(media_id, {"attributes": attrs})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", help="Tator host URL", default="https://cloud.tator.io"
    )
    parser.add_argument("--token", required=True, help="Tator token")
    parser.add_argument("media_ids", type=str, help="Media ID to finalize")
    parser.add_argument("state_type_id", type=int, help="State type ID for QR codes")
    args = parser.parse_args()
    api = tator.get_api(host=args.host, token=args.token)
    state_type_obj = api.get_state_type(args.state_type_id)
    project = state_type_obj.project
    raw_media_ids = args.media_ids.split(",")
    media_ids = []
    for idx in raw_media_ids:
        if idx.isdigit():
            media_ids.append(int(idx))
        elif idx.find("section:") == 0:
            section_id = int(idx.split("section:")[1])
            section_list = api.get_media_list(project, section=section_id)
            media_ids.extend([m.id for m in section_list])

    for media_id in media_ids:
        qr_code_foi(
            media_id,
            host=args.host,
            token=args.token,
            state_type_id=args.state_type_id,
        )
