def create_metadata(seed, feature, filename, text, my_text, image_features=None, attribute_values=None):
    metadata = {
        "name": f"NFT Artwork {seed}-{feature}",
        "description": text,
        "my_text": my_text,
        "image": filename,
        "image_features": image_features,
        "attributes": attribute_values
    }

    return metadata
    if attribute_values:
        metadata["attributes"] = attribute_values
    return metadata

def save_metadata_to_json(metadata, output_folder, seed, feature):
    json_filename = os.path.join(output_folder, f"nft_{seed}_metadata_{feature}.json")

    with open(json_filename, "w") as json_file:
        json.dump(metadata, json_file, indent=4)

    return json_filename

def save_metadata_to_csv(metadata, csv_filename):
    metadata_list = [metadata]  # Convert metadata to a list containing a single dictionary
    keys = metadata_list[0].keys()

    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        dict_writer = csv.DictWriter(csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(metadata_list)


