filename = "your_filename_here"
def create_metadata(seed, feature, filename, text, my_text, attribute_values=None):

    # Your function implementation here

    metadata = create_metadata(
    title="NFT Artwork",
    artist="Viktor S. Kristensen",
    description=text,

    seed=seed,
    feature=feature,
    filename=filename,  # Replace this with the correct variable
    text=text           # Replace this with the correct variable
) 
    return metadata

def save_metadata_to_json(metadata, output_folder, seed, feature):
    json_filename = os.path.join(output_folder, f"nft_{seed}_metadata_{feature}.json")

    with open(json_filename, "w") as json_file:
        json.dump(metadata, json_file, indent=4)

    return json_filename

def save_metadata_to_csv(metadata_list, output_folder, csv_filename="metadata.csv"):
    keys = metadata_list[0].keys()
    csv_file_path = os.path.join(output_folder, csv_filename)

    with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
        dict_writer = csv.DictWriter(csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(metadata_list)
