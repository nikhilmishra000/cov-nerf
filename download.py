import gdown


def main():
    for folder, url in {
        "checkpoints": "https://drive.google.com/drive/folders/15dVBBGj2NvTqbiv1dGnj3_x_XZfz3MWB",
        "assets": "https://drive.google.com/drive/folders/1QE2UeSYwxgqutVeWliQC1eXAO_loOq_5",
    }.items():
        gdown.download_folder(url, output=folder)


if __name__ == "__main__":
    main()
