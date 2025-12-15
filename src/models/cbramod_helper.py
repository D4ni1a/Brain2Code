import requests

def download_file(url, local_filename):
    """
    Downloads a file from a given URL and saves it to a local file.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The name of the local file to save the content to.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filt/er out keep-alive new chunks
                    f.write(chunk)
        print(f"File '{local_filename}' downloaded successfully from '{url}'.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")



if __name__ == "__main__":
    # Load model by clonning https://github.com/wjq-learning/CBraMod

    file_url = "https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth"
    output_file = "./pretrained_weights/pretrained_weights.pth"

    download_file(file_url, output_file)

    import torch
    import torch.nn as nn
    from models.cbramod import CBraMod
    from einops.layers.torch import Rearrange

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CBraMod().to(device)
    model.load_state_dict(
        torch.load('/content/local/pretrained_weights/pretrained_weights.pth', map_location=device, weights_only=False))
    model.proj_out = nn.Identity()

    for param in model.parameters():
        param.requires_grad = False
