import requests
import subprocess
import os

from .env import DATAPATH, ROSSMANN_PATH


def download_rossmann():
    """
    Code from:
    https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py

    Thanks to FAST.AI to host the dataset for us!
    """

    # Thanks for fast.ai to host the dataset.
    url = "http://files.fast.ai/part2/lesson14/rossmann.tgz"
    try:
        if not os.path.exists(DATAPATH):
            os.mkdir(DATAPATH)

        r = requests.get(url, stream=True)
        output_file = open(ROSSMANN_PATH, "wb")

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                output_file.write(chunk)
        output_file.close()

        subprocess.call(("tar xvf %s -C %s" % (ROSSMANN_PATH, DATAPATH)).split(" "))
    except:
        print("Download Failed, " "please manually download data in " "%s" % DATAPATH)
