import os
import tempfile
import uuid
import errno
import shutil
from tqdm import tqdm
from typing import List
from urllib.request import Request, urlopen
from urllib.parse import urljoin

READ_DATA_CHUNK = 128 * 1024


# The code below is copied from torch.hub.download_url_to_file
def download_url_to_file(
    url: str,
    dst: str,
    progress: bool = True,
) -> None:
    r"""Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    req = Request(url, headers={"User-Agent": "mblt_model_zoo"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    # We deliberately do not use NamedTemporaryFile to avoid restrictive
    # file permissions being applied to the downloaded file.
    dst = os.path.expanduser(dst)
    for seq in range(tempfile.TMP_MAX):
        tmp_dst = dst + "." + uuid.uuid4().hex + ".partial"
        try:
            f = open(tmp_dst, "w+b")
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")

    try:
        with tqdm(
            desc=os.path.split(dst)[1],
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(READ_DATA_CHUNK)
                if len(buffer) == 0:
                    break
                f.write(buffer)  # type: ignore[possibly-undefined]
                pbar.update(len(buffer))

        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def download_url_to_folder(
    url_dir: str,
    url_filelist: List[str],
    dst: str,
    progress: bool = True,
) -> None:
    os.makedirs(dst, exist_ok=True)

    for url_file in url_filelist:
        u = urljoin(url_dir, url_file)
        d = os.path.join(dst, url_file)
        try:
            if not os.path.exists(d):
                download_url_to_file(u, d, progress)

        except Exception:
            print(f"{url_file} download failed")
