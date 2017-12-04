import bz2
import urllib.request
import os
import os.path

DATA_DIR = 'data'
OPEN_CORPORA_URL = 'http://opencorpora.org/files/export/annot/annot.opcorpora.no_ambig.xml.bz2'
OPEN_CORPORA_DEST_FILE = 'data/annot.opcorpora.no_ambig.xml'
CHUNK = 16 * 1024


def download_and_unbzip(url, dest_file):
    source = urllib.request.urlopen(url)
    decompressor = bz2.BZ2Decompressor()
    with open(dest_file, 'wb') as dest_file:
        while True:
            data = source.read(CHUNK)
            if not data:
                break
            dest_file.write(decompressor.decompress(data))


if __name__ == '__main__':
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)
    download_and_unbzip(OPEN_CORPORA_URL, OPEN_CORPORA_DEST_FILE)
