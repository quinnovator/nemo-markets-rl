import hashlib

import pytest

from deadline_offer import data


@pytest.mark.parametrize("split", data.SPLITS)
def test_generated_split_matches_published_bytes(split):
    digest = hashlib.sha256()
    for line in data.lines(split):
        digest.update(line.encode())
    assert digest.hexdigest() == data.SHA256[split]


def test_write_split(tmp_path):
    path = data.write_split("test", tmp_path)
    assert sum(1 for _ in path.open()) == data.SPLITS["test"][2]
