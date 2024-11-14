from setuptools import setup
import re
import io
import os

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

META_PATH = os.path.join("tmaven", "__init__.py")
META_FILE = read(META_PATH)

def find_meta(meta):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.MULTILINE
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError(f"Unable to find __{meta}__ string.")

if __name__ == "__main__":
    setup(
        version=find_meta("version"),
        description=find_meta("description"),
        license=find_meta("license"),
        url=find_meta("url"),
        author=find_meta("author"),
    )