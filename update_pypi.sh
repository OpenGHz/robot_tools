#! /usr/bin/env bash
set -e

python setup.py clean --all
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*