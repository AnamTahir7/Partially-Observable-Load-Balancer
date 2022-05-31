#!/bin/bash
version=$(python -c "import platform; print(platform.python_version())")
if [ "$version" != "3.6.8" ]
then
	echo "Requires Python 3.6.8"
	exit 1
fi
pip install -r requirements.txt

