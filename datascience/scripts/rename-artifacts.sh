#!/bin/bash

# This script looks into all the yaml files of the mlruns
# directoy to remove the current path.
# It is used to prepare the containerization of the datascience project
# and make the artifact usable.


regex="s#$(pwd)##g"
grep --include *.yaml -rl "$(pwd)" mlruns | xargs sed -i "" -r $regex
