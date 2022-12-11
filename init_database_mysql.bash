#!/bin/bash -eu
python ./src/mysql/init_database.py $1 $2
echo initialized database $1
