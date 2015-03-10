##!/bin/bash

# connect to ftp server (host=10.11.12.13 port=22 username=helloFTP password=world)
HOST='10.11.12.13'
LOGIN='helloFTP'
PASSWORD='world'
PORT=22
ftp $HOST $PORT << END_SCRIPT
quote USER $LOGIN
quote PASS $PASSWORD

# download all files that have their name started with "webtrekk_marketing" into "home/Marketing Report/Data/"
mget webtrekk_marketing* "home/Marketing Report/Data/"
quit

# run ZMR.py which is located in "home/Marketing Report/Scripts/"
python "home/Marketing Report/Scripts/ZMR.py"

# run UpdateWebtrekk.sql which is located in "home/Marketing Report/Scripts/" on a PostgreSQL DB (host=10.11.12.13 port=5439 database=zalora username=helloDB password=world
export PGPASSWORD=world
psql -h 10.11.12.13 -d zalora -p 5439 -U helloDB -f "home/Marketing Report/Scripts/UpdateWebtrekk.sql"

END_SCRIPT
exit 0