=============================================================================
#### 1. Python / Haskell / Perl / Your prefered scripting language.<br>
a) array = [['a','b','c'],['d','e','f']]<br>
How do I get this output?<br>
a<br>
b<br>
c<br>
d<br>
e<br>
f<br>
<br>
```python
print(*[j for i in array for j in i], sep = '\n')
```

b) Have a look at "programming-tasks/top10_sample.csv"<br>
Each line in this file represents a list of Brand in our store.<br>
Write a script to print out a list of brand names and their occurrence counts (sorted).<br>
```python
import csv
dataPath = '/home/roms/Desktop/Zalora/'
from collections import Counter
brands = Counter()
with open(dataPath + 'programming-tasks/top10_sample.csv', 'r') as datafile:
    datareader = csv.reader(datafile)
    for row in datareader:
    	 for brand in row[0][1:-1].split(','):
          brands[brand] += 1
brands.most_common(len(brands))
```

=============================================================================
#### 2. SQL<br>
a) What is the relation between Database, Schema, Tables, View in PostgreSQL / MySQL?<br>
```
A Database is a set of Tables and Views. 
Its organization (relations between tables) is based on a Schema.
```
<br>
b) What is the difference between a table and a view?<br>
```
A view is a virtual table and is the result of a given SELECT query.
```
<br>
c) Table reporting.items has 4 columns: Item_Code - Date - Visits - Orders<br>
Write a query to get total number of Visit over all Item_Codes for the day '2013-01-12'.<br>
```sql
SELECT Item_codes, Visits
FROM reporting.items
WHERE Date = '2013-01-12';
```
Write a query to get total number of visit over all Item_Codes for every year?.<br>
```sql
SELECT Item_codes, YEAR(Date), SUM(Visits)
FROM reporting.items
GROUP BY Item_codes, YEAR(Date);
```
<br>
d) As a DBA: in PostgreSQL DB, write query(s) needed to give account "buying" access to all tables currently in schema "sales", and all future Tables created in schema "sales".<br>
```sql
GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES 
IN SCHEMA sales 
TO buying;
```

=============================================================================
#### 3. Bash scripting<br>
a) Write a bash script for the below set of tasks:<br>
- connect to ftp server (host=10.11.12.13 port=22 username=helloFTP password=world)
- download all files that have their name started with "webtrekk_marketing" into "home/Marketing Report/Data/"
- run ZMR.py which is located in "home/Marketing Report/Scripts/"
- run UpdateWebtrekk.sql which is located in "home/Marketing Report/Scripts/" on a PostgreSQL DB (host=10.11.12.13 port=5439 database=zalora username=helloDB password=world)
```shell
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

END_SCRIPT
exit 0
```
How would you schedule the above as a cron job every day at 2.35am?

b) Have a look at the folder "/programming-tasks/bash/"
- Write a bash script to rename all files below from "zalora-*" to "Zalora-*"
```shell
##!/bin/bash
for file in zalora-* ; do mv "$file" Z"${file#z}" ; done
```
- All Zalora-* files contain a single string: "this is a test." (with a new line at the end):
    Write a shell script to change the content of those files to all uppercase.
```shell
##!/bin/bash

```
    Write a shell script to remove all dot character (.) within those files.
```shell
##!/bin/bash

```
