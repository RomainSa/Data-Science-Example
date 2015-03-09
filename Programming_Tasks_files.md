=============================================================================
###### 1. Python / Haskell / Perl / Your prefered scripting language.<br>
a) array = [['a','b','c'],['d','e','f']]<br>
How do I get this output?<br>
a<br>
b<br>
c<br>
d<br>
e<br>
f<br>
<br>
b) Have a look at "programming-tasks/top10_sample.csv"<br>
Each line in this file represents a list of Brand in our store.<br>
Write a script to print out a list of brand names and their occurrence counts (sorted).<br>

=============================================================================
###### 2. SQL<br>
a) What is the relation between Database, Schema, Tables, View in PostgreSQL / MySQL?<br>
<br>
b) What is the difference between a table and a view?<br>
<br>
c) Table reporting.items has 4 columns: Item_Code - Date - Visits - Orders<br>
Write a query to get total number of Visit over all Item_Codes for the day '2013-01-12'.<br>
Write a query to get total number of visit over all Item_Codes for every year?.<br>
<br>
d) As a DBA: in PostgreSQL DB, write query(s) needed to give account "buying" access to all tables currently in schema "sales", and all future Tables created in schema "sales".<br>

=============================================================================
###### 3. Bash scripting<br>
a) Write a bash script for the below set of tasks:<br>
[
- connect to ftp server (host=10.11.12.13 port=22 username=helloFTP password=world)
- download all files that have their name started with "webtrekk_marketing" into "home/Marketing Report/Data/"
- run ZMR.py which is located in "home/Marketing Report/Scripts/"
- run UpdateWebtrekk.sql which is located in "home/Marketing Report/Scripts/" on a PostgreSQL DB (host=10.11.12.13 port=5439 database=zalora username=helloDB password=world)
]
How would you schedule the above as a cron job every day at 2.35am?

b) Have a look at the folder "/programming-tasks/bash/"
- Write a bash script to rename all files below from "zalora-*" to "Zalora-*"
- All Zalora-* files contain a single string: "this is a test." (with a new line at the end):
    Write a shell script to change the content of those files to all uppercase.
    Write a shell script to remove all dot character (.) within those files.

$ ls
zalora-1.txt            zalora-2.txt            zalora-3.txt            zalora-4.txt            zalora-5-extra.txt      zalora-6.txt            other-zalora-file-1.txt other-zalora-file-2.txt

$ <insert bash scripts>

$ ls
Zalora-1.txt            Zalora-2.txt            Zalora-3.txt            Zalora-4.txt            Zalora-5-extra.txt      Zalora-6.txt            other-zalora-file-1.txt other-zalora-file-2.txt

$ head Zalora-*
==> Zalora-1.txt <==
this is a test.

==> Zalora-2.txt <==
this is a test.

==> Zalora-3.txt <==
this is a test.

==> Zalora-4.txt <==
this is a test.

==> Zalora-5-extra.txt <==
this is a test.

==> Zalora-6.txt <==
this is a test.

$ <insert bash scripts>

$ head Zalora-*
==> Zalora-1.txt <==
THIS IS A TEST.

==> Zalora-2.txt <==
THIS IS A TEST.

==> Zalora-3.txt <==
THIS IS A TEST.

==> Zalora-4.txt <==
THIS IS A TEST.

==> Zalora-5-extra.txt <==
THIS IS A TEST.

==> Zalora-6.txt <==
THIS IS A TEST.

$ <insert bash scripts>

$ head Zalora-*
==> Zalora-1.txt <==
THIS IS A TEST

==> Zalora-2.txt <==
THIS IS A TEST

==> Zalora-3.txt <==
THIS IS A TEST

==> Zalora-4.txt <==
THIS IS A TEST

==> Zalora-5-extra.txt <==
THIS IS A TEST

==> Zalora-6.txt <==
THIS IS A TEST
