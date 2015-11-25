SELECT SUM(Visits)
FROM reporting.items
WHERE Date = '2013-01-12';

SELECT YEAR(Date), SUM(Visits)
FROM reporting.items
GROUP BY YEAR(Date);