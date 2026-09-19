-- Tourism Experience Analytics
-- Expected tables: "Transaction" and "Item".
-- "Transaction" is quoted because it is a reserved SQL keyword in SQLite.

-- 1. Basic dataset statistics
SELECT COUNT(*) AS total_transactions
FROM "Transaction";

SELECT COUNT(DISTINCT UserId) AS total_users
FROM "Transaction";

SELECT COUNT(DISTINCT AttractionId) AS total_attractions
FROM "Transaction";

-- 2. Average rating
SELECT AVG(Rating) AS average_rating
FROM "Transaction";

-- 3. Rating distribution
SELECT Rating, COUNT(*) AS rating_count
FROM "Transaction"
GROUP BY Rating
ORDER BY Rating;

-- 4. Visit mode distribution
SELECT VisitMode, COUNT(*) AS transaction_count
FROM "Transaction"
GROUP BY VisitMode
ORDER BY transaction_count DESC;

-- 5. Yearly and monthly activity
SELECT VisitYear, COUNT(*) AS transaction_count
FROM "Transaction"
GROUP BY VisitYear
ORDER BY VisitYear;

SELECT VisitMonth, COUNT(*) AS transaction_count
FROM "Transaction"
GROUP BY VisitMonth
ORDER BY VisitMonth;

-- 6. Top attractions
SELECT AttractionId, COUNT(*) AS visit_count
FROM "Transaction"
GROUP BY AttractionId
ORDER BY visit_count DESC
LIMIT 10;

SELECT
    AttractionId,
    AVG(Rating) AS average_rating,
    COUNT(*) AS number_of_reviews
FROM "Transaction"
GROUP BY AttractionId
ORDER BY average_rating DESC;

-- 7. Rating and visit-mode analysis
SELECT
    VisitMode,
    AVG(Rating) AS average_rating,
    COUNT(*) AS transaction_count
FROM "Transaction"
GROUP BY VisitMode
ORDER BY average_rating DESC;

SELECT VisitMode, Rating, COUNT(*) AS transaction_count
FROM "Transaction"
GROUP BY VisitMode, Rating
ORDER BY VisitMode, Rating;

-- 8. Attraction performance
SELECT
    AttractionId,
    COUNT(*) AS visits,
    AVG(Rating) AS average_rating,
    MIN(Rating) AS minimum_rating,
    MAX(Rating) AS maximum_rating
FROM "Transaction"
GROUP BY AttractionId
ORDER BY visits DESC;

SELECT AttractionId, AVG(Rating) AS average_rating, COUNT(*) AS number_of_reviews
FROM "Transaction"
GROUP BY AttractionId
HAVING AVG(Rating) >= 4.5
ORDER BY average_rating DESC;

SELECT
    AttractionId,
    COUNT(*) AS visit_count,
    AVG(Rating) AS average_rating
FROM "Transaction"
GROUP BY AttractionId
HAVING COUNT(*) >= 100
ORDER BY average_rating DESC, visit_count DESC;

SELECT
    AttractionId,
    COUNT(*) AS visit_count,
    AVG(Rating) AS average_rating
FROM "Transaction"
GROUP BY AttractionId
HAVING COUNT(*) >= 500 AND AVG(Rating) >= 4.0
ORDER BY visit_count DESC;

-- 9. User activity
SELECT
    UserId,
    COUNT(*) AS transaction_count,
    COUNT(DISTINCT AttractionId) AS unique_attractions,
    AVG(Rating) AS average_rating
FROM "Transaction"
GROUP BY UserId
ORDER BY transaction_count DESC;

SELECT UserId, COUNT(*) AS transaction_count
FROM "Transaction"
GROUP BY UserId
ORDER BY transaction_count DESC
LIMIT 10;

SELECT
    UserId,
    AVG(Rating) AS average_rating,
    COUNT(*) AS number_of_ratings
FROM "Transaction"
GROUP BY UserId
HAVING COUNT(*) >= 2
ORDER BY average_rating DESC;

-- 10. Attraction type analysis
SELECT
    i.AttractionTypeId,
    COUNT(t.TransactionId) AS visit_count,
    AVG(t.Rating) AS average_rating
FROM "Transaction" AS t
JOIN "Item" AS i ON t.AttractionId = i.AttractionId
GROUP BY i.AttractionTypeId
ORDER BY visit_count DESC;

-- 11. Time and visit-mode analysis
SELECT VisitYear, VisitMode, COUNT(*) AS transaction_count
FROM "Transaction"
GROUP BY VisitYear, VisitMode
ORDER BY VisitYear, transaction_count DESC;

SELECT
    VisitYear,
    AVG(Rating) AS average_rating,
    COUNT(*) AS transaction_count
FROM "Transaction"
GROUP BY VisitYear
ORDER BY VisitYear;
