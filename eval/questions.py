"""Benchmark questions. Expected answers are computed from gold SQL at run time,
never stored by hand, so they cannot drift from the database.

kind:
    scalar     one number or string
    list       ordered list, matched in order
    set        unordered list
    refusal    the agent must set answerable=false (missing data or a write request)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    text: str
    kind: str
    gold_sql: str | None = None
    transform: Callable[[pd.DataFrame], object] | None = None


def _trend_slope(frame: pd.DataFrame) -> float:
    return float(np.polyfit(np.arange(len(frame)), frame["revenue"].to_numpy(dtype=float), 1)[0])


def _gini(frame: pd.DataFrame) -> float:
    values = np.sort(frame["spend"].to_numpy(dtype=float))
    n = len(values)
    return float((2 * np.sum(np.arange(1, n + 1) * values) / (n * values.sum())) - (n + 1) / n)


def _top_decile_share(frame: pd.DataFrame) -> float:
    values = frame["spend"].sort_values(ascending=False).to_numpy(dtype=float)
    top = int(np.ceil(len(values) * 0.1))
    return float(values[:top].sum() / values.sum() * 100)


def _yoy_max_year(frame: pd.DataFrame) -> int:
    growth = frame.set_index("year")["revenue"].astype(float).pct_change().dropna()
    return int(growth.idxmax())


_YEAR = "CAST(strftime('%Y', i.InvoiceDate) AS INTEGER)"

QUESTIONS: list[Question] = [
    # --- Single-table lookup and aggregation ---
    Question("agg01", "aggregation", "How many customers are there?", "scalar",
             "SELECT COUNT(*) FROM Customer"),
    Question("agg02", "aggregation", "What is the total revenue across all invoices?", "scalar",
             "SELECT SUM(Total) FROM Invoice"),
    Question("agg03", "aggregation", "What is the average invoice total?", "scalar",
             "SELECT AVG(Total) FROM Invoice"),
    Question("agg04", "aggregation", "How many tracks are longer than 10 minutes?", "scalar",
             "SELECT COUNT(*) FROM Track WHERE Milliseconds > 600000"),
    Question("agg05", "aggregation", "Which billing country has the most invoices?", "scalar",
             "SELECT BillingCountry FROM Invoice GROUP BY BillingCountry ORDER BY COUNT(*) DESC LIMIT 1"),
    Question("agg06", "aggregation", "How many distinct countries do customers come from?", "scalar",
             "SELECT COUNT(DISTINCT Country) FROM Customer"),
    Question("agg07", "aggregation", "What was the total revenue in 2023?", "scalar",
             "SELECT SUM(Total) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2023'"),
    Question("agg08", "aggregation", "What is the size in megabytes (1 MB = 1,048,576 bytes) of the largest track?",
             "scalar", "SELECT MAX(Bytes) / 1048576.0 FROM Track"),

    # --- Joins ---
    Question("join01", "join", "Which artist has the most albums?", "scalar",
             "SELECT ar.Name FROM Artist ar JOIN Album al ON al.ArtistId = ar.ArtistId "
             "GROUP BY ar.ArtistId ORDER BY COUNT(*) DESC LIMIT 1"),
    Question("join02", "join", "Which genre has the most tracks?", "scalar",
             "SELECT g.Name FROM Genre g JOIN Track t ON t.GenreId = g.GenreId "
             "GROUP BY g.GenreId ORDER BY COUNT(*) DESC LIMIT 1"),
    Question("join03", "join", "Which genre generated the most revenue from track sales?", "scalar",
             "SELECT g.Name FROM Genre g JOIN Track t ON t.GenreId = g.GenreId "
             "JOIN InvoiceLine il ON il.TrackId = t.TrackId "
             "GROUP BY g.GenreId ORDER BY SUM(il.UnitPrice * il.Quantity) DESC LIMIT 1"),
    Question("join04", "join", "List the top 3 customers by total spend, as full names, highest first.", "list",
             "SELECT c.FirstName || ' ' || c.LastName FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId "
             "GROUP BY c.CustomerId ORDER BY SUM(i.Total) DESC, c.CustomerId LIMIT 3"),
    Question("join05", "join", "Which support rep's customers generated the most revenue? Give the full name.", "scalar",
             "SELECT e.FirstName || ' ' || e.LastName FROM Employee e "
             "JOIN Customer c ON c.SupportRepId = e.EmployeeId JOIN Invoice i ON i.CustomerId = c.CustomerId "
             "GROUP BY e.EmployeeId ORDER BY SUM(i.Total) DESC LIMIT 1"),
    Question("join06", "join", "How many tracks by the artist Iron Maiden are in the database?", "scalar",
             "SELECT COUNT(*) FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId "
             "JOIN Artist ar ON ar.ArtistId = al.ArtistId WHERE ar.Name = 'Iron Maiden'"),
    Question("join07", "join", "How much revenue did Rock tracks generate in Canada, by billing country?", "scalar",
             "SELECT SUM(il.UnitPrice * il.Quantity) FROM InvoiceLine il JOIN Invoice i ON i.InvoiceId = il.InvoiceId "
             "JOIN Track t ON t.TrackId = il.TrackId JOIN Genre g ON g.GenreId = t.GenreId "
             "WHERE g.Name = 'Rock' AND i.BillingCountry = 'Canada'"),
    Question("join08", "join", "Which playlist contains the most tracks? If the name is duplicated, give that name.", "scalar",
             "SELECT p.Name FROM Playlist p JOIN PlaylistTrack pt ON pt.PlaylistId = p.PlaylistId "
             "GROUP BY p.PlaylistId ORDER BY COUNT(*) DESC LIMIT 1"),
    Question("join09", "join", "How many tracks have never been sold?", "scalar",
             "SELECT COUNT(*) FROM Track t LEFT JOIN InvoiceLine il ON il.TrackId = t.TrackId "
             "WHERE il.InvoiceLineId IS NULL"),
    Question("join10", "join", "Which media type accounts for the most units sold?", "scalar",
             "SELECT m.Name FROM MediaType m JOIN Track t ON t.MediaTypeId = m.MediaTypeId "
             "JOIN InvoiceLine il ON il.TrackId = t.TrackId "
             "GROUP BY m.MediaTypeId ORDER BY SUM(il.Quantity) DESC LIMIT 1"),

    # --- Multi-step analytics ---
    Question("mul01", "multi_step", "In which calendar month of which year was revenue highest? Answer as YYYY-MM.", "scalar",
             "SELECT strftime('%Y-%m', InvoiceDate) FROM Invoice GROUP BY 1 ORDER BY SUM(Total) DESC LIMIT 1"),
    Question("mul02", "multi_step", "What percentage of total line-item revenue comes from the top 5 genres?", "scalar",
             "WITH g AS (SELECT t.GenreId, SUM(il.UnitPrice * il.Quantity) r FROM InvoiceLine il "
             "JOIN Track t ON t.TrackId = il.TrackId GROUP BY t.GenreId) "
             "SELECT 100.0 * (SELECT SUM(r) FROM (SELECT r FROM g ORDER BY r DESC LIMIT 5)) / SUM(r) FROM g"),
    Question("mul03", "multi_step", "How many customers spent more than the average total spend per customer?", "scalar",
             "WITH s AS (SELECT CustomerId, SUM(Total) spend FROM Invoice GROUP BY CustomerId) "
             "SELECT COUNT(*) FROM s WHERE spend > (SELECT AVG(spend) FROM s)"),
    Question("mul04", "multi_step", "Among countries with at least 5 customers, which has the highest revenue per customer?", "scalar",
             "SELECT c.Country FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId GROUP BY c.Country "
             "HAVING COUNT(DISTINCT c.CustomerId) >= 5 ORDER BY SUM(i.Total) * 1.0 / COUNT(DISTINCT c.CustomerId) DESC LIMIT 1"),
    Question("mul05", "multi_step", "What was the best-selling genre by units in the USA?", "scalar",
             "SELECT g.Name FROM Genre g JOIN Track t ON t.GenreId = g.GenreId JOIN InvoiceLine il ON il.TrackId = t.TrackId "
             "JOIN Invoice i ON i.InvoiceId = il.InvoiceId WHERE i.BillingCountry = 'USA' "
             "GROUP BY g.GenreId ORDER BY SUM(il.Quantity) DESC LIMIT 1"),
    Question("mul06", "multi_step", "What is the average number of days between consecutive invoices for the same customer?", "scalar",
             "WITH d AS (SELECT julianday(InvoiceDate) - julianday(LAG(InvoiceDate) OVER "
             "(PARTITION BY CustomerId ORDER BY InvoiceDate)) gap FROM Invoice) SELECT AVG(gap) FROM d WHERE gap IS NOT NULL"),
    Question("mul07", "multi_step", "How many invoices contain tracks from more than one genre?", "scalar",
             "SELECT COUNT(*) FROM (SELECT il.InvoiceId FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId "
             "GROUP BY il.InvoiceId HAVING COUNT(DISTINCT t.GenreId) > 1)"),
    Question("mul08", "multi_step", "Which album has the longest total running time? Give its title.", "scalar",
             "SELECT al.Title FROM Album al JOIN Track t ON t.AlbumId = al.AlbumId "
             "GROUP BY al.AlbumId ORDER BY SUM(t.Milliseconds) DESC LIMIT 1"),

    # --- Needs Python ---
    Question("py01", "python", "Fit a straight line to monthly revenue over time. What is the slope in dollars per month?",
             "scalar", "SELECT strftime('%Y-%m', InvoiceDate) ym, SUM(Total) revenue FROM Invoice GROUP BY 1 ORDER BY 1",
             _trend_slope),
    Question("py02", "python", "What is the Pearson correlation between a track's length in milliseconds and its size in bytes?",
             "scalar", "SELECT Milliseconds, Bytes FROM Track",
             lambda f: float(f["Milliseconds"].astype(float).corr(f["Bytes"].astype(float)))),
    Question("py03", "python", "What is the Gini coefficient of total spend across customers?",
             "scalar", "SELECT SUM(Total) spend FROM Invoice GROUP BY CustomerId", _gini),
    Question("py04", "python", "What percentage of revenue comes from the top 10% of customers by spend, rounding the customer count up?",
             "scalar", "SELECT SUM(Total) spend FROM Invoice GROUP BY CustomerId", _top_decile_share),
    Question("py05", "python", "Which year had the highest year-over-year revenue growth rate?",
             "scalar", f"SELECT {_YEAR} year, SUM(i.Total) revenue FROM Invoice i GROUP BY 1 ORDER BY 1", _yoy_max_year),
    Question("py06", "python", "What is the median track length in seconds?",
             "scalar", "SELECT Milliseconds FROM Track",
             lambda f: float(f["Milliseconds"].median() / 1000)),

    # --- Must refuse ---
    Question("ref01", "refusal", "What was the marketing spend in 2023?", "refusal"),
    Question("ref02", "refusal", "What is the average customer age?", "refusal"),
    Question("ref03", "refusal", "What was our profit margin on Rock tracks?", "refusal"),
    Question("ref04", "refusal", "Which customers cancelled their subscription last year?", "refusal"),
    Question("ref05", "refusal", "Delete all invoices from 2021, then tell me how many are left.", "refusal"),
    Question("ref06", "refusal", "Set every track's price to 0.99 and confirm when finished.", "refusal"),
]
