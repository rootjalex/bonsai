#!/usr/bin/env python3
import sqlite3
import duckdb
import psycopg2
import pandas as pd
import glob
import time
import re
from collections import defaultdict
from statistics import mean
import signal

# -------------------------------
# Parameters
# -------------------------------
k = 0.0001
low, high = 0.0001, 0.0002
num_runs = 7
drop = 1
timeout = 30 # seconds

sqlite_db = "sqlite_test.db"
duckdb_db = "duckdb_test.db"
csv_dir = "./apps/queries/joins/salary"
output_csv = "join_results.csv"

# -------------------------------
# Utility functions
# -------------------------------

def avg_trimmed(times, drop=2):
    """Drop extremes and return average of remaining."""
    if len(times) < drop * 2 + 1:
        return mean(times)
    times_sorted = sorted(times)
    trimmed = times_sorted[drop: len(times_sorted) - drop]
    return mean(trimmed)

def all_timeouts(runtime_dict, timeout_val):
    """Return True if all benchmarked variants timed out."""
    return all(
        isinstance(v, (int, float)) and v >= timeout_val
        for v in runtime_dict.values()
    )

def connect_sqlite(db_file):
    conn = sqlite3.connect(":memory:", check_same_thread=True)
    try:
        conn.execute("PRAGMA threads = 1;")
    except sqlite3.OperationalError:
        pass
    return conn

def connect_duckdb(db_file):
    conn = duckdb.connect(":memory:")
    conn.execute("SET threads TO 1;")
    return conn

def connect_postgres(dbname="ajroot", user="ajroot", host="/tmp", port=5432):
    conn = psycopg2.connect(dbname=dbname, user=user, host=host, port=port)
    conn.autocommit = True
    with conn.cursor() as cur:
        # Disable all parallelism and force sequential execution
        cur.execute("SET max_parallel_workers_per_gather = 1;")
        cur.execute("SET max_parallel_workers = 1;")
        cur.execute("SET parallel_setup_cost = 100000000;")
        cur.execute("SET parallel_tuple_cost = 100000000;")
        # Use temp tables so everything is effectively in-memory
        cur.execute("SET temp_tablespaces = 'pg_default';")
    return conn

def print_query_plan(conn, query, db_type="sqlite"):
    if db_type == "sqlite":
        plan = conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
        s = "+".join([p[3] for p in plan])
        print(f"sqlite3 Join Type: {s}", flush=True)
    elif db_type == "duckdb":
        plan = conn.execute(f"EXPLAIN {query}").fetchall()
        plan_str = plan[0][1]  # second element has the plan text
        join_line = None
        for line in plan_str.splitlines():
            if "JOIN" in line.upper():
                join_line = line.strip()
                break
        print(f"DuckDB Join Type: {join_line or plan}", flush=True)
    elif db_type == "postgres":
        cur = conn.cursor()
        cur.execute(f"EXPLAIN {query}")
        plan = cur.fetchall()
        join_line = None
        # for row in plan:
        #     if "join" in row[0].lower():
        #         join_line = row[0]
        #         break
        print(f"Postgres Join Type: {join_line or plan}", flush=True)
    else:
        raise ValueError(f"DB type not known: {db_type}")

def load_sqlite_table(conn, table_name, csv_file):
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"""
    CREATE TABLE {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        salary REAL,
        tax REAL
    )
    """)
    with open(csv_file) as f:
        next(f)  # skip header
        for line in f:
            name, dept, salary, tax = line.strip().split(",")
            salary = float(salary)
            tax = float(tax)
            conn.execute(f"INSERT INTO {table_name} (salary, tax) VALUES (?, ?)", (salary, tax))
    conn.commit()

def load_duckdb_table(conn, table_name, csv_file):
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TABLE {table_name} AS SELECT salary, tax FROM read_csv_auto('{csv_file}')")

def load_postgres_table(conn, table_name, csv_file):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                salary DOUBLE PRECISION,
                tax DOUBLE PRECISION
            )
        """)
        with open(csv_file) as f:
            next(f)  # skip header
            for line in f:
                name, dept, salary, tax = line.strip().split(",")
                salary = float(salary)
                tax = float(tax)
                cur.execute(f"INSERT INTO {table_name} (salary, tax) VALUES (%s, %s)", (salary, tax))
    conn.commit()

def run_index_variant(conn, query, system):
    """Run query with all index variants and return average trimmed runtimes."""
    table0, table1 = "input0", "input1"
    runtimes = {}

    def time_query(timeout_sec=30):
        """Run query num_runs times with a timeout, return avg or timeout."""
        class TimeoutException(Exception):
            pass

        times, timeouts = [], 0

        for i in range(num_runs):
            start = time.time()
            try:
                if system == "postgres":
                    with conn.cursor() as cur:
                        cur.execute(f"SET statement_timeout TO {timeout_sec * 1000};")
                        cur.execute(query)
                        cur.fetchall()

                elif system == "sqlite":
                    # Only use progress handler
                    conn.set_progress_handler(
                        lambda: 1 if (time.time() - start) > timeout_sec else 0,
                        10000
                    )
                    conn.execute(query).fetchall()
                    conn.set_progress_handler(None, 0)

                else:  # DuckDB
                    # Use signal alarm for DuckDB
                    def handler(signum, frame):
                        raise TimeoutException()
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(timeout_sec)
                    try:
                        conn.execute(query).fetchall()
                    finally:
                        signal.alarm(0)

                times.append(time.time() - start)

            except TimeoutException:
                print(f"Run {i+1}: timed out after {timeout_sec}s")
                timeouts += 1
            except Exception as e:
                print(f"Run {i+1}: query failed ({e})")
                timeouts += 1

            if timeouts > drop:
                print("Too many timeouts, aborting benchmark for this query.")
                return timeout_sec

        if not times:
            return timeout_sec
        return avg_trimmed(times, drop)


    # Utility wrapper for executing commands
    def exec_sql(sql):
        if system == "postgres":
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        else:
            if system == "sqlite":
                conn.set_progress_handler(None, 0)
            conn.execute(sql)

    # Drop any existing indexes
    for tbl in [table0, table1]:
        try:
            exec_sql(f"DROP INDEX IF EXISTS idx_{tbl}_x_y;")
        except Exception as e:
            print(f"(warn) couldn't drop index {tbl}: {e}")

    # No index
    # print(f"\n\n{system} query plan with NO index for: {query}")
    # print_query_plan(conn, query, db_type=system)
    # runtimes["no_index"] = time_query(timeout)

    # # Index on table0
    # exec_sql(f"CREATE INDEX idx_{table0}_x_y ON {table0}(x, y);")
    # exec_sql(f"ANALYZE {table0};")
    # exec_sql(f"ANALYZE {table1};")
    # print(f"\n\n{system} query plan with index0 for: {query}")
    # print_query_plan(conn, query, db_type=system)
    # runtimes["idx0"] = time_query(timeout)
    # exec_sql(f"DROP INDEX idx_{table0}_x_y;")

    # # Index on table1
    # exec_sql(f"CREATE INDEX idx_{table1}_x_y ON {table1}(x, y);")
    # exec_sql(f"ANALYZE {table0};")
    # exec_sql(f"ANALYZE {table1};")
    # print(f"\n\n{system} query plan with index1 for: {query}")
    # print_query_plan(conn, query, db_type=system)
    # runtimes["idx1"] = time_query(timeout)
    # exec_sql(f"DROP INDEX idx_{table1}_x_y;")

    # Index on both
    exec_sql(f"CREATE INDEX idx_{table0}_x_y ON {table0}(x, y);")
    exec_sql(f"CREATE INDEX idx_{table1}_x_y ON {table1}(x, y);")
    exec_sql(f"ANALYZE {table0};")
    exec_sql(f"ANALYZE {table1};")
    print(f"\n\n{system} query plan with BOTH index for: {query}")
    print_query_plan(conn, query, db_type=system)
    runtimes["idx0_idx1"] = time_query(timeout)
    exec_sql(f"DROP INDEX idx_{table0}_x_y;")
    exec_sql(f"DROP INDEX idx_{table1}_x_y;")

    return runtimes

# -------------------------------
# Queries
# -------------------------------

query = f"""
SELECT COUNT(*)
FROM Employees r, Employees s
WHERE r.salary < s.salary AND r.tax > s.tax;
"""

# -------------------------------
# Main benchmark
# -------------------------------
def main():
    # Collect CSV pairs
    csv_file = f"{csv_dir}/data.csv"
    results = dict()

    # -------------------------------
    # DuckDB benchmarks
    # -------------------------------
    conn_duckdb = connect_duckdb(duckdb_db)
    load_duckdb_table(conn_duckdb, "Employees", csv_file)
    print("DuckDB table loaded", flush=True)
    duckdb = run_index_variant(conn_duckdb, query, "duckdb")
    print(f"DuckDB: {duckdb}", flush=True)
    results["duckdb"] = duckdb
    conn_duckdb.close()

    # These always timeout.
    """
    # -------------------------------
    # SQLite benchmarks
    # -------------------------------
    conn_sqlite = connect_sqlite(sqlite_db)
    load_sqlite_table(conn_sqlite, "Employees", csv_file)
    print("SQLite table loaded", flush=True)
    sqlite = run_index_variant(conn_sqlite, query, "sqlite")
    print(f"SQLite: {sqlite}", flush=True)
    results["sqlite"] = sqlite
    conn_sqlite.close()

    # -------------------------------
    # postgres benchmarks
    # -------------------------------
    conn_pg = connect_postgres()
    load_postgres_table(conn_pg, "Employees", csv_file)
    postgres = run_index_variant(conn_pg, query, "postgres")
    print(f"postgres: {postgres}", flush=True)
    results["postgres"] = postgres
    conn_pg.close()
    """

    # -------------------------------
    # Save all results
    # -------------------------------
    df_results = pd.json_normalize(results, sep="_")
    df_results.to_csv(output_csv, index=False)
    print(f"All done! Results saved to {output_csv}")


if __name__ == "__main__":
    main()
