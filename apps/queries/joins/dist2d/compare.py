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
num_runs = 9
drop = 2
timeout = 60 # seconds

sqlite_db = "sqlite_test.db"
duckdb_db = "duckdb_test.db"
csv_dir = "../pldi-data"
output_csv = "join_runtime_comparison_indexed.csv"

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

def connect_sqlite(db_file):
    conn = sqlite3.connect(db_file, check_same_thread=True)
    try:
        conn.execute("PRAGMA threads = 1;")
    except sqlite3.OperationalError:
        pass
    return conn

def connect_duckdb(db_file):
    conn = duckdb.connect(db_file)
    conn.execute("SET threads TO 1;")
    return conn

def connect_postgres(dbname="ajroot", user="ajroot", host="/tmp", port=5432):
    conn = psycopg2.connect(dbname=dbname, user=user, host=host, port=port)
    conn.autocommit = True
    return conn

def print_query_plan(conn, query, db_type="sqlite"):
    if db_type == "sqlite":
        plan = conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
        s = "+".join([p[3] for p in plan])
        print(f"sqlite3 Join Type: {s}")
    elif db_type == "duckdb":
        plan = conn.execute(f"EXPLAIN {query}").fetchall()
        plan_str = plan[0][1]  # second element has the plan text
        join_line = None
        for line in plan_str.splitlines():
            if "JOIN" in line.upper():
                join_line = line.strip()
                break
        print(f"DuckDB Join Type: {join_line or plan}")
    elif db_type == "postgres":
        cur = conn.cursor()
        cur.execute(f"EXPLAIN {query}")
        plan = cur.fetchall()
        join_line = None
        for row in plan:
            if "join" in row[0].lower():
                join_line = row[0]
                break
        print(f"Postgres Join Type: {join_line or plan}")
    else:
        raise ValueError(f"DB type not known: {db_type}")

def load_sqlite_table(conn, table_name, csv_file):
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"""
    CREATE TABLE {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        x REAL,
        y REAL
    )
    """)
    with open(csv_file) as f:
        next(f)  # skip header
        for line in f:
            x, y = map(float, line.strip().split(","))
            conn.execute(f"INSERT INTO {table_name} (x, y) VALUES (?, ?)", (x, y))
    conn.commit()

def load_duckdb_table(conn, table_name, csv_file):
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_file}')")

def load_postgres_table(conn, table_name, csv_file):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                x DOUBLE PRECISION,
                y DOUBLE PRECISION
            )
        """)
        with open(csv_file) as f:
            next(f)  # skip header
            for line in f:
                x, y = map(float, line.strip().split(","))
                cur.execute(f"INSERT INTO {table_name} (x, y) VALUES (%s, %s)", (x, y))
    conn.commit()

def run_index_variant(conn, query, system):
    """Run query with all index variants and return average trimmed runtimes."""
    table0, table1 = "input0", "input1"
    runtimes = {}

    def time_query(timeout_sec=60):
        """Run query num_runs times with a timeout, return avg or timeout."""
        class TimeoutException(Exception):
            pass

        def handler(signum, frame):
            raise TimeoutException()

        signal.signal(signal.SIGALRM, handler)
        times, timeouts = [], 0

        for i in range(num_runs):
            signal.alarm(timeout_sec)
            start = time.time()
            try:
                if system == "postgres":
                    with conn.cursor() as cur:
                        cur.execute(f"SET statement_timeout TO {timeout_sec * 1000};")
                        cur.execute(query)
                        cur.fetchall()
                else:
                    conn.execute(query).fetchall()
                times.append(time.time() - start)
            except TimeoutException:
                print(f"Run {i+1}: timed out after {timeout_sec}s")
                timeouts += 1
            except Exception as e:
                print(f"Run {i+1}: query failed ({e})")
                timeouts += 1
            finally:
                signal.alarm(0)
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
            conn.execute(sql)

    # Drop any existing indexes
    for tbl in [table0, table1]:
        try:
            exec_sql(f"DROP INDEX IF EXISTS idx_{tbl}_x_y;")
        except Exception as e:
            print(f"(warn) couldn't drop index {tbl}: {e}")

    # No index
    print(f"\n\n{system} query plan with NO index for: {query}")
    print_query_plan(conn, query, db_type=system)
    runtimes["no_index"] = time_query(timeout)

    # Index on table0
    exec_sql(f"CREATE INDEX idx_{table0}_x_y ON {table0}(x, y);")
    exec_sql(f"ANALYZE {table0};")
    exec_sql(f"ANALYZE {table1};")
    print(f"\n\n{system} query plan with index0 for: {query}")
    print_query_plan(conn, query, db_type=system)
    runtimes["idx0"] = time_query(timeout)
    exec_sql(f"DROP INDEX idx_{table0}_x_y;")

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

# Chebyshev variant 1: max(abs(...), abs(...)) <= k
query_cheb_max_sqlite = f"""
SELECT *
FROM input0 AS a
JOIN input1 AS b
ON MAX(ABS(a.x - b.x), ABS(a.y - b.y)) <= {k};
"""

query_cheb_max_duck = f"""
SELECT *
FROM input0 AS a
JOIN input1 AS b
ON GREATEST(ABS(a.x - b.x), ABS(a.y - b.y)) <= {k};
"""

# Chebyshev variant 2: range conditions (x in [b.x-k, b.x+k] and y in [b.y-k, b.y+k])
query_cheb_range = f"""
SELECT *
FROM input0 AS a
JOIN input1 AS b
ON a.x BETWEEN b.x - {k} AND b.x + {k}
AND a.y BETWEEN b.y - {k} AND b.y + {k};
"""

# Donut join remains the same
query_donut = f"""
SELECT *
FROM input0 AS a
JOIN input1 AS b
ON SQRT((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y))
    BETWEEN {low} AND {high};
"""

# -------------------------------
# Main benchmark
# -------------------------------
def main():
    # Collect CSV pairs
    csv_files = glob.glob(f"{csv_dir}/*.csv")
    pattern = re.compile(r"^(.*)_input([01])_(\d+)\.csv$")
    size_to_files = defaultdict(dict)

    for f in csv_files:
        fname = f.split("/")[-1]
        m = pattern.match(fname)
        if m:
            prefix, input_num, size = m.groups()
            size_to_files[size][f"input{input_num}"] = f
        else:
            print(f"Warning: filename does not match pattern: {fname}")

    results = []

    # -------------------------------
    # DuckDB benchmarks
    # -------------------------------
    conn_duckdb = connect_duckdb(duckdb_db)

    for size, files_dict in sorted(size_to_files.items(), key=lambda x: int(x[0])):
        if "input0" not in files_dict or "input1" not in files_dict:
            print(f"Skipping size {size}, missing input0 or input1")
            continue

        print(f"DuckDB benchmarking size {size}...")

        # Load CSVs
        load_duckdb_table(conn_duckdb, "input0", files_dict["input0"])
        load_duckdb_table(conn_duckdb, "input1", files_dict["input1"])

        # Run benchmarks
        duckdb_cheb_max = run_index_variant(conn_duckdb, query_cheb_max_duck, "duckdb")
        duckdb_cheb_range = run_index_variant(conn_duckdb, query_cheb_range, "duckdb")
        duckdb_donut = run_index_variant(conn_duckdb, query_donut, "duckdb")

        # Store intermediate results
        result_entry = {
            "size": size,
            "duckdb_cheb_max": duckdb_cheb_max,
            "duckdb_cheb_range": duckdb_cheb_range,
            "duckdb_donut": duckdb_donut
        }
        results.append(result_entry)

        print(f"DuckDB intermediate results for size {size}:")
        for key, val in result_entry.items():
            if key == "size":
                continue
            print(f"  {key}: {val}")

    conn_duckdb.close()


    # -------------------------------
    # SQLite benchmarks
    # -------------------------------
    conn_sqlite = connect_sqlite(sqlite_db)

    for size, files_dict in sorted(size_to_files.items(), key=lambda x: int(x[0])):
        if "input0" not in files_dict or "input1" not in files_dict:
            print(f"Skipping size {size}, missing input0 or input1")
            continue

        print(f"SQLite benchmarking size {size}...")

        # Load CSVs
        load_sqlite_table(conn_sqlite, "input0", files_dict["input0"])
        load_sqlite_table(conn_sqlite, "input1", files_dict["input1"])

        # Only run queries supported in SQLite (skip GREATEST/MAX variant)
        sqlite_cheb_max = run_index_variant(conn_sqlite, query_cheb_max_sqlite, "sqlite")
        sqlite_cheb_range = run_index_variant(conn_sqlite, query_cheb_range, "sqlite")
        sqlite_donut = run_index_variant(conn_sqlite, query_donut, "sqlite")

        # Merge results into existing entry
        for r in results:
            if r["size"] == size:
                r.update({
                    "sqlite_cheb_max": sqlite_cheb_max,
                    "sqlite_cheb_range": sqlite_cheb_range,
                    "sqlite_donut": sqlite_donut
                })
                break

        print(f"SQLite intermediate results for size {size}:")
        print(f"  cheb_max: {sqlite_cheb_max}")
        print(f"  cheb_range: {sqlite_cheb_range}")
        print(f"  donut: {sqlite_donut}", flush=True)

    conn_sqlite.close()

    conn_pg = connect_postgres()

    for size, files_dict in sorted(size_to_files.items(), key=lambda x: int(x[0])):
        if "input0" not in files_dict or "input1" not in files_dict:
            print(f"Skipping size {size}, missing input0 or input1")
            continue

        print(f"PostgreSQL benchmarking size {size}...")

        # Load CSVs
        load_postgres_table(conn_pg, "input0", files_dict["input0"])
        load_postgres_table(conn_pg, "input1", files_dict["input1"])

        # Run benchmarks (only queries PostgreSQL supports directly)
        pg_cheb_max = run_index_variant(conn_pg, query_cheb_max_duck, "postgres")
        pg_cheb_range = run_index_variant(conn_pg, query_cheb_range, "postgres")
        pg_donut = run_index_variant(conn_pg, query_donut, "postgres")

        # Merge results
        for r in results:
            if r["size"] == size:
                r.update({
                    "postgres_cheb_max": pg_cheb_max,
                    "postgres_cheb_range": pg_cheb_range,
                    "postgres_donut": pg_donut
                })
                break

        print(f"Postgres intermediate results for size {size}:")
        print(f"  cheb_max: {pg_cheb_max}")
        print(f"  cheb_range: {pg_cheb_range}")
        print(f"  donut: {pg_donut}", flush=True)
    
    conn_pg.close()



    # -------------------------------
    # Save all results
    # -------------------------------
    df_results = pd.json_normalize(results, sep="_")
    df_results.to_csv(output_csv, index=False)
    print(f"All done! Results saved to {output_csv}")


if __name__ == "__main__":
    main()
