"""
Migration: Enable Row Level Security on every table in the public schema.

Supabase serves public tables over its REST API (/rest/v1) to anyone holding
the anon key, which the frontend ships to browsers. With RLS off, that key can
read and write these tables directly. Enabling RLS with no policies denies the
anon/authenticated roles, while the backend keeps working because it connects
straight to Postgres as the table owner (owners and BYPASSRLS roles skip RLS).

Safety: the script aborts without changing anything if the backend DB user
would itself be blocked by RLS on any table.

Idempotent; only touches tables that don't have RLS yet.
    python db_youtube_transcripts/migration_enable_rls.py          # report only
    python db_youtube_transcripts/migration_enable_rls.py --apply  # make changes

Undo for one table: ALTER TABLE public.<table> DISABLE ROW LEVEL SECURITY;
"""

import os
import sys

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

load_dotenv()

apply = "--apply" in sys.argv

connection = psycopg2.connect(
    database=os.getenv("DB_NAME_YOUTUBE_TRANSCRIPTS"),
    host=os.getenv("DB_HOST_YOUTUBE_TRANSCRIPTS"),
    user=os.getenv("DB_USERNAME_YOUTUBE_TRANSCRIPTS"),
    password=os.getenv("DB_PASSWORD_YOUTUBE_TRANSCRIPTS"),
    port=os.getenv("DB_PORT_YOUTUBE_TRANSCRIPTS"),
)
connection.autocommit = True

with connection.cursor() as c:
    c.execute(
        "SELECT current_user, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
    )
    user, is_superuser, bypasses_rls = c.fetchone()
    print(f"Backend DB user: {user} (superuser={is_superuser}, bypassrls={bypasses_rls})")

    c.execute(
        """
        SELECT tablename, tableowner, rowsecurity
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename
        """
    )
    tables = c.fetchall()

    to_enable = [name for name, _, has_rls in tables if not has_rls]
    blocked = [
        name
        for name, owner, _ in tables
        if owner != user and not (is_superuser or bypasses_rls)
    ]

    print(f"\n{'table':<30} {'owner':<25} rls")
    for name, owner, has_rls in tables:
        print(f"{name:<30} {owner:<25} {'on' if has_rls else 'OFF'}")

    if blocked:
        print(
            f"\n✗ Aborting: {user} doesn't own and can't bypass RLS on: "
            f"{', '.join(blocked)}. Enabling RLS would block the backend."
        )
        sys.exit(1)

    if not to_enable:
        print("\n✓ RLS already enabled on every public table")
    elif not apply:
        print(f"\nWould enable RLS on {len(to_enable)} tables. Re-run with --apply.")
    else:
        for name in to_enable:
            c.execute(
                sql.SQL("ALTER TABLE public.{} ENABLE ROW LEVEL SECURITY").format(
                    sql.Identifier(name)
                )
            )
            print(f"✓ RLS enabled on {name}")

connection.close()
