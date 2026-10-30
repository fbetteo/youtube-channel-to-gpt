"""
Drive an isolated local instance of the transcript API for verification.

    poetry run python .claude/skills/verify/verify.py <command> ...

Commands (see SKILL.md):
    up                      start Postgres (docker) + schema + API on :8765
    doctor                  read-only health check of the instance this tool started
    down                    stop the API and Postgres this tool started, delete its Stripe test clocks
    jwt USER_ID             print a website JWT for USER_ID (UUID)
    call METHOD PATH [--user UUID | --api-key KEY | --bearer KEY] [--json BODY] [--run NAME]
    sql "SELECT ..." [--run NAME]
    stripe-subscribe --user UUID [--lookup-key KEY] [--run NAME]
    stripe-advance --days N [--run NAME]
    stripe-cancel [--run NAME]
    stripe-forward [--run NAME]  re-deliver this customer's undelivered Stripe events

Isolation: the API gets a throwaway database, a Stripe *test* key, a local JWT
secret and webhook secret, and deliberately invalid AWS credentials, so it can
never touch production data, S3, Lambda or live Stripe even though .env points
there. Evidence goes to scripts/e2e_results/verify/<run>/ and survives `down`.
"""

import argparse
import hashlib
import hmac
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from base64 import b64encode
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STATE_DIR = REPO / "scripts" / "e2e_results" / "verify"
STATE_FILE = STATE_DIR / "instance.json"
API_PORT = 8765
API = f"http://127.0.0.1:{API_PORT}"
PG_PORT = 55432
PG_CONTAINER = "ytx-verify-pg"
PG_PASSWORD = "verify"
JWT_SECRET = "ytx-verify-jwt-secret"
WEBHOOK_SECRET = "whsec_ytx_verify_local"
PROBE_USER = "00000000-0000-4000-8000-00000000d0c0"
WELCOME_CREDITS = 100  # src/transcript_api.py WELCOME_CREDITS

# The table production gets from Supabase; no repo script creates it.
USER_CREDITS_SCAFFOLD = """
CREATE TABLE IF NOT EXISTS user_credits (
    user_id UUID PRIMARY KEY,
    credits INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def dotenv():
    from dotenv import dotenv_values

    return dotenv_values(REPO / ".env")


def instance_env():
    env = dict(os.environ)
    stripe_key = dotenv().get("STRIPE_SECRET_KEY_TEST_TRANSCRIPTS") or ""
    if not stripe_key.startswith(("sk_test_", "rk_test_")):
        stripe_key = "sk_test_unavailable"
    env.update(
        PYTHONIOENCODING="utf-8",
        # src.transcript_api imports sibling modules (youtube_service) top-level.
        PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO)]),
        DB_HOST_YOUTUBE_TRANSCRIPTS="127.0.0.1",
        DB_PORT_YOUTUBE_TRANSCRIPTS=str(PG_PORT),
        DB_NAME_YOUTUBE_TRANSCRIPTS="postgres",
        DB_USERNAME_YOUTUBE_TRANSCRIPTS="postgres",
        DB_PASSWORD_YOUTUBE_TRANSCRIPTS=PG_PASSWORD,
        SUPABASE_SECRET_YOUTUBE_TRANSCRIPTS=JWT_SECRET,
        STRIPE_SECRET_KEY_LIVE=stripe_key,
        STRIPE_WEBHOOK_SECRET_TRANSCRIPTS=WEBHOOK_SECRET,
        # Invalid on purpose: S3 cache errors fall back to live fetches,
        # Lambda dispatch fails instead of running production workers.
        AWS_ACCESS_KEY_ID="ytx-verify-disabled",
        AWS_SECRET_ACCESS_KEY="ytx-verify-disabled",
        AWS_SHARED_CREDENTIALS_FILE=str(STATE_DIR / "no-aws-credentials"),
        AWS_CONFIG_FILE=str(STATE_DIR / "no-aws-config"),
        AWS_EC2_METADATA_DISABLED="true",
        S3_BUCKET_NAME="ytx-verify-disabled",
        S3_BUCKET_NAME_FALLBACK="",
        LAMBDA_FUNCTION_NAME="ytx-verify-disabled",
        LAMBDA_RESULTS_QUEUE_URL="",
        SUMMARY_OPENAI_API_KEY="",
        OPENAI_API_KEY="",
        DOCS_SECRET_KEY="ytx-verify-docs",
        PROXY_SHARED_SECRET="ytx-verify-proxy",
        API_BASE_URL=API,
        FRONTEND_URL_YOUTUBE_TRANSCRIPTS="http://localhost:3000",
    )
    return env


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else None


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def run_dir(name):
    path = STATE_DIR / (name or datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def record(run, kind, entry):
    entry = {"at": datetime.now().isoformat(timespec="seconds"), "kind": kind, **entry}
    with open(run_dir(run) / "evidence.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def pid_alive(pid):
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def port_listener_pid(port):
    """PID listening on 127.0.0.1:port, or None (Windows netstat / POSIX lsof)."""
    if os.name == "nt":
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                return int(parts[4])
        return None
    out = subprocess.run(
        ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True
    ).stdout.split()
    return int(out[0]) if out else None


def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def docker(*args, check=True):
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def make_jwt(user_id):
    from jose import jwt

    now = int(time.time())
    return jwt.encode(
        {"sub": user_id, "email": f"{user_id[:8]}@verify.local", "iat": now, "exp": now + 6 * 3600},
        JWT_SECRET,
        algorithm="HS256",
    )


def http(method, path, user=None, api_key=None, body=None, headers=None, raw=None, bearer=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(API + path, data=data, method=method.upper())
    req.add_header("Content-Type", "application/json")
    if user:
        req.add_header("Authorization", f"Bearer {make_jwt(user)}")
    if api_key:
        req.add_header("X-API-Key", api_key)
    if bearer:  # hosted MCP takes the API key as a bearer token
        req.add_header("Authorization", f"Bearer {bearer}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            status, payload = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, payload = e.code, e.read()
    text = payload.decode("utf-8", errors="replace")
    try:
        return status, json.loads(text)
    except ValueError:
        return status, text


def pg():
    import psycopg2

    conn = psycopg2.connect(
        dbname="postgres", host="127.0.0.1", port=PG_PORT, user="postgres", password=PG_PASSWORD
    )
    conn.autocommit = True
    return conn


# ---------------------------------------------------------------- lifecycle


def cmd_up(args):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if state and pid_alive(state["api_pid"]):
        sys.exit(f"An instance is already running (pid {state['api_pid']}); run `down` first.")
    if port_in_use(API_PORT) or port_in_use(PG_PORT):
        sys.exit(f"Port {API_PORT} or {PG_PORT} is in use by something this tool did not start.")
    if docker("version", "--format", "{{.Server.Version}}", check=False).returncode:
        sys.exit("Docker is not running. Start Docker Desktop, then retry.")
    if docker("ps", "-aq", "-f", f"name=^{PG_CONTAINER}$").stdout.strip():
        sys.exit(f"Container {PG_CONTAINER} exists from an earlier run; run `down` first.")

    print("Starting Postgres...")
    docker("run", "-d", "--rm", "--name", PG_CONTAINER, "--label", "ytx-verify=1",
           "-e", f"POSTGRES_PASSWORD={PG_PASSWORD}", "-p", f"{PG_PORT}:5432", "pgvector/pgvector:pg15")
    state = {"pg_container": PG_CONTAINER, "api_pid": None, "stripe_test_clocks": [],
             "started": datetime.now().isoformat(timespec="seconds")}
    save_state(state)
    for _ in range(60):  # pg_isready passes during the image's init restart; probe a real query
        try:
            with pg() as conn, conn.cursor() as c:
                c.execute("SELECT 1")
            break
        except Exception:
            time.sleep(1)
    else:
        sys.exit("Postgres did not become ready; run `down`.")

    print("Applying schema...")
    with pg() as conn, conn.cursor() as c:
        c.execute(USER_CREDITS_SCAFFOLD)
    env = instance_env()
    if args.openai:
        env["SUMMARY_OPENAI_API_KEY"] = dotenv().get("SUMMARY_OPENAI_API_KEY") or ""
    db_dir = REPO / "db_youtube_transcripts"
    modules = ["schema", "schema_api_keys"] + sorted(p.stem for p in db_dir.glob("migration_*.py"))
    for name in modules:  # as modules: some migrations use package-relative imports
        result = subprocess.run([sys.executable, "-m", f"db_youtube_transcripts.{name}"], cwd=REPO, env=env,
                                capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            sys.exit(f"{name} failed:\n{result.stdout}\n{result.stderr}")
        print(f"  {name} ok")

    log_path = STATE_DIR / f"server-{datetime.now():%Y%m%d-%H%M%S}.log"
    flags = {}
    if os.name == "nt":
        flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        flags["start_new_session"] = True
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.transcript_api:app",
             "--host", "127.0.0.1", "--port", str(API_PORT)],
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **flags)
    state.update(api_pid=proc.pid, server_log=str(log_path),
                 stripe_mode="test" if env["STRIPE_SECRET_KEY_LIVE"] != "sk_test_unavailable" else "unavailable")
    save_state(state)

    print(f"Starting API (pid {proc.pid}, log {log_path})...")
    for _ in range(90):
        if proc.poll() is not None:
            sys.exit(f"API exited early; see {log_path}")
        try:
            status, body = http("GET", "/user/credits", user=PROBE_USER)
            if status == 200 and body.get("credits") == WELCOME_CREDITS:
                print(f"Ready: {API} (probe user has {WELCOME_CREDITS} welcome credits)")
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    sys.exit(f"API did not become ready; see {log_path}, then run `down`.")


def cmd_doctor(args):
    state = load_state()
    checks = []
    checks.append(("instance state file", state is not None, str(STATE_FILE)))
    if state:
        alive = bool(state.get("api_pid")) and pid_alive(state["api_pid"])
        checks.append(("API process alive", alive, f"pid {state.get('api_pid')}"))
        listener = port_listener_pid(API_PORT)
        checks.append(("port 8765 owned by our API", listener == state.get("api_pid"),
                       f"listener pid {listener}"))
        running = docker("ps", "-q", "-f", f"name=^{PG_CONTAINER}$", check=False).stdout.strip()
        checks.append(("Postgres container running", bool(running), PG_CONTAINER))
        try:
            with pg() as conn, conn.cursor() as c:
                c.execute("SELECT to_regclass('user_subscriptions') IS NOT NULL, "
                          "to_regclass('api_keys') IS NOT NULL")
                schema_ok = all(c.fetchone())
        except Exception as e:
            schema_ok = False
        checks.append(("schema applied", schema_ok, "user_subscriptions, api_keys"))
        try:
            status, body = http("GET", "/user/credits", user=PROBE_USER)
            checks.append(("JWT auth + DB via API", status == 200, f"{status} {body}"))
        except Exception as e:
            checks.append(("JWT auth + DB via API", False, repr(e)))
        checks.append(("Stripe key is test mode", state.get("stripe_mode") == "test",
                       state.get("stripe_mode")))
    for name, ok, detail in checks:
        print(f"{'OK  ' if ok else 'FAIL'} {name}: {detail}")
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


def cmd_down(args):
    state = load_state()
    if not state:
        print("No instance recorded; nothing to stop.")
        return
    pid = state.get("api_pid")
    if pid and pid_alive(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(pid, 15)
        print(f"Stopped API pid {pid}")
    if state.get("stripe_test_clocks"):
        stripe = stripe_client()
        for clock in state["stripe_test_clocks"]:
            try:
                stripe.test_helpers.TestClock.delete(clock)
                print(f"Deleted Stripe test clock {clock}")
            except Exception as e:
                print(f"Could not delete test clock {clock}: {e}")
    if docker("ps", "-aq", "-f", f"name=^{PG_CONTAINER}$", "-f", "label=ytx-verify=1",
              check=False).stdout.strip():
        docker("rm", "-f", PG_CONTAINER)
        print(f"Removed container {PG_CONTAINER}")
    STATE_FILE.unlink()
    print(f"Evidence kept in {STATE_DIR}")


# ---------------------------------------------------------------- driving


def cmd_jwt(args):
    print(make_jwt(args.user_id))


def cmd_call(args):
    if ":" in args.path.split("?")[0]:
        sys.exit(f"Path {args.path!r} was rewritten by Git Bash; prefix the command with MSYS_NO_PATHCONV=1.")
    body = json.loads(args.json) if args.json else None
    status, payload = http(args.method, args.path, user=args.user, api_key=args.api_key, body=body,
                           bearer=args.bearer)
    record(args.run, "http", {"request": {"method": args.method, "path": args.path,
                                          "user": args.user, "api_key": bool(args.api_key or args.bearer), "body": body},
                              "status": status, "response": payload})
    print(status)
    print(json.dumps(payload, indent=2) if not isinstance(payload, str) else payload)


def cmd_sql(args):
    query = args.query.strip()
    if not query.lower().startswith(("select", "with")):
        sys.exit("sql is read-only: SELECT/WITH only. Change state through the API.")
    with pg() as conn, conn.cursor() as c:
        c.execute(query)
        cols = [d[0] for d in c.description]
        rows = [dict(zip(cols, r)) for r in c.fetchall()]
    record(args.run, "sql", {"query": query, "rows": rows})
    print(json.dumps(rows, indent=2, default=str))


CREATOR_COLUMNS = "channel_id, slug, name, handle, category, avatar_url, tagline"
CHUNK_COLUMNS = ("video_id, chunk_index, channel_id, video_title, published_at, "
                 "start_seconds, end_seconds, text, embedding, embedding_model")


def cmd_seed_creator(args):
    """Copy one creator and its embedded chunks from production (read-only) and enable it."""
    import psycopg2

    env = dotenv()
    source = psycopg2.connect(
        dbname=env["DB_NAME_YOUTUBE_TRANSCRIPTS"], host=env["DB_HOST_YOUTUBE_TRANSCRIPTS"],
        user=env["DB_USERNAME_YOUTUBE_TRANSCRIPTS"], password=env["DB_PASSWORD_YOUTUBE_TRANSCRIPTS"],
        port=env["DB_PORT_YOUTUBE_TRANSCRIPTS"],
    )
    source.set_session(readonly=True)
    with source, source.cursor() as c:
        c.execute(f"SELECT {CREATOR_COLUMNS} FROM creators WHERE slug = %s", (args.slug,))
        creator = c.fetchone()
        if not creator:
            sys.exit(f"No creator '{args.slug}' in production. Add and index it with scripts/creators.py.")
        c.execute(f"SELECT {CHUNK_COLUMNS.replace('embedding,', 'embedding::text,')} "
                  "FROM transcript_chunks WHERE channel_id = %s", (creator[0],))
        chunks = c.fetchall()
    source.close()
    with pg() as conn, conn.cursor() as c:
        c.execute(f"INSERT INTO creators ({CREATOR_COLUMNS}, enabled) VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE) "
                  "ON CONFLICT (channel_id) DO UPDATE SET enabled = TRUE", creator)
        c.executemany(f"INSERT INTO transcript_chunks ({CHUNK_COLUMNS}) VALUES "
                      "(%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s) ON CONFLICT DO NOTHING", chunks)
    print(f"Seeded {creator[2]} ('{args.slug}', enabled) with {len(chunks)} chunks")


def stripe_client():
    import stripe

    key = instance_env()["STRIPE_SECRET_KEY_LIVE"]
    if key == "sk_test_unavailable":
        sys.exit("STRIPE_SECRET_KEY_TEST_TRANSCRIPTS is missing or not a test key in .env.")
    stripe.api_key = key
    return stripe


def forward_events(run, customer, since):
    """Deliver this customer's Stripe test events to /payments/webhook, signed, oldest first."""
    key = instance_env()["STRIPE_SECRET_KEY_LIVE"]
    req = urllib.request.Request(f"https://api.stripe.com/v1/events?limit=100&created[gte]={since}")
    req.add_header("Authorization", "Basic " + b64encode(f"{key}:".encode()).decode())
    with urllib.request.urlopen(req, timeout=30) as resp:
        events = json.loads(resp.read())["data"]
    state = load_state()
    delivered = set(state.get("delivered_events", []))
    for event in reversed(events):
        obj = event["data"]["object"]
        if event["id"] in delivered or obj.get("customer") != customer:
            continue
        payload = json.dumps(event).encode()
        ts = int(time.time())
        sig = hmac.new(WEBHOOK_SECRET.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
        status, body = http("POST", "/payments/webhook", raw=payload,
                            headers={"Stripe-Signature": f"t={ts},v1={sig}"})
        delivered.add(event["id"])
        detail = obj.get("billing_reason") or obj.get("status") or ""
        print(f"  {event['type']:<32} {detail:<24} -> {status} {body}")
        record(run, "webhook", {"event": event["id"], "type": event["type"], "detail": detail,
                                "status": status, "response": body})
    state["delivered_events"] = sorted(delivered)
    save_state(state)


def cmd_stripe_subscribe(args):
    """What a completed subscription Checkout creates, on a Stripe test clock."""
    stripe = stripe_client()
    state = load_state()
    since = int(time.time()) - 5
    clock = stripe.test_helpers.TestClock.create(frozen_time=int(time.time()), name="ytx verify")
    state["stripe_test_clocks"].append(clock.id)
    save_state(state)
    customer = stripe.Customer.create(
        email=f"{args.user[:8]}@verify.local", test_clock=clock.id, payment_method="pm_card_visa",
        invoice_settings={"default_payment_method": "pm_card_visa"})
    price = stripe.Price.list(lookup_keys=[args.lookup_key], active=True).data[0]
    sub = stripe.Subscription.create(
        customer=customer.id, items=[{"price": price.id}],
        metadata={"project": "transcript-api", "user_id": args.user})
    state = load_state()
    state["stripe"] = {"clock": clock.id, "customer": customer.id, "subscription": sub.id,
                       "user": args.user, "since": since}
    save_state(state)
    record(args.run, "stripe", {"action": "subscribe", "clock": clock.id, "customer": customer.id,
                                "subscription": sub.id, "price": price.id})
    print(f"Subscribed {args.user}: {sub.id} ({sub.status}); delivering events...")
    time.sleep(4)
    forward_events(args.run, customer.id, since)


def cmd_stripe_advance(args):
    stripe = stripe_client()
    s = load_state()["stripe"]
    clock = stripe.test_helpers.TestClock.retrieve(s["clock"])
    target = clock.frozen_time + args.days * 86400
    stripe.test_helpers.TestClock.advance(s["clock"], frozen_time=target)
    while stripe.test_helpers.TestClock.retrieve(s["clock"]).status != "ready":
        time.sleep(2)
    record(args.run, "stripe", {"action": "advance", "days": args.days, "clock": s["clock"]})
    print(f"Advanced test clock {args.days} days; waiting for renewal invoice...")
    for _ in range(20):
        time.sleep(3)
        invoices = stripe.Invoice.list(customer=s["customer"]).data
        if sum(i.status == "paid" for i in invoices) >= 2:
            break
    time.sleep(3)
    forward_events(args.run, s["customer"], s["since"])


def cmd_stripe_forward(args):
    s = load_state()["stripe"]
    forward_events(args.run, s["customer"], s["since"])


def cmd_stripe_cancel(args):
    stripe = stripe_client()
    s = load_state()["stripe"]
    stripe.Subscription.cancel(s["subscription"])
    record(args.run, "stripe", {"action": "cancel", "subscription": s["subscription"]})
    print(f"Canceled {s['subscription']}; delivering events...")
    time.sleep(4)
    forward_events(args.run, s["customer"], s["since"])


def main():
    p = argparse.ArgumentParser(description="Drive an isolated local transcript API.")
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up")
    u.add_argument("--openai", action="store_true",
                   help="use the real SUMMARY_OPENAI_API_KEY from .env (spends money)")
    u.set_defaults(fn=cmd_up)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("down").set_defaults(fn=cmd_down)
    j = sub.add_parser("jwt")
    j.add_argument("user_id")
    j.set_defaults(fn=cmd_jwt)
    c = sub.add_parser("call")
    c.add_argument("method")
    c.add_argument("path")
    c.add_argument("--user")
    c.add_argument("--api-key")
    c.add_argument("--bearer")
    c.add_argument("--json")
    c.add_argument("--run")
    c.set_defaults(fn=cmd_call)
    q = sub.add_parser("sql")
    q.add_argument("query")
    q.add_argument("--run")
    q.set_defaults(fn=cmd_sql)
    sc = sub.add_parser("seed-creator")
    sc.add_argument("slug")
    sc.set_defaults(fn=cmd_seed_creator)
    s = sub.add_parser("stripe-subscribe")
    s.add_argument("--user", required=True)
    s.add_argument("--lookup-key", default="transcript_monthly_1000")
    s.add_argument("--run")
    s.set_defaults(fn=cmd_stripe_subscribe)
    a = sub.add_parser("stripe-advance")
    a.add_argument("--days", type=int, default=32)
    a.add_argument("--run")
    a.set_defaults(fn=cmd_stripe_advance)
    f = sub.add_parser("stripe-forward")
    f.add_argument("--run")
    f.set_defaults(fn=cmd_stripe_forward)
    x = sub.add_parser("stripe-cancel")
    x.add_argument("--run")
    x.set_defaults(fn=cmd_stripe_cancel)
    args = p.parse_args()
    os.chdir(REPO)
    args.fn(args)


if __name__ == "__main__":
    main()
