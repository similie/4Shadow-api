#!/usr/bin/env bash
# Updates a 4Shadow forecast server to the newest code and checks that it answers.
# Run it on the server, as a user that may restart the services (root, or with sudo):
#
#   deploy/deploy.sh status     what's running, and what's new on the branch
#   deploy/deploy.sh update     fast-forward, install requirements, check the code imports, restart the
#                               API, restart ingest only if its code changed, then smoke-test
#   deploy/deploy.sh check      smoke tests only
#   deploy/deploy.sh rollback   back to the commit before the last update
#
# Settings, from the environment (defaults suit deploy/*.service):
#   APP_DIR      the checkout                    (default: the one this script is in)
#   BRANCH       branch to follow                (main)
#   API          the API, from this machine      (http://127.0.0.1:5003)
#   API_UNIT     systemd unit of the API         (forecast-api)
#   INGEST_UNIT  systemd unit of the ingest      (forecast-ingest)
#   INGEST       auto | restart | skip           (auto: restart only if ingest code changed; a restart
#                                                 begins any run being downloaded again)
#   PYTHON       the service's Python            ($APP_DIR/.venv/bin/python)
#   SYSTEMCTL    how to reach systemd            (systemctl as root, else sudo systemctl)
#
# Never needs the network beyond `git fetch`, PyPI, and the API on this machine. Nothing secret is read.
set -euo pipefail

APP_DIR=${APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BRANCH=${BRANCH:-main}
API=${API:-http://127.0.0.1:5003}
API_UNIT=${API_UNIT:-forecast-api}
INGEST_UNIT=${INGEST_UNIT:-forecast-ingest}
INGEST=${INGEST:-auto}
PYTHON=${PYTHON:-$APP_DIR/.venv/bin/python}
STATE=$APP_DIR/.deploy
# Code the ingest loop runs. A change here needs an ingest restart to take effect.
INGEST_PATHS=(forecast/ingest.py forecast/sources forecast/catalog.py forecast/store.py forecast/grid.py
  forecast/config.py forecast/requirements.txt)

cd "$APP_DIR"
if [ -n "${SYSTEMCTL:-}" ]; then
  read -r -a SYSTEMCTL <<<"$SYSTEMCTL"
elif [ "$(id -u)" -eq 0 ]; then
  SYSTEMCTL=(systemctl)
else
  SYSTEMCTL=(sudo systemctl)
fi

say() { printf '%s\n' "$*"; }
die() {
  printf 'deploy: %s\n' "$*" >&2
  exit 1
}

# ------------------------------------------------------------------------------------------- checks

# Waits until the API answers at all (200, or 503 while no run has been ingested yet).
wait_for_api() {
  local code
  for _ in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$API/v2/health" || true)
    [ "$code" = 200 ] || [ "$code" = 503 ] && return 0
    sleep 0.5
  done
  die "the API didn't answer at $API within 30 s ($API_UNIT: $("${SYSTEMCTL[@]}" is-active "$API_UNIT" || true))"
}

# One request: prints "  name  code  bytes  seconds" and returns 0 if the code is one of the expected.
probe() {
  local name=$1 expect=$2 url=$3
  shift 3
  local out code
  out=$(curl -s -o /dev/null -w '%{http_code} %{size_download} %{time_total}' "$@" "$url" || echo "000 0 0")
  code=${out%% *}
  printf '  %-22s %s\n' "$name" "$out"
  [[ " $expect " == *" $code "* ]]
}

smoke() {
  say "Smoke tests against $API"
  local failed=0
  say "  health: $(curl -s "$API/v2/health" | head -c 240)"
  probe "health" "200" "$API/v2/health" || failed=1
  probe "forecast (Dili)" "200" "$API/v2/forecast?lat=-8.556&lon=125.56&hours=6&daily=false" || failed=1
  probe "forecast by name" "200 503" "$API/v2/forecast?q=Dili&hours=1&daily=false" || failed=1
  probe "places" "200 503" "$API/v2/places?q=Dili&limit=1" || failed=1
  probe "tile" "200" "$API/v2/gfs/tiles/latest/t2m/f006/2/3/2.png" || failed=1
  local field
  field=$(mktemp)
  if curl -sf -o "$field" "$API/v2/gfs/fields/latest/f006?vars=u10,v10&stride=4" &&
    "$PYTHON" -c 'import sys; from forecast.fields import decode_field
h, a = decode_field(open(sys.argv[1], "rb").read())
print("  field                 run", h["run"], {k: round(float(abs(v).max()), 1) for k, v in a.items()})' "$field"; then
    :
  else
    say "  field                 FAILED"
    failed=1
  fi
  rm -f "$field"
  probe "mcp initialize" "200" "$API/v2/mcp" -X POST -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"deploy","version":"0"}}}' || failed=1
  probe "legacy v1" "200 404" "$API/legacy/v1/forecast?lat=-8.55&lon=125.57" || failed=1
  if [ "$failed" = 0 ]; then say "All good."; else say "Some checks failed (above)."; fi
  return "$failed"
}

# ------------------------------------------------------------------------------------------- steps

# True when tracked files match the commit (untracked files, such as rotated logs, don't count).
clean_checkout() { git diff --quiet && git diff --cached --quiet; }

status() {
  git fetch -q origin "$BRANCH"
  say "Checkout   $APP_DIR on $(git branch --show-current || echo '(detached)') at $(git log --oneline -1)"
  say "Services   $API_UNIT: $("${SYSTEMCTL[@]}" is-active "$API_UNIT" || true), $INGEST_UNIT: $("${SYSTEMCTL[@]}" is-active "$INGEST_UNIT" || true)"
  local new
  new=$(git log --oneline "HEAD..origin/$BRANCH")
  if [ -n "$new" ]; then
    say "New on origin/$BRANCH:"
    printf '%s\n' "$new" | sed 's/^/  /'
  else
    say "Up to date with origin/$BRANCH."
  fi
  say "API        $(curl -s "$API/v2/health" | head -c 240)"
}

restart_ingest_if_needed() {
  local from=$1 to=$2 why=""
  case "$INGEST" in
  skip) return 0 ;;
  restart) why="INGEST=restart" ;;
  auto) [ -n "$(git diff --name-only "$from" "$to" -- "${INGEST_PATHS[@]}")" ] && why="its code changed" ;;
  *) die "INGEST must be auto, restart or skip" ;;
  esac
  if [ -n "$why" ]; then
    say "Restarting $INGEST_UNIT ($why). A run being downloaded starts again."
    "${SYSTEMCTL[@]}" restart "$INGEST_UNIT"
    sleep 2
    "${SYSTEMCTL[@]}" is-active "$INGEST_UNIT"
  else
    say "Ingest code unchanged: $INGEST_UNIT keeps running."
  fi
}

install() {
  "$PYTHON" -m pip install -q --disable-pip-version-check -r requirements.txt || return 1
  "$PYTHON" -c "import forecast.api, forecast.ingest; print('Code imports: forecast', forecast.__version__)"
}

update() {
  clean_checkout || die "the checkout has local changes; commit or stash them first"
  git fetch -q origin "$BRANCH"
  local old new
  old=$(git rev-parse HEAD)
  if [ "$(git branch --show-current)" != "$BRANCH" ]; then
    # Only a move forward: the running code must be part of the branch we switch to.
    git merge-base --is-ancestor HEAD "origin/$BRANCH" ||
      die "origin/$BRANCH doesn't contain the running commit $(git log --oneline -1); switch by hand"
    if git show-ref -q --verify "refs/heads/$BRANCH"; then
      git merge-base --is-ancestor "$BRANCH" "origin/$BRANCH" ||
        die "the local $BRANCH branch has commits that origin/$BRANCH lacks; switch by hand"
    fi
    say "Switching the checkout from $(git branch --show-current || echo '(detached)') to $BRANCH."
    git checkout -q -B "$BRANCH" "origin/$BRANCH"
    git branch -q --set-upstream-to "origin/$BRANCH"
  fi
  git merge --ff-only -q "origin/$BRANCH" || die "the checkout and origin/$BRANCH have diverged; nothing was changed"
  new=$(git rev-parse HEAD)
  if [ "$old" = "$new" ]; then
    say "Already at $(git log --oneline -1); nothing to deploy."
    return 0
  fi
  say "Code: $(git log --oneline -1 "$old") -> $(git log --oneline -1)"
  git log --oneline "$old..$new" | sed 's/^/  /'

  if ! install; then
    say "The new code doesn't import; going back to $(git log --oneline -1 "$old"). Nothing was restarted."
    git reset -q --hard "$old"
    install || true
    exit 1
  fi

  mkdir -p "$STATE"
  echo "$old" >"$STATE/previous"
  echo "$(date -u +%FT%TZ) update $old -> $new" >>"$STATE/history"

  say "Restarting $API_UNIT."
  "${SYSTEMCTL[@]}" restart "$API_UNIT"
  wait_for_api
  restart_ingest_if_needed "$old" "$new"
  smoke || die "deployed, but checks failed. To go back: deploy/deploy.sh rollback"
}

rollback() {
  [ -s "$STATE/previous" ] || die "no previous deploy recorded in $STATE/previous"
  clean_checkout || die "the checkout has local changes; commit or stash them first"
  local from to
  from=$(git rev-parse HEAD)
  to=$(cat "$STATE/previous")
  say "Rolling back: $(git log --oneline -1 "$from") -> $(git log --oneline -1 "$to")"
  git reset -q --hard "$to"
  install
  rm -f "$STATE/previous" # one step back only; `update` moves forward again
  echo "$(date -u +%FT%TZ) rollback $from -> $to" >>"$STATE/history"
  "${SYSTEMCTL[@]}" restart "$API_UNIT"
  wait_for_api
  restart_ingest_if_needed "$from" "$to"
  smoke
}

case "${1:-}" in
status) status ;;
update) update ;;
check) smoke ;;
rollback) rollback ;;
*)
  sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
