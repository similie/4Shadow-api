from typing import Dict, cast

import redis


def ensure_expired_events_enabled(r: redis.Redis) -> bool:
    required = {"E", "x"}

    cfg = cast(Dict[str, str], r.config_get("notify-keyspace-events"))
    current = cfg.get("notify-keyspace-events", "")

    if required.issubset(current):
        print("[redis] notify-keyspace-events already enabled:", current)
        return True

    merged = "".join(sorted(set(current) | required))

    try:
        r.config_set("notify-keyspace-events", merged)
        cfg_after = cast(Dict[str, str], r.config_get("notify-keyspace-events"))
        after = cfg_after.get("notify-keyspace-events", "")
        print("[redis] Updated notify-keyspace-events:", after)
        return required.issubset(after)
    except Exception as e:
        print("[redis] CONFIG SET failed:", e)
        return False
