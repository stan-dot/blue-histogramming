import redis


def check_redis_connection(host="dragonfly", port=6379, db=0):
    try:
        r = redis.Redis(host=host, port=port, db=db)
        r.ping()
        print(f"✅ Connected to Redis at {host}:{port}")
        return True
    except redis.ConnectionError as e:
        print(f"❌ Could not connect to Redis at {host}:{port} - {e}")
        return False


if __name__ == "__main__":
    check_redis_connection()
