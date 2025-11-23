import json
import redis

# Connect to Redis (make sure Redis server is running: brew services start redis)
try:
    _redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    _redis_client.ping()
except redis.ConnectionError:
    print("Warning: Redis not available. Run: brew install redis && brew services start redis")
    _redis_client = None

BUF_KEY = "detection_buffer"
MAX_LEN = 20


class SharedBuffer:
    """Redis-backed buffer for inter-process communication."""

    def __init__(self):
        self.client = _redis_client

    def append(self, item):
        """Append item to buffer (person_count, luggage_count, hawaii_prob)."""
        if self.client is None:
            return

        # Serialize tuple to JSON
        data = json.dumps(item)

        # Add to Redis list
        self.client.rpush(BUF_KEY, data)

        # Trim to max length
        self.client.ltrim(BUF_KEY, -MAX_LEN, -1)

    def get_all(self):
        """Get all items from buffer."""
        if self.client is None:
            return []

        items = self.client.lrange(BUF_KEY, 0, -1)
        return [json.loads(item) for item in items]

    def clear(self):
        """Clear the buffer."""
        if self.client is None:
            return
        self.client.delete(BUF_KEY)

    def __len__(self):
        """Get buffer length."""
        if self.client is None:
            return 0
        return self.client.llen(BUF_KEY)


buf = SharedBuffer()
