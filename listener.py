import redis

r = redis.Redis()
pubsub = r.pubsub()
pubsub.subscribe("paddle", "ball")

for message in pubsub.listen():
    if message['type'] == 'message':
        print(f'{message["channel"]}: {message["data"]}')
