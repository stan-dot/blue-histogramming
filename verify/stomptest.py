import sys
import time

import stomp


class MyListener(stomp.ConnectionListener):
    def on_error(self, frame):
        print(f'received an error "{frame.body}"')

    def on_message(self, frame):
        print(f'received a message "{frame.body}"')


conn = stomp.Connection([("rmq", 61613)], auto_content_length=False)
conn.set_listener("", MyListener())
conn.connect("user", "password", wait=True)
conn.subscribe(destination="/queue/test", id=1, ack="auto")
conn.send(body=" ".join(sys.argv[1:]), destination="/queue/test")
time.sleep(2)
conn.disconnect()
