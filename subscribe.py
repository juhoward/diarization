import asyncio
import sys
import websockets
import logging
import os
from azure.messaging.webpubsubservice import WebPubSubServiceClient

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger('subscriber')

async def connect(url):
    async with websockets.connect(url) as ws:
        print('connected')
        while True:
            print('Received message: ' + await ws.recv())

if __name__ == '__main__':
    os.environ['WEBPUBSUB_CONNECTION_STRING'] = 'Endpoint=https://do-chatbot.webpubsub.azure.com;AccessKey=Qgfns+ZEoNIhpJvhOPvnon1FFjc1x+aI+Kv96W1sYBo=;Version=1.0;'
    try:
        connection_string = os.environ['WEBPUBSUB_CONNECTION_STRING']
    except KeyError:
        LOG.error("Missing environment variable 'WEBPUBSUB_CONNECTION_STRING' - please set if before running the example")
        exit()

    hub_name = "Sample_ChatApp"

    service = WebPubSubServiceClient.from_connection_string(connection_string, hub=hub_name)
    token = service.get_client_access_token()

    try:
        asyncio.get_event_loop().run_until_complete(connect(token['url']))
    except KeyboardInterrupt:
        pass
