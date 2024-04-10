import threading
import logging
import time
import json
import os
from websocket import WebSocketApp
from typing import List, Optional
from azure.messaging.webpubsubservice import WebPubSubServiceClient

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger('pubsub')

class WebsocketClientsManager:
    '''
    This class contains multiple websocket clients which are connected to Azure Web PubSub Services.
    '''
    def __init__(self) -> None:
        self.clients = []
        self.connection_ids = []
        self.recv_messages = []
        

    def add_client(self, service: WebPubSubServiceClient, user_id: str, groups: Optional[List[str]] = None):
        # define how to process messages when they are received.
        def on_message(websocket_app: WebSocketApp, message: str): 
            message = json.loads(message)

            if message["type"] == "message":
                self.recv_messages.append(message["data"])

            if message["type"] == "system" and message["event"] == "connected":
                self.connection_ids.append(message["connectionId"])

            # LOG.debug(message)
        # message to show that connection exists
        def on_open(websocket_app: WebSocketApp): 
            LOG.debug("connected")
        # generate an access token for a client
        token = service.get_client_access_token(groups=groups, user_id=user_id)
        LOG.debug(f"token\n{token}")
        # create a websocket with the token url and protocols
        client = WebSocketApp(token["url"], subprotocols=['json.webpubsub.azure.v1'], on_open=on_open, on_message=on_message)
        self.clients.append(client)

    def start_all(self):
        for client in self.clients:
            wst = threading.Thread(target=client.run_forever, daemon=True)
            wst.start()        

        LOG.debug("Waiting for all clients connected...")
        while len(self.connection_ids) != self.client_number:
            pass

    @property
    def client_number(self):
        return len(self.clients)


def test_overall_integration(webpubsub_connection_string: str, hub:str):
    # build a service client from the connection string.
    service = WebPubSubServiceClient.from_connection_string(webpubsub_connection_string, hub=hub, logging_enable=False)

    # build multiple websocket clients connected to the Web PubSub service
    clients = WebsocketClientsManager()
    # add 5 clients to "InitGroup" for testing
    for i in range(5):  
        clients.add_client(service, user_id="User%d" % clients.client_number, groups=["InitGroup"])
    clients.start_all()
    for idx in range(len(clients.clients)):
        user_id = f'User{idx}'
        LOG.debug(f"adding {user_id} to 'InitGroup'")
        service.add_user_to_group("InitGroup", user_id)
    try:
        # test naive send_to_all
        service.send_to_all(message='Message_For_All', content_type='text/plain') # N messages

        # test if generating token with the initial group is working
        service.send_to_group(group="InitGroup", message='Message_For_InitGroup', content_type='text/plain') # N messages

        # test if parameter "filter" in send is working
        # service.send_to_all("Message_Not_For_User0", filter="userId ne 'User0'", content_type='text/plain') # N - 1 messages

        # test if remove_connection_from_all_groups works
        group_names = ["Group%d" % i for i in range(3)]
        for group in group_names:
            service.add_connection_to_group(group, clients.connection_ids[0])
            service.send_to_group(group, "Message_For_RemoveFromAll", content_type='text/plain')  
        for group in group_names:
            for connection in clients.connection_ids:
                service.remove_connection_from_group(group,connection)

        for group in group_names:
            service.send_to_group(group, "Message_For_RemoveFromAll", content_type='text/plain')

        # other tests
        service.send_to_user("User0", message='Message_For_User0', content_type='text/plain') # 1 messages
    except BaseException as e:
        print(e)
    time.sleep(5)
    for msg in clients.recv_messages:
        LOG.info(f"Message: {msg}")
    # LOG.info(f"Received Message: {clients.recv_messages}")
    assert service.group_exists("InitGroup") == True
    assert clients.recv_messages.count("Message_For_All") == clients.client_number
    assert clients.recv_messages.count("Message_For_InitGroup") == clients.client_number
    # assert clients.recv_messages.count("Message_Not_For_User0") == clients.client_number - 1
    assert clients.recv_messages.count("Message_For_User0") == 1
    assert clients.recv_messages.count("Message_For_RemoveFromAll") == 3
    LOG.info("Complete All Integration Test Successfully")

if __name__ == "__main__":
    os.environ['WEBPUBSUB_CONNECTION_STRING'] = 'Endpoint=https://do-chatbot.webpubsub.azure.com;AccessKey=Qgfns+ZEoNIhpJvhOPvnon1FFjc1x+aI+Kv96W1sYBo=;Version=1.0;'
    try:
        connection_string = os.environ['WEBPUBSUB_CONNECTION_STRING']
    except KeyError:
        LOG.error("Missing environment variable 'WEBPUBSUB_CONNECTION_STRING' - please set if before running the example")
        exit()
    # service = WebPubSubServiceClient.from_connection_string(connection_string=connection_string, 
    #                                                         hub='Sample_ChatApp')
    # service.send_to_all(message = {
    #         'from': 'user1',
    #         'data': 'Hello world'
    #     })
    test_overall_integration(connection_string, 'Sample_ChatApp')