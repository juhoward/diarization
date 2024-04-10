import asyncio
import json
import websockets

class ChatHub:
    def __init__(self):
        self.clients = set()

    async def handle_connection(self, websocket):
        try:
            username = await websocket.recv()
            self.clients.add(websocket)
            print(f"{username} has joined the chat.")
            await self.broadcast(f"{username} has joined the chat.")

            async for message in websocket:
                message_dict = json.loads(message)
                message_type = message_dict["type"]
                message_content = message_dict.get("content")

                if message_type == "chat":
                    await self.broadcast(f"{username}: {message_content}")
                elif message_type == "private":
                    recipient_username = message_dict.get("recipient")
                    recipient = next((c for c in self.clients if c != websocket and username in c.recv_queue), None)
                    if recipient:
                        await recipient.send(json.dumps({"type": "private", "sender": username, "content": message_content}))
                    else:
                        await websocket.send(json.dumps({"type": "error", "message": "Recipient not found"}))
                else:
                    await websocket.send(json.dumps({"type": "error", "message": "Invalid message type"}))

        except websockets.ConnectionClosed:
            print(f"{username} has left the chat.")
            self.clients.remove(websocket)
            await self.broadcast(f"{username} has left the chat.")

    async def broadcast(self, message):
        for client in self.clients:
            await client.send(json.dumps({"type": "message", "content": message}))

async def main():
    chat_hub = ChatHub()
    async with websockets.serve(chat_hub.handle_connection, port=8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
