import asyncio
import websockets

async def handler(websocket):
    async for message in websocket:
        print(f"Received message: {message}")
        await websocket.send("You said: " + message)

async def hello(websocket):
    name = await websocket.recv()
    print(f"<<< {name}")

    greeting = f"Hello {name}!"

    await websocket.send(greeting)
    print(f">>> {greeting}")

async def start_server():
    async with websockets.serve(hello, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        pass
