import asyncio
import websockets
import json


def request_one_instrument(instrument = 'BTC-PERPETUAL'):


    msg =  {
  "jsonrpc" : "2.0",
  "id" : 9344,
  "method" : "public/get_book_summary_by_instrument",
  "params" : {
    "instrument_name" : "BTC-PERPETUAL",
    }
    }

    async def call_api(msg):
        async with websockets.connect('wss://www.deribit.com/ws/api/v2/') as websocket:
            await websocket.send(msg)
            while websocket.open:
                response = await websocket.recv()
                # do something with the response...
                print(response)

    asyncio.get_event_loop().run_until_complete(call_api(json.dumps(msg)))

request_one_instrument()