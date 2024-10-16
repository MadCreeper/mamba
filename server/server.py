import argparse
import time
import json
import random
import asyncio
from threading import Thread
import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers.generation import TextStreamer

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel




# model initialize

# parser = argparse.ArgumentParser(description="Generation benchmarking")
# parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
# parser.add_argument("--prompt", type=str, default=None)
# parser.add_argument("--promptlen", type=int, default=100)
# parser.add_argument("--genlen", type=int, default=100)
# parser.add_argument("--temperature", type=float, default=1.0)
# parser.add_argument("--topk", type=int, default=1)
# parser.add_argument("--topp", type=float, default=1.0)
# parser.add_argument("--minp", type=float, default=0.0)
# parser.add_argument("--repetition-penalty", type=float, default=1.0)
# parser.add_argument("--batch", type=int, default=1)
# args = parser.parse_args()
class DefaultArgs(BaseModel):
    model_name: str = "state-spaces/mamba-2.8b"
    promptlen: int = 100
    genlen: int = 100
    temperature: float = 1.0
    topk: int = 1
    topp: float = 1.0
    minp: float = 0.0
    repetition_penalty: float = 1.0
    batch: int = 1
    seed: int = 42

args = DefaultArgs()

repeats = 1
device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba") or args.model_name.startswith("state-spaces/transformerpp")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# streamer = TextStreamer(tokenizer)
class AsyncTextStreamer(TextStreamer):
    def __init__(self, tokenizer, output_stream, event_loop):
        super().__init__(tokenizer)
        self.output_stream = output_stream
        self.loop = event_loop

    def on_finalized_text(self, text: str, stream_end: bool = False):
        super().on_finalized_text(text, stream_end)
        self.loop.call_soon_threadsafe(self.loop.create_task, self.output_stream.put(text))


# Define FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL if hosted on a different domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Request schema using Pydantic
class GenerateRequest(BaseModel):
    input_text: str = "Once upon a time"
    promptlen: int = 100
    genlen: int = 100
    max_length: int = 100
    temperature: float = 1.0
    topk: int = 1
    topp: float = 1.0
    minp: float = 0.0
    repetition_penalty: float = 1.0
    seed: int = 42


async def fake_data_streamer():
    for i in range(10):
        yield b'some fake data\n\n'
        await asyncio.sleep(0.5)

@app.post("/generate/")
async def generate_text(request: GenerateRequest):
    # Set the seed for reproducibility
    torch.manual_seed(request.seed)
    random.seed(request.seed)

    # Encode the input

    if request.input_text is None:
        input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
    else:
        tokens = tokenizer(request.input_text, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        attn_mask = tokens.attention_mask.to(device=device)

    # Create an async queue for streaming tokens
    output_stream = asyncio.Queue()
    loop = asyncio.get_running_loop()
    streamer = AsyncTextStreamer(tokenizer, output_stream, loop)

    max_length = input_ids.shape[1] + request.genlen
        # Generate the text synchronously but stream it asynchronously

    # Run model.generate() in a non-blocking manner using a thread
    def run_generation():
        model.generate(
            input_ids=input_ids,
            max_length=request.max_length,
            top_k=request.topk,
            top_p=request.topp,
            temperature=request.temperature,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=request.repetition_penalty,
            streamer=streamer  # Pass the async streamer
        )
        # Indicate that the generation is complete by putting None into the queue
        asyncio.run(output_stream.put(None))

    # Start the generation in a background thread
    thread = Thread(target=run_generation)
    thread.start()
    stop_word = tokenizer.decode(tokenizer.eos_token_id)
    # Asynchronous generator to yield tokens
    async def async_word_generator():
        while True:
            token = await output_stream.get()
            if token == stop_word or token is None:  # Check if generation is finished
                break
            yield token

    # Return StreamingResponse
    return StreamingResponse(async_word_generator(), media_type='text/event-stream')


