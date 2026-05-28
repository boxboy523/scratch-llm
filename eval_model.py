import torch
from data.dataset import get_tokenizer, KoIterableDataset
from model.lm import KoLLM
from train.checkpoint import load_checkpoint
import config

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))

def token_indices_to_string(tokenizer, indices):
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    tokens = tokenizer.convert_ids_to_tokens(indices)
    text_bytes = bytes([byte_decoder.get(c, ord(c)) for c in "".join(tokens)])
    return text_bytes.decode("utf-8", errors="ignore")
device = torch.device("cuda")
tokenizer = get_tokenizer(config.TOKENIZER_DIR)
model = KoLLM(
            vocab_size=config.VOCAB_SIZE,
            context_len=config.CONTEXT_LEN,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            n_kv_heads=config.N_KV_HEADS,
            n_layers=config.N_LAYERS,
            d_ffn=config.D_FFN
        )
checkpoint = torch.load("artifacts/checkpoints/checkpoint_step_1000.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

dataset = KoIterableDataset(tokenizer, config.CONTEXT_LEN)
batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))

inputs = batch[:, :-1].to(device)
with torch.no_grad():
    logits, _ = model(inputs)
    pred = logits.argmax(-1)

print("IN :", token_indices_to_string(tokenizer, inputs[0].tolist()))
print("OUT:", token_indices_to_string(tokenizer, pred[0].tolist()))
