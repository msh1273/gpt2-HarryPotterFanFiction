from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from flask import Flask, request, Response, jsonify
from torch.nn import functional as F
from queue import Queue, Empty
import time
import threading

# Server & Handling Setting
app = Flask(__name__)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

tokenizer = AutoTokenizer.from_pretrained("ceostroff/harry-potter-gpt2-fanfiction")
model = AutoModelWithLMHead.from_pretrained("ceostroff/harry-potter-gpt2-fanfiction", return_dict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Queue 핸들링
def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in requests_batch:
                if len(requests['input']) == 2 :
                    requests['output'] = run_word(requests['input'][0], requests['input'][1])
                elif len(requests['input']) == 3 :
                    requests['output'] = run_generation(requests['input'][0], requests['input'][1], requests['input'][2])

# 쓰레드
threading.Thread(target=handle_requests_by_batch).start()


def run_generation(sequence, num_samples, length):
    try:
        sequence = sequence.strip()
        input_ids = tokenizer.encode(sequence, return_tensors='pt')

        # input_ids also need to apply gpu device!
        input_ids = input_ids.to(device)

        min_length = len(input_ids.tolist()[0])
        length += min_length

        sample_outputs = model.generate(input_ids, pad_token_id=50256,
                                        do_sample=True,
                                        max_length=length,
                                        min_length=length,
                                        top_k=40,
                                        num_return_sequences=num_samples)

        result = dict()
        for idx, sample_output in enumerate(sample_outputs):
            result[idx] = tokenizer.decode(sample_output.tolist()[min_length:], skip_special_tokens=True)

        return result

    except Exception as e:
        print(e)
        return 500


# Running GPT-2
def run_word(sequence, num_samples):
    try:
        input_ids = tokenizer.encode(sequence, return_tensors="pt")
        tokens_tensor = input_ids.to(device)

        # get logits of last hidden state
        next_token_logits = model(tokens_tensor).logits[:, -1, :]
        # filter
        filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
        # sample
        probs = F.softmax(filtered_next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=num_samples)

        result = dict()
        for idx, token in enumerate(next_token.tolist()[0]):
            result[idx] = tokenizer.decode(token)

        return result

    except Exception as e:
        print(e)
        return 500


@app.route("/gpt2-harrypotter/<type>", methods=['POST'])
def gpt2_harrypotter(type):
    if type != 'short' and type != 'long' :
        return jsonify({'error': 'This is the wrong address.'}), 400

    # 큐에 쌓여있을 경우,
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'Too Many Requests'}), 429

    try:
        args = []
        text = request.form['text']
        num_samples = int(request.form['num_samples'])

        args.append(text)
        args.append(num_samples)

        if type == 'long':
            length = int(request.form['length'])
            args.append(length)

    except Exception:
        print("Empty Text")
        return Response("fail", status=400)

    # Queue - put data
    req = {
        'input': args
    }
    requests_queue.put(req)

    # Queue - wait & check
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return req['output']


# Health Check
@app.route("/healthz", methods=["GET"])
def healthCheck():
    return "", 200


if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)