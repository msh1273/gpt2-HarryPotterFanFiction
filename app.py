# External module.
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
from flask import Flask, request, Response, jsonify, render_template
import torch
from torch.nn import functional as F

# Internal module.
from queue import Queue, Empty
import threading
import time

app = Flask(__name__)

# tokenizer and model loading.
tokenizer = AutoTokenizer.from_pretrained("ceostroff/harry-potter-gpt2-fanfiction")
model = AutoModelForCausalLM.from_pretrained("ceostroff/harry-potter-gpt2-fanfiction", return_dict=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # gpu check.
model.to(device)

requests_queue = Queue()    # request queue.
BATCH_SIZE = 1              # max request size.
CHECK_INTERVAL = 0.1


##
# Request handler.
# GPU app can process only one request in one time.
def handle_requests_by_batch():
    while True:
        request_batch = []

        while not (len(request_batch) >= BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in request_batch:
                if len(requests['input']) == 2:
                    requests["output"] = run_short(requests['input'][0], requests['input'][1])
                elif len(requests['input']) == 3:
                    requests["output"] = run_long(requests['input'][0], requests['input'][1], requests['input'][2])


handler = threading.Thread(target=handle_requests_by_batch).start()


##
# short gpt-2 text generation.
# Generate one word.
def run_short(text, samples):
    try:
        input_ids = tokenizer.encode(text, return_tensors='pt')

        input_ids = input_ids.to(device)

        next_token_logits = model(input_ids).logits[:, -1, :]

        filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

        probs = F.softmax(filtered_next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=samples)

        result = dict()

        for idx, token in enumerate(next_token.tolist()[0]):
            result[idx] = tokenizer.decode(token)

        return result

    except Exception as e:
        print('Error occur in short generating!', e)
        return jsonify({'error': e}), 500


##
# long gpt-2 text generation.
# Generate long sting as size of 'length'.
def run_long(sequence, num_samples, length):
    try:
        sequence = sequence.strip()
        input_ids = tokenizer.encode(sequence, return_tensors='pt')

        # input_ids also need to apply gpu device!
        input_ids = input_ids.to(device)

        min_length = len(input_ids.tolist()[0])
        length += min_length

        length = length if length > 20 else 20

        # model generating
        sample_outputs = model.generate(input_ids, pad_token_id=50256,
                                        do_sample=True,
                                        max_length=length,
                                        min_length=min_length,
                                        top_k=40,
                                        num_return_sequences=num_samples)

        result = dict()

        for idx, sample_output in enumerate(sample_outputs):
            result[idx] = tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)

        return result

    except Exception as e:
        print('Error occur in long generating!', e)
        return jsonify({'error': e}), 500


##
# Get post request page.
@app.route('/GPT2-HarryPotterFF/<types>', methods=['POST'])
def generate(types):
    # type checking.
    if types != 'short' and types != 'long':
        return jsonify({'Error': 'The wrong address.'}), 400

    # GPU app can process only one request in one time.
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'Error': 'Too Many Requests'}), 429

    try:
        args = []

        text = request.form['text']
        samples = int(request.form['samples'])

        args.append(text)
        args.append(samples)

        if types == 'long':
            length = int(request.form['length'])
            args.append(length)

    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    # input a request on queue
    req = {'input': args}
    requests_queue.put(req)

    # wait
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return jsonify(req['output'])


##
# Queue deadlock error debug page.
@app.route('/queue_clear')
def queue_clear():
    with requests_queue.mutex:
        requests_queue.queue.clear()

    return "Clear", 200


##
# Sever health checking page.
@app.route('/healthz', methods=["GET"])
def health_check():
    return "Health", 200


##
# Main page.
@app.route('/')
def main():
    return render_template('index.html'), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)