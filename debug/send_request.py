import requests
import argparse
import json

req = {
    "content": {
        "Content1":["How can I start programming from zero level?", "How do I configure a gmail account for company?"],
        "Content2":["How do I start learning programming language? Which one to start with?", "Why can't you have a gmail alias?"],
    }
}

def send_input_text(input_text, infer_url):
    response = requests.post(infer_url, json=req)
    return response.json()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i")
    parser.add_argument("--port", "-p", default=8080)
    args = parser.parse_args()

    server_url = 'http://127.0.0.1:{}/autotable/predict'.format(args.port)
    print(send_input_text(args.input, server_url))
