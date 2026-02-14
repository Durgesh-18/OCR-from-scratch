import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from ocr import OCRNeuralNetwork

PORT = 8000
nn = OCRNeuralNetwork(use_file=True)

class OCRRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            content = self.rfile.read(content_length)
            payload = json.loads(content.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {
                "type": "error",
                "message": f"Invalid JSON: {str(e)}"
            })
            return

        # ---- TRAIN ----
        if payload.get("train"):
            try:
                nn.train(payload["trainArray"])
                nn.save()
                self._send_json(200, {
                    "type": "train",
                    "status": "ok"
                })
            except Exception as e:
                self._send_json(500, {
                    "type": "error",
                    "message": f"Training failed: {str(e)}"
                })

        # ---- PREDICT ----
        elif payload.get("predict"):
            try:
                result = nn.predict(payload["image"])
                self._send_json(200, {
                    "type": "test",
                    "result": result
                })
            except Exception as e:
                self._send_json(500, {
                    "type": "error",
                    "message": f"Prediction failed: {str(e)}"
                })

        # ---- INVALID ----
        else:
            self._send_json(400, {
                "type": "error",
                "message": "Invalid request"
            })

    def _send_json(self, status_code, obj):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

def run():
    server = HTTPServer(("", PORT), OCRRequestHandler)
    print(f"Server running on port {PORT}")
    server.serve_forever()

if __name__ == "__main__":
    run()
