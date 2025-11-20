from RangeHTTPServer import RangeRequestHandler
from http.server import HTTPServer


class CORSRangeRequestHandler(RangeRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "OK")
        self.end_headers()


if __name__ == "__main__":
    port = 8000
    server = HTTPServer(("localhost", port), CORSRangeRequestHandler)
    print(f"Serving with Range support on http://localhost:{port}")
    server.serve_forever()
