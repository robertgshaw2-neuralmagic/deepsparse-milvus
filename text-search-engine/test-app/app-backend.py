import requests, argparse, uvicorn
from fastapi import FastAPI

parser = argparse.ArgumentParser(
    prog="app-backend.py",
    description="Backend Side of Toy Example Testing Out A VPC Connection"
)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=5001)

def launch_app(host, port):
    app = FastAPI()

    @app.get("/backend-test")
    def test_route():
        print("Got request from frontend.")
        return "Hello from the backend!"
        
    uvicorn.run(app=app, host=host, port=port, workers=1)

if __name__ == "__main__":
    args = parser.parse_args()
    launch_app(args.host, args.port)