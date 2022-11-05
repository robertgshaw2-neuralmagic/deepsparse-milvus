import requests, argparse, uvicorn
from fastapi import FastAPI

parser = argparse.ArgumentParser(
    prog="app-frontend.py",
    description="Frontend Side of Toy Example Testing Out A VPC Deployment"
)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("backend_url", type=str)

def launch_app(backend_url, host, port):
    app = FastAPI()

    @app.get("/")
    def index_route():
        return "The correct path is '/test'"

    @app.get("/test")
    def test_route():

        print("Making request to app backend")
        response = requests.get(backend_url + "/backend-test")

        if response.status_code == 200:
            print("Request to app server succeeded")
            return f'Backend Response: {response.text}'
        else:
            print("Request to app server failed")
            return f'Backend Response: {response.text}'

    uvicorn.run(app=app, host=host, port=port, workers=1)

if __name__ == "__main__":
    args = parser.parse_args()
    launch_app(args.backend_url, args.host, args.port)