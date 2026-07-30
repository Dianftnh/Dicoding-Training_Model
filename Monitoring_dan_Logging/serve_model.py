import mlflow.pyfunc
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import json

MODEL_PATH = "C:/Users/use/Downloads/Dicoding-Training_Model/Membangun_model/mlruns/294663521044745902/models/m-6016bce7ec7542a3afbdf58789bf6ce7/artifacts"

print("Loading model...")
model = mlflow.pyfunc.load_model(MODEL_PATH)
print("Model loaded!")

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/invocations")
async def invocations(request: Request):
    body = await request.body()
    data = json.loads(body)
    instances = data.get("instances")
    if instances is not None:
        result = model.predict(np.array(instances))
        if hasattr(result, "tolist"):
            result = result.tolist()
        return JSONResponse(content={"predictions": result})
    df_records = data.get("dataframe_records")
    if df_records is not None:
        import pandas as pd
        df = pd.DataFrame(df_records)
        result = model.predict(df)
        if hasattr(result, "tolist"):
            result = result.tolist()
        return JSONResponse(content={"predictions": result})
    return JSONResponse(content={"error": "unsupported format"}, status_code=400)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
