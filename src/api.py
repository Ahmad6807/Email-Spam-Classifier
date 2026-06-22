from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from classifier import predict_message

app = FastAPI(title='Spam Classifier API', version='1.0.0')


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    label: str
    confidence: float | None = None
    model: str
    is_spam: bool


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict', response_model=MessageResponse)
def predict(req: MessageRequest):
    result = predict_message(req.message)
    return MessageResponse(
        label=result['label'],
        confidence=result['confidence'],
        model=result['model'],
        is_spam=result['label'].lower() == 'spam',
    )


if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=True)
