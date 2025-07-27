from celery.result import AsyncResult
from fastapi import FastAPI

# Import the Celery task we defined in worker.py
from worker import train_model_task

api = FastAPI()


@api.post("/train/", status_code=202)
async def start_training(ticker: str, num_days: int = 5000):
    """
    Accepts a training request, dispatches it to the Celery worker,
    and immediately returns a task ID.
    """
    print(
        f"API: Received training request for {ticker} for training on {num_days} days of data."
    )
    # .delay() is the magic that sends the task to the queue.
    # It doesn't run the function here; it just sends a message.
    task = train_model_task.delay(ticker=ticker, num_days=num_days)
    print(f"API: Dispatched task to worker. Task ID: {task.id}")
    return {"message": "Training task started.", "task_id": task.id}


@api.get("/train/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Allows the client to check the status of a previously started task.
    """
    task_result = AsyncResult(task_id, app=train_model_task.app)

    if task_result.ready():
        if task_result.successful():
            return {"status": "SUCCESS", "result": task_result.get()}
        else:
            # The result of a failed task is the exception that was raised.
            return {"status": "FAILURE", "error": str(task_result.result)}
    else:
        return {"status": "PENDING"}
