"""Application-wide serial task queue with cooperative cancellation.

Non-interruptible library calls finish before shutdown; threads are never killed.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock


class TaskManager:
    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-task")
        self._lock = Lock()
        self._jobs = {}
        self._closing = False

    @property
    def busy(self):
        with self._lock:
            return bool(self._jobs)

    def submit(self, owner, target, args=(), on_cancel=None):
        token = Event()
        with self._lock:
            if self._closing:
                raise RuntimeError("任务管理器正在关闭")
            if owner in self._jobs:
                raise RuntimeError("该页面仍有任务未结束")
            self._jobs[owner] = token
        owner.status_changed.emit("任务已排队，等待其他后台任务结束")

        def run():
            try:
                if token.is_set():
                    if on_cancel:
                        on_cancel()
                    return
                owner.status_changed.emit("后台任务开始")
                target(*args)
            finally:
                with self._lock:
                    self._jobs.pop(owner, None)
        self._pool.submit(run)

    def cancel(self, owner):
        with self._lock:
            token = self._jobs.get(owner)
            if token:
                token.set()
        event = getattr(owner, "_stop_event", None)
        if event is not None:
            event.set()

    def shutdown(self):
        with self._lock:
            self._closing = True
            owners = list(self._jobs)
        for owner in owners:
            self.cancel(owner)
        self._pool.shutdown(wait=False)


TASKS = TaskManager()
