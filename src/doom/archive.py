"""Keep the full multiday audit in hourly compressed files, without rotation loss."""
import gzip
import logging
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path


class AuditArchive(logging.Handler):
    def __init__(self,directory,run_id):
        super().__init__();self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.run_id=run_id;self.hour=None;self.stream=None;self.last_flush=0

    def emit(self,record):
        hour=datetime.now(timezone.utc).strftime('%Y%m%dT%H')
        if hour!=self.hour:
            if self.stream:self.stream.close()
            if shutil.disk_usage(self.directory).free<1024**3:raise RuntimeError('Less than 1 GiB free for scientific audit')
            self.stream=gzip.open(self.directory/f'{hour}-{self.run_id}.jsonl.gz','at',encoding='utf-8')
            self.hour=hour
        self.stream.write(record.getMessage()+'\n')
        if time.monotonic()-self.last_flush>10:
            self.stream.flush();self.last_flush=time.monotonic()

    def close(self):
        if self.stream:self.stream.close();self.stream=None
        super().close()
