import base64
import asyncio
import aiohttp
from loguru import logger


def to_b64(file):
    with open(file, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


async def do_parse(file, url, **kwargs):
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        data = {'file': to_b64(file), 'kwargs': kwargs}
        async with session.post(url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result | {'file_path': file}
            else:
                raise Exception(await response.text())


async def worker(url, queue, **kwargs):
    while not queue.empty():
        try:
            file = queue.get_nowait()
            info = f'File: {file} - Info: '
            result = await do_parse(file, url, **kwargs)
            logger.success(info + result['output_dir'])
        except Exception as e:
            logger.error(info + str(e))
        finally:
            queue.task_done()


async def run_tasks(files, urls, workers_per_url):
    queue = asyncio.Queue()
    for file in files:
        queue.put_nowait(file)

    max_workers = min(len(files), workers_per_url * len(urls))
    workers_per_url = max(1, max_workers // len(urls))

    workers = []
    for url in urls:
        for _ in range(workers_per_url):
            workers.append(worker(url, queue))

    await asyncio.gather(*workers)


async def main():
    urls = [
        'http://127.0.0.1:8000/predict',
    ]

    files = [
        'demo/small_ocr.pdf',
    ]

    await run_tasks(files, urls, workers_per_url=16)


if __name__ == '__main__':
    logger.add('{time:%Y%m%d_%H%M%S}.log')
    asyncio.run(main())
