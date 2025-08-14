import asyncio
import time

async def test_math():
    # time.sleep(1) for sync func
    await asyncio.sleep(1)
    return 2+3

async def test_english():

    # time.sleep(2)
    await asyncio.sleep(2)
    return "Welcome to parallelization"

# global interpreter log, at one time one run
async def main():
    start_time = time.time()
    #test_english()
    #test_math()
    await test_english()
    await test_math()
    #await asyncio.gather(test_math(),test_english())
    end_time = time.time()
    print(f"time taken: {end_time-start_time} seconds")

if __name__ == '__main__':
    asyncio.run(main())