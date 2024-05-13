import argparse
import os
import time
import logging
import sys
from src.cluster import get_restart_index

def main():
    parser = argparse.ArgumentParser(description='Example command-line parser')

    parser.add_argument('--path', type=str, required=True, help='Path to output directory')
    parser.add_argument('--happy', action='store_true', default=False, help='Set this flag to disable failure mode')
    args = parser.parse_args()

    print(f"Training PID {os.getpid()}")

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s", level=logging.INFO
    )
    RANK = int(os.environ["RANK"])
    from supervisor import get_message_queue

    if RANK == 0:
        q = get_message_queue()

    restart_index = get_restart_index()
    print(restart_index)

    path = args.path
    with open(f"{path}/rank_{RANK}_output.txt", "a+") as f:
        f.write(f"rank {RANK} Training, attempt {restart_index}\n")

    # if RANK == 0:
    #     from supervisor import get_message_queue

    #     message_queue = get_message_queue()

    for i in range(10):
        logger.info(f"{RANK} Training {i}")

        # if RANK == 0:
        #     message_queue.send_pyobj(i)
        #     if i == 5:
        #         time.sleep(100)
        if not args.happy and RANK == 1 and i == 5:
            with open(f"/tmp/supervisor_reply_file_{RANK}", "w") as f:
                f.write('{"message": "hi"}\n')
            raise Exception("Rank 0 failure!")
        if RANK == 0:
            q.send_pyobj(i)
        time.sleep(1)

if  __name__ == '__main__':
    main()
