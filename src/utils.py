import re


def parse_checkpoint(filename):
    match = re.match(r'epoch=(\d+)-step=(\d+)\.ckpt', filename)
    if match:
        epoch, step = map(int, match.groups())
        return epoch, step
    return -1, -1
