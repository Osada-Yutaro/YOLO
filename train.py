import sys
import yolo

args = sys.argv
yolo.train.trainer.fit(args[1], args[2], int(args[3]), float(args[4]), int(args[5]))
