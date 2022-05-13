import torch
import json
import torch.backends.cudnn as cudnn
from datetime import datetime

from config import args
from trainer import Trainer
from logger_config import logger


def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    start_time = datetime.now().strftime("%d.%m.%Y_%H:%M")
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    with open(f'{args.model_dir}/SimKGC_args_{start_time}.json', 'w') as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent = 4)
    f.close()
    trainer.train_loop()
    logger.info('Done with training.')


if __name__ == '__main__':
    main()
