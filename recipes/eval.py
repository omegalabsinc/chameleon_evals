import sys
import time
import argparse
import logging
from typing import Dict, List, Tuple
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

import lmms_eval
from lmms_eval.evaluator import evaluate
from lmms_eval import utils as lmms_utils
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model

import torch

from chameleon.inference.chameleon import ChameleonInferenceModel

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())
level = getattr(logging, "DEBUG")
logger.setLevel(level)


@register_model("chameleon")
class EvalWrapper(lmms):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.model = ChameleonInferenceModel(
            cfg.model.model_path,
            cfg.model.tokenizer_path,
            cfg.model.vqgan_config_path,
            cfg.model.vqgan_ckpt_path,
        )

    # not implemented
    @torch.no_grad()
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float | bool]]:
        return super().loglikelihood(requests)

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]
    
        re_ords = lmms_utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits = zip(*chunk)
            breakpoint()


        return None
    


class MMEvalRecipe():
    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._limit = self._cfg.limit
        self._tasks = list(self._cfg.tasks)
        self._model = EvalWrapper(cfg)

    @torch.no_grad()
    def evaluate(self) -> None:
        t1 = time.time()
        # Task initialization API changed between v0.4.1 and 0.4.2
        try:
            lmms_eval.tasks.initialize_tasks()
        except Exception:
            pass

        logger.info(f"Running evaluation on {self._tasks} tasks.")

        task_dict = lmms_eval.tasks.get_task_dict(self._tasks, model_name="chameleon")
        for task_name in task_dict.keys():
            task_obj = task_dict[task_name]
            if type(task_obj) == tuple:
                group, task_obj = task_obj
                if task_obj is None:
                    continue
            self._model.task_dict[task_name] = task_obj.dataset

        eleuther_output = evaluate(
            lm=self._model,
            task_dict=task_dict,
            limit=self._limit,
            cli_args=self._cfg.mm_eval_args
        )

        for task_name in self._tasks:
            logger.info(f'Incorrect answers for {task_name}')
            for result in [r for r in eleuther_output['samples'][task_name] if r['exact_match'] != 1.0]:
                logger.info(f"{result['doc_id']}:6d] {result['filtered_resps'][0]:10s} != {result['target']}")

        logger.info(f"Eval completed in {time.time() - t1:.02f} seconds.")
        for task, res in eleuther_output["results"].items():
            logger.info(f"{task}: {res}")
        with open('mmeval.txt', 'a') as f:
            for task, res in eleuther_output["results"].items():
                f.write(f"{task}: {res}\n")


def recipe_main(cfg: DictConfig) -> None:
    """Entry point for the recipe."""
    recipe = MMEvalRecipe(cfg=cfg)
    recipe.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMEval Recipe")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    sys.exit(recipe_main(cfg=cfg))