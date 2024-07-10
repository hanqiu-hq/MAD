import contextlib
import copy
import io
import itertools
import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.logger import create_small_table

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


class COCOCapEvaluator(DatasetEvaluator):
    def __init__(
            self,
            dataset_name,
            distributed=True,
            output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset.
            allow_cached_coco (bool): Whether to use cached coco json from previous validation
                runs. You should set this to False if you need to use different validation data.
                Defaults to True.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        assert hasattr(self._metadata, "json_file")

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'results')
            if not os.path.exists(self.output_dir) and comm.is_main_process():
                os.mkdir(self.output_dir)
        else:
            self.output_dir = None

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            assert "captions" in output
            prediction["caption"] = output["captions"]

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOCapEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "caption_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        self._eval_captions(predictions)

        return copy.deepcopy(self._results)

    def _eval_captions(self, predictions):

        coco_resutls = self._coco_api.loadRes(predictions)
        cocoEval = COCOEvalCap(self._coco_api, coco_resutls)
        cocoEval.evaluate()
        res_dict = cocoEval.eval

        self._logger.info(
            "Evaluation results for image caption: \n" + create_small_table(res_dict)
        )
        self._results["caption"] = res_dict
