import gzip
import io
import logging
import os
import pickle
import shutil
import tempfile
from abc import ABC

from ray.tune import Trainable
from ray.tune.logger import TFLogger, UnifiedLogger

logger = logging.getLogger(__name__)


class ExtendedTrainable(Trainable, ABC):
    """
    A trainable that is compatible with tensorflow saving
    """

    # noinspection PyProtectedMember
    def find_tf_logger(self):
        if isinstance(self._result_logger, UnifiedLogger):
            for logger in self._result_logger._loggers:
                if isinstance(logger, TFLogger):
                    return logger._file_writer
        return None

    def save(self, checkpoint_dir=None):
        """Saves the current model state to a checkpoint.

        Subclasses should override ``_save()`` instead to save state.
        This method dumps additional metadata alongside the saved path.

        Args:
            checkpoint_dir (str): Optional dir to place the checkpoint.

        Returns:
            Checkpoint path that may be passed to restore().
        """

        checkpoint_dir = os.path.join(checkpoint_dir or self.logdir, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint = self._save(checkpoint_dir)
        saved_as_dict = False
        if isinstance(checkpoint, str):
            if (not checkpoint.startswith(checkpoint_dir)
                    or checkpoint == checkpoint_dir):
                raise ValueError(
                    "The returned checkpoint path must be within the "
                    "given checkpoint dir {}: {}".format(
                        checkpoint_dir, checkpoint))
            if not os.path.exists(checkpoint):
                raise ValueError(
                    "The returned checkpoint path does not exist: {}".format(
                        checkpoint))
            checkpoint_path = checkpoint
        elif isinstance(checkpoint, dict):
            saved_as_dict = True
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)
        else:
            raise ValueError(
                "`_save` must return a dict or string type: {}".format(
                    str(type(checkpoint))))
        with open(checkpoint_path + ".tune_metadata", "wb") as f:
            pickle.dump({
                "experiment_id": self._experiment_id,
                "iteration": self._iteration,
                "timesteps_total": self._timesteps_total,
                "time_total": self._time_total,
                "episodes_total": self._episodes_total,
                "saved_as_dict": saved_as_dict
            }, f)
        return checkpoint_path

    def save_to_object(self):
        """Saves the current model state to a Python object. It also
        saves to disk but does not return the checkpoint path.

        Returns:
            Object holding checkpoint data.
        """

        tmpdir = tempfile.mkdtemp("save_to_object", dir=self.logdir)
        checkpoint_prefix = self.save(tmpdir)

        data = {}
        base_dir = os.path.dirname(checkpoint_prefix)
        for path in os.listdir(base_dir):
            path = os.path.join(base_dir, path)
            if path.startswith(checkpoint_prefix):
                with open(path, "rb") as file:
                    data[os.path.basename(path)] = file.read()

        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode="wb") as f:
            compressed = pickle.dumps({
                "checkpoint_name": os.path.basename(checkpoint_prefix),
                "data": data,
            })
            if len(compressed) > 10e6:  # getting pretty large
                logger.info("Checkpoint size is {} bytes".format(
                    len(compressed)))
            f.write(compressed)

        shutil.rmtree(tmpdir)
        return out.getvalue()
