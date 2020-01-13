import os
from pyleaves.utils import ensure_dir_exists


class Config:
    """Config class with global project variables."""

    def __init__(self, dataset_name='Fossil', local_tfrecords=None, **kwargs):
        """Global config file for normalization experiments."""
        self.dataset_name = 'Fossil'
        self.project_directory = '/media/data/jacob/fossil_experiments/'
        self.tfrecords = os.path.join(
            self.project_directory,
            'tf_records')  # Alternative, slow I/O path for tfrecords.
        self.local_tfrecords = local_tfrecords or self.tfrecords  # Primary path for tfrecords.
        self.checkpoints = os.path.join(
            self.project_directory,
            'checkpoints')
        self.summaries = os.path.join(
            self.project_directory,
            'summaries')
        self.experiment_evaluations = os.path.join(
            self.project_directory,
            'experiment_evaluations')
        self.plots = os.path.join(
            self.project_directory,
            'plots')
        self.results = 'results'
        self.log_dir = os.path.join(self.project_directory, 'logs')

        # Create directories if they do not exist
        check_dirs = [
            self.tfrecords,
            self.local_tfrecords,
            self.checkpoints,
            self.summaries,
            self.experiment_evaluations,
            self.plots,
            self.log_dir,
        ]
        [ensure_dir_exists(x) for x in check_dirs]
        
        self.seed = 1085

    def __getitem__(self, name):
        """Get item from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains field."""
        return hasattr(self, name)