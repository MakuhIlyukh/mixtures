import mlflow
from mlflow.utils.git_utils import get_git_commit
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT


def set_commit_tag():
    """ Sets commit tag under the current run.
    
    By default, mlflow doesn't set commit tag in interactive mode
    (jupyter notebooks or ipython session)
    """
    comm_hash = get_git_commit(".")
    mlflow.set_tag(MLFLOW_GIT_COMMIT, comm_hash)