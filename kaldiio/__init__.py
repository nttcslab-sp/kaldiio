# flake8: noqa F401
import pkg_resources

from .arkio import load_ark
from .arkio import load_mat
from .arkio import load_scp
from .arkio import load_wav_scp
from .arkio import save_ark
from .arkio import save_mat
from .highlevel import parse_specifier
from .highlevel import ReadHelper
from .highlevel import WriteHelper
from .utils import open_like_kaldi

try:
    __version__ = pkg_resources.get_distribution('kaldiio').version
except Exception:
    __version__ = None
del pkg_resources
