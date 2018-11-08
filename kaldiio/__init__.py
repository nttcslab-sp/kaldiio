import pkg_resources

from .arkio import load_ark
from .arkio import load_scp
from .arkio import load_wav_scp
from .arkio import save_ark
from .arkio import load_mat
from .arkio import save_mat
from .utils import open_like_kaldi
from .highlevel import parse_specifier
from .highlevel import WriteHelper
from .highlevel import ReadHelper

try:
    __version__ = pkg_resources.get_distribution('kaldiio').version
except:
    __version__ = None
del pkg_resources
