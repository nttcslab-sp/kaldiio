# flake8: noqa F401
import pkg_resources

from kaldiio.matio import load_ark
from kaldiio.matio import load_mat
from kaldiio.matio import load_scp
from kaldiio.matio import load_scp_sequential
from kaldiio.matio import load_wav_scp
from kaldiio.matio import save_ark
from kaldiio.matio import save_mat
from kaldiio.highlevel import ReadHelper
from kaldiio.highlevel import WriteHelper
from kaldiio.utils import open_like_kaldi
from kaldiio.utils import parse_specifier

try:
    __version__ = pkg_resources.get_distribution('kaldiio').version
except Exception:
    __version__ = None
del pkg_resources
